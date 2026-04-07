#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

/*
 * server.c
 *
 * Protocol summary (all messages are length-prefixed binary messages):
 *   header = 4 uint32_t fields in network byte order:
 *     type, payload_len, round, worker_id
 *
 * Message types:
 *   MSG_HELLO    : client -> server, empty payload
 *   MSG_ASSIGN   : server -> client, payload = assign_payload_t
 *   MSG_READY    : client -> server, empty payload
 *   MSG_WEIGHTS  : server -> client, payload = float[dim]
 *   MSG_GRADIENT : client -> server, payload = float[dim]
 *   MSG_FINISH   : server -> client, payload = float[dim]
 *   MSG_ERROR    : either direction, payload = UTF-8/ASCII error string
 *
 * Assumption: all processes run on the same architecture, so float arrays are
 * sent as raw IEEE-754 bytes. For a more portable version, serialize floats as
 * uint32_t bit patterns in network byte order.
 */

#define MAX_CLIENTS   64
#define INBUF_SIZE    (1024*1024)
#define OUTBUF_SIZE   (1024*1024)
#define BACKLOG       16

#define MSG_HELLO     1u
#define MSG_ASSIGN    2u
#define MSG_READY     3u
#define MSG_WEIGHTS   4u
#define MSG_GRADIENT  5u
#define MSG_FINISH    6u
#define MSG_ERROR     7u

#define INVALID_FD    (-1)

typedef struct {
    uint32_t type;
    uint32_t payload_len;
    uint32_t round;
    uint32_t worker_id;
} msg_header_t;

typedef struct {
    uint32_t worker_id;
    uint32_t dim;
    uint32_t max_rounds;
    uint32_t reserved;
    float learning_rate;
} assign_payload_t;

typedef struct {
    int fd;
    bool active;
    bool got_hello;
    bool ready;
    bool got_gradient;
    uint32_t worker_id;

    unsigned char inbuf[INBUF_SIZE];
    size_t in_used;

    unsigned char outbuf[OUTBUF_SIZE];
    size_t out_used;
    size_t out_sent;

    float *gradient;
} client_t;

typedef struct {
    int listen_fd;
    uint16_t port;
    int expected_workers;
    int dim;
    float learning_rate;
    int max_rounds;

    bool training_started;
    bool training_finished;
    uint32_t round;

    float *weights;
    client_t clients[MAX_CLIENTS];
    uint32_t next_worker_id;
} server_state_t;

static void die(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

static int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL);
    if (flags == -1) {
        return -1;
    }
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
        return -1;
    }
    return 0;
}

static void init_client(client_t *c) {
    c->fd = INVALID_FD;
    c->active = false;
    c->got_hello = false;
    c->ready = false;
    c->got_gradient = false;
    c->worker_id = UINT32_MAX;
    c->in_used = 0;
    c->out_used = 0;
    c->out_sent = 0;
    c->gradient = NULL;
}

static void close_client(client_t *c) {
    if (c->fd != INVALID_FD) {
        close(c->fd);
    }
    free(c->gradient);
    init_client(c);
}

static int active_client_count(const server_state_t *s) {
    int count = 0;
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (s->clients[i].active) {
            count++;
        }
    }
    return count;
}

static int ready_client_count(const server_state_t *s) {
    int count = 0;
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (s->clients[i].active && s->clients[i].ready) {
            count++;
        }
    }
    return count;
}

static int gradient_count_for_round(const server_state_t *s) {
    int count = 0;
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (s->clients[i].active && s->clients[i].ready && s->clients[i].got_gradient) {
            count++;
        }
    }
    return count;
}

static int participating_worker_count(const server_state_t *s) {
    int count = 0;
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (s->clients[i].active && s->clients[i].ready) {
            count++;
        }
    }
    return count;
}

static int queue_bytes(client_t *c, const void *data, size_t len) {
    size_t pending = c->out_used - c->out_sent;
    if (pending + len > OUTBUF_SIZE) {
        return -1;
    }

    if (pending > 0 && c->out_sent > 0) {
        memmove(c->outbuf, c->outbuf + c->out_sent, pending);
    }
    c->out_used = pending;
    c->out_sent = 0;

    memcpy(c->outbuf + c->out_used, data, len);
    c->out_used += len;
    return 0;
}

static int queue_message(client_t *c,
                         uint32_t type,
                         uint32_t round,
                         uint32_t worker_id,
                         const void *payload,
                         uint32_t payload_len) {
    msg_header_t hdr;
    hdr.type = htonl(type);
    hdr.payload_len = htonl(payload_len);
    hdr.round = htonl(round);
    hdr.worker_id = htonl(worker_id);

    if (queue_bytes(c, &hdr, sizeof(hdr)) == -1) {
        return -1;
    }
    if (payload_len > 0 && payload != NULL) {
        if (queue_bytes(c, payload, payload_len) == -1) {
            return -1; 
        }
    }
    return 0;
}

static int queue_error(client_t *c, const char *msg) {
    return queue_message(c, MSG_ERROR, 0, 0, msg, (uint32_t) strlen(msg) + 1u);
}

static void flush_client_output(client_t *c) {
    while (c->out_sent < c->out_used) {
        ssize_t n = send(c->fd,
                         c->outbuf + c->out_sent,
                         c->out_used - c->out_sent,
                         0);
        if (n > 0) {
            c->out_sent += (size_t) n;
        } else if (n == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return;
        } else {
            c->active = false;
            return;
        }
    }

    c->out_used = 0;
    c->out_sent = 0;
}

static int create_listen_socket(uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == -1) {
        return -1;
    }

    int one = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) == -1) { //quick restart after server exit
        close(fd);
        return -1;
    }

    if (set_nonblocking(fd) == -1) {
        close(fd);
        return -1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    if (bind(fd, (struct sockaddr *) &addr, sizeof(addr)) == -1) {
        close(fd);
        return -1;
    }

    if (listen(fd, BACKLOG) == -1) {
        close(fd);
        return -1;
    }

    return fd;
}

static client_t *alloc_client_slot(server_state_t *s) {
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (!s->clients[i].active) {
            return &s->clients[i];
        }
    }
    return NULL;
}

static int all_expected_workers_ready(const server_state_t *s) {
    return ready_client_count(s) == s->expected_workers;
}

static int all_gradients_arrived(const server_state_t *s) {
    int participants = participating_worker_count(s);
    return participants > 0 && gradient_count_for_round(s) == participants;
}

static void broadcast_weights(server_state_t *s) {
    uint32_t payload_len = (uint32_t) (s->dim * (int) sizeof(float));

    for (int i = 0; i < MAX_CLIENTS; i++) {
        client_t *c =&s->clients[i];
        if (!c->active || !c->ready) {
            continue;
        }
        c->got_gradient = false;
        if (queue_message(c,
                          MSG_WEIGHTS,
                          s->round,
                          c->worker_id,
                          s->weights,
                          payload_len) == -1) {
            fprintf(stderr, "server: output buffer overflow for worker %u\n", c->worker_id);
            c->active = false;
        }
    }

    printf("server: broadcast weights for round %u\n", s->round);
}

static void broadcast_finish(server_state_t *s) {
    uint32_t payload_len = (uint32_t) (s->dim * (int) sizeof(float));
    for (int i = 0; i < MAX_CLIENTS; i++) {
        client_t *c = &s->clients[i];
        if (!c->active || !c->ready) {
            continue;
        }
        if (queue_message(c,
                          MSG_FINISH,
                          s->round,
                          c->worker_id,
                          s->weights,
                          payload_len) == -1) {
            c->active = false;
        }
    }
}

static void aggregate_and_update(server_state_t *s) {
    int participants = participating_worker_count(s);
    if (participants <= 0) {
        return;
    }

    float *avg = calloc((size_t) s->dim, sizeof(float));
    if (avg == NULL) {
        die("calloc avg");
    }

    for (int i = 0; i < MAX_CLIENTS; i++) {
        client_t *c = &s->clients[i];
        if (!c->active || !c->ready || !c->got_gradient) {
            continue;
        }
        for (int j = 0; j < s->dim; j++) {
            avg[j] += c->gradient[j];
        }
    }

    for (int j = 0; j < s->dim; j++) {
        avg[j] /= (float) participants;
        s->weights[j] -= s->learning_rate * avg[j];
    }

    free(avg);

    s->round++;
    printf("server: completed round %u\n", s->round);

    if ((int) s->round >= s->max_rounds) {
        s->training_finished = true;
        broadcast_finish(s);
        printf("server: training finished\n");
    } else {
        broadcast_weights(s);
    }
}

static void maybe_start_or_advance_training(server_state_t *s) {
    if (!s->training_started) {
        if (all_expected_workers_ready(s)) {
            s->training_started = true;
            s->round = 0;
            printf("server: all %d workers ready, starting training\n", s->expected_workers);
            broadcast_weights(s);
        }
        return;
    }

    if (!s->training_finished && all_gradients_arrived(s)) {
        aggregate_and_update(s);
    }
}

static void remove_dead_clients(server_state_t *s) {
    for (int i = 0; i < MAX_CLIENTS; i++) {
        client_t *c = &s->clients[i];
        if (!c->active && c->fd != INVALID_FD) {
            printf("server: removing worker %u\n", c->worker_id);
            close_client(c);
        }
    }
}

static void handle_disconnect(server_state_t *s, client_t *c) {
    fprintf(stderr, "server: worker %u disconnected\n", c->worker_id);
    c->active = false;

    if (!s->training_started) {
        return;
    }

    if (participating_worker_count(s) == 0) {
        s->training_finished = true;
        return;
    }

    if (!s->training_finished && all_gradients_arrived(s)) {
        aggregate_and_update(s);
    }
}

static int handle_hello(server_state_t *s, client_t *c) {
    if (c->got_hello) {
        return queue_error(c, "duplicate HELLO");
    }

    c->got_hello = true;
    c->worker_id = s->next_worker_id++;

    assign_payload_t payload;
    payload.worker_id = htonl(c->worker_id);
    payload.dim = htonl((uint32_t) s->dim);
    payload.max_rounds = htonl((uint32_t) s->max_rounds);
    payload.reserved = 0;
    payload.learning_rate = s->learning_rate;

    printf("server: assigned worker id %u\n", c->worker_id);
    return queue_message(c, MSG_ASSIGN, 0, c->worker_id, &payload, (uint32_t) sizeof(payload));
}

static int handle_ready(server_state_t *s, client_t *c) {
    (void) s;
    if (!c->got_hello) {
        return queue_error(c, "READY before HELLO/ASSIGN");
    }
    if (c->ready) {
        return queue_error(c, "duplicate READY");
    }
    c->ready = true;
    printf("server: worker %u is ready\n", c->worker_id);
    return 0;
}

static int handle_gradient(server_state_t *s,
                           client_t *c,
                           uint32_t msg_round,
                           const unsigned char *payload,
                           uint32_t payload_len) {
    uint32_t expected = (uint32_t) (s->dim * (int) sizeof(float));

    if (!s->training_started) {
        return queue_error(c, "GRADIENT before training starts");
    }
    if (!c->ready) {
        return queue_error(c, "GRADIENT from non-ready worker");
    }
    if (msg_round != s->round) {
        return queue_error(c, "GRADIENT for wrong round");
    }
    if (payload_len != expected) {
        return queue_error(c, "GRADIENT payload_len mismatch");
    }
    if (c->got_gradient) {
        return queue_error(c, "duplicate GRADIENT for round");
    }

    memcpy(c->gradient, payload, payload_len);
    c->got_gradient = true;
    printf("server: received gradient from worker %u for round %u\n",
           c->worker_id,
           msg_round);
    return 0;
}

static int dispatch_message(server_state_t *s,
                            client_t *c,
                            uint32_t type,
                            uint32_t round,
                            uint32_t worker_id,
                            const unsigned char *payload,
                            uint32_t payload_len) {
    if (worker_id != 0 && c->got_hello && worker_id != c->worker_id) {
        return queue_error(c, "worker_id mismatch");
    }

    switch (type) {
        case MSG_HELLO:
            if (payload_len != 0) {
                return queue_error(c, "HELLO payload must be empty");
            }
            return handle_hello(s, c);

        case MSG_READY:
            if (payload_len != 0) {
                return queue_error(c, "READY payload must be empty");
            }
            return handle_ready(s, c);

        case MSG_GRADIENT:
            return handle_gradient(s, c, round, payload, payload_len);

        default:
            return queue_error(c, "unknown message type");
    }
}

static void process_client_buffer(server_state_t *s, client_t *c) {
    size_t offset = 0;

    while (c->in_used - offset >= sizeof(msg_header_t)) {
        msg_header_t net_hdr;
        memcpy(&net_hdr, c->inbuf + offset, sizeof(net_hdr));

        uint32_t type = ntohl(net_hdr.type);
        uint32_t payload_len = ntohl(net_hdr.payload_len);
        uint32_t round = ntohl(net_hdr.round);
        uint32_t worker_id = ntohl(net_hdr.worker_id);

        if (payload_len > INBUF_SIZE) {
            queue_error(c, "payload too large");
            c->active = false;
            return;
        }

        size_t total = sizeof(msg_header_t) + payload_len;
        if (c->in_used - offset < total) {
            break;
        }

        const unsigned char *payload = c->inbuf + offset + sizeof(msg_header_t);
        if (dispatch_message(s, c, type, round, worker_id, payload, payload_len) == -1) {
            fprintf(stderr, "server: failed to queue response to worker %u\n", c->worker_id);
            c->active = false;
            return;
        }

        offset += total;
    }

    if (offset > 0) {
        memmove(c->inbuf, c->inbuf + offset, c->in_used - offset);
        c->in_used -= offset;
    }
}

static void handle_client_readable(server_state_t *s, client_t *c) {
    while (1) {
        ssize_t n = recv(c->fd, c->inbuf + c->in_used, INBUF_SIZE - c->in_used, 0);
        if (n > 0) {
            c->in_used += (size_t) n;
            process_client_buffer(s, c);
            if (!c->active) {
                handle_disconnect(s, c);
                return;
            }
            if (c->in_used == INBUF_SIZE) {
                queue_error(c, "input buffer full");
                c->active = false;
                handle_disconnect(s, c);
                return;
            }
        } else if (n == 0) {
            handle_disconnect(s, c);
            return;
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return;
        } else {
            handle_disconnect(s, c);
            return;
        }
    }
}

static void accept_new_clients(server_state_t *s) {
    while (1) {
        struct sockaddr_in addr;
        socklen_t addrlen = sizeof(addr);
        int fd = accept(s->listen_fd, (struct sockaddr *) &addr, &addrlen);
        if (fd == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return;
            }
            perror("accept");
            return;
        }

        if (s->training_started || active_client_count(s) >= s->expected_workers) {
            const char *msg = "server not accepting more workers\n";
            (void) send(fd, msg, strlen(msg), 0);
            close(fd);
            continue;
        }

        if (set_nonblocking(fd) == -1) {
            close(fd);
            continue;
        }

        client_t *slot = alloc_client_slot(s);
        if (slot == NULL) {
            close(fd);
            continue;
        }

        init_client(slot);
        slot->fd = fd;
        slot->active = true;
        slot->gradient = calloc((size_t) s->dim, sizeof(float));
        if (slot->gradient == NULL) {
            close_client(slot);
            continue;
        }

        char ip[INET_ADDRSTRLEN] = {0};
        inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip));
        printf("server: accepted connection from %s:%d\n",
               ip,
               ntohs(addr.sin_port));
    }
}

static void init_server(server_state_t *s,
                        uint16_t port,
                        int expected_workers,
                        int dim,
                        float learning_rate,
                        int max_rounds) {
    memset(s, 0, sizeof(*s));
    s->listen_fd = INVALID_FD;
    s->port = port;
    s->expected_workers = expected_workers;
    s->dim = dim;
    s->learning_rate = learning_rate;
    s->max_rounds = max_rounds;
    s->training_started = false;
    s->training_finished = false;
    s->round = 0;
    s->next_worker_id = 0;

    for (int i = 0; i < MAX_CLIENTS; i++) {
        init_client(&s->clients[i]);
    }

    s->weights = calloc((size_t) dim, sizeof(float));
    if (s->weights == NULL) {
        die("calloc weights");
    }

    // Initialize weights with random values (match our nn's Xavier initialization)
    srand(42);

    int in = 784;
    int hid = 64;
    int out = 10;

    int off = 0;

    /* W1 */
    float limit1 = sqrtf(6.0f / (float)(in + hid));
    for (int i = 0; i < in * hid; i++) {
        s->weights[off + i] =
            ((float)rand() / (float)RAND_MAX) * 2.0f * limit1 - limit1;
    }
    off += in * hid;

    /* b1 = 0 */
    for (int i = 0; i < hid; i++) {
        s->weights[off + i] = 0.0f;
    }
    off += hid;

    /* W2 */
    float limit2 = sqrtf(6.0f / (float)(hid + out));
    for (int i = 0; i < hid * out; i++) {
        s->weights[off + i] =
            ((float)rand() / (float)RAND_MAX) * 2.0f * limit2 - limit2;
    }
    off += hid * out;

    /* b2 = 0 */
    for (int i = 0; i < out; i++) {
        s->weights[off + i] = 0.0f;
    }
    s->listen_fd = create_listen_socket(port);
    if (s->listen_fd == -1) {
        die("create_listen_socket");
    }
}

static void destroy_server(server_state_t *s) {
    if (s->listen_fd != INVALID_FD) {
        close(s->listen_fd);
    }
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (s->clients[i].fd != INVALID_FD) {
            close_client(&s->clients[i]);
        }
    }
    free(s->weights);
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr,
                "Usage: %s <port> <num_workers> <dim> <learning_rate> <max_rounds>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    uint16_t port = (uint16_t) strtoul(argv[1], NULL, 10);
    int expected_workers = atoi(argv[2]);
    int dim = atoi(argv[3]);
    float learning_rate = strtof(argv[4], NULL);
    int max_rounds = atoi(argv[5]);

    if (expected_workers <= 0 || expected_workers > MAX_CLIENTS || dim <= 0 || max_rounds <= 0) {
        fprintf(stderr, "invalid arguments\n");
        return EXIT_FAILURE;
    }

    server_state_t *s = malloc(sizeof(server_state_t));
    init_server(s, port, expected_workers, dim, learning_rate, max_rounds);

    printf("server: listening on port %u\n", port);
    printf("server: expecting %d workers, dim=%d, lr=%f, max_rounds=%d\n",
           expected_workers,
           dim,
           learning_rate,
           max_rounds);

    while (1) {
        fd_set readfds;
        fd_set writefds;
        FD_ZERO(&readfds);
        FD_ZERO(&writefds);

        int maxfd = s->listen_fd;
        FD_SET(s->listen_fd, &readfds);

        for (int i = 0; i < MAX_CLIENTS; i++) {
            client_t *c = &s->clients[i];
            if (!c->active) {
                continue;
            }
            FD_SET(c->fd, &readfds);
            if (c->out_used > c->out_sent) {
                FD_SET(c->fd, &writefds);
            }
            if (c->fd > maxfd) {
                maxfd = c->fd;
            }
        }

        int ready = select(maxfd + 1, &readfds, &writefds, NULL, NULL);
        if (ready == -1) {
            if (errno == EINTR) {
                continue;
            }
            die("select");
        }

        if (FD_ISSET(s->listen_fd, &readfds)) {
            accept_new_clients(s);
        }

        for (int i = 0; i < MAX_CLIENTS; i++) {
            client_t *c = &s->clients[i];
            if (!c->active) {
                continue;
            }

            if (FD_ISSET(c->fd, &readfds)) {
                handle_client_readable(s, c);
            }

            if (c->active && FD_ISSET(c->fd, &writefds)) {
                flush_client_output(c);
                if (!c->active) {
                    handle_disconnect(s, c);
                }
            }
        }

        remove_dead_clients(s);
        maybe_start_or_advance_training(s);

        if (s->training_finished) {
            bool all_outputs_flushed = true;
            for (int i = 0; i < MAX_CLIENTS; i++) {
                client_t *c = &s->clients[i];
                if (c->active && c->out_used > c->out_sent) {
                    all_outputs_flushed = false;
                    break;
                }
            }
            if (all_outputs_flushed) {
                break;
            }
        }
    }

    destroy_server(s);
    return EXIT_SUCCESS;
}
