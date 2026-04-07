#include "worker.h"

#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>


//  Network I/O helpers

int send_all(int fd, const void *buf, size_t len) {
    const unsigned char *p = buf;
    size_t remaining = len;

    while (remaining > 0) {
        ssize_t n = send(fd, p, remaining, 0);
        if (n > 0) {
            p += n;
            remaining -= (size_t)n;
        } else if (n == -1 && errno == EINTR) {
            continue;   /* interrupted by signal, retry */
        } else {
            perror("send_all");
            return -1;
        }
    }
    return 0;
}

int recv_all(int fd, void *buf, size_t len) {
    unsigned char *p = buf;
    size_t remaining = len;

    while (remaining > 0) {
        ssize_t n = recv(fd, p, remaining, 0);
        if (n > 0) {
            p += n;
            remaining -= (size_t)n;
        } else if (n == 0) {
            fprintf(stderr, "recv_all: server closed connection\n");
            return -1;
        } else if (errno == EINTR) {
            continue;
        } else {
            perror("recv_all");
            return -1;
        }
    }
    return 0;
}

int send_message(int fd,
                 uint32_t type,
                 uint32_t round,
                 uint32_t worker_id,
                 const void *payload,
                 uint32_t payload_len) {
    msg_header_t hdr;
    hdr.type       = htonl(type);
    hdr.payload_len = htonl(payload_len);
    hdr.round      = htonl(round);
    hdr.worker_id  = htonl(worker_id);

    if (send_all(fd, &hdr, sizeof(hdr)) == -1) {
        return -1;
    }
    if (payload_len > 0 && payload != NULL) {
        if (send_all(fd, payload, payload_len) == -1) {
            return -1;
        }
    }
    return 0;
}

int recv_message(int fd,
                 uint32_t *type,
                 uint32_t *round,
                 uint32_t *worker_id,
                 unsigned char **payload_out,
                 uint32_t *payload_len) {
    msg_header_t hdr;

    if (recv_all(fd, &hdr, sizeof(hdr)) == -1) {
        return -1;
    }

    *type       = ntohl(hdr.type);
    *payload_len = ntohl(hdr.payload_len);
    *round      = ntohl(hdr.round);
    *worker_id  = ntohl(hdr.worker_id);

    if (*payload_len == 0) {
        *payload_out = NULL;
        return 0;
    }

    /* Sanity check: reject absurdly large payloads */
    if (*payload_len > 64 * 1024 * 1024) {
        fprintf(stderr, "recv_message: payload too large (%u bytes)\n",
                *payload_len);
        return -1;
    }

    *payload_out = malloc(*payload_len);
    if (*payload_out == NULL) {
        perror("malloc payload");
        return -1;
    }

    if (recv_all(fd, *payload_out, *payload_len) == -1) {
        free(*payload_out);
        *payload_out = NULL;
        return -1;
    }

    return 0;
}

//  Data shard loading

int shard_load(DataShard *shard, const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror(path);
        return -1;
    }

    int32_t meta[3];
    if (fread(meta, sizeof(int32_t), 3, f) != 3) {
        fprintf(stderr, "shard_load: failed to read metadata from %s\n", path);
        fclose(f);
        return -1;
    }

    shard->n_samples  = meta[0];
    shard->n_features = meta[1];
    shard->n_classes  = meta[2];

    if (shard->n_samples <= 0 || shard->n_features <= 0 || shard->n_classes <= 0) {
        fprintf(stderr, "shard_load: invalid metadata in %s\n", path);
        fclose(f);
        return -1;
    }

    size_t x_count = (size_t)shard->n_samples * shard->n_features;
    shard->X = malloc(x_count * sizeof(float));
    shard->Y = malloc((size_t)shard->n_samples * sizeof(int));

    if (shard->X == NULL || shard->Y == NULL) {
        perror("malloc shard data");
        shard_free(shard);
        fclose(f);
        return -1;
    }

    if (fread(shard->X, sizeof(float), x_count, f) != x_count) {
        fprintf(stderr, "shard_load: failed to read features from %s\n", path);
        shard_free(shard);
        fclose(f);
        return -1;
    }

    /* Labels are stored as int32_t on disk for portability */
    int32_t *labels_buf = malloc((size_t)shard->n_samples * sizeof(int32_t));
    if (labels_buf == NULL) {
        perror("malloc labels_buf");
        shard_free(shard);
        fclose(f);
        return -1;
    }

    if (fread(labels_buf, sizeof(int32_t), (size_t)shard->n_samples, f)
            != (size_t)shard->n_samples) {
        fprintf(stderr, "shard_load: failed to read labels from %s\n", path);
        free(labels_buf);
        shard_free(shard);
        fclose(f);
        return -1;
    }

    for (int i = 0; i < shard->n_samples; i++) {
        shard->Y[i] = (int)labels_buf[i];
    }
    free(labels_buf);

    fclose(f);
    printf("worker: loaded shard %s (%d samples, %d features, %d classes)\n",
           path, shard->n_samples, shard->n_features, shard->n_classes);
    return 0;
}

void shard_free(DataShard *shard) {
    free(shard->X); shard->X = NULL;
    free(shard->Y); shard->Y = NULL;
    shard->n_samples  = 0;
    shard->n_features = 0;
    shard->n_classes  = 0;
}

//  Connection setup

static int connect_to_server(WorkerState *w) {
    w->sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (w->sockfd == -1) {
        perror("socket");
        return -1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(w->server_port);

    if (inet_pton(AF_INET, w->server_host, &addr.sin_addr) != 1) {
        fprintf(stderr, "worker: invalid server address '%s'\n", w->server_host);
        close(w->sockfd);
        w->sockfd = -1;
        return -1;
    }

    if (connect(w->sockfd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        perror("connect");
        close(w->sockfd);
        w->sockfd = -1;
        return -1;
    }

    printf("worker: connected to %s:%u\n", w->server_host, w->server_port);
    return 0;
}

// Protocol handshake 

/* Send MSG_HELLO, receive MSG_ASSIGN, allocate net + buffers. */
static int do_handshake(WorkerState *w) {
    /* --- Send HELLO --- */
    if (send_message(w->sockfd, MSG_HELLO, 0, 0, NULL, 0) == -1) {
        fprintf(stderr, "worker: failed to send HELLO\n");
        return -1;
    }

    /* --- Receive ASSIGN --- */
    uint32_t type, round, wid, plen;
    unsigned char *payload = NULL;

    if (recv_message(w->sockfd, &type, &round, &wid, &payload, &plen) == -1) {
        fprintf(stderr, "worker: failed to receive ASSIGN\n");
        return -1;
    }

    if (type == MSG_ERROR) {
        fprintf(stderr, "worker: server error: %.*s\n", (int)plen, payload);
        free(payload);
        return -1;
    }

    if (type != MSG_ASSIGN || plen != sizeof(assign_payload_t)) {
        fprintf(stderr, "worker: expected ASSIGN, got type=%u len=%u\n",
                type, plen);
        free(payload);
        return -1;
    }

    assign_payload_t ap;
    memcpy(&ap, payload, sizeof(ap));
    free(payload);

    w->worker_id     = ntohl(ap.worker_id);
    w->dim           = (int)ntohl(ap.dim);
    w->max_rounds    = (int)ntohl(ap.max_rounds);
    w->learning_rate = ap.learning_rate;
    w->assigned      = true;

    printf("worker %u: assigned (dim=%d, max_rounds=%d, lr=%f)\n",
           w->worker_id, w->dim, w->max_rounds, w->learning_rate);

    w->input_size  = w->shard.n_features;
    w->output_size = w->shard.n_classes;

    int expected_dim = w->input_size * w->hidden_size + w->hidden_size
                     + w->hidden_size * w->output_size + w->output_size;

    if (expected_dim != w->dim) {
        fprintf(stderr,
                "worker %u: dimension mismatch! server dim=%d, "
                "expected=%d (in=%d, hid=%d, out=%d)\n",
                w->worker_id, w->dim, expected_dim,
                w->input_size, w->hidden_size, w->output_size);
        return -1;
    }

    if (net_init(&w->net, w->input_size, w->hidden_size, w->output_size) != 0) {
        fprintf(stderr, "worker %u: net_init failed\n", w->worker_id);
        return -1;
    }
    if (grads_init(&w->grads, &w->net) != 0) {
        fprintf(stderr, "worker %u: grads_init failed\n", w->worker_id);
        return -1;
    }
    if (cache_init(&w->cache, &w->net) != 0) {
        fprintf(stderr, "worker %u: cache_init failed\n", w->worker_id);
        return -1;
    }

    w->param_buf = malloc((size_t)w->dim * sizeof(float));
    if (w->param_buf == NULL) {
        perror("malloc param_buf");
        return -1;
    }

    /* --- Send READY --- */
    if (send_message(w->sockfd, MSG_READY, 0, w->worker_id, NULL, 0) == -1) {
        fprintf(stderr, "worker %u: failed to send READY\n", w->worker_id);
        return -1;
    }

    printf("worker %u: sent READY, waiting for training to begin\n",
           w->worker_id);
    return 0;
}

//  Training loop

static int train_one_round(WorkerState *w, const float *weights_payload) {
    /* Unpack server weights into local net */
    memcpy(w->param_buf, weights_payload, (size_t)w->dim * sizeof(float));
    net_unpack_params(&w->net, w->param_buf);

    /* Zero gradients for this round */
    grads_zero(&w->grads, &w->net);

    /* Forward + backward on every sample in the shard */
    float total_loss = 0.0f;
    int   correct    = 0;

    for (int i = 0; i < w->shard.n_samples; i++) {
        const float *x = w->shard.X + i * w->shard.n_features;
        int label = w->shard.Y[i];

        net_forward(&w->net, x, &w->cache);
        total_loss += net_backward(&w->net, x, label, &w->cache, &w->grads);

        if (net_predict(&w->cache, w->output_size) == label) {
            correct++;
        }
    }

    float avg_loss = total_loss / (float)w->shard.n_samples;
    float accuracy = 100.0f * (float)correct / (float)w->shard.n_samples;

    printf("worker %u: round %u | loss: %.4f | accuracy: %.1f%% (%d/%d)\n",
        w->worker_id, w->current_round,
        avg_loss, accuracy, correct, w->shard.n_samples);

    /* Average gradients over local samples before sending to server */
    float inv_n = 1.0f / (float)w->shard.n_samples;

    int in  = w->net.input_size;
    int hid = w->net.hidden_size;
    int out = w->net.output_size;

    for (int i = 0; i < in * hid; i++) {
        w->grads.dW1[i] *= inv_n;
    }
    for (int i = 0; i < hid; i++) {
        w->grads.db1[i] *= inv_n;
    }
    for (int i = 0; i < hid * out; i++) {
        w->grads.dW2[i] *= inv_n;
    }
    for (int i = 0; i < out; i++) {
        w->grads.db2[i] *= inv_n;
    }

    grads_pack(&w->grads, &w->net, w->param_buf);

    /* Send gradients to server */
    uint32_t payload_len = (uint32_t)(w->dim * (int)sizeof(float));
    if (send_message(w->sockfd, MSG_GRADIENT, w->current_round,
                     w->worker_id, w->param_buf, payload_len) == -1) {
        fprintf(stderr, "worker %u: failed to send GRADIENT\n", w->worker_id);
        return -1;
    }

    return 0;
}

static int training_loop(WorkerState *w) {
    while (!w->finished) {
        /* Receive next message from server */
        uint32_t type, round, wid, plen;
        unsigned char *payload = NULL;

        if (recv_message(w->sockfd, &type, &round, &wid, &payload, &plen) == -1) {
            fprintf(stderr, "worker %u: lost connection to server\n",
                    w->worker_id);
            return -1;
        }

        switch (type) {
        case MSG_WEIGHTS: {
            uint32_t expected = (uint32_t)(w->dim * (int)sizeof(float));
            if (plen != expected) {
                fprintf(stderr,
                        "worker %u: WEIGHTS payload mismatch (%u != %u)\n",
                        w->worker_id, plen, expected);
                free(payload);
                return -1;
            }

            w->current_round = round;
            if (train_one_round(w, (const float *)payload) == -1) {
                free(payload);
                return -1;
            }
            break;
        }

        case MSG_FINISH: {
            printf("worker %u: received FINISH at round %u\n",
                   w->worker_id, round);

            /* Unpack final weights */
            uint32_t expected = (uint32_t)(w->dim * (int)sizeof(float));
            if (plen == expected) {
                net_unpack_params(&w->net, (const float *)payload);
                printf("worker %u: final weights received\n", w->worker_id);

                char model_path[128];
                snprintf(model_path, sizeof(model_path),
                        "trained_model_worker_%u.bin", w->worker_id);

                if (net_save(&w->net, model_path) == 0) {
                    printf("worker %u: saved model to %s\n",
                        w->worker_id, model_path);
                } else {
                    fprintf(stderr, "worker %u: failed to save model\n",
                            w->worker_id);
                }
            }

            w->finished = true;
            break;
        }

        case MSG_ERROR:
            fprintf(stderr, "worker %u: server error: %.*s\n",
                    w->worker_id, (int)plen, payload);
            free(payload);
            return -1;

        default:
            fprintf(stderr, "worker %u: unexpected message type %u\n",
                    w->worker_id, type);
            free(payload);
            return -1;
        }

        free(payload);
    }

    return 0;
}

//  Public API

int worker_init(WorkerState *w,
                const char *host,
                uint16_t port,
                int hidden_size,
                const char *shard_path) {
    memset(w, 0, sizeof(*w));

    w->sockfd = -1;
    w->assigned = false;
    w->finished = false;

    /* Copy server address */
    strncpy(w->server_host, host, sizeof(w->server_host) - 1);
    w->server_host[sizeof(w->server_host) - 1] = '\0';
    w->server_port = port;

    w->hidden_size = hidden_size;

    /* Load data shard early so we know input_size and output_size
     * before the handshake (needed to verify dim). */
    if (shard_load(&w->shard, shard_path) == -1) {
        return -1;
    }

    return 0;
}

int worker_run(WorkerState *w) {
    if (connect_to_server(w) == -1) {
        return -1;
    }

    if (do_handshake(w) == -1) {
        return -1;
    }

    if (training_loop(w) == -1) {
        return -1;
    }

    printf("worker %u: finished training\n", w->worker_id);
    return 0;
}

void worker_destroy(WorkerState *w) {
    if (w->sockfd != -1) {
        close(w->sockfd);
        w->sockfd = -1;
    }

    if (w->assigned) {
        net_free(&w->net);
        grads_free(&w->grads);
        cache_free(&w->cache);
        free(w->param_buf);
        w->param_buf = NULL;
    }

    shard_free(&w->shard);
}

// Main

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr,
                "Usage: %s <server_ip> <port> <hidden_size> <shard_file>\n",
                argv[0]);
        return EXIT_FAILURE;
    }
 
    const char *host   = argv[1];
    uint16_t    port   = (uint16_t)strtoul(argv[2], NULL, 10);
    int hidden_size    = atoi(argv[3]);
    const char *shard  = argv[4];
 
    if (hidden_size <= 0) {
        fprintf(stderr, "error: hidden_size must be positive\n");
        return EXIT_FAILURE;
    }
 
    WorkerState w;
    if (worker_init(&w, host, port, hidden_size, shard) == -1) {
        fprintf(stderr, "error: worker_init failed\n");
        return EXIT_FAILURE;
    }
 
    int result = worker_run(&w);
    worker_destroy(&w);
 
    return result == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}