#ifndef WORKER_H
#define WORKER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../common/neural_network.h"

/*
 * worker.h
 *
 * Client-side worker for the distributed parameter server.
 *
 * The worker connects to the server, receives its assignment (worker_id,
 * model dimension, learning rate, max rounds), loads a local data shard,
 * and enters a training loop:
 *
 *   1. Receive weights from server   (MSG_WEIGHTS)
 *   2. Unpack weights into local Net
 *   3. Forward + backward on local data shard (one full pass)
 *   4. Pack gradients and send to server (MSG_GRADIENT)
 *   5. Repeat until MSG_FINISH
 *
 */

//  Protocol constants (from server.c)

#define MSG_HELLO     1u
#define MSG_ASSIGN    2u
#define MSG_READY     3u
#define MSG_WEIGHTS   4u
#define MSG_GRADIENT  5u
#define MSG_FINISH    6u
#define MSG_ERROR     7u

typedef struct {
    uint32_t type;
    uint32_t payload_len;
    uint32_t round;
    uint32_t worker_id;
} msg_header_t;

/* Payload of MSG_ASSIGN (from server). 
 * Fields are in network byte order 
 * except learning_rate which is raw IEEE-754 float
 * same-architecture
 * assumption matching server.c). */
typedef struct {
    uint32_t worker_id;
    uint32_t dim;
    uint32_t max_rounds;
    uint32_t reserved;
    float    learning_rate;
} assign_payload_t;

//  Data shard

/* A container for one worker's portion of the training data.
 * Each sample is a flat float vector of length 'n_features', and
 * each label is an integer class index. */
typedef struct {
    float *X;          /* row-major: n_samples x n_features  */
    int   *Y;          /* class labels: n_samples            */
    int    n_samples;
    int    n_features;
    int    n_classes;
} DataShard;

/* Load a data shard from a binary file.
 *
 * File format (little-endian):
 *   int32: n_samples
 *   int32: n_features
 *   int32: n_classes
 *   float[n_samples * n_features]: feature data (row-major)
 *   int32[n_samples]: labels
 *
 * Returns 0 on success, -1 on failure. */
int  shard_load(DataShard *shard, const char *path);
void shard_free(DataShard *shard);

//  Worker state

typedef struct {
    /* Network connection */
    int      sockfd;
    char     server_host[256];
    uint16_t server_port;

    /* Identity (assigned by server via MSG_ASSIGN) */
    uint32_t worker_id;
    int      dim;            /* total number of model floats */
    int      max_rounds;
    float    learning_rate;
    bool     assigned;       /* true after receiving MSG_ASSIGN */

    /* Neural network */
    Net      net;
    Grads    grads;
    Cache    cache;
    int      input_size;
    int      hidden_size;
    int      output_size;

    /* Serialization buffers (dim floats each) */
    float   *param_buf;      /* for receiving weights / sending gradients */

    /* Local training data */
    DataShard shard;

    /* Training state */
    uint32_t current_round;
    bool     finished;
} WorkerState;

//  Worker lifecycle

/* Initialise the worker state. 
 * Does NOT connect or allocate the net yet
 * (dimensions are unknown until MSG_ASSIGN arrives).
 * 
 * hidden_size: the hidden layer width (must match server's expectation) */
int  worker_init(WorkerState *w,
                 const char *host,
                 uint16_t port,
                 int hidden_size,
                 const char *shard_path);

/* Connect to the server, HELLO → ASSIGN → READY,
 * then enter the training loop until MSG_FINISH or an error. */
int  worker_run(WorkerState *w);

/* Release all resources. */
void worker_destroy(WorkerState *w);

//  Network I/O helpers

/* Blocking send/recv of exactly 'len' bytes.  Returns 0 on success,
 * -1 on error or unexpected EOF.  
 * The worker uses blocking I/O (unlike the server's non-blocking select loop) 
 * because each worker only talks
 * to one peer — there is nothing else to multiplex. */
int  send_all(int fd, const void *buf, size_t len);
int  recv_all(int fd, void *buf, size_t len);

/* Send a complete message (header + optional payload). */
int  send_message(int fd,
                  uint32_t type,
                  uint32_t round,
                  uint32_t worker_id,
                  const void *payload,
                  uint32_t payload_len);

/* Receive a complete message.  Fills in header fields (converted to host
 * byte order) and malloc's *payload_out if payload_len > 0.  Caller must
 * free *payload_out.  Sets *payload_out = NULL if payload_len == 0. */
int  recv_message(int fd,
                  uint32_t *type,
                  uint32_t *round,
                  uint32_t *worker_id,
                  unsigned char **payload_out,
                  uint32_t *payload_len);

#endif