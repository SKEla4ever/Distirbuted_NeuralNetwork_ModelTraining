#ifndef NN_H
#define NN_H

# include <stdint.h>

 /* 
 A simple 2-layer fully-connected neural network.
 * Architecture: input_size -> hidden_size (ReLU) -> output_size (Softmax)
 */


typedef struct {
    int input_size;
    int hidden_size;
    int output_size;

// Learnable parameters 
    float *W1;   /* shape: input_size  x hidden_size  */
    float *b1;   /* shape: hidden_size                */
    float *W2;   /* shape: hidden_size x output_size  */
    float *b2;   /* shape: output_size                */
} Net;


// Gradients
typedef struct {
    float *dW1;  /* shape: input_size  x hidden_size */
    float *db1;  /* shape: hidden_size               */
    float *dW2;  /* shape: hidden_size x output_size */
    float *db2;  /* shape: output_size               */
} Grads;


// Intermediate values saved during forward pass, needed for backward
typedef struct {
    float *Z1;   /* pre-activation hidden  (hidden_size)  */
    float *A1;   /* post-ReLU hidden       (hidden_size)  */
    float *Z2;   /* pre-activation output  (output_size)  */
    float *Y_hat; /* softmax output        (output_size)  */
} Cache;


// Initialisation of structs needed
int  net_init(Net *net, int input_size, int hidden_size, int output_size);
void net_free(Net *net);

int  grads_init(Grads *g, const Net *net);
void grads_free(Grads *g);
void grads_zero(Grads *g, const Net *net);

int  cache_init(Cache *c, const Net *net);
void cache_free(Cache *c);


// Forward & Backward pass

/*
 * Forward pass for one sample.
 *   x      : input vector  (input_size floats)
 *   cache  : filled in by this function
 *
 * After this call, cache->Y_hat contains the softmax probabilities.
 */
void net_forward(const Net *net, const float *x, Cache *cache);

/*
 * Backward pass for one sample.  Computes gradients and ACCUMULATES them
 * into 'grads'.
 *
 *   x      : the same input used in forward
 *   label  : ground-truth class index (0-based)
 *   cache  : the cache populated by net_forward
 *   grads  : gradients are ADDED to existing values
 *
 * Returns the cross-entropy loss for this sample.
 */
float net_backward(const Net *net, const float *x, int label,
                   const Cache *cache, Grads *grads);


// Weight update

/*
 * Apply gradients:  W -= (lr / batch_size) * dW   for every parameter.
 * Then zeros out grads so they are ready for the next batch.
 */
void net_update(Net *net, Grads *grads, float lr, int batch_size);


// Other utility

// Return the index of the largest element in Y_hat (the predicted class)
int net_predict(const Cache *cache, int output_size);

// Total number of learnable floats in the network
int net_param_count(const Net *net);

/* Pack all parameters (W1,b1,W2,b2) into a contiguous buffer.
 * `buf` must hold at least net_param_count(net) floats.  */
void net_pack_params(const Net *net, float *buf);

// Unpack a contiguous buffer back into the network's parameters
void net_unpack_params(Net *net, const float *buf);

// Pack/unpack for gradients
void grads_pack(const Grads *g, const Net *net, float *buf);
void grads_unpack(Grads *g, const Net *net, const float *buf);

// Save/load the network parameters to/from a file (for checkpointing)
int net_save(const Net *net, const char *path);
int net_load(Net *net, const char *path);

#endif 