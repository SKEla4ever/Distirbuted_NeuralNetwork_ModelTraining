#include "neural_network.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>


/* Xavier initialisation: uniform in [-limit, +limit] */
static void xavier_init(float *w, int fan_in, int fan_out, int count) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < count; i++) {
        w[i] = ((float)rand() / (float)RAND_MAX) * 2.0f * limit - limit;
    }
}


int net_init(Net *net, int input_size, int hidden_size, int output_size) {
    net->input_size  = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;

    net->W1 = calloc((size_t)input_size * hidden_size, sizeof(float));
    net->b1 = calloc((size_t)hidden_size, sizeof(float));
    net->W2 = calloc((size_t)hidden_size * output_size, sizeof(float));
    net->b2 = calloc((size_t)output_size, sizeof(float));

    if (!net->W1 || !net->b1 || !net->W2 || !net->b2) {
        net_free(net);
        return -1;
    }

    xavier_init(net->W1, input_size, hidden_size, input_size * hidden_size);
    xavier_init(net->W2, hidden_size, output_size, hidden_size * output_size);
    /* biases start at zero (already calloc'd) */

    return 0;
}

void net_free(Net *net) {
    free(net->W1); net->W1 = NULL;
    free(net->b1); net->b1 = NULL;
    free(net->W2); net->W2 = NULL;
    free(net->b2); net->b2 = NULL;
}

int grads_init(Grads *g, const Net *net) {
    int in  = net->input_size;
    int hid = net->hidden_size;
    int out = net->output_size;

    g->dW1 = calloc((size_t)in * hid, sizeof(float));
    g->db1 = calloc((size_t)hid, sizeof(float));
    g->dW2 = calloc((size_t)hid * out, sizeof(float));
    g->db2 = calloc((size_t)out, sizeof(float));

    if (!g->dW1 || !g->db1 || !g->dW2 || !g->db2) {
        grads_free(g);
        return -1;
    }
    return 0;
}

void grads_free(Grads *g) {
    free(g->dW1); g->dW1 = NULL;
    free(g->db1); g->db1 = NULL;
    free(g->dW2); g->dW2 = NULL;
    free(g->db2); g->db2 = NULL;
}

void grads_zero(Grads *g, const Net *net) {
    memset(g->dW1, 0, (size_t)net->input_size * net->hidden_size * sizeof(float));
    memset(g->db1, 0, (size_t)net->hidden_size * sizeof(float));
    memset(g->dW2, 0, (size_t)net->hidden_size * net->output_size * sizeof(float));
    memset(g->db2, 0, (size_t)net->output_size * sizeof(float));
}

int cache_init(Cache *c, const Net *net) {
    c->Z1    = calloc((size_t)net->hidden_size, sizeof(float));
    c->A1    = calloc((size_t)net->hidden_size, sizeof(float));
    c->Z2    = calloc((size_t)net->output_size, sizeof(float));
    c->Y_hat = calloc((size_t)net->output_size, sizeof(float));

    if (!c->Z1 || !c->A1 || !c->Z2 || !c->Y_hat) {
        cache_free(c);
        return -1;
    }
    return 0;
}

void cache_free(Cache *c) {
    free(c->Z1);    c->Z1    = NULL;
    free(c->A1);    c->A1    = NULL;
    free(c->Z2);    c->Z2    = NULL;
    free(c->Y_hat); c->Y_hat = NULL;
}

// Forward pass

void net_forward(const Net *net, const float *x, Cache *cache) {
    int in  = net->input_size;
    int hid = net->hidden_size;
    int out = net->output_size;

    /* Z1 = X dot W1 + b1   (vector-matrix multiplication) */
    for (int j = 0; j < hid; j++) {
        float sum = net->b1[j];
        for (int i = 0; i < in; i++) {
            sum += x[i] * net->W1[i * hid + j];
        }
        cache->Z1[j] = sum;
    }

    /* A1 = ReLU(Z1) */
    for (int j = 0; j < hid; j++) {
        cache->A1[j] = cache->Z1[j] > 0.0f ? cache->Z1[j] : 0.0f;
    }

    /* Z2 = A1 dot W2 + b2 */
    for (int j = 0; j < out; j++) {
        float sum = net->b2[j];
        for (int k = 0; k < hid; k++) {
            sum += cache->A1[k] * net->W2[k * out + j];
        }
        cache->Z2[j] = sum;
    }

    /* Y_hat = softmax(Z2)  — numerically stable version */
    float max_z = cache->Z2[0];
    for (int j = 1; j < out; j++) {
        if (cache->Z2[j] > max_z) max_z = cache->Z2[j];
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < out; j++) {
        cache->Y_hat[j] = expf(cache->Z2[j] - max_z);
        sum_exp += cache->Y_hat[j];
    }
    for (int j = 0; j < out; j++) {
        cache->Y_hat[j] /= sum_exp;
    }
}


// Backward pass

float net_backward(const Net *net, const float *x, int label,
                   const Cache *cache, Grads *grads) {
    int in  = net->input_size;
    int hid = net->hidden_size;
    int out = net->output_size;

    /* Cross-entropy loss */
    float prob = cache->Y_hat[label];
    if (prob < 1e-7f) prob = 1e-7f;          /* avoid log(0) */
    float loss = -logf(prob);

    /* dZ2 = Y_hat - Y_onehot   (softmax + cross-entropy shortcut) */
    float *dZ2 = malloc((size_t)out * sizeof(float));
    for (int j = 0; j < out; j++) {
        dZ2[j] = cache->Y_hat[j];
    }
    dZ2[label] -= 1.0f;

    /* Accumulate dW2 += A1^T dot dZ2  and  db2 += dZ2 */
    for (int k = 0; k < hid; k++) {
        for (int j = 0; j < out; j++) {
            grads->dW2[k * out + j] += cache->A1[k] * dZ2[j];
        }
    }
    for (int j = 0; j < out; j++) {
        grads->db2[j] += dZ2[j];
    }

    /* dA1 = dZ2 dot W2^T */
    float *dA1 = calloc((size_t)hid, sizeof(float));
    for (int k = 0; k < hid; k++) {
        for (int j = 0; j < out; j++) {
            dA1[k] += dZ2[j] * net->W2[k * out + j];
        }
    }

    /* dZ1 = dA1 * ReLU'(Z1)   (element-wise) */
    float *dZ1 = malloc((size_t)hid * sizeof(float));
    for (int k = 0; k < hid; k++) {
        dZ1[k] = cache->Z1[k] > 0.0f ? dA1[k] : 0.0f;
    }

    /* Accumulate dW1 += X^T dot dZ1  and  db1 += dZ1 */
    for (int i = 0; i < in; i++) {
        for (int k = 0; k < hid; k++) {
            grads->dW1[i * hid + k] += x[i] * dZ1[k];
        }
    }
    for (int k = 0; k < hid; k++) {
        grads->db1[k] += dZ1[k];
    }

    free(dZ2);
    free(dA1);
    free(dZ1);
    return loss;
}

// Weight update

void net_update(Net *net, Grads *grads, float lr, int batch_size) {
    float scale = lr / (float)batch_size;
    int in  = net->input_size;
    int hid = net->hidden_size;
    int out = net->output_size;

    for (int i = 0; i < in * hid; i++)
        net->W1[i] -= scale * grads->dW1[i];
    for (int i = 0; i < hid; i++)
        net->b1[i] -= scale * grads->db1[i];
    for (int i = 0; i < hid * out; i++)
        net->W2[i] -= scale * grads->dW2[i];
    for (int i = 0; i < out; i++)
        net->b2[i] -= scale * grads->db2[i];

    /* Zero grads for next batch */
    grads_zero(grads, net);
}

// Other utility

int net_predict(const Cache *cache, int output_size) {
    int best = 0;
    for (int j = 1; j < output_size; j++) {
        if (cache->Y_hat[j] > cache->Y_hat[best]) best = j;
    }
    return best;
}

int net_param_count(const Net *net) {
    return net->input_size * net->hidden_size + net->hidden_size
         + net->hidden_size * net->output_size + net->output_size;
}

void net_pack_params(const Net *net, float *buf) {
    int in  = net->input_size;
    int hid = net->hidden_size;
    int out = net->output_size;
    int off = 0;

    memcpy(buf + off, net->W1, (size_t)in * hid * sizeof(float));  off += in * hid;
    memcpy(buf + off, net->b1, (size_t)hid * sizeof(float));       off += hid;
    memcpy(buf + off, net->W2, (size_t)hid * out * sizeof(float)); off += hid * out;
    memcpy(buf + off, net->b2, (size_t)out * sizeof(float));
}

void net_unpack_params(Net *net, const float *buf) {
    int in  = net->input_size;
    int hid = net->hidden_size;
    int out = net->output_size;
    int off = 0;

    memcpy(net->W1, buf + off, (size_t)in * hid * sizeof(float));  off += in * hid;
    memcpy(net->b1, buf + off, (size_t)hid * sizeof(float));       off += hid;
    memcpy(net->W2, buf + off, (size_t)hid * out * sizeof(float)); off += hid * out;
    memcpy(net->b2, buf + off, (size_t)out * sizeof(float));
}

void grads_pack(const Grads *g, const Net *net, float *buf) {
    int in  = net->input_size;
    int hid = net->hidden_size;
    int out = net->output_size;
    int off = 0;

    memcpy(buf + off, g->dW1, (size_t)in * hid * sizeof(float));  off += in * hid;
    memcpy(buf + off, g->db1, (size_t)hid * sizeof(float));       off += hid;
    memcpy(buf + off, g->dW2, (size_t)hid * out * sizeof(float)); off += hid * out;
    memcpy(buf + off, g->db2, (size_t)out * sizeof(float));
}

void grads_unpack(Grads *g, const Net *net, const float *buf) {
    int in  = net->input_size;
    int hid = net->hidden_size;
    int out = net->output_size;
    int off = 0;

    memcpy(g->dW1, buf + off, (size_t)in * hid * sizeof(float));  off += in * hid;
    memcpy(g->db1, buf + off, (size_t)hid * sizeof(float));       off += hid;
    memcpy(g->dW2, buf + off, (size_t)hid * out * sizeof(float)); off += hid * out;
    memcpy(g->db2, buf + off, (size_t)out * sizeof(float));
}

int net_save(const Net *net, const char *path) {
    if (net == NULL || path == NULL) {
        fprintf(stderr, "net_save: invalid argument\n");
        return -1;
    }

    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        perror("net_save fopen");
        return -1;
    }

    int32_t meta[3];
    meta[0] = net->input_size;
    meta[1] = net->hidden_size;
    meta[2] = net->output_size;

    if (fwrite(meta, sizeof(int32_t), 3, f) != 3) {
        fprintf(stderr, "net_save: failed to write metadata\n");
        fclose(f);
        return -1;
    }

    int param_count = net_param_count(net);
    float *buf = malloc((size_t)param_count * sizeof(float));
    if (buf == NULL) {
        perror("net_save malloc");
        fclose(f);
        return -1;
    }

    net_pack_params(net, buf);

    if (fwrite(buf, sizeof(float), (size_t)param_count, f) != (size_t)param_count) {
        fprintf(stderr, "net_save: failed to write parameters\n");
        free(buf);
        fclose(f);
        return -1;
    }

    free(buf);
    fclose(f);
    return 0;
}

int net_load(Net *net, const char *path) {
    if (net == NULL || path == NULL) {
        fprintf(stderr, "net_load: invalid argument\n");
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror("net_load fopen");
        return -1;
    }

    int32_t meta[3];
    if (fread(meta, sizeof(int32_t), 3, f) != 3) {
        fprintf(stderr, "net_load: failed to read metadata\n");
        fclose(f);
        return -1;
    }

    int input_size = meta[0];
    int hidden_size = meta[1];
    int output_size = meta[2];

    if (input_size <= 0 || hidden_size <= 0 || output_size <= 0) {
        fprintf(stderr, "net_load: invalid metadata\n");
        fclose(f);
        return -1;
    }

    if (net_init(net, input_size, hidden_size, output_size) != 0) {
        fprintf(stderr, "net_load: net_init failed\n");
        fclose(f);
        return -1;
    }

    int param_count = net_param_count(net);
    float *buf = malloc((size_t)param_count * sizeof(float));
    if (buf == NULL) {
        perror("net_load malloc");
        net_free(net);
        fclose(f);
        return -1;
    }

    if (fread(buf, sizeof(float), (size_t)param_count, f) != (size_t)param_count) {
        fprintf(stderr, "net_load: failed to read parameters\n");
        free(buf);
        net_free(net);
        fclose(f);
        return -1;
    }

    net_unpack_params(net, buf);

    free(buf);
    fclose(f);
    return 0;
}