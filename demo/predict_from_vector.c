#include "../common/neural_network.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int load_vector_784(const char *path, float *x) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror("load_vector_784 fopen");
        return -1;
    }

    size_t n = fread(x, sizeof(float), 784, f);
    fclose(f);

    if (n != 784) {
        fprintf(stderr, "load_vector_784: expected 784 floats, got %zu\n", n);
        return -1;
    }

    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <model.bin> <vector.bin>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *vector_path = argv[2];

    Net net;
    memset(&net, 0, sizeof(net));

    if (net_load(&net, model_path) != 0) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    if (net.input_size != 784) {
        fprintf(stderr, "Model input_size is %d, expected 784 for MNIST\n", net.input_size);
        net_free(&net);
        return 1;
    }

    float x[784];
    if (load_vector_784(vector_path, x) != 0) {
        net_free(&net);
        return 1;
    }

    Cache cache;
    memset(&cache, 0, sizeof(cache));
    if (cache_init(&cache, &net) != 0) {
        fprintf(stderr, "cache_init failed\n");
        net_free(&net);
        return 1;
    }

    net_forward(&net, x, &cache);
    int pred = net_predict(&cache, net.output_size);

    printf("Predicted digit: %d\n", pred);
    printf("Probabilities:\n");
    for (int i = 0; i < net.output_size; i++) {
        printf("%d: %.6f\n", i, cache.Y_hat[i]);
    }

    cache_free(&cache);
    net_free(&net);
    return 0;
}