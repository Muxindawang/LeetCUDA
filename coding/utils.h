#pragma once

#include <cuda_runtime.h>
#include <system_error>

#define CUDACHECK(call) {                                                  \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
        printf("ERROR: %s:%d, ", __FILE__, __LINE__);                      \
        printf("CODE:%d, DETAIL:%s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                           \
    }                                                                      \
}

#define BLOCKSIZE 16

void init_matrix(float* data, int size, int low, int high, int seed);
void printMat(float* data, int size);
void compareMat(float* h_data, float* d_data, int size);
