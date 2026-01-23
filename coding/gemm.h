#pragma once
#include <cuda_runtime.h>

#define CEIL_DIV(m, b) ((m + b -1) / b)

void mat_mul_on_device(float *mat_a, float *mat_b, float *mat_c, int h, int w, int k);