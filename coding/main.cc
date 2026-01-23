#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <iostream>
#include "gemm.h"
#include "utils.h"
#include "timer.h"



void mat_mul_on_host(float *mat_a, float *mat_b, float *mat_c, int h, int w, int k) {
  // h: mat_a 行数, w: mat_a列数/mat_b行数, k: mat_b列数
  for (int row = 0; row < h; row++) {
    for (int col = 0; col < k; col++) {
      float sum = 0.0f;  // 每列计算前重置sum
      for (int i = 0; i < w; i++) {
        sum += mat_a[row * w + i] * mat_b[i * k + col];
      }
      mat_c[row * k + col] = sum;
    }
  }
}

int main() {
  int h = 1 << 10,
      w = 1 << 9,
      k = 1 << 11;


  float* mat_a = new float[h * w];
  float* mat_b = new float[w * k];
  float* mat_c = new float[h * k];
  int seed = 0;
  init_matrix(mat_a, h * w, 0, 1, seed);
  seed++;
  init_matrix(mat_b, w * k, 0, 1, seed);
  float* mat_c_host = new float[h * k];

  Timer timer;

  timer.start();
  mat_mul_on_host(mat_a, mat_b, mat_c_host, h, w, k);
  timer.stop();
  timer.duration<Timer::ms>("matmul in cpu");


  for (int i = 0; i < 100; i++) {
    mat_mul_on_device(mat_a, mat_b, mat_c, h, w, k);
  }
  timer.start();
  mat_mul_on_device(mat_a, mat_b, mat_c, h, w, k);
  timer.stop();
  timer.duration<Timer::ms>("matmul in gpu(warmup)");
  printMat(mat_c_host, 20);
  printMat(mat_c, 20);
  compareMat(mat_c, mat_c_host, h * k);
  
  delete[] mat_a;
  delete[] mat_b;
  delete[] mat_c;
  return 0;

}