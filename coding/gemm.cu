#include "gemm.h"
#include <stdio.h>
#include "utils.h"

#define TILE_SIZE 32  // 通常选 16, 32；需满足 TILE_SIZE <= blockDim.x/y

__global__ void MatmulKernel(float* mat_a_device, float* mat_b_device, float* mat_c_device, int M, int K, int N) {
  // x 和 y 分别表示输出矩阵mat_c_devie中的列和行的index
  // mat_c_device[y][x] = mat_a_device[y, :] * mat_b_device[: ,x] 这是矩阵的计算方式
  // x沿着宽度方向
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= N || y >= M) return;
  float sum = 0.;
#pragma unroll
  for (int i = 0; i < K; i++) {
    float a = mat_a_device[K * y + i];
    float b = mat_b_device[i * N + x];
    sum += a * b;
  }
  mat_c_device[y * N + x] = sum;
}


// 模板参数：定义分块大小（编译期常量，比运行时变量更高效）
// BM: C矩阵每个分块的行数（也是A矩阵分块的行数）
// BN: C矩阵每个分块的列数（也是B矩阵分块的列数）
// BK: K维度的分块大小（A矩阵分块的列数 = B矩阵分块的行数）
template<const int BM = 32, const int BK = 32, const int BN = 32>
__global__ void sgemm_sliced_k_f32_kernel(float *A, float *B, float *C, int M, int K, int N) {
  __shared__ float As[BM][BK+ 1], Bs[BK + 1][BN];
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;

  int load_gmem_a_m = by * BM + ty;  // global row of a and c  要加载到共享内存的、矩阵 A 的、全局内存中的行索引
  int load_gmem_b_n = bx * BN + tx;  // global col of b and c

  float sum = 0.f;
  // 每个block要处理的数据是C中的BM*BN个数据，需要A中的BM*K个数据和B中的K*BN个数据
  for (int tile = 0; tile < (K + BK - 1) / BK; tile++) {
    int load_gmem_a_k = tile * BK + tx;

    if (load_gmem_a_m < M && load_gmem_a_k < K) {
      As[ty][tx] = A[load_gmem_a_m * K + load_gmem_a_k];
    } else {
      As[ty][tx] = 0.f;
    }


    int load_gmem_b_k = tile * BK + ty;
    if (load_gmem_b_k < K && load_gmem_b_n < N) {
      Bs[ty][tx] = B[load_gmem_b_k * N + load_gmem_b_n];
    } else {
      Bs[ty][tx] = 0.f;
    }
    __syncthreads();
#pragma unroll 
    for (int j = 0; j < BK; j++) {
      sum += As[ty][j] * Bs[j][tx];
    }
    __syncthreads();
  }
  if (load_gmem_a_m < M && load_gmem_b_n < N) {
    C[load_gmem_a_m * N + load_gmem_b_n] = sum;
  }
}



void sgemm_sliced_k_fp32(float *A, float *B, float *C, int M, int K, int N) {
  constexpr int BM = 32;
  constexpr int BK = 32;
  constexpr int BN = 32;
  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_sliced_k_f32_kernel<BM, BN, BK><<<grid, block>>>(A, B, C, M, K, N);
}


void mat_mul_on_device(float *mat_a, float *mat_b, float *mat_c, int M, int K, int N) {
  int block_size = 32;
  float* mat_a_device;
  CUDACHECK(cudaMalloc(&mat_a_device, M * K * sizeof(float)));
  
  float* mat_b_device;
  CUDACHECK(cudaMalloc(&mat_b_device, K * N * sizeof(float)));
  
  float* mat_c_device;
  CUDACHECK(cudaMalloc(&mat_c_device, M * N * sizeof(float)));

  CUDACHECK(cudaMemcpy(mat_a_device, mat_a, M * K * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(mat_b_device, mat_b, K * N * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));

  sgemm_sliced_k_fp32(mat_a_device, mat_b_device, mat_c_device, M, K, N);
  cudaDeviceSynchronize();
  CUDACHECK(cudaMemcpy(mat_c, mat_c_device, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));

  cudaFree(mat_a_device);
  cudaFree(mat_b_device);
  cudaFree(mat_c_device);
}