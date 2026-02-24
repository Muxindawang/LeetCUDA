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



#define BLOCK_SIZE 32
#define CEIL(a, b) (((a) + (b) - 1) / (b))

__global__ void sgemm_clear(const float* A, const float* B, float* C, int M, int K, int N) {
    // 1. 【坐标计算】直接使用 row 和 col，拒绝迷惑
    // .y 对应行 (Row)，范围 0 ~ M-1
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // .x 对应列 (Col)，范围 0 ~ N-1
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (row >= M || col >= N) return;

    // 2. 【共享内存】
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 3. 【指针初始化】
    // A 是 [M, K]，当前线程负责第 row 行。起始位置是该行的开头。
    const float* a_ptr = A + row * K;
    
    // B 是 [K, N]，当前线程负责第 col 列。起始位置是该列的开头（即第0行第col列）。
    // 注意：B 是行优先存储，第0行第col列的偏移量就是 col。
    const float* b_ptr = B + col;

    float accum = 0.0f;

    // 4. 【K 维度循环】
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        
        // --- 加载阶段 (Load) ---
        // 每个线程把自己负责的那个元素搬进去
        // As: 取第 row 行，从 k 开始的 BLOCK_SIZE 个数据
        // 线程 (ty, tx) 搬运到 As[ty][tx]
        As[threadIdx.y][threadIdx.x] = a_ptr[threadIdx.x];

        // Bs: 取第 k 行开始的 BLOCK_SIZE 行，每行取第 col 列的数据
        // 线程 (ty, tx) 负责搬运 B 中 "第 ty 行，第 col 列" 的数据
        // B 的行跨度是 N，所以偏移是 ty * N + col
        Bs[threadIdx.y][threadIdx.x] = b_ptr[threadIdx.y * N];

        __syncthreads();

        // --- 计算阶段 (Compute) ---
        // 矩阵乘法核心：Row_i * Col_j
        // As 的第 threadIdx.y 行 点乘 Bs 的第 threadIdx.x 列
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            accum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();

        // --- 指针移动 (Shift) ---
        // a_ptr 向右移 (列方向 +1)
        a_ptr += BLOCK_SIZE;
        // b_ptr 向下移 (行方向 +N)
        b_ptr += BLOCK_SIZE * N;
    }

    // 5. 【写回】
    // C[row][col] -> 行优先偏移：row * N + col
    C[row * N + col] = accum;
}


void sgemm_sliced_k_fp32(float *A, float *B, float *C, int M, int K, int N) {
  constexpr int BM = 32;
  constexpr int BK = 32;
  constexpr int BN = 32;
  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_sliced_k_f32_kernel<BM, BK, BN><<<grid, block>>>(A, B, C, M, K, N);
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
  // dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  // dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  // sgemm_clear<<<grid, block>>>(mat_a_device, mat_b_device, mat_c_device, M, K, N);
  cudaDeviceSynchronize();
  CUDACHECK(cudaMemcpy(mat_c, mat_c_device, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));

  cudaFree(mat_a_device);
  cudaFree(mat_b_device);
  cudaFree(mat_c_device);
}