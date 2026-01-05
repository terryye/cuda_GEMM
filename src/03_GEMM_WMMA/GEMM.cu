#include <stdio.h>
#include <mma.h>

using namespace nvcuda;

// WMMA tile dimensions: M, N, K = 16, 16, 16 for fp16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// For compatibility with existing code
#define TILE_WIDTH 16

__global__ void GEMM(float* d_A, float* d_B, float* d_C, int M, int K, int N, float alpha, float beta, bool transA, bool transB) {
    // Each 16x16 thread block handles one 16x16 output tile
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Warp and lane IDs
    // int warpM = (threadIdx.x / 32);
    // int warpN = 0;

    // Declare the fragments with appropriate layouts
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Shared memory for loading tiles
    __shared__ half sA[WMMA_M][WMMA_K];
    __shared__ half sB[WMMA_K][WMMA_N];

    // Loop over K dimension in WMMA_K chunks
    for (int k_tile = 0; k_tile < (K + WMMA_K - 1) / WMMA_K; k_tile++) {
        // Collaborative loading of A tile into shared memory
        int a_row = by * WMMA_M + threadIdx.y;
        int a_col = k_tile * WMMA_K + threadIdx.x;

        if (!transA) {
            // A is M x K, load normally
            if (a_row < M && a_col < K) {
                sA[threadIdx.y][threadIdx.x] = __float2half(d_A[a_row * K + a_col]);
            }
            else {
                sA[threadIdx.y][threadIdx.x] = __float2half(0.0f);
            }
        }
        else {
            // A is K x M (transposed), need to read from transposed location
            int a_row_t = k_tile * WMMA_K + threadIdx.y;
            int a_col_t = by * WMMA_M + threadIdx.x;
            if (a_row_t < K && a_col_t < M) {
                sA[threadIdx.x][threadIdx.y] = __float2half(d_A[a_row_t * M + a_col_t]);
            }
            else {
                sA[threadIdx.x][threadIdx.y] = __float2half(0.0f);
            }
        }

        // Collaborative loading of B tile into shared memory
        int b_row = k_tile * WMMA_K + threadIdx.y;
        int b_col = bx * WMMA_N + threadIdx.x;

        if (!transB) {
            // B is K x N, load normally
            if (b_row < K && b_col < N) {
                sB[threadIdx.y][threadIdx.x] = __float2half(d_B[b_row * N + b_col]);
            }
            else {
                sB[threadIdx.y][threadIdx.x] = __float2half(0.0f);
            }
        }
        else {
            // B is N x K (transposed), need to read from transposed location
            int b_row_t = bx * WMMA_N + threadIdx.y;
            int b_col_t = k_tile * WMMA_K + threadIdx.x;
            if (b_row_t < N && b_col_t < K) {
                sB[threadIdx.x][threadIdx.y] = __float2half(d_B[b_row_t * K + b_col_t]);
            }
            else {
                sB[threadIdx.x][threadIdx.y] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Load the tiles into fragments
        wmma::load_matrix_sync(a_frag, &sA[0][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &sB[0][0], WMMA_N);

        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    // Load C tile and compute final result
    __shared__ float sC[WMMA_M][WMMA_N];

    int c_row = by * WMMA_M + threadIdx.y;
    int c_col = bx * WMMA_N + threadIdx.x;

    if (c_row < M && c_col < N) {
        sC[threadIdx.y][threadIdx.x] = d_C[c_row * N + c_col];
    }
    else {
        sC[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Load C into fragment
    wmma::load_matrix_sync(c_frag, &sC[0][0], WMMA_N, wmma::mem_row_major);

    // Apply alpha and beta: result = alpha * (A * B) + beta * C
    for (int i = 0; i < c_frag.num_elements; i++) {
        acc_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store result back to shared memory
    wmma::store_matrix_sync(&sC[0][0], acc_frag, WMMA_N, wmma::mem_row_major);

    __syncthreads();

    // Write back to global memory
    if (c_row < M && c_col < N) {
        d_C[c_row * N + c_col] = sC[threadIdx.y][threadIdx.x];
    }
}