#pragma once

#include <stdio.h>
#include <math.h>
#include <cfloat>
#include <math.h>
#include <assert.h>
#include <float.h>


#define FULL_MASK 0xffffffff
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
// Helper function to get number of blocks
__host__ __device__ inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// Warp-level reduction for sum
__device__ float warp_reduce_sum(float val) {
    //#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        //_sync so we don't have to worry about divergence, no need to __syncwarp() here
        val += __shfl_xor_sync(FULL_MASK, val, mask);
    }
    return val;
}


// Add this warp reduction function for max at the top of the file with other helpers
__device__ float warp_reduce_max(float val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, mask));
    }
    return val;
}

/**
 * Debugging kernel to print an array from a single thread
 * @param arr Pointer to the array
 * @param n Number of elements in the array
 * @param name const char*  name of the array
 */
__device__ void print_array_single_thread(float* arr, int n, const char* name) {
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("%s Array contents: \n", name);
        for (int i = 0; i < n; i++) {
            printf("arr[%d] = %f ", i, arr[i]);
            if ((i + 1) % 8 == 0) printf("\n");  // 8 elements per line
        }
        printf("\n");
    }
    __syncthreads();

}


__global__ void init_array(float* arr, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = value;
    }
}

__device__ __host__ inline void swap_float_ptrs(float** a, float** b) {
    float* temp = *a;
    *a = *b;
    *b = temp;
}


#define CHECK_CUDA(call)   \
    do {            \
        cudaError_t err = call;         \
        if (err != cudaSuccess) {        \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err)); \
            assert(0); \
        } \
    } while(0)
