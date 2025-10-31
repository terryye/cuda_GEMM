#include "util/cuda_shim.h"
#define TILE_WIDTH 16

__global__ void GEMM(float *d_A, float *d_B, float *d_C, int M, int K, int N, float alpha, float beta, bool transA, bool transB) {
   // printf("current block: (%d,%d), thread (%d,%d) , block dim (%d, %d)\n", blockIdx.x,  blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x, blockDim.y);
   __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH]; //tile size 
   __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

   //check if the blockDim is set correctly
   if (TILE_WIDTH != blockDim.x || TILE_WIDTH != blockDim.y){
       if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0){
           printf("Error: TILE_WIDTH != blockDim.x or blockDim.y, please set blockDim to (%d,%d) \n", TILE_WIDTH, TILE_WIDTH);
       }
       return;
    }
    /*
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    */
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y; // row
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x; // col

    //loop over tiles
    float val = 0.0f;
    for( int ph = 0; ph < (K - 1 + TILE_WIDTH) / TILE_WIDTH ; ph++){
        
        int yA, xA, yB, xB, HeightA, WidthA, HeightB, WidthB;

        if (transA){
            yA = ph * TILE_WIDTH + threadIdx.x;
            xA = y;
            HeightA = K;
            WidthA = M;
        } else{
            yA = y;
            xA = ph * TILE_WIDTH + threadIdx.x;
            HeightA = M;
            WidthA = K;
        }

        if (transB){
            yB = x;
            xB = ph * TILE_WIDTH + threadIdx.y;
            HeightB = N;
            WidthB = K;
        } else{ 
            yB = ph * TILE_WIDTH + threadIdx.y;
            xB = x;
            HeightB = K;
            WidthB = N;
        }

        //boundary check for A and B
        //load one tile of A and B into shared memory

        if( (yA >= HeightA) || (xA >= WidthA) ){
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }else{
            tile_A[threadIdx.y][threadIdx.x] = d_A[yA * WidthA + xA]; // row: y; col: i
        }

        if( (yB >= HeightB) || (xB >= WidthB) ){
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        } else{
            tile_B[threadIdx.y][threadIdx.x] = d_B[yB * WidthB + xB]; // row : i;  col: x
        }

        __syncthreads();
        
        /*
        
        if (blockIdx.x == 0 && blockIdx.y == 0){
            printf("[threadIdx.x, threadIdx.y] = [%d, %d], [[yA,xA]=[%d,%d], [yB,xB]=[%d,%d], [HeightB, WidthB] = [%d,%d] \n", threadIdx.x, threadIdx.y, yA, xA, yB, xB, HeightB, WidthB);
            printf("loading tile for phase %d, block (%d,%d) \n", ph, blockIdx.x, blockIdx.y);
            printf("tile_B: \n%f, %f; \n%f, %f \n", tile_B[0][0], tile_B[0][1], tile_B[1][0], tile_B[1][1]);
        }
        */

        for(int k = 0; k < TILE_WIDTH; k++){
            val += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();

    }
    // Compute D = alpha * A * B + beta * C

    // check boundaries
    if (y >= M  || x >= N) return;

    //a*(A*B) + beta * C
    val = alpha * val +  d_C [y * N + x] * beta;  // row : row;  col: col

    //updates $C$ in place instead of requiring an additional allocation $D$
    d_C[y * N + x] = val;

    //printf("value in cuda: %f", d_D[row * N + col]);
}