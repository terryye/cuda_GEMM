__global__ void GEMM(float *d_A, float *d_B, float *d_C, int M, int K, int N, float alpha, float beta, bool transA, bool transB) {
   // printf("current block: (%d,%d), thread (%d,%d) , block dim (%d, %d)\n", blockIdx.x,  blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x, blockDim.y);

    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col

    // check boundaries
    if (y >= M  || x >= N) return;

    // Compute D = alpha * A * B + beta * C
    //alpha * A * B 
    float val = 0.0f;
    for(int i = 0; i < K; i++){

        float a = transA ? d_A[i * M + y] : d_A[y * K + i]; // row: y; col: i
        float b = transB ? d_B[x * K + i] : d_B[i * N + x]; // row : i;  col: x
        val +=  a * b;
    }
    //a*(A*B) + beta * C
    val = alpha * val +  d_C [y * N + x] * beta;  // row : row;  col: col

    //updates $C$ in place instead of requiring an additional allocation $D$
    d_C[y * N + x] = val;

    //printf("value in cuda: %f", d_D[row * N + col]);
}