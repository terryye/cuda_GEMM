#include <stdio.h>
#include "../../util/cuda_shim.h"
#include "../../util/color.h"
#include "../../util/float_eq.h"
#include "../../util/time.h"
#include "./GEMM.cu"

void print_matrix(float* matrix, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", matrix[i * col + j]);
        }
        printf("\n");
    }
}
__global__ void mykernel() {
    printf("current block: %d, thread %d , block dim %d, grid dim %d\n", blockIdx.x, threadIdx.x,  blockDim.x, gridDim.x);
    // blockDim.x  the "dimension" of a block refers to # of threads in a block. 
    // gridDim.x the "dimension" of the grid refers to # of blocks in a grid
}
/**
      size_t M = 2  
        , N = 5
        , K = 3;  
 */


void init_matrix(float** h_matrix, int row, int col) {
    size_t size = row * col;
    *h_matrix = (float*)malloc(sizeof(float) * size);
    assertc(h_matrix != NULL);
}

float* copy2cuda(float* h_matrix, int row, int col) {
    size_t size = row * col * sizeof(float);
    float* d_matrix;
    chk(cudaMalloc((void**)&d_matrix, size));
    chk(cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice));
    return d_matrix;
}

void cp2cuda(float* d_matrix, float* h_matrix, int row, int col) {
    size_t size = row * col * sizeof(float);
    chk(cudaMemcpy( d_matrix, h_matrix, size, cudaMemcpyHostToDevice));
}

float* init_matrix_with_value(int row, int col, float value) {
    size_t size = row * col;
    float* h_matrix = (float*)malloc(sizeof(float) * size);
    assertc(h_matrix != NULL);
    for (int i = 0; i < size; i++) {
        h_matrix[i] = value;
    }
    return h_matrix;
}

int randomInt(int min, int max) {
    return min + (rand() % (max - min + 1));
}

int randomCheck(float* matrix, int row, int col, float value) {
    size_t size = row * col;
    for (int i = 0; i < 10000 && i < size; i++) {
        int idx = randomInt(0, size - 1);
        if( !float_equal(matrix[idx], value)){
            printf("FAILED: matrix[%d] = %f, expected %f\n", idx, matrix[idx], value);
            return 0;
        }
    }
 
    return 1;
}

void test_kernel() {
    #ifndef __INTELLISENSE__
    mykernel<<<2, 4>>>();
    #endif
    chk(cudaGetLastError());
    chk(cudaDeviceSynchronize());
    green("PASSED: base env check\n");
}

void test(int m, int n, int k){
    printf("Test with M=%d, N=%d, K=%d\n", m, n, k);
    float v_a = 1.0f;
    float v_b = 2.0f;
    float v_c = 3.0f;
    float co_a = 2.0f;
    float co_b = 6.0f;
    //init
    float* h_A = init_matrix_with_value(m, k, v_a);
    float* h_A_T = init_matrix_with_value(k, m, v_a);
    float* h_B = init_matrix_with_value(k, n, v_b);
    float* h_B_T = init_matrix_with_value(n, k, v_b);
    float* h_C = init_matrix_with_value(m, n, v_c);
    float* h_O = init_matrix_with_value(m, n, 0.0f);

    float* d_A = copy2cuda(h_A, m, k);
    float* d_B = copy2cuda(h_B, k, n);
    float* d_C = copy2cuda(h_C, m, n);

    float* d_A_T = copy2cuda(h_A_T, k, m);
    float* d_B_T = copy2cuda(h_B_T, n, k);

    // Launch kernel
    dim3 threadsPerBlock = new_dim3(16, 16, 1);
    int blocks_x = (n - 1 ) / threadsPerBlock.x + 1;
    int blocks_y = (m - 1 ) / threadsPerBlock.y + 1;
    dim3 blocksPerGrid  = new_dim3(blocks_x, blocks_y, 1);


    printf("check no transpose \n");
    #ifndef __INTELLISENSE__
    GEMM<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m , k , n , co_a, co_b, false, false);
    #endif
    //cp back
    chk(cudaMemcpy(h_O, d_C,  m * n * sizeof(float), cudaMemcpyDeviceToHost));
    chk(cudaGetLastError());
    chk(cudaDeviceSynchronize());   
    int ret = randomCheck( h_O, m, n, co_a * k * v_a * v_b + co_b * v_c);
    if (ret){
        green("PASSED\n");
    } else {
        red("FAILED\n");
    }

    printf("check transpose A\n");
    //reset C
//    cudaMemcpy( d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cp2cuda( d_C, h_C, m, n);

    #ifndef __INTELLISENSE__
    GEMM<<<blocksPerGrid, threadsPerBlock>>>(d_A_T, d_B, d_C, m , k , n , co_a, co_b, true, false);
    #endif
    chk(cudaMemcpy(h_O, d_C,  m * n * sizeof(float), cudaMemcpyDeviceToHost));
    
    chk(cudaGetLastError());
    chk(cudaDeviceSynchronize());   
    ret = randomCheck( h_O, m, n, co_a * k * v_a * v_b + co_b * v_c);
    if (ret){
        green("PASSED\n");
    } else {
        red("FAILED\n");
    }



    printf("check transpose B\n");
    cp2cuda( d_C, h_C, m, n);
    #ifndef __INTELLISENSE__
    GEMM<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B_T, d_C, m , k , n , co_a, co_b, false, true);
    #endif
    chk(cudaMemcpy(h_O, d_C,  m * n * sizeof(float), cudaMemcpyDeviceToHost));
    
    chk(cudaGetLastError());
    chk(cudaDeviceSynchronize());   
    ret = randomCheck( h_O, m, n, co_a * k * v_a * v_b + co_b * v_c);
    if (ret){
        green("PASSED\n");
    } else {
        red("FAILED\n");
    }


    printf("check transpose A & B\n");
    cp2cuda( d_C, h_C, m, n);
    #ifndef __INTELLISENSE__
    GEMM<<<blocksPerGrid, threadsPerBlock>>>(d_A_T, d_B_T, d_C, m , k , n , co_a, co_b, true, true);
    #endif
    chk(cudaMemcpy(h_O, d_C,  m * n * sizeof(float), cudaMemcpyDeviceToHost));
    
    chk(cudaGetLastError());
    chk(cudaDeviceSynchronize());   
    ret = randomCheck( h_O, m, n, co_a * k * v_a * v_b + co_b * v_c);
    if (ret){
        green("PASSED\n");
    } else {
        red("FAILED\n");
    }

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_O);
    free(h_A_T);
    free(h_B_T);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_T);
    cudaFree(d_B_T);

    
}

void test_example() {
    printf("Test with example data\n");
    size_t M = 2  
        , N = 5
        , K = 3;  

    float co_a = 2.0f;
    float co_b = 6.0f;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // M x K = (2, 3)
    float Adata[] = {
        1.0f, 2.0f, 3.0f,
        7.0f, 11.0f, 13.0f,
    };

    // K * M = (3, 2)
    float Adata_T[] = {
        1.0f, 7.0f,
        2.0f, 11.0f, 
        3.0f, 13.0f,
    };

    // K x N = (3, 5)
    float Bdata[] = {
        1.0f,   2.0f,   3.0f,    7.0f,  11.0f,
       13.0f,  17.0f,  19.0f,   23.0f,  29.0f,
       31.0f,  37.0f,  41.0f,   43.0f,  51.0f,
    };

    // N x K  = (5, 3)
    float Bdata_T[] = {
        1.0f, 13.0f, 31.0f,
        2.0f, 17.0f, 37.0f,
        3.0f, 19.0f, 41.0f,
        7.0f, 23.0f, 43.0f,
       11.0f, 29.0f, 51.0f,
    };

    // M x N = (2, 5)
    float Cdata[] = {
        120.0f,   147.0f,  164.0f,  182.0f,  222.0f,
        553.0f,  682.0f,   763.0f,  861.0f,  1059.0f
    };
    
    float Ddata[] = {
        960.0,    1176.0,   1312.0,   1456.0,   1776.0,
        4424.0,   5456.0,   6104.0,   6888.0,   8472.0 
    }  ;


    float *h_A = 0, *h_B= 0, *h_C= 0; // host matrices
    float *h_A_T, *h_B_T; // host matrices

    init_matrix(&h_A, M, K);
    init_matrix(&h_B, K, N);
    init_matrix(&h_C, M, N);
    
    init_matrix(&h_A_T, K, M);
    init_matrix(&h_B_T, N, K);

    memcpy(h_A, Adata, size_A);
    memcpy(h_B, Bdata, size_B);
    memcpy(h_C, Cdata, size_C);

    memcpy(h_A_T, Adata_T, size_A);
    memcpy(h_B_T, Bdata_T, size_B);


    //cp to cuda    
    float *d_A, *d_B, *d_C; // device matrices
    float *d_A_T, *d_B_T; // device matrices

    d_A = copy2cuda(h_A, M, K);
    d_B = copy2cuda(h_B, K, N);
    d_C = copy2cuda(h_C, M, N);

    d_A_T = copy2cuda(h_A_T, K, M);
    d_B_T = copy2cuda(h_B_T, N, K);
    

    // Launch kernel
    dim3 threadsPerBlock = new_dim3(16, 16, 1);
    int blocksPerGridX = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blocksPerGridY = (M + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 blocksPerGrid = new_dim3(blocksPerGridX, blocksPerGridY, 1);


    printf("check no transpose \n");
    #ifndef __INTELLISENSE__
    GEMM<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M , K , N , co_a, co_b, false, false);
    #endif
    //cp back
    chk(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    check_float_array_eq( h_C, Ddata, M * N);
    printf("check transpose A\n");
    d_C = copy2cuda(Cdata, M, N);
    #ifndef __INTELLISENSE__
    GEMM<<<blocksPerGrid, threadsPerBlock>>>(d_A_T, d_B, d_C, M , K , N , co_a, co_b, true, false);
    #endif

    chk(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    print_matrix(h_C, M, N);

    check_float_array_eq( h_C, Ddata, M * N);

    printf("check transpose B\n");
    d_C = copy2cuda(Cdata, M, N);

    #ifndef __INTELLISENSE__
    GEMM<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B_T, d_C, M , K , N , co_a, co_b, false, true);
    #endif

    chk(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    check_float_array_eq( h_C, Ddata, M * N);



    printf("check transpose A & B\n");
    d_C = copy2cuda(Cdata, M, N);

    #ifndef __INTELLISENSE__
    GEMM<<<blocksPerGrid, threadsPerBlock>>>(d_A_T, d_B_T, d_C, M , K , N , co_a, co_b, true, true);
    #endif

    chk(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    check_float_array_eq( h_C, Ddata, M * N);

    chk(cudaDeviceSynchronize());
    chk(cudaGetLastError());

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_A_T);
    free(h_B_T);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_T);
    cudaFree(d_B_T);

    green("Finished\n");

}


int main() { 
    test_kernel();
    test_example();
    test(1,1,1);
    test(8,2,10);
    test(17,16,160);
    test(800,200,100);
    test(1024,1024,1024); 
    return 0;
}