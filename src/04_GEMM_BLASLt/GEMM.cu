#include <stdio.h>
#include <cublasLt.h>
#include "util/cuda_helper.h"

// Host-side GEMM using cuBLASLt (row-major layouts).

void GEMM(float* d_A, float* d_B, float* d_C, int M, int K, int N,
    float alpha, float beta, bool transA, bool transB) {
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasLtMatmulDesc_t opDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // Matrix layouts in row-major order.
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    int rowsA = transA ? K : M;
    int colsA = transA ? M : K;
    int rowsB = transB ? N : K;
    int colsB = transB ? K : N;

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, rowsA, colsA, colsA));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, rowsB, colsB, colsB));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, M, N, N));

    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));

    const size_t workspaceBytes = 1 << 22; // 4 MB workspace cap.
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceBytes, sizeof(workspaceBytes)));

    cublasLtMatmulHeuristicResult_t heuristic;
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle, opDesc, layoutA, layoutB, layoutC, layoutC,
        preference, 1, &heuristic, &returnedResults));

    void* workspace = nullptr;
    if (heuristic.workspaceSize > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, heuristic.workspaceSize));
    }

    if (returnedResults == 0) {
        printf("cuBLASLt heuristic failed to return an algorithm.\n");
        if (workspace) cudaFree(workspace);
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(opDesc);
        cublasLtDestroy(handle);
        return;
    }

    CHECK_CUBLAS(cublasLtMatmul(
        handle, opDesc, &alpha, d_A, layoutA, d_B, layoutB,
        &beta, d_C, layoutC, d_C, layoutC,
        &heuristic.algo, workspace, heuristic.workspaceSize, 0));

    if (workspace) cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtDestroy(handle);
}