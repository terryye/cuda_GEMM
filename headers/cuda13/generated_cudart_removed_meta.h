// This file is generated.  Any changes you make will be lost during the next clean build.

// CUDA public interface, for type definitions and api function prototypes
#include "cudart_removed.h"

// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

// Currently used parameter trace structures
typedef struct cudaStreamDestroy_v3020_params_st {
    cudaStream_t stream;
} cudaStreamDestroy_v3020_params;

typedef struct cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6000_params_st {
    int *numBlocks;
    const void *func;
    size_t numDynamicSmemBytes;
} cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6000_params;

typedef struct cudaConfigureCall_v3020_params_st {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem  __dv;
    cudaStream_t stream  __dv;
} cudaConfigureCall_v3020_params;

typedef struct cudaSetupArgument_v3020_params_st {
    const void *arg;
    size_t size;
    size_t offset;
} cudaSetupArgument_v3020_params;

typedef struct cudaLaunch_v3020_params_st {
    const void *func;
} cudaLaunch_v3020_params;

typedef struct cudaLaunch_ptsz_v7000_params_st {
    const void *func;
} cudaLaunch_ptsz_v7000_params;

typedef struct cudaStreamSetFlags_v10200_params_st {
    cudaStream_t hStream;
    unsigned int flags;
} cudaStreamSetFlags_v10200_params;

typedef struct cudaStreamSetFlags_ptsz_v10200_params_st {
    cudaStream_t hStream;
    unsigned int flags;
} cudaStreamSetFlags_ptsz_v10200_params;

typedef struct cudaProfilerInitialize_v4000_params_st {
    const char *configFile;
    const char *outputFile;
    cudaOutputMode_t outputMode;
} cudaProfilerInitialize_v4000_params;

typedef struct cudaThreadSetLimit_v3020_params_st {
    enum cudaLimit limit;
    size_t value;
} cudaThreadSetLimit_v3020_params;

typedef struct cudaThreadGetLimit_v3020_params_st {
    size_t *pValue;
    enum cudaLimit limit;
} cudaThreadGetLimit_v3020_params;

typedef struct cudaThreadGetCacheConfig_v3020_params_st {
    enum cudaFuncCache *pCacheConfig;
} cudaThreadGetCacheConfig_v3020_params;

typedef struct cudaThreadSetCacheConfig_v3020_params_st {
    enum cudaFuncCache cacheConfig;
} cudaThreadSetCacheConfig_v3020_params;

typedef struct cudaSetDoubleForDevice_v3020_params_st {
    double *d;
} cudaSetDoubleForDevice_v3020_params;

typedef struct cudaSetDoubleForHost_v3020_params_st {
    double *d;
} cudaSetDoubleForHost_v3020_params;

typedef struct cudaCreateTextureObject_v2_v11080_params_st {
    cudaTextureObject_t *pTexObject;
    const struct cudaResourceDesc *pResDesc;
    const struct cudaTextureDesc *pTexDesc;
    const struct cudaResourceViewDesc *pResViewDesc;
} cudaCreateTextureObject_v2_v11080_params;

typedef struct cudaGetTextureObjectTextureDesc_v2_v11080_params_st {
    struct cudaTextureDesc *pTexDesc;
    cudaTextureObject_t texObject;
} cudaGetTextureObjectTextureDesc_v2_v11080_params;

typedef struct cudaBindTexture_v3020_params_st {
    size_t *offset;
    const struct textureReference *texref;
    const void *devPtr;
    const struct cudaChannelFormatDesc *desc;
    size_t size  __dv;
} cudaBindTexture_v3020_params;

typedef struct cudaBindTexture2D_v3020_params_st {
    size_t *offset;
    const struct textureReference *texref;
    const void *devPtr;
    const struct cudaChannelFormatDesc *desc;
    size_t width;
    size_t height;
    size_t pitch;
} cudaBindTexture2D_v3020_params;

typedef struct cudaBindTextureToArray_v3020_params_st {
    const struct textureReference *texref;
    cudaArray_const_t array;
    const struct cudaChannelFormatDesc *desc;
} cudaBindTextureToArray_v3020_params;

typedef struct cudaBindTextureToMipmappedArray_v5000_params_st {
    const struct textureReference *texref;
    cudaMipmappedArray_const_t mipmappedArray;
    const struct cudaChannelFormatDesc *desc;
} cudaBindTextureToMipmappedArray_v5000_params;

typedef struct cudaUnbindTexture_v3020_params_st {
    const struct textureReference *texref;
} cudaUnbindTexture_v3020_params;

typedef struct cudaGetTextureAlignmentOffset_v3020_params_st {
    size_t *offset;
    const struct textureReference *texref;
} cudaGetTextureAlignmentOffset_v3020_params;

typedef struct cudaGetTextureReference_v3020_params_st {
    const struct textureReference **texref;
    const void *symbol;
} cudaGetTextureReference_v3020_params;

typedef struct cudaBindSurfaceToArray_v3020_params_st {
    const struct surfaceReference *surfref;
    cudaArray_const_t array;
    const struct cudaChannelFormatDesc *desc;
} cudaBindSurfaceToArray_v3020_params;

typedef struct cudaGetDeviceProperties_v3020_params_st {
    struct cudaDeviceProp *prop;
    int deviceOrdinal;
} cudaGetDeviceProperties_v3020_params;

typedef struct cudaGetSurfaceReference_v3020_params_st {
    const struct surfaceReference **surfref;
    const void *symbol;
} cudaGetSurfaceReference_v3020_params;

typedef struct cudaGraphInstantiate_v10000_params_st {
    cudaGraphExec_t *pGraphExec;
    cudaGraph_t graph;
    cudaGraphNode_t *pErrorNode;
    char *pLogBuffer;
    size_t bufferSize;
} cudaGraphInstantiate_v10000_params;

typedef struct cudaLaunchCooperativeKernelMultiDevice_v9000_params_st {
    struct cudaLaunchParams *launchParamsList;
    unsigned int numDevices;
    unsigned int flags  __dv;
} cudaLaunchCooperativeKernelMultiDevice_v9000_params;

typedef struct cudaMemAdvise_v8000_params_st {
    const void *devPtr;
    size_t count;
    enum cudaMemoryAdvise advice;
    int device;
} cudaMemAdvise_v8000_params;

typedef struct cudaMemPrefetchAsync_v8000_params_st {
    const void *devPtr;
    size_t count;
    int dstDevice;
    cudaStream_t stream;
} cudaMemPrefetchAsync_v8000_params;

typedef struct cudaMemPrefetchAsync_ptsz_v8000_params_st {
    const void *devPtr;
    size_t count;
    int dstDevice;
    cudaStream_t stream;
} cudaMemPrefetchAsync_ptsz_v8000_params;

typedef struct cudaEventElapsedTime_v3020_params_st {
    float *ms;
    cudaEvent_t start;
    cudaEvent_t end;
} cudaEventElapsedTime_v3020_params;

typedef struct cudaGraphGetEdges_v10000_params_st {
    cudaGraph_t graph;
    cudaGraphNode_t *from;
    cudaGraphNode_t *to;
    size_t *numEdges;
} cudaGraphGetEdges_v10000_params;

typedef struct cudaGraphNodeGetDependencies_v10000_params_st {
    cudaGraphNode_t node;
    cudaGraphNode_t *pDependencies;
    size_t *pNumDependencies;
} cudaGraphNodeGetDependencies_v10000_params;

typedef struct cudaGraphNodeGetDependentNodes_v10000_params_st {
    cudaGraphNode_t node;
    cudaGraphNode_t *pDependentNodes;
    size_t *pNumDependentNodes;
} cudaGraphNodeGetDependentNodes_v10000_params;

typedef struct cudaGraphAddDependencies_v10000_params_st {
    cudaGraph_t graph;
    const cudaGraphNode_t *from;
    const cudaGraphNode_t *to;
    size_t numDependencies;
} cudaGraphAddDependencies_v10000_params;

typedef struct cudaGraphRemoveDependencies_v10000_params_st {
    cudaGraph_t graph;
    const cudaGraphNode_t *from;
    const cudaGraphNode_t *to;
    size_t numDependencies;
} cudaGraphRemoveDependencies_v10000_params;

typedef struct cudaGraphAddNode_v12020_params_st {
    cudaGraphNode_t *pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t *pDependencies;
    size_t numDependencies;
    struct cudaGraphNodeParams *nodeParams;
} cudaGraphAddNode_v12020_params;

typedef struct cudaStreamGetCaptureInfo_v10010_params_st {
    cudaStream_t stream;
    enum cudaStreamCaptureStatus *captureStatus_out;
    unsigned long long *id_out  __dv;
    cudaGraph_t *graph_out  __dv;
    const cudaGraphNode_t **dependencies_out  __dv;
    size_t *numDependencies_out  __dv;
} cudaStreamGetCaptureInfo_v10010_params;

typedef struct cudaStreamGetCaptureInfo_ptsz_v10010_params_st {
    cudaStream_t stream;
    enum cudaStreamCaptureStatus *captureStatus_out;
    unsigned long long *id_out  __dv;
    cudaGraph_t *graph_out  __dv;
    const cudaGraphNode_t **dependencies_out  __dv;
    size_t *numDependencies_out  __dv;
} cudaStreamGetCaptureInfo_ptsz_v10010_params;

typedef struct cudaStreamGetCaptureInfo_v2_v11030_params_st {
    cudaStream_t stream;
    enum cudaStreamCaptureStatus *captureStatus_out;
    unsigned long long *id_out  __dv;
    cudaGraph_t *graph_out  __dv;
    const cudaGraphNode_t **dependencies_out  __dv;
    size_t *numDependencies_out  __dv;
} cudaStreamGetCaptureInfo_v2_v11030_params;

typedef struct cudaStreamGetCaptureInfo_v2_ptsz_v11030_params_st {
    cudaStream_t stream;
    enum cudaStreamCaptureStatus *captureStatus_out;
    unsigned long long *id_out  __dv;
    cudaGraph_t *graph_out  __dv;
    const cudaGraphNode_t **dependencies_out  __dv;
    size_t *numDependencies_out  __dv;
} cudaStreamGetCaptureInfo_v2_ptsz_v11030_params;

typedef struct cudaStreamUpdateCaptureDependencies_v11030_params_st {
    cudaStream_t stream;
    cudaGraphNode_t *dependencies;
    size_t numDependencies;
    unsigned int flags  __dv;
} cudaStreamUpdateCaptureDependencies_v11030_params;

typedef struct cudaStreamUpdateCaptureDependencies_ptsz_v11030_params_st {
    cudaStream_t stream;
    cudaGraphNode_t *dependencies;
    size_t numDependencies;
    unsigned int flags  __dv;
} cudaStreamUpdateCaptureDependencies_ptsz_v11030_params;

typedef struct cudaSignalExternalSemaphoresAsync_v10000_params_st {
    const cudaExternalSemaphore_t *extSemArray;
    const struct cudaExternalSemaphoreSignalParams_v1 *paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream  __dv;
} cudaSignalExternalSemaphoresAsync_v10000_params;

typedef struct cudaSignalExternalSemaphoresAsync_ptsz_v10000_params_st {
    const cudaExternalSemaphore_t *extSemArray;
    const struct cudaExternalSemaphoreSignalParams_v1 *paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream  __dv;
} cudaSignalExternalSemaphoresAsync_ptsz_v10000_params;

typedef struct cudaWaitExternalSemaphoresAsync_v10000_params_st {
    const cudaExternalSemaphore_t *extSemArray;
    const struct cudaExternalSemaphoreWaitParams_v1 *paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream  __dv;
} cudaWaitExternalSemaphoresAsync_v10000_params;

typedef struct cudaWaitExternalSemaphoresAsync_ptsz_v10000_params_st {
    const cudaExternalSemaphore_t *extSemArray;
    const struct cudaExternalSemaphoreWaitParams_v1 *paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream  __dv;
} cudaWaitExternalSemaphoresAsync_ptsz_v10000_params;

typedef struct cudaMemcpyBatchAsync_v12080_params_st {
    void **dsts;
    void **srcs;
    size_t *sizes;
    size_t count;
    struct cudaMemcpyAttributes *attrs;
    size_t *attrsIdxs;
    size_t numAttrs;
    size_t *failIdx;
    cudaStream_t stream;
} cudaMemcpyBatchAsync_v12080_params;

typedef struct cudaMemcpyBatchAsync_ptsz_v12080_params_st {
    void **dsts;
    void **srcs;
    size_t *sizes;
    size_t count;
    struct cudaMemcpyAttributes *attrs;
    size_t *attrsIdxs;
    size_t numAttrs;
    size_t *failIdx;
    cudaStream_t stream;
} cudaMemcpyBatchAsync_ptsz_v12080_params;

typedef struct cudaMemcpy3DBatchAsync_v12080_params_st {
    size_t numOps;
    struct cudaMemcpy3DBatchOp *opList;
    size_t *failIdx;
    unsigned long long flags;
    cudaStream_t stream;
} cudaMemcpy3DBatchAsync_v12080_params;

typedef struct cudaMemcpy3DBatchAsync_ptsz_v12080_params_st {
    size_t numOps;
    struct cudaMemcpy3DBatchOp *opList;
    size_t *failIdx;
    unsigned long long flags;
    cudaStream_t stream;
} cudaMemcpy3DBatchAsync_ptsz_v12080_params;

// Parameter trace structures for removed functions


// End of parameter trace structures
