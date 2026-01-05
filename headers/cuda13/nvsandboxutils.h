/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NVSANDBOXUTILS_H__
#define __NVSANDBOXUTILS_H__

#ifdef __cplusplus
extern "C" {
#endif

#define INPUT_LENGTH            256
#define MAX_FILE_PATH           256
#define MAX_NAME_LENGTH         256


/***************************************************************************************************/
/** @defgroup enums Enumerations
 *  @{
 */
/***************************************************************************************************/

/**
 * Return types
 */
typedef enum
{
    NVSANDBOXUTILS_SUCCESS                              = 0,         //!< The operation was successful
    NVSANDBOXUTILS_ERROR_UNINITIALIZED                  = 1,         //!< The library wasn't successfully initialized
    NVSANDBOXUTILS_ERROR_NOT_SUPPORTED                  = 2,         //!< The requested operation is not supported on target device
    NVSANDBOXUTILS_ERROR_INVALID_ARG                    = 3,         //!< A supplied argument is invalid
    NVSANDBOXUTILS_ERROR_INSUFFICIENT_SIZE              = 4,         //!< A supplied argument is not large enough
    NVSANDBOXUTILS_ERROR_VERSION_NOT_SUPPORTED          = 5,         //!< Requested library version is not supported
    NVSANDBOXUTILS_ERROR_LIBRARY_LOAD                   = 6,         //!< The library load failed
    NVSANDBOXUTILS_ERROR_FUNCTION_NOT_FOUND             = 7,         //!< Called function was not found
    NVSANDBOXUTILS_ERROR_DEVICE_NOT_FOUND               = 8,         //!< Target device was not found
    NVSANDBOXUTILS_ERROR_NVML_LIB_CALL                  = 9,         //!< NVML library call failed
    NVSANDBOXUTILS_ERROR_OUT_OF_MEMORY                  = 10,        //!< There is insufficient memory
    NVSANDBOXUTILS_ERROR_FILEPATH_NOT_FOUND             = 11,        //!< A supplied file path was not found
    NVSANDBOXUTILS_ERROR_UNKNOWN                        = 0xFFFF,    //!< Unknown error occurred
} nvSandboxUtilsRet_t;

/**
 * Return if there is an error
 */
#define RETURN_ON_SANDBOX_ERROR(result) \
    if ((result) != NVSANDBOXUTILS_SUCCESS) { \
        NVSANDBOXUTILS_ERROR_MSG("%s %d result=%d", __func__, __LINE__, result); \
        return result; \
    }

/**
 * Log levels
 */
typedef enum
{
    NVSANDBOXUTILS_LOG_LEVEL_FATAL         = 0,                       //!< Log fatal errors
    NVSANDBOXUTILS_LOG_LEVEL_ERROR         = 1,                       //!< Log all errors
    NVSANDBOXUTILS_LOG_LEVEL_WARN          = 2,                       //!< Log all warnings
    NVSANDBOXUTILS_LOG_LEVEL_DEBUG         = 3,                       //!< Log all debug messages
    NVSANDBOXUTILS_LOG_LEVEL_INFO          = 4,                       //!< Log all info messages
    NVSANDBOXUTILS_LOG_LEVEL_NONE          = 0xFFFF,                  //!< Log none
} nvSandboxUtilsLogLevel_t;

/**
 * Input rootfs to help access files inside the driver container
 */
typedef enum
{
    NV_ROOTFS_DEFAULT,                                               //!< Default no rootfs
    NV_ROOTFS_PATH,                                                  //!< /run/nvidia/driver
    NV_ROOTFS_PID,                                                   //!< /proc/PID/mountinfo
} nvSandboxUtilsRootfsInputType_t;

/**
 * File type
 */
typedef enum
{
    NV_DEV,                                                          //!< /dev file system
    NV_PROC,                                                         //!< /proc file system
    NV_SYS,                                                          //!< /sys file system
    NV_FILE_TYPE_NONE = 0xFFFF,
} nvSandboxUtilsFileType_t;

/**
 * File subtype
 */
typedef enum
{
    NV_DEV_NVIDIA,                                                   //!< /dev/nvidia0
    NV_DEV_DRI_CARD,                                                 //!< /dev/dri/card1
    NV_DEV_DRI_RENDERD,                                              //!< /dev/dri/renderD128
    NV_DEV_DRI_CARD_SYMLINK,                                         //!< /dev/dri/by-path/pci-0000:41:00.0-card
    NV_DEV_DRI_RENDERD_SYMLINK,                                      //!< /dev/dri/by-path/pci-0000:41:00.0-render
    NV_DEV_NVIDIA_UVM,                                               //!< /dev/nvidia-uvm
    NV_DEV_NVIDIA_UVM_TOOLS,                                         //!< /dev/nvidia-uvm-tools
    NV_DEV_NVIDIA_MODESET,                                           //!< /dev/nvidia-uvm-modeset
    NV_DEV_NVIDIA_CTL,                                               //!< /dev/nvidiactl
    NV_DEV_GDRDRV,                                                   //!< /dev/gdrdrv
    NV_DEV_NVIDIA_CAPS_NVIDIA_CAP,                                   //!< /dev/nvidia-caps/nvidia-cap22
    NV_PROC_DRIVER_NVIDIA_GPUS_PCIBUSID,                             //!< /proc/driver/nvidia/gpus/0000:2d:00.0
    NV_PROC_DRIVER_NVIDIA_GPUS,                                      //!< /proc/driver/nvidia/gpus (for mask out)
    NV_PROC_NVIDIA_PARAMS,                                           //!< /proc/driver/nvidia/params
    NV_PROC_NVIDIA_CAPS_MIG_MINORS,                                  //!< /proc/driver/nvidia-caps/mig-minors
    NV_PROC_DRIVER_NVIDIA_CAPABILITIES_GPU,                          //!< /proc/driver/nvidia/capabilities/gpu0
    NV_PROC_DRIVER_NVIDIA_CAPABILITIES,                              //!< /proc/driver/nvidia/capabilities (for mask out)
    NV_PROC_DRIVER_NVIDIA_CAPABILITIIES_GPU_MIG_CI_ACCESS,           //!< proc/driver/nvidia/capabilities/gpu0/mig/gi2/ci0/access
    NV_SYS_MODULE_NVIDIA_DRIVER_PCIBUSID,                            //!< /sys/module/nvidia/drivers/pci:nvidia/0000:2d:00.0
    NV_SYS_MODULE_NVIDIA_DRIVER,                                     //!< /sys/module/nvidia/drivers/pci:nvidia (for mask out)
    NV_NUM_SUBTYPE, // always at the end.
    NV_FILE_SUBTYPE_NONE = 0xFFFF,
} nvSandboxUtilsFileSystemSubType_t;

/**
 * File module
 */
typedef enum
{
    NV_GPU,                                                           //!< Target device
    NV_MIG,                                                           //!< Target device- MIG
    NV_DRIVER_NVIDIA,                                                 //!< NVIDIA kernel driver
    NV_DRIVER_NVIDIA_UVM,                                             //!< NVIDIA kernel driver-UVM
    NV_DRIVER_NVIDIA_MODESET,                                         //!< NVIDIA kernel driver-modeset
    NV_DRIVER_GDRDRV,                                                 //!< GDRDRV driver
    NV_SYSTEM,                                                        //!< System module
    NV_FILE_MODULE_NONE = 0xFFFF,
} nvSandboxUtilsFileModule_t;

/**
 * Flag to provide additional details about the file
 */
typedef enum
{
    NV_FILE_FLAG_NONE            =  0,
    NV_FILE_FLAG_HINT            = (1 << 0),                         //!< Default no hint
    NV_FILE_FLAG_MASKOUT         = (1 << 1),                         //!< For /proc/driver/nvidia/gpus
    NV_FILE_FLAG_CONTENT         = (1 << 2),                         //!< For /proc/driver/nvidia/params
                                                                     //!< For SYMLINK
                                                                     //!< Use \p nvSandboxUtilsGetFileContent to get name of the linked file
    NV_FILE_FLAG_DEPRECATED      = (1 << 3),                         //!< For all the FIRMWARE GSP file
    NV_FILE_FLAG_CANDIDATES      = (1 << 4),                         //!< For libcuda.so
} nvSandboxUtilsFileFlag_t;

/**
 * Input type of the target device
 */
typedef enum
{
    NV_GPU_INPUT_GPU_UUID,                                           //!< GPU UUID
    NV_GPU_INPUT_MIG_UUID,                                           //!< MIG UUID
    NV_GPU_INPUT_PCI_ID,                                             //!< PCIe DBDF ID
    NV_GPU_INPUT_PCI_INDEX,                                          //!< PCIe bus order (0 points to the GPU that has lowest PCIe BDF)
} nvSandboxUtilsGpuInputType_t;

typedef enum
{
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_LIB                         = 0,
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_BINARY                      = 1,
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_DATA                        = 2,
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_ICD                         = 3,
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_PROFILE                     = 4,
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_JSON                        = 5,
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_FIRMWARE                    = 6,
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_MODPROBE                    = 7,
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_SYMLINK                     = 8,
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_XMODULE_SHARED_LIB          = 9,
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_GLX_MODULE_SHARED_LIB       = 10,
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_IPC                         = 11,
    NVSANDBOXUTILS_NUM_DRIVER_FILE_TYPE,
    NVSANDBOXUTILS_DRIVER_FILE_TYPE_INVALID                     = 0xFFFF,
} nvSandboxUtilsDriverFileType_t;

typedef enum
{
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_FIRMWARE                = 0,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_MODPROBE                = 1,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_NVPD                    = 2,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_UTILS                   = 3,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_NVML                    = 4,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_NVAPI                   = 5,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_CUDA                    = 6,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_OPENCL                  = 7,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_OPTIX                   = 8,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_VIDEO                   = 9,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_EGL                     = 10,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_EGL_HEADLESS            = 11,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_EGL_GBM                 = 12,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_EGL_WAYLAND             = 13,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_EGL_X11                 = 14,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_GLX                     = 15,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_GBM                     = 16,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_VULKAN                  = 17,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_POWERD                  = 18,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_XDRIVER                 = 19,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_NGX                     = 20,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_VGPU                    = 21,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_NVSANDBOXUTILS          = 22,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_NVPRESENT               = 23,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_IPC                     = 24,
    NVSANDBOXUTILS_NUM_DRIVER_FILE_CATEGORY,
    NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_INVALID                 = 0xFFFF,
} nvSandboxUtilsDriverFileCategory_t;

typedef enum
{
    NVSANDBOXUTILS_DRIVER_FILE_SUBTYPE_IPC_SOCKET_MPS_PIPE      = 0,
    NVSANDBOXUTILS_DRIVER_FILE_SUBTYPE_IPC_SOCKET_FM            = 1,
    NVSANDBOXUTILS_DRIVER_FILE_SUBTYPE_IPC_SOCKET_PERSISTENCED  = 2,
    NVSANDBOXUTILS_NUM_DRIVER_FILE_SUBTYPE                      = 3,
    NVSANDBOXUTILS_DRIVER_FILE_SUBTYPE_INVALID                  = 0xFFFF,
} nvSandboxUtilsDriverFileSubtype_t;

/** @} */

/***************************************************************************************************/
/** @defgroup dataTypes Structures and Unions
 *  @{
 */
/***************************************************************************************************/

/**
 * Initalization input v1
 */
typedef struct
{
    unsigned int version;                                            //!< Version for the structure
    nvSandboxUtilsRootfsInputType_t type;                            //!< One of \p nvSandboxUtilsRootfsInputType_t
    char value[INPUT_LENGTH];                                        //!< String representation of input
} nvSandboxUtilsInitInput_v1_t;

typedef nvSandboxUtilsInitInput_v1_t nvSandboxUtilsInitInput_t;

/**
 * File system information
 */
typedef struct nvSandboxUtilsGpuFileInfo_v1_t
{
    struct nvSandboxUtilsGpuFileInfo_v1_t *next;                     //!< Pointer to the next node in the linked list
    nvSandboxUtilsFileType_t fileType;                               //!< One of \p nvSandboxUtilsFileType_t
    nvSandboxUtilsFileSystemSubType_t fileSubType;                   //!< One of \p nvSandboxUtilsFileSystemSubType_t
    nvSandboxUtilsFileModule_t module;                               //!< One of \p nvSandboxUtilsFileModule_t
    nvSandboxUtilsFileFlag_t flags;                                  //!< One of \p nvSandboxUtilsFileFlag_t
    char *filePath;                                                  //!< Absolute file path
}nvSandboxUtilsGpuFileInfo_v1_t;

/**
 * GPU resource request v1
 */
typedef struct
{
    unsigned int version;                                           //!< Version for the structure
    nvSandboxUtilsGpuInputType_t inputType;                         //!< One of \p nvSandboxUtilsGpuInputType_t
    char input[INPUT_LENGTH];                                       //!< String representation of input
    nvSandboxUtilsGpuFileInfo_v1_t *files;                          //!< Linked list of \ref nvSandboxUtilsGpuFileInfo_v1_t
} nvSandboxUtilsGpuRes_v1_t;

typedef nvSandboxUtilsGpuRes_v1_t nvSandboxUtilsGpuRes_t;

/*
 * Generic bitmask to hold 255 bits, represented by 8 elements of 32 bits
 */
#define NVSANDBOXUTILS_255_MASK_BITS_PER_ELEM     32
#define NVSANDBOXUTILS_255_MASK_NUM_ELEMS         8
#define NVSANDBOXUTILS_MAX_CATEGORIES             NVSANDBOXUTILS_255_MASK_NUM_ELEMS * NVSANDBOXUTILS_255_MASK_BITS_PER_ELEM

#define NVSANDBOXUTILS_255_MASK_BIT_SET(index, nvSandboxUtilsMask)                      \
    nvSandboxUtilsMask.mask[index / NVSANDBOXUTILS_255_MASK_BITS_PER_ELEM] |= (1 << (index % NVSANDBOXUTILS_255_MASK_BITS_PER_ELEM))

#define NVSANDBOXUTILS_255_MASK_BIT_GET(index, nvSandboxUtilsMask)                      \
    nvSandboxUtilsMask.mask[index / NVSANDBOXUTILS_255_MASK_BITS_PER_ELEM] & (1 << (index % NVSANDBOXUTILS_255_MASK_BITS_PER_ELEM))

#define NVSANDBOXUTILS_255_MASK_BIT_SET_PTR(index, nvSandboxUtilsMask)                  \
    nvSandboxUtilsMask->mask[index / NVSANDBOXUTILS_255_MASK_BITS_PER_ELEM] |= (1 << (index % NVSANDBOXUTILS_255_MASK_BITS_PER_ELEM))

#define NVSANDBOXUTILS_255_MASK_BIT_GET_PTR(index, nvSandboxUtilsMask)                  \
    nvSandboxUtilsMask->mask[index / NVSANDBOXUTILS_255_MASK_BITS_PER_ELEM] & (1 << (index % NVSANDBOXUTILS_255_MASK_BITS_PER_ELEM))

typedef struct
{
    unsigned int mask[NVSANDBOXUTILS_255_MASK_NUM_ELEMS];            //<! Array to hold 255 bits
} nvSandboxUtilsMask255_t;

#define  NVSANDBOXUTILS_DRIVER_FILE_CATEGORY_ALL_MASK                ((1 << 256) - 1) //<! 64F To request all categories

typedef struct nvSandboxUtilsDriverFile_v1_t
{
    struct nvSandboxUtilsDriverFile_v1_t *next;
    nvSandboxUtilsDriverFileType_t type;
    nvSandboxUtilsDriverFileSubtype_t subtype;
    nvSandboxUtilsMask255_t categories;                              //<! Bit mask that can hold multiple categories
    nvSandboxUtilsMask255_t flags;                                   //<! Bit mask that can hold multiple flags
    char *filePath;                                                  //<! Absolute file path
} nvSandboxUtilsDriverFile_v1_t;

typedef struct
{
    unsigned int version;
    nvSandboxUtilsDriverFile_v1_t *files;
    nvSandboxUtilsMask255_t requestFlags;
} nvSandboxUtilsDriverFileInfo_v1_t;

typedef nvSandboxUtilsDriverFileInfo_v1_t nvSandboxUtilsDriverFileInfo_t;

/** @} */

/***************************************************************************************************/
/** @defgroup funcs Functions
 *  @{
 */
/***************************************************************************************************/

/* *************************************************
 * Initialize library
 * *************************************************
 */
/**
 * Prepare library resources before library API can be used.
 * This initialization will not fail if one of the initialization prerequisites fails.
 * @param           input  Reference to the called-supplied input struct that has initialization fields
 *
 * @returns         @ref NVSANDBOXUTILS_SUCCESS                      on success
 * @returns         @ref NVSANDBOXUTILS_ERROR_INVALID_ARG            if \p input->value isn't a valid rootfs path
 * @returns         @ref NVSANDBOXUTILS_ERROR_VERSION_NOT_SUPPORTED  if \p input->version isn't supported by the library
 * @returns         @ref NVSANDBOXUTILS_ERROR_FILEPATH_NOT_FOUND     if any of the required file paths are not found during initialization
 * @returns         @ref NVSANDBOXUTILS_ERROR_OUT_OF_MEMORY          if there is insufficient system memory during initialization
 * @returns         @ref NVSANDBOXUTILS_ERROR_LIBRARY_LOAD           on any error during loading the library
 */
nvSandboxUtilsRet_t nvSandboxUtilsInit(nvSandboxUtilsInitInput_t *input);

/* *************************************************
 * Shutdown library
 * *************************************************
 */
/**
 * Clean up library resources created by init call
 *
 * @returns         @ref NVSANDBOXUTILS_SUCCESS                      on success
 */
nvSandboxUtilsRet_t nvSandboxUtilsShutdown(void);

/* *************************************************
 * Get NVIDIA RM driver version
 * *************************************************
 */
/**
 * Get NVIDIA RM driver version
 * @param           version  Reference to caller-supplied buffer to return driver version string
 * @param           length   The maximum allowed length of the string returned in \p version
 *
 * @returns         @ref NVSANDBOXUTILS_SUCCESS                      on success
 * @returns         @ref NVSANDBOXUTILS_ERROR_INVALID_ARG            if \p version is NULL
 * @returns         @ref NVSANDBOXUTILS_ERROR_INSUFFICIENT_SIZE      if \p length is not large enough to hold the version
 */
nvSandboxUtilsRet_t nvSandboxUtilsGetDriverVersion(char *version, unsigned int length);

/* *************************************************
 * Get /dev, /proc, /sys file system information
 * *************************************************
 */
/**
 * Get /dev, /proc, /sys file system information
 * @param           request  Reference to caller-supplied request struct to return the file system information
 *
 * @returns         @ref NVSANDBOXUTILS_SUCCESS                      on success
 * @returns         @ref NVSANDBOXUTILS_ERROR_INVALID_ARG            if \p request->input doesn't match any device
 * @returns         @ref NVSANDBOXUTILS_ERROR_VERSION_NOT_SUPPORTED  if \p request->version isn't supported by the library
 */
nvSandboxUtilsRet_t nvSandboxUtilsGetGpuResource(nvSandboxUtilsGpuRes_t *request);

/* *************************************************
 * Get content of given file path
 * *************************************************
 */
/**
 * Get file content of input file path
 * @param           filePath     Reference to the file path
 * @param           content      Reference to the caller-supplied buffer to return the file content
 * @param           contentSize  Reference to the maximum allowed size of content. It is updated to the actual size of the content on return
 *
 * @returns         @ref NVSANDBOXUTILS_SUCCESS                      on success
 * @returns         @ref NVSANDBOXUTILS_ERROR_INVALID_ARG            if \p filePath or \p content is NULL
 * @returns         @ref NVSANDBOXUTILS_ERROR_INSUFFICIENT_SIZE      if \p contentSize is too small
 * @returns         @ref NVSANDBOXUTILS_ERROR_FILEPATH_NOT_FOUND     on an error while obtaining the content for the file path
 */
nvSandboxUtilsRet_t nvSandboxUtilsGetFileContent(char *filePath, char *content, unsigned int *contentSize);

/* *************************************************
 * Get driver files information
 * *************************************************
 */
/**
 * Get driver files information
 * @param           request  Reference to caller-supplied request struct to return the driver files information
 *
 * @returns         @ref NVSANDBOXUTILS_SUCCESS                      on success
 * @returns         @ref NVSANDBOXUTILS_ERROR_INVALID_ARG            if \p request->requestFlags doesn't have a valid value
 * @returns         @ref NVSANDBOXUTILS_ERROR_VERSION_NOT_SUPPORTED  if \p request->version isn't supported by the library
 */
nvSandboxUtilsRet_t nvSandboxUtilsGetDriverFileInfo(nvSandboxUtilsDriverFileInfo_t *request);

/** @} */

#ifdef __cplusplus
}
#endif
#endif // __NVSANDBOXUTILS_H__
