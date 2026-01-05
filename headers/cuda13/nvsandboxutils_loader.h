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

#ifndef __NVSANDBOXUTILS_LOADER_H__
#define __NVSANDBOXUTILS_LOADER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "nvsandboxutils.h"

typedef struct
{
    unsigned int version;                 // version for files structure
    nvSandboxUtilsRootfsInputType_t type;   // default, rootfs filepath, driver container pid. 
    char value[INPUT_LENGTH];             // string representation of input
} nvSandboxUtilsLoadLibraryInput_v1_t;

typedef nvSandboxUtilsLoadLibraryInput_v1_t nvSandboxUtilsLoadLibraryInput_t;

nvSandboxUtilsRet_t nvSandboxUtilsLoadLibrary(nvSandboxUtilsLoadLibraryInput_t *input);

#ifdef __cplusplus
}
#endif

#endif // __NVSANDBOXUTILS_LOADER_H__
