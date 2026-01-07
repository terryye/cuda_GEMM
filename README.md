# CUDA GEMM Implementations

An learning walk through general matrix multiplication (GEMM) on NVIDIA GPUs, progressing from a naive CUDA kernel to tiled shared-memory, Tensor Core WMMA, and a cuBLASLt-backed implementation. The goal is to understand GEMM deeply by inspecting and running increasingly optimized kernels.

## Overview

-   Four self-contained implementations in `src/01_GEMM`, `02_GEMM_TILING`, `03_GEMM_WMMA`, and `04_GEMM_BLASLt`.
-   Each folder has a minimal `GEMM.cu` and `main.cu` plus a `run.sh` that builds and runs a sanity test.
-   Utilities in `src/util` provide timing, error checking, and simple validation helpers.
-   Headers under `headers/` enable good IntelliSense on macOS; actual execution requires an NVIDIA GPU.

## Development Environment

-   Visual Studio Code (local or Remote SSH) with CUDA syntax support.
-   Nsight Visual Studio Code Edition for profiling/debugging (local or remote).
-   The `.vscode` settings and `headers/` folder are tuned for macOS editing while building/running on a Linux box with GPUs.

## Prerequisites (for running on a GPU machine)

### For Modal (Cloud Execution)

-   Python 3.10+ with pip
-   Modal account and CLI: `pip install modal && python3 -m modal setup`

### For Local Debug Execution (Self Managed linux system with NVIDIA GPUs, e.g. Vast.ai **kvm** instances, or your own server)

-   NVIDIA CUDA Toolkit 12.x or 13.x
-   NVIDIA NCCL library
-   OpenMPI 4.x
-   C++17 compatible compiler (clang++ or g++)
-   NVIDIA NVSHMEM (optional, for advanced implementations)
-   NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
-   run `scripts/install.sh` to set up the linux environment

## Getting Started

### Quick Start

```bash
# Navigate to any GEMM implementation
cd src/01_GEMM        # or 02_GEMM_TILING, 03_GEMM_WMMA, 04_GEMM_BLASLt

# Run on Modal (Cloud)
./run.sh

# Run on self-managed GPU system
./run_local.sh
```

## Implementations

-   `src/01_GEMM`: Baseline CUDA GEMM (global-memory heavy) to establish correctness.
-   `src/02_GEMM_TILING`: Shared-memory tiling for better data reuse and bandwidth.
-   `src/03_GEMM_WMMA`: Tensor Core (WMMA) variant to explore warp-level matrix ops.
-   `src/04_GEMM_BLASLt`: Library-backed GEMM using cuBLASLt with explicit layout descriptors; includes a row-major wrapper over `cublasGemmEx`.

## Utilities

-   `src/util/cuda_helper.h`: CUDA error checking and small helpers.
-   `src/util/time.h`: Simple timing utilities.
-   `src/util/float_eq.h`: Approximate float comparisons for validation.
-   `src/util/color.h`: Colorized console output.

## Scripts

-   `scripts/install.sh`: Convenience setup for a Linux CUDA environment.
-   `scripts/local_gpu.sh`: Example launcher for local GPU runs.
-   `scripts/modal_nvcc.py`: Modal helper for cloud-based compilation/testing.
-   `scripts/testall.sh`: Run all implementations in sequence on chosen backend (Modal or local).
    -   Usage: `./test_all.sh [modal|local]` (default is `modal`)

## Project Structure

```
src/
	01_GEMM/
	02_GEMM_TILING/
	03_GEMM_WMMA/
	04_GEMM_BLASLt/
	util/
headers/
scripts/
```

## Tips

-   Start with `01_GEMM` to confirm correctness, then move to tiled and WMMA versions to see the impact of memory hierarchy and Tensor Cores.
-   Use the cuBLASLt version as a reference for production-grade GEMM and for understanding row-major vs. column-major handling in cuBLAS.
