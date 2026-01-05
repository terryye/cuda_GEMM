#!/bin/bash
set -ex

## Make sure NVIDIA GPU drivers already installed.
## Vast.ai cloud instances have them by default.

### Run this script on an Ubuntu 22 / 24 system

## CUDA toolkit, NCCL, OpenMPI, and NVSHMEM will be installed.
## CUDA 12 / CUDA 13 are supported. 

CUDA_VERSION=$(nvidia-smi | grep "CUDA" | grep -oP 'CUDA Version: \K[0-9.]+')

CUDA_VER_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_VER_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

#check if this is a X64 system
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
  echo "This script only supports x86_64 architecture."
  exit 1
fi

#chek if apt has cuda package cuda-toolkit-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} 
if ! apt-cache show cuda-toolkit-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} &> /dev/null ; then
  #check if system is in WSL 
  if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null ; then
    CUDA_REPO="wsl-ubuntu"
  #check if system is ubuntu 22.04
  elif [[ $(lsb_release -rs) == "22.04" ]]; then
    CUDA_REPO="ubuntu2204"
  #check if system is ubuntu 24.04
  elif [[ $(lsb_release -rs) == "24.04" ]]; then
    CUDA_REPO="ubuntu2404"
  else
    echo "This script only supports Ubuntu 22.04 and 24.04 or WSL on Windows with CUDA installed."
    exit 1
  fi
  # Add NVIDIA package repositories
  wget https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo rm cuda-keyring_1.1-1_all.deb

fi

# Install dependencies
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget \
    build-essential \
    cmake \
    cuda-toolkit-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} \

cat >> ~/.bashrc << 'EOF'

# NVIDIA CUDA Toolkit
export PATH=/usr/local/cuda/bin:${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

EOF

# show install success message
echo -e "\033[1;32mNVIDIA CUDA Tool have been successfully installed.\033[0m"
# remind user to source bashrc, in yellow text
echo -e "Please restart your terminal or run '\033[1;33msource ~/.bashrc\033[0m' to apply environment changes."