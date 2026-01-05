#!/bin/bash

set -Eeuoa pipefail


# Check OS and show alert if Mac or Windows
os=$(uname -s)
if [[ "$os" == "Darwin" ]] || [[ "$os" == *"MINGW"* ]] || [[ "$os" == *"CYGWIN"* ]] || [[ "$os" == "MSYS"* ]]; then
    echo -e "\033[33m Alert: local execution ONLY supported on Linux \033[0m"
    exit 1
fi

mkdir -p ./bin
rm -rf ./bin/output.bin
rm -rf ./bin/*.bin.dSYM

# Check that there are 1 or 2 arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 1 ]; then
  echo "Usage: $0 <path_to_cuda_file> "
  exit 1
fi

# Check that $1 is a path to an existing file
if [ ! -f "$1" ]; then
  echo "Error: '$1' is not a valid file."
  exit 2
fi

# Get the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# Get the src directory 
SRC_DIR=$(dirname "$SCRIPT_DIR")/src

# compile and run the code
nvcc -DCUDA=1 -g -G -rdc=true \
       -lstdc++ \
       -lm \
       -lmpi \
       -lcudart \
       -lcudadevrt \
        -I/usr/lib/x86_64-linux-gnu/openmpi/include \
        -I ${SRC_DIR} \
        -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
       -o ./bin/output.bin $1

# run the program
./bin/output.bin

