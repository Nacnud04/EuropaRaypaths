#!/bin/bash

if [ "$(hostname)" = "hpdl380g6" ]; then
    echo "Compiling for hpdl380g6"
    nvcc sim.cu -O3 -lineinfo -o sim -lcufft \
    --compiler-bindir=/usr/bin/g++-8 \
    -gencode=arch=compute_30,code=sm_30

elif [ "$(hostname)" = "CCAR-L-006" ]; then
    echo "Compiling for CCAR-L-006"
    nvcc sim.cu -O3 -lineinfo -o sim -lcufft \
    -allow-unsupported-compiler
fi
