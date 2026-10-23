#!/bin/bash
set -euo pipefail

# Run from the directory containing this script and the .cu files.
cd "$(dirname "$0")"

# Source files compiled separately.
SOURCES=(
    sim.cu
    chirp.cu
    elementary_kernels.cu
    facet_funcs.cu
    file_io.cu
    geometry_kernels.cu
    radar_kernels.cu
    scattering.cu
)

# Keep object files in a dedicated directory.
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

# Common compilation and device-link options.
COMMON_FLAGS=(-O3 -lineinfo -rdc=true)
LINK_FLAGS=(-rdc=true -lcufft)

case "$(hostname)" in
    hpdl380g6)
        echo "Compiling for hpdl380g6"
        NVCC_FLAGS=(--compiler-bindir=/usr/bin/g++-8
                    -gencode=arch=compute_30,code=sm_30)
        LINK_FLAGS+=(-lstdc++fs)
        ;;
    CCAR-L-006)
        echo "Compiling for CCAR-L-006"
        NVCC_FLAGS=(-allow-unsupported-compiler)
        ;;
    passiveradarlab)
        echo "Compiling for passiveradarlab"
        NVCC_FLAGS=(-ccbin g++-10)
        ;;
    *)
        echo "Error: unsupported hostname: $(hostname)" >&2
        exit 1
        ;;
esac

# Compile each .cu file into a relocatable object file.
OBJECTS=()
for src in "${SOURCES[@]}"; do
    obj="$BUILD_DIR/${src%.cu}.o"
    echo "Compiling $src -> $obj"
    nvcc -c "${COMMON_FLAGS[@]}" "${NVCC_FLAGS[@]}" \
        "$src" -o "$obj"
    OBJECTS+=("$obj")
done

# Link all objects. nvcc performs the device-link step for RDC code.
echo "Linking executable: sim"
nvcc "${COMMON_FLAGS[@]}" "${NVCC_FLAGS[@]}" \
    "${OBJECTS[@]}" "${LINK_FLAGS[@]}" -o sim

echo "Build successful: ./sim"

