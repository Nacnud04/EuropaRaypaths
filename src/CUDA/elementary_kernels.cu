/******************************************************************************
 * File:        elementary_kernels.cu
 * Author:      Duncan Byrne
 * Institution: Univeristy of Colorado Boulder
 * Department:  Aerospace Engineering Sciences
 * Email:       duncan.byrne@colorado.edu
 * Date:        2025-11-07
 *
 * Description:
 *    File providing elementary CUDA kernels for mathematical functions
 *
 * Contents:
 *    - Implementations of many trig functions
 *    - Vector operations such as distance and dot product
 *
 * Usage:
 *    #include "elementary_kernels.cu"
 * 
 * Notes:
 *
 ******************************************************************************/

// FFT for chirp convolution
#include <cufft.h>

#define THREADS 256 // threads per block

// --- TRIG FUNCTIONS ---

// slow but accurate sin function
__device__ void slowSinBulk(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sinf(input[idx]);
    }
}
__device__ float slowSin(float input) {
    return sinf(input);
}
__device__ float sinGPU(float x) {
    return slowSin(x);
    // Approximate sin(x) using a Taylor series expansion
    // sin(x) ≈ x - x^3/6 + x^5/120 - x^7/5040 for small x
    //float x2 = x * x;
    //return x * (1 - x2 / 6 + x2 * x2 / 120 - x2 * x2 * x2 / 5040);
}

// sinc function
__device__ float sinc(float x) {
    if (fabsf(x) < 1e-6f) return 1.0f;
    return slowSin(M_PI * x) / (M_PI * x);
}

// slow but accurate arcsin function
__device__ void slowArcSinBulk(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = asinf(input[idx]);
    }
}
__device__ float slowArcSin(float input) {
    return asinf(input);
}

// slow but accurate cos function
__device__ void slowCosBulk(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = cosf(input[idx]);
    }
}
__device__ float slowCos(float input) {
    return cosf(input);
}
__device__ float cosGPU(float x) {
    return slowCos(x);
    // Approximate cos(x) using a Taylor series expansion
    // cos(x) ≈ 1 - x^2/2 + x^4/24 - x^6/720 for small x
    //float x2 = x * x;
    //return 1 - x2 / 2 + x2 * x2 / 24 - x2 * x2 * x2 / 720;
}

// slow but accurate arccos function
__device__ void slowArcCosBulk(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = acosf(input[idx]);
    }
}
__device__ float slowArcCos(float input) {
    return acosf(input);
}

// slow but accurate atan2 function
__device__ float slowAtan2(float y, float x) {
    return atan2f(y, x);
}

// --- VECTOR OPERATIONS ---

// distance between a 3d point and a 3d point array
__device__ void pointDistanceBulk(float x1,  float y1,  float z1, 
                              float* x2, float* y2, float* z2,
                              float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float dx = x2[idx] - x1;
        float dy = y2[idx] - y1;
        float dz = z2[idx] - z1;
        output[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
    }
}

// distance between two 3D point arrays
__device__ void pointDistanceVecBulk(float* x1, float* y1, float* z1, 
                              float* x2, float* y2, float* z2,
                              float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float dx = x2[idx] - x1[idx];
        float dy = y2[idx] - y1[idx];
        float dz = z2[idx] - z1[idx];
        output[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
    }
}

// dot product of two 3D vector arrays
__device__ void dotProductBulk(float* x1, float* y1, float* z1, 
                           float* x2, float* y2, float* z2,
                           float* output, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = x1[idx] * x2[idx] + y1[idx] * y2[idx] + z1[idx] * z2[idx];
    }
}
__device__ float dotProduct(float x1, float y1, float z1, 
                            float x2, float y2, float z2) {
    return x1 * x2 + y1 * y2 + z1 * z2;
}
__host__ float dotProductHost(float x1, float y1, float z1,
                              float x2, float y2, float z2) {
    return x1 * x2 + y1 * y2 + z1 * z2;
}

// dot product of two 3D vector arrays with negate option
__device__ void dotProductNegateBulk(float* x1, float* y1, float* z1, 
                                 float* x2, float* y2, float* z2,
                                 float* output, int n, bool negate) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = x1[idx] * x2[idx] + y1[idx] * y2[idx] + z1[idx] * z2[idx];
        if (negate) {
            output[idx] = -output[idx];
        }
    }
}

// magnitude of 3D vector
__host__ float vectorMagnitudeHost(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z);
}

// find angle between normalized source and target vectors
__host__ float angleSourceNormTargetPosHost(float snx, float sny, float snz,
                                            float  tx, float  ty, float  tz) {
    // NOTE: This function requires normalized target position vector
    float dp = dotProductHost(snx, sny, snz, tx, ty, tz);
    float theta = acosf(dp);
    return theta;
}


// --- GENERIC MEMORY OPERATIONS ---

// take every other sample of a 1D array (float)
__global__ void takeEveryOtherFloats(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[2 * idx];
    }
}

// take every other sample of a 1D array (cuFloatComplex)
__global__ void takeEveryOtherComplex(cuFloatComplex* input,
                                     cuFloatComplex* output,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[2 * idx];
    }
}

// -- SCALE ARRAY ---

__global__ void scaleComplex(cuFloatComplex* data, int N, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}


// --- COMPLEX POINTWISE MULTIPLICATION ---

__global__ void complexPointwiseMul(cuFloatComplex* a, const cuFloatComplex* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        cuFloatComplex A = a[i];
        cuFloatComplex B = b[i];
        a[i] = make_cuFloatComplex(A.x * B.x - A.y * B.y,
                                   A.x * B.y + A.y * B.x);
    }
}


// --- CONVOLUTION ---


// convolve two 1D complex signals of the same length. Downsample at end.
void convolveComplex(cuFloatComplex* d_signal, cuFloatComplex* d_kernel,
                     cuFloatComplex* d_output, int nr) {

    // convolution in here
    int nrPad = 2 * nr; // zero-pad to avoid circular convolution

    // allocate device memory for padded signal and kernel
    cuFloatComplex* d_signalPad;
    cuFloatComplex* d_kernelPad;
    cudaMalloc(&d_signalPad, nrPad * sizeof(cuFloatComplex));
    cudaMalloc(&d_kernelPad, nrPad * sizeof(cuFloatComplex));

    // fill both arrays with zeros
    cudaMemset(d_signalPad, 0, nrPad * sizeof(cuFloatComplex));
    cudaMemset(d_kernelPad, 0, nrPad * sizeof(cuFloatComplex));

    // move signal and kernel into padded arrays
    cudaMemcpy(d_signalPad, d_signal, nr * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_kernelPad, d_kernel, nr * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);

    // create FFT plan for padded length
    cufftHandle plan;
    cufftPlan1d(&plan, nrPad, CUFFT_C2C, 1);

    // forward FFT on both signal and kernel
    cufftExecC2C(plan, d_signalPad, d_signalPad, CUFFT_FORWARD);
    cufftExecC2C(plan, d_kernelPad, d_kernelPad, CUFFT_FORWARD);

    // pointwise multiply in frequency domain
    int blocks = (nrPad + THREADS - 1) / THREADS;
    complexPointwiseMul<<<blocks, THREADS>>>(d_signalPad, d_kernelPad, nrPad);

    // inverse FFT to get the convolved signal
    cufftExecC2C(plan, d_signalPad, d_signalPad, CUFFT_INVERSE);

    // normalize by the padded length
    float scale = 1.0f / nrPad;
    scaleComplex<<<blocks, THREADS>>>(d_signalPad, nrPad, scale);

    // downsample the convolved signal by taking every other sample of valid region
    takeEveryOtherComplex<<<blocks, THREADS>>>(d_signalPad, d_output, nr);

    // free device memory and destroy FFT plan
    cudaFree(d_signalPad);
    cudaFree(d_kernelPad);
    cufftDestroy(plan);

}