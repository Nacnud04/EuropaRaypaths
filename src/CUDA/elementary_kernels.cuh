/******************************************************************************
 * File:        elementary_kernels.cuh
 * Author:      Duncan Byrne
 * Institution: Univeristy of Colorado Boulder
 * Department:  Aerospace Engineering Sciences
 * Email:       duncan.byrne@colorado.edu
 * Date:        2026-10-22
 *
 * Description:
 *    header file for elementary_kernels.cu
 *
 * Usage:
 *    #include "elementary_kernels.cuh"
 * 
 * Notes:
 *
 ******************************************************************************/

#pragma once

#include "file_io.cuh"

#include <iostream>

#include <stdio.h>
#include <cuComplex.h>
#include <cufft.h>


// --- CUDA ERROR CHECKING ---

void checkCUDAError(const char* msg);


// --- TRIG FUNCTIONS ---

// sin
 __device__ void slowSinBulk(float* input, float* output, int n);
 __device__ float slowSin(float input);
 __device__ float sinGPU(float x);

// sinc
__device__ float sinc(float x);

// arcsin
__device__ void slowArcSinBulk(float* input, float* output, int n);
__device__ float slowArcSin(float input);

// cos
__device__ void slowCosBulk(float* input, float* output, int n);
__device__ float slowCos(float input);
__device__ float cosGPU(float x);

// arccos
__device__ void slowArcCosBulk(float* input, float* output, int n);
__device__ float slowArcCos(float input);
__device__ float slowAtan2(float y, float x);


// --- VECTOR OPERATIONS ---

// point distances
__device__ void pointDistanceBulk(float x1,  float y1,  float z1, 
                                  float* x2, float* y2, float* z2,
                                  float* output, int n);
__device__ void pointDistanceVecBulk(float* x1, float* y1, float* z1, 
                                     float* x2, float* y2, float* z2,
                                     float* output, int n);

// dot products
__device__ void dotProductBulk(float* x1, float* y1, float* z1, 
                           float* x2, float* y2, float* z2,
                           float* output, int n);
__device__ float dotProduct(float x1, float y1, float z1, 
                            float x2, float y2, float z2);
__host__ float dotProductHost(float x1, float y1, float z1,
                              float x2, float y2, float z2);
__device__ void dotProductNegateBulk(float* x1, float* y1, float* z1, 
                                 float* x2, float* y2, float* z2,
                                 float* output, int n, bool negate);

// vector magnitudes
__device__ float vectorMagnitude(float x, float y, float z);
__host__ float vectorMagnitudeHost(float x, float y, float z);

// angles
__device__ float angleSourceNormTargetPos(float snx, float sny, float snz,
                                          float  tx, float  ty, float  tz);
__host__ float angleSourceNormTargetPosHost(float snx, float sny, float snz,
                                            float  tx, float  ty, float  tz);


// --- MEMORY / TYPE CHANGE OPERATIONS --

// memory
__global__ void takeEveryOtherFloats(float* input, float* output, int n);
__global__ void takeEveryOtherComplex(cuFloatComplex* input,
                                     cuFloatComplex* output,
                                     int n);

// type
__global__ void realToComplex(const float* realSignal, cuFloatComplex* complexSignal, int N);


// --- SCALING ---

__global__ void normalizeAndExtractReal(
    const cufftComplex* d_input, float* d_output,
    int N);
__global__ void scaleComplex(cuFloatComplex* data, int N, float scale);


// --- COMPLEX MATH ---

// complex pointwise multiplication
__global__ void complexPointwiseMul(cuFloatComplex* a, const cuFloatComplex* b, int N);

// squaring
__global__ void squareComplex(cuFloatComplex* in,
                              cuFloatComplex* out,
                              int n);
void launchSquare(cuFloatComplex* d_in,
                  cuFloatComplex* d_out,
                  int n);


// --- CONVOLUTION ---

__global__ void cropSpectrum(cuFloatComplex *X,
                  int N,
                  float Fs,
                  float B);
void convolveComplex(cuFloatComplex* d_signal, cuFloatComplex* d_kernel,
                     cuFloatComplex* d_output, SimulationParameters par, 
                     const char* argv[], int is, int it);
void convolveComplexAsync(cuFloatComplex* d_signal,
                          cuFloatComplex* d_kernel,
                          cuFloatComplex* d_output,
                          cuFloatComplex* d_signalPad,
                          cuFloatComplex* d_kernelPad,
                          cufftHandle plan,
                          SimulationParameters par,
                          cudaStream_t st);
