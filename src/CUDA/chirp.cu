/******************************************************************************
 * File:        scattering.cu
 * Author:      Duncan Byrne
 * Institution: Univeristy of Colorado Boulder
 * Department:  Aerospace Engineering Sciences
 * Email:       duncan.byrne@colorado.edu
 * Date:        2026-10-22
 *
 * Description:
 *    Generating chirps with windowing functions
 *
 * Contents:
 *
 * Usage:
 *    #include "chirp.cu"
 * 
 * Notes:
 *
 ******************************************************************************/


#include <cufft.h>

#include "elementary_kernels.cuh"
#include "file_io.cuh"

#define BLOCK_SIZE 256
#define pi 3.1415926535897932384


// Generalized cosine window coefficients
__device__ __constant__ float BLACKMAN_WEIGHTS[] = {
    0.42f, 0.50f, 0.08f
};

__device__ __constant__ float BLACKMAN_EXACT_WEIGHTS[] = {
    0.42659f, 0.49656f, 0.076849f
};

__device__ __constant__ float NUTTALL_WEIGHTS[] = {
    0.355768f, 0.487396f, 0.144232f, 0.012604f
};

__device__ __constant__ float BLACKMAN_NUTTALL_WEIGHTS[] = {
    0.3635819f, 0.4891775f, 0.1365995f, 0.0106411f
};


// generate a generic chirp function
__global__ void genericChirpPadded(
    float* d_chirp,
    float dr,
    int nr,
    int paddedNr,
    float range_res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < paddedNr) {
        int kernel_center = nr / 2;

        float offset_bins  = (float)(i - kernel_center);
        float offset_range = offset_bins * dr;

        d_chirp[i] = sinc(offset_range / range_res);
    }
}


// GENERALIZED COSINE WINDOW
// Implements:
// w(r) = sum_m (-1)^m * a[m] * cos(2*pi*m*r)
//
// r is normalized from 0 at the center to 1 at the edges.
// Coefficients are supplied as positive values.
__device__ float raisedCosineWindow(
    const float* weights,
    float r,
    int Nweights
) {
    float amplitude = 0.0f;

    for (int m = 0; m < Nweights; m++) {
        float sign = (m % 2 == 0) ? 1.0f : -1.0f;

        amplitude += sign * weights[m] *
            cosf(2.0f * pi * m * r);
    }

    return amplitude;
}


// Apply a generalized cosine window centered at DC.
// bandwidth: full window bandwidth in Hz
// sampleRate: sampling rate in Hz
__global__ void applyFrequencyWindow(
    cufftComplex* d_spectrum,
    int N,
    float bandwidth,
    float sampleRate,
    WindowTypes wintype
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {

        // Signed FFT frequency bin
        int k = (i <= N / 2) ? i : i - N;

        // Convert bandwidth to a half-width in FFT bins.
        // FFT bin spacing = sampleRate / N.
        float halfWidthBins =
            (bandwidth * (float)N) / (2.0f * sampleRate);

        float window = 0.0f;
        float absK = fabsf((float)k);

        if (halfWidthBins > 0.0f && absK <= halfWidthBins) {

            // Normalize distance from DC to [0, 1].
            float r = 0.5 + absK / (2.0f * halfWidthBins);

            switch (wintype) {
                case BLACKMAN_WINDOW:
                    window = raisedCosineWindow(BLACKMAN_WEIGHTS, r, 3);
                    break;

                case BLACKMAN_EXACT_WINDOW:
                    window = raisedCosineWindow(BLACKMAN_EXACT_WEIGHTS, r, 3);
                    break;

                case NUTTALL_WINDOW:
                    window = raisedCosineWindow(NUTTALL_WEIGHTS, r, 4);
                    break;

                case BLACKMAN_NUTTALL_WINDOW:
                    window = raisedCosineWindow(BLACKMAN_NUTTALL_WEIGHTS, r, 4);
                    break;

                case NO_WINDOW:
                default:
                    window = 1.0f;
                    break;
            }
   
        }

        d_spectrum[i].x *= window;
        d_spectrum[i].y *= window;
    }
}

// function which generates the transmit chirp
void genTransmitSignal(float* d_chirp, 
                       SimulationParameters par,
                       int window, float res_scale) {
    
    // first generate the chirp
    
    // to begin we define how many blocks/threads to call
    int paddedNr = 2 * par.nr;
    int nBlocks  = (paddedNr + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // generate generic chirp in time domain
    genericChirpPadded<<<nBlocks, BLOCK_SIZE>>>(
        d_chirp, par.dr, par.nr, paddedNr, par.rng_res * res_scale
    );
    checkCUDAError("genCenteredChirpPadded kernel");

    // if there is no windowing, no FFT is needed
    std::cout << "Window type: " << par.window << std::endl;
    if (par.window == NO_WINDOW) {
        return;
    }

    // if there is windowing, we need to compute the FFT
    cuFloatComplex* d_spectrum = nullptr;
    cudaMalloc(
        &d_spectrum,
        paddedNr * sizeof(cuFloatComplex)
    );

    // make the fft plan
    cufftHandle plan;
    cufftPlan1d(&plan, paddedNr, CUFFT_C2C, 1);
    
    // convert chirp to complex data type
    realToComplex<<<nBlocks, BLOCK_SIZE>>>(
        d_chirp, d_spectrum, paddedNr
    );

    // do forward FFT
    cufftExecC2C(plan, d_spectrum, d_spectrum, CUFFT_FORWARD);

    // APPLY THE FREQUENCY WINDOW
    applyFrequencyWindow<<<nBlocks, BLOCK_SIZE>>>(
        d_spectrum, paddedNr, par.B / res_scale, par.smpl, par.window
    );

    // do ifft and normalize
    cufftExecC2C(plan, d_spectrum, d_spectrum, CUFFT_INVERSE);
    normalizeAndExtractReal<<<nBlocks, BLOCK_SIZE>>>(
        d_spectrum, d_chirp, paddedNr
    );

    // destroy fft plan
    cufftDestroy(plan);
    cudaFree(d_spectrum);
    
}