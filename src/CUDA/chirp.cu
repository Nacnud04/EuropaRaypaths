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

#define BLOCK_SIZE 256
#define pi 3.1415926535897932384


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


// GENERIC GENERALIZED COSINE WINDOW
// weights contain the unsigned coefficients a_m.
// Alternating signs are applied inside the summation.
__device__ float raisedCosineWindow(
    const float* weights,
    int nIdx,
    int Ntotal,
    int Nweights
) {
    float amplitude = 0.0f;

    for (int m = 0; m < Nweights; m++) {

        float sign = (m % 2 == 0) ? 1.0f : -1.0f;

        amplitude += sign * weights[m] *
            cosf((2.0f * pi * m * nIdx) / Ntotal);
    }

    return amplitude;
}


// Apply a generalized cosine window centered at DC
__global__ void applyFrequencyWindow(
    cufftComplex* d_spectrum,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {

        // Map FFT ordering to a window centered at DC.
        // Assumes even N.
        int n = (i + N / 2) % N;

        // Standard Blackman coefficients (unsigned).
        const float blackmanWeights[3] = {
            0.42f, 0.50f, 0.08f
        };

        float window = raisedCosineWindow(
            blackmanWeights, n, N, 3
        );

        d_spectrum[i].x *= window;
        d_spectrum[i].y *= window;
    }
}


// function which generates the transmit chirp
void genTransmitSignal(float* d_chirp, float dr, int nr, 
                      float range_res, int window) {
    
    // first generate the chirp
    
    // to begin we define how many blocks/threads to call
    int paddedNr = 2 * nr;
    int nBlocks  = (paddedNr + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // generate generic chirp in time domain
    genericChirpPadded<<<nBlocks, BLOCK_SIZE>>>(
        d_chirp, dr, nr, paddedNr, range_res
    );
    checkCUDAError("genCenteredChirpPadded kernel");

    // if there is no windowing, no FFT is needed
    /*if (window == 0) {
        return;
    }*/

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
        d_spectrum, paddedNr
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