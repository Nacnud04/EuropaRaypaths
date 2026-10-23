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
    if (window == 0) {
        return;
    }

    // if there is windowing, things get more complex.
    // TODO: Implement windowing functions...
    
}