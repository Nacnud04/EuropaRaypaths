#pragma once

#include <cufft.h>

#include "elementary_kernels.cuh"


__global__ void genericChirpPadded(
    float* d_chirp,
    float dr,
    int nr,
    int paddedNr,
    float range_res);

void genTransmitSignal(float* d_chirp, 
                       SimulationParameters par,
                       int window, float res_scale);