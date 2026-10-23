#pragma once

#include <cuComplex.h>
#include <math.h>

// thrust includes for phasor computation
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/complex.h>


// FFT for chirp convolution
#include <cufft.h>

#include "geometry_kernels.cuh"
#include "file_io.cuh"


struct cuComplexAdd {
    __host__ __device__
    cuFloatComplex operator()(const cuFloatComplex& a, const cuFloatComplex& b) const {
        return make_cuFloatComplex(a.x + b.x, a.y + b.y);
    }
};


__device__ float rerad_funct(int funcnum, float th1, float th2);

__device__ int reradiate_index(int id0);

__global__ void combineRadarSignals(cuFloatComplex* refl_sig, cuFloatComplex* refr_sig,
                                    cuFloatComplex* total_sig, int nr);

__global__ void genRefrPhasor(cuFloatComplex* refr_phasor, short* refr_rbs,
                              float* d_fRfrSR, float* d_fRefrEI, float* d_fRefrEO, 
                              float* d_Targetth, float* d_Ttd, int target_fun,
                              float P, float G, float lam, float fs, int nfacets,
                              float rst, float dr, int nr, float c, float c2);

void genPhasorTrace(cuFloatComplex* d_phasorTrace,
                    short* d_rbs, cuFloatComplex* d_phasors,
                    int nfacets, int nr);

void genPhasorTraceAsync(cuFloatComplex* d_phasorTrace,
                         short* d_rbs,
                         cuFloatComplex* d_phasors,
                         int nfacets, int nr,
                         cudaStream_t st);

__device__ float chirp(float cen, float offset, float rng_res);

__global__ void genChirp(float* d_chirp, float rst, float dr, int nr, float range_res);

__global__ void genCenteredChirp(float* d_chirp, float rst, float dr, int nr, float range_res);

__global__ void genCenteredChirpPadded(float* d_chirp, float dr, int nr, int paddedNr, float range_res);

__global__ void addComplexArrays(cuFloatComplex* dest, const cuFloatComplex* src, int N);

void convolvePhasorChirp(cuFloatComplex* d_phasorTrace, float* d_chirp, 
                         cuFloatComplex* d_sig, int nr);

void convolvePhasorChirpLinear(cuFloatComplex* d_phasorTrace, float* d_chirp,
                               cuFloatComplex* d_sig, int nr, SimulationParameters par, 
			       const char* argv[], int is, int it);

void convolvePhasorChirpLinearAsync(cuFloatComplex* d_phasorTrace, float* d_chirp,
                                    cuFloatComplex* d_sig, int nr, SimulationParameters par,
                                    const char* argv[], int is, int it,
                                    cudaStream_t s);

__device__ float hertz_dipole(float th);

__device__ float facet_G(float th, float ph,  
                                  float lam, float fs);

__device__ cuFloatComplex phasor(float r, float lam);

__device__ cuFloatComplex randomPhasor(int seed);

__global__ void surfacePT(cuFloatComplex* d_refl_P, short* d_refl_rbs, float* d_Ith, float* d_Iph, float* d_Itd,
                          float* d_fReflE, SimulationParameters par, int nfacets);

__global__ void accumulateTarget(cuFloatComplex* d_refr_P, short* d_refr_rbs, 
                                 float* d_Ith, float* d_Iph, float* d_Itd,
                                 float* d_Tth, float* d_Tph, float* d_Ttd, 
                                 float* d_Rth,
                                 float* d_Tarth, float* d_fRefrEI, float* d_fRfrSR, 
                                 float* d_fx, float* d_fy, float* d_fz, 
                                 SimulationParameters par, int nfacets);

__global__ void radiateTarget(cuFloatComplex* d_refr_P, short* d_refr_rbs, 
                              float* d_Ith, float* d_Iph, float* d_Itd,
                              float* d_Tth, float* d_Tph, float* d_Ttd,
                              float* d_Rth,
                              float* d_Tarth, float* d_fRefrEO, float* d_fRfrSR, 
                              SimulationParameters par, int nfacets);

__global__ void evaluateSubsurface(cuFloatComplex* signal,
                                   float* d_Ith, float* d_Iph, float* d_Itd,
                                   float* d_Tth, float* d_Tph, float* d_Ttd,
                                   float* d_Rth, float* d_Tarth, float* d_fRefrEO,
                                   float* d_fRfrSR, SimulationParameters par, int nfacets);

