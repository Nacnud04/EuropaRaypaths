#pragma once

#include <cuComplex.h>
#include <math.h>
#include <limits>

// thrust includes for aperture cropping
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

#include "elementary_kernels.cuh"
#include "scattering.cuh"


__host__ int nIlluminatedFacets(float sz, float fz, float fs, float theta, float buff);

__global__ void compSourceInclination(float sx, float sy, float sz,
                                      float* d_fx, float* d_fy, float* d_fz,
                                      float snx, float sny, float snz,
                                      float* d_Itd, float* d_Sth, int totFacets);

__global__ void compIncidentRays(float sx, float sy, float sz,
                                 float* d_fx, float* d_fy, float* d_fz,
                                 float* d_fnx, float* d_fny, float* d_fnz,
                                 float* d_fux, float* d_fuy, float* d_fuz,
                                 float* d_fvx, float* d_fvy, float* d_fvz,
                                 float* d_Itd, float* d_Iph, float* d_Ith,
                                 int nfacets);


struct is_below_threshold
{
    const float thresh;
    __host__ __device__
    is_below_threshold(float t) : thresh(t) {}

    __host__ __device__
    bool operator()(const float x) const {
        return x < thresh;
    }
};


int cropByAperture(int totfacets, int nfacets, float aperture, float* d_FSth,
                    float* d_Ffx,  float* d_Ffy,  float* d_Ffz,
                    float* d_Ffnx, float* d_Ffny, float* d_Ffnz,
                    float* d_Ffux, float* d_Ffuy, float* d_Ffuz,
                    float* d_Ffvx, float* d_Ffvy, float* d_Ffvz,
                    float* d_FItd,
                    float* d_fx,  float* d_fy,  float* d_fz,
                    float* d_fnx, float* d_fny, float* d_fnz,
                    float* d_fux, float* d_fuy, float* d_fuz,
                    float* d_fvx, float* d_fvy, float* d_fvz,
                    float* d_Itd);

__device__ float prismIntersectionDistance(float ox, float oy, float oz,
                                          float dx, float dy, float dz,
                                          float xmin, float xmax,
                                          float ymin, float ymax,
                                          float zmin, float zmax,
                                          float t);

__global__ void compTargetRays(float tx, float ty, float tz,
                               float tnx, float tny, float tnz,
                               float* d_fx, float* d_fy, float* d_fz,
                               float* d_fnx, float* d_fny, float* d_fnz,
                               float* d_fux, float* d_fuy, float* d_fuz,
                               float* d_fvx, float* d_fvy, float* d_fvz,
                               float* d_Ttd, float* d_Tph, float* d_Tth,
                               float* d_TargetTh,
                               int nfacets,
                               float* d_attXmin, float* d_attXmax,
                               float* d_attYmin, float* d_attYmax,
                               float* d_attZmin, float* d_attZmax,
                               float* d_alphas, float alpha2, int nAttenPrisms,
                               float* d_fRefrEI, float* d_fRefrEO);

__device__ float facetReradiation(float dist, float th, float ph,  
                                  float lam, float fs);

__device__ float radarEq(float P, float G, float fs, float lam, float dist, int nfacets);

__device__ float beaconEq(float P, float G, float fs, float lam, float dist, int nfacets);

__device__ float beaconPowerDensity(float P, float G, float dist);

__device__ float friis(float P, float Gt, float Gr, float lam, float dist);

__device__ float friisSubsurf(float P, float Gt, float Gr, float lam, float d1, float d2);

__device__ float doubleFriis(float P, float lam, float D1, float D2);

__device__ float effectiveArea(float G, float lam);

__host__   float fresnelAngle(float lam, float h);

__global__ void compReflectedEnergy(float* d_Itd, float* d_Ith, float* d_Iph,
                                    float* d_fRe, float* d_Rth, float* d_fRfrC,
                                    SimulationParameters par, int nfacets);

__device__ float snellsLaw(float th, float eps_1, float eps_2);

__global__ void compRefractedRays(float* d_Ith, float* d_Iph,
                                  float* d_Rth, 
                                  float* d_fx, float* d_fy, float* d_fz,
                                  float eps_1, float eps_2, int nfacets);

__global__ void compRefrEnergyIn(
                    float* d_Rth, float* d_Itd, float* d_Iph,
                    float* d_Ttd, float* d_Tth, float* d_Tph, float* d_fRfrC,
                    float* d_fRefrEI, float* d_fRfrSR,
                    SimulationParameters par, int nfacets);

__global__ void compRefrEnergyOut(float* d_Itd, float* d_Iph,
                                  float* d_Ttd, float* d_Tth, float* d_Tph,
                                  float* d_fRefrEO, float* d_fRfrC, 
                                  SimulationParameters par, int nfacets);

__global__ void findTargetsInAperture(float* d_tx, float* d_ty, float* d_tz,
                                      float sx, float sy, float sz,
                                      float snx, float sny, float snz,
                                      bool* tInApt,
                                      float aperture, int nTargets);

