#pragma once

#include <cuComplex.h>
#include <math.h>

#include "elementary_kernels.cuh"


const float pi = 3.14159265f;


__device__ float scatterSpecular(float ks);

__device__ float coherentSpecular(float ks, float th);

__device__ float simpleBackscatter(float rms_h, float lam);

__device__ float roughnessWrapper(RoughnessModel roughness_model, 
                                  float ks, float rms_h, float lam, float th);