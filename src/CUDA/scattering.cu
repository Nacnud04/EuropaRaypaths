/******************************************************************************
 * File:        scattering.cu
 * Author:      Duncan Byrne
 * Institution: Univeristy of Colorado Boulder
 * Department:  Aerospace Engineering Sciences
 * Email:       duncan.byrne@colorado.edu
 * Date:        2026-10-21
 *
 * Description:
 *    CUDA device calls for various surface scattering models
 *
 * Contents:
 *    - scatterSpecular float
 *    - coherentSpecular float: coherent fraction retained in specular direction
 *    - simpleBackscatter float: simple model for coherent + incoherent backscattered energy 
 *
 * Usage:
 *    #include "scattering.cu"
 * 
 * Notes:
 *
 ******************************************************************************/

#include <cuComplex.h>
#include <math.h>

const float pi = 3.14159265f;

// MODEL: Specular Scattering
// if it is purely specular, returned power depends on radiation pattern
// so the roughness loss is a 1.
__device__ float scatterSpecular(float ks){
    return 1.0f;
}

// MODEL: Coherent Loss
// this is the coherent fraction retained in the specular direction. this 
// is likely not suitable for simulation except for in rare instances
__device__ float coherentSpecular(float ks, float th) {
    return  expf(-4 *
                ((ks*cosGPU(th)) *
                (ks*cosGPU(th)))
            );
}

// MODEL: Simple Backscatter
// most suitable, describes incoherent + coherent backscattered energy
__device__ float simpleBackscatter(float rms_h, float lam) {
    float g = (4 * pi * rms_h) / lam;
    return expf(-1.0 * g * g) * 
           powf(cyl_bessel_i0f(0.5 * g * g), 2.0f);
}


// function to wrap all roughness calculations
__device__ float roughnessWrapper(RoughnessModel roughness_model, 
                                  float ks, float rms_h, float lam, float th) {
    switch (roughness_model) {
        case SPECULAR:
            return scatterSpecular(ks);

        case COHERENT_SPECULAR:
            return coherentSpecular(ks, th);

        case SIMPLE_BACKSCATTER:
            return simpleBackscatter(rms_h, lam);

        default:
            return 1.0f;  // fallback
    }
}
