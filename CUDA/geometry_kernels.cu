#include <cuComplex.h>
#include <math.h>

#include "elementary_kernels.cu"

const float pi = 3.14159265f;

// function which approximates the amount of illuminated facets within the aperture
// assuming the facets are arranged in a grid
__host__ int nIlluminatedFacets(float sz, float fz, float fs, float theta) {
    
    // get radius of illuminated area on surface
    float r;
    r = (sz - fz) * sinf((pi/180)*theta);

    // turn into area
    float A;
    A = pi * r * r;

    // get the amount of illuminated facets
    float nfacets;
    nfacets = A / (fs * fs);

    return nfacets;

}

__global__ void compIncidentRays(float sx, float sy, float sz,
                                 float* d_fx, float* d_fy, float* d_fz,
                                 float* d_fnx, float* d_fny, float* d_fnz,
                                 float* d_fux, float* d_fuy, float* d_fuz,
                                 float* d_fvx, float* d_fvy, float* d_fvz,
                                 float* d_Itd, float* d_Iph, float* d_Ith,
                                 int nfacets) {
    
    // first get the distance betweem the source and all facets
    pointDistanceBulk(sx, sy, sz,
                  d_fx, d_fy, d_fz,
                  d_Itd, nfacets);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfacets) {

        // get incident cartesian vector
        float Ix = (d_fx[idx] - sx) / d_Itd[idx];
        float Iy = (d_fy[idx] - sy) / d_Itd[idx];
        float Iz = (d_fz[idx] - sz) / d_Itd[idx];

        // incident inclination
        d_Ith[idx] = slowArcCos(dotProduct(-1*Ix, -1*Iy, -1*Iz, 
                                          d_fnx[idx], d_fny[idx], d_fnz[idx]));

        // incident azimuth
        d_Iph[idx] = slowAtan2(
            dotProduct(Ix, Iy, Iz, d_fux[idx], d_fuy[idx], d_fuz[idx]),
            dotProduct(Ix, Iy, Iz, d_fvx[idx], d_fvy[idx], d_fvz[idx])
        );

    }    

}

__global__ void compTargetRays(float tx, float ty, float tz,
                               float* d_fx, float* d_fy, float* d_fz,
                               float* d_fnx, float* d_fny, float* d_fnz,
                               float* d_fux, float* d_fuy, float* d_fuz,
                               float* d_fvx, float* d_fvy, float* d_fvz,
                               float* d_Ttd, float* d_Tph, float* d_Tth,
                               int nfacets) {
    
    // first get the distance betweem the source and all facets
    pointDistanceBulk(tx, ty, tz,
                      d_fx, d_fy, d_fz,
                      d_Ttd, nfacets);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfacets) {

        // get incident cartesian vector
        float Ix = (d_fx[idx] - tx) / d_Ttd[idx];
        float Iy = (d_fy[idx] - ty) / d_Ttd[idx];
        float Iz = (d_fz[idx] - tz) / d_Ttd[idx];

        // incident inclination
        d_Tth[idx] = slowArcCos(dotProduct(-1*Ix, -1*Iy, -1*Iz, 
                                          d_fnx[idx], d_fny[idx], d_fnz[idx]));

        // incident azimuth
        d_Tph[idx] = slowAtan2(
            dotProduct(Ix, Iy, Iz, d_fux[idx], d_fuy[idx], d_fuz[idx]),
            dotProduct(Ix, Iy, Iz, d_fvx[idx], d_fvy[idx], d_fvz[idx])
        );

    }    

}


__device__ float facetReradiation(float dist, float th, float ph,  
                                  float lam, float fs){
    // NOTE: dist is the observer distance from the facet aperture

    // start with k
    float k = 2 * 3.14159265f / lam;

    // now make complex c = ((1j * r**2)/lam) * (np.exp(-1j*k*R)/R) * k 
    // (1j * r**2)/lam
    cuFloatComplex i_r2_over_lam = make_cuFloatComplex(0.0f, fs*fs / lam);
    // exp(-1j*k*R)/R
    float neg_kR = -1 * k * dist;
    cuFloatComplex exp_term = make_cuFloatComplex(cosf(neg_kR)/dist, sinf(neg_kR)/dist);
    // combine into c
    cuFloatComplex c_val = cuCmulf(i_r2_over_lam, exp_term);
    c_val = cuCmulf(c_val, make_cuFloatComplex(k, 0.0f));

    // now get sinc components:
    // np.sinc(((r) / lam) * np.sin(ph) * np.cos(th))
    float sinc1 = sinc((fs / lam) * sinGPU(th) * cosGPU(ph));
    // np.sinc(((r) / lam) * np.sin(ph) * np.sin(th))
    float sinc2 = sinc((fs / lam) * sinGPU(th) * sinGPU(ph));

    // combine all together
    return cuCabsf(cuCmulf(c_val, make_cuFloatComplex(sinc1 * sinc2, 0.0f)));

}


__device__ float radarEq(float P, float G, float sigma, float lam, float dist, float fs){
    
    // Pr = (Pt * G^2 * lam^2 * sigma) / ((4*pi)^3 * R^4)
    float num = P * G * G * lam * lam * sigma;
    float denom = pow(4 * 3.14159, 3) * dist * dist * dist * dist; // (4*pi)^3
    return (num / denom) * fs * fs;

}


__global__ void compReflectedEnergy(float* d_Itd, float* d_Ith, float* d_Iph,
                                    float* d_fRe, float* d_Rth, float* d_fRfrC,
                                    float P, float G, float sigma, float fs, float lam, 
                                    float nu1, float nu2, float alpha1, 
                                    float ks, int polarization, int nfacets){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfacets) {

        // first get facet reradiation
        // we double inclination angle to as the center of the beam pattern is in the
        // exact opposite direction as the incident ray. 
        d_fRe[idx] = facetReradiation(d_Itd[idx], 2*d_Ith[idx], -1*d_Iph[idx], lam, fs);

        // losses from radar equation
        d_fRe[idx] = d_fRe[idx] * radarEq(P, G, sigma, lam, d_Itd[idx], fs);

        // reflection coefficient
        // horizontal pol.
        float rho;
        if (polarization == 0){ 
            rho = (nu2 * cosGPU(d_Ith[idx]) - nu1 * cosGPU(d_Rth[idx])) /
                    (nu2 * cosGPU(d_Ith[idx]) + nu1 * cosGPU(d_Rth[idx]));
            d_fRfrC[idx] = 1 - (rho * rho);
        } 
        // vertical pol.
        else if (polarization == 1) {
            rho = (nu2 * cosGPU(d_Rth[idx]) - nu1 * cosGPU(d_Ith[idx])) /
                    (nu2 * cosGPU(d_Rth[idx]) + nu1 * cosGPU(d_Ith[idx]));
            d_fRfrC[idx] = 1 - (rho * rho);
        }
        d_fRe[idx] = d_fRe[idx] * rho * rho;

        // signal attenuation
        d_fRe[idx] = d_fRe[idx] * expf(-2.0f * alpha1 * d_Itd[idx]);

        // surface roughness losses
        float rough_loss = expf(-4*((ks*cosGPU(d_Ith[idx]))*(ks*cosGPU(d_Ith[idx]))));
        d_fRe[idx] = d_fRe[idx] * rough_loss;

    }

}


__device__ float snellsLaw(float th, float eps_1, float eps_2) {

    float sin_th = sinGPU(th);
    float sin_Rth = sqrtf(eps_1/eps_2) * sin_th;
    if (sin_Rth > 1.0f) {
        // total internal reflection set to something more than pi/2 so it is easy to identify
        return 1e4;
    } else {
        return slowArcSin(sin_Rth);
    }

}


__global__ void compRefractedRays(float* d_Ith, float* d_Iph,
                                  float* d_Rth, float* d_Rtd,
                                  float* d_fx, float* d_fy, float* d_fz,
                                  float tx, float ty, float tz,
                                  float eps_1, float eps_2, int nfacets) {

    pointDistanceBulk(tx, ty, tz,
                      d_fx, d_fy, d_fz,
                      d_Rtd, nfacets);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfacets) {

        // refracted inclination
        d_Rth[idx] = snellsLaw(d_Ith[idx], eps_1, eps_2);

    }    

}


__global__ void compRefrEnergyIn(
                    float* d_Rtd, float* d_Rth, float* d_Itd, float* d_Iph,
                    float* d_Ttd, float* d_Tth, float* d_Tph, float* d_fRfrC,
                    float* d_fRefrEI, float* d_fRfrSR,
                    float ks, int nfacets, float alpha2, float c1, float c2,
                    float fs, float P, float G, float lam) {

    float c = 299792458.0f; // speed of light in m/s

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nfacets) {

        // start with facet reradiation
        // we need to get delta theta between refracted and forced ray as follows:
        float delta_th = (pi - d_Rth[id]) - d_Tth[id];
        // now we do similar for phi
        float delta_ph = d_Iph[id] - d_Tph[id];
        // compute facet reradiation
        d_fRefrEI[id] = facetReradiation(d_Ttd[id], delta_th, delta_ph, lam, fs);

        // refraction coefficient
        d_fRefrEI[id] = d_fRefrEI[id] * d_fRfrC[id];

        // signal attenuation
        d_fRefrEI[id] = d_fRefrEI[id] * expf(-2 * alpha2 * d_Rtd[id]);

        // surface roughness losses
        float rough_loss = expf(-4*((ks*cosGPU(d_Rth[id]))*(ks*cosGPU(d_Rth[id]))));
        d_fRefrEI[id] = d_fRefrEI[id] * rough_loss;

        // total travel slant range
        d_fRfrSR[id] = d_Itd[id] + d_Rtd[id];

    }

}


__global__ void compRefrEnergyOut(float* d_Itd, float* d_Iph,
                                  float* d_Ttd, float* d_Tth, float* d_Tph,
                                  float* d_fRefrEO, float* d_fRfrC, 
                                  float ks, int nfacets, float alpha1, float alpha2, float c1, float c2,
                                  float fs, float G, float lam, float eps_1, float eps_2){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfacets) {

        // for facet reradiation for upward transmitted, we can assume that
        // RrTh is the upward transmitted inclination angle
        float RrTh = snellsLaw(pi - d_Tth[idx], eps_2, eps_1);

        d_fRefrEO[idx] = facetReradiation(d_Itd[idx], RrTh, d_Tph[idx]-d_Iph[idx], lam, fs);

        // for transmission coefficient use refraction cofficient from before
        d_fRefrEO[idx] = d_fRefrEO[idx] * d_fRfrC[idx];

        // signal attenuation
        // first above surface
        d_fRefrEO[idx] = d_fRefrEO[idx] * expf(-2.0f * alpha1 * d_Itd[idx]);
        // then in subsurface
        d_fRefrEO[idx] = d_fRefrEO[idx] * expf(-2.0f * alpha2 * d_Ttd[idx]);

        // surface roughness losses
        float rough_loss = expf(-4*((ks*cosGPU(d_Tth[idx]))*(ks*cosGPU(d_Tth[idx]))));
        d_fRefrEO[idx] = d_fRefrEO[idx] * rough_loss;

    }

}