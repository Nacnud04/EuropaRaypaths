/******************************************************************************
 * File:        geometry_kernels.cu
 * Author:      Duncan Byrne
 * Institution: Univeristy of Colorado Boulder
 * Department:  Aerospace Engineering Sciences
 * Email:       duncan.byrne@colorado.edu
 * Date:        2025-11-07
 *
 * Description:
 *    File providing CUDA functions relevant to raypath geometry calculation
 *
 * Contents:
 *    - nIlluminatedFacets: approximates number of illuminated facets within aperture
 *    - compSourceInclination: computes source inclination angle for all facets
 *    - compIncidentRays: computes incident ray parameters for facets
 *    - cropByAperture: crops facets based on aperture angle using Thrust
 *    - compTargetRays: computes target ray parameters for facets
 *    - facetReradiation: computes facet reradiated field strength
 *    - radarEq: computes radar equation losses
 *    - compReflectedEnergy: computes reflected energy constant from each facet
 *    - snellsLaw: computes refracted angle using Snell's law
 *    - compRefractedRays: computes refracted ray parameters for facets
 *    - compRefrEnergyIn: computes refracted energy constant into the subsurface
 *    - compRefrEnergyOut: computed refracted energy constant out of the subsurface
 *
 * Usage:
 *    #include "geometry_kernels.cu"
 * 
 * Notes:
 *
 ******************************************************************************/

#include <cuComplex.h>
#include <math.h>

// thrust includes for aperture cropping
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

#include "elementary_kernels.cu"

const float pi = 3.14159265f;

// function which approximates the amount of illuminated facets within the aperture
// assuming the facets are arranged in a grid
__host__ int nIlluminatedFacets(float sz, float fz, float fs, float theta, float buff) {
    
    // get radius of illuminated area on surface
    float r;
    r = (sz - fz) * sinf((pi/180)*theta);

    // turn into area
    float A;
    A = pi * r * r;

    // get the amount of illuminated facets
    float nfacets;
    nfacets = A / (fs * fs);

    return nfacets * buff;

}


__global__ void compSourceInclination(float sx, float sy, float sz,
                                      float* d_fx, float* d_fy, float* d_fz,
                                      float snx, float sny, float snz,
                                      float* d_Itd, float* d_Sth, int totFacets){

    // first get the distance betweem the source and all facets
    pointDistanceBulk(sx, sy, sz,
                      d_fx, d_fy, d_fz,
                      d_Itd, totFacets);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totFacets) {

        // get incident cartesian vector
        float Ix = (d_fx[idx] - sx) / d_Itd[idx];
        float Iy = (d_fy[idx] - sy) / d_Itd[idx];
        float Iz = (d_fz[idx] - sz) / d_Itd[idx];

        // source inclination
        d_Sth[idx] = slowArcCos(dotProduct(-1*Ix, -1*Iy, -1*Iz, 
                                          snx, sny, snz));

    }

}


__global__ void compIncidentRays(float sx, float sy, float sz,
                                 float* d_fx, float* d_fy, float* d_fz,
                                 float* d_fnx, float* d_fny, float* d_fnz,
                                 float* d_fux, float* d_fuy, float* d_fuz,
                                 float* d_fvx, float* d_fvy, float* d_fvz,
                                 float* d_Itd, float* d_Iph, float* d_Ith,
                                 int nfacets) {

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
        // NOTE: this is clamped to -1 to 1 b/c floating point errors can cause perfectly
        // incident facets to return nan
        d_Ith[idx] = fmaxf(-1.0f, fminf(1.0f, 
                            slowArcCos(dotProduct(-1*Ix, -1*Iy, -1*Iz, 
                            d_fnx[idx], d_fny[idx], d_fnz[idx]))));

        // incident azimuth
        d_Iph[idx] = slowAtan2(
            dotProduct(Ix, Iy, Iz, d_fux[idx], d_fuy[idx], d_fuz[idx]),
            dotProduct(Ix, Iy, Iz, d_fvx[idx], d_fvy[idx], d_fvz[idx])
        );

    }    

}


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
                    float* d_Itd){

    // wrap raw pointers with the thrust device pointer
    thrust::device_ptr<float>   Ffx(d_Ffx),   Ffy(d_Ffy),   Ffz(d_Ffz);
    thrust::device_ptr<float> Ffnx(d_Ffnx), Ffny(d_Ffny), Ffnz(d_Ffnz);
    thrust::device_ptr<float> Ffux(d_Ffux), Ffuy(d_Ffuy), Ffuz(d_Ffuz);
    thrust::device_ptr<float> Ffvx(d_Ffvx), Ffvy(d_Ffvy), Ffvz(d_Ffvz);
    thrust::device_ptr<float> FItd(d_FItd);
    thrust::device_ptr<float> FSth(d_FSth);

    // define output thrust device pointers (point at the destination/cropped arrays)
    thrust::device_ptr<float>   fx(d_fx),   fy(d_fy),   fz(d_fz);
    thrust::device_ptr<float> fnx(d_fnx), fny(d_fny), fnz(d_fnz);
    thrust::device_ptr<float> fux(d_fux), fuy(d_fuy), fuz(d_fuz);
    thrust::device_ptr<float> fvx(d_fvx), fvy(d_fvy), fvz(d_fvz);
    thrust::device_ptr<float> Itd(d_Itd);

    // define predicate
    is_below_threshold pred((pi/180)*aperture);

    // copy_if using d_FIth as a stencil
    auto end_fx = thrust::copy_if(Ffx, Ffx + totfacets, FSth, fx, pred);
    auto end_fy = thrust::copy_if(Ffy, Ffy + totfacets, FSth, fy, pred);
    auto end_fz = thrust::copy_if(Ffz, Ffz + totfacets, FSth, fz, pred);

    auto end_fnx = thrust::copy_if(Ffnx, Ffnx + totfacets, FSth, fnx, pred);
    auto end_fny = thrust::copy_if(Ffny, Ffny + totfacets, FSth, fny, pred);
    auto end_fnz = thrust::copy_if(Ffnz, Ffnz + totfacets, FSth, fnz, pred);

    auto end_fux = thrust::copy_if(Ffux, Ffux + totfacets, FSth, fux, pred);
    auto end_fuy = thrust::copy_if(Ffuy, Ffuy + totfacets, FSth, fuy, pred);
    auto end_fuz = thrust::copy_if(Ffuz, Ffuz + totfacets, FSth, fuz, pred);

    auto end_fvx = thrust::copy_if(Ffvx, Ffvx + totfacets, FSth, fvx, pred);
    auto end_fvy = thrust::copy_if(Ffvy, Ffvy + totfacets, FSth, fvy, pred);
    auto end_fvz = thrust::copy_if(Ffvz, Ffvz + totfacets, FSth, fvz, pred);

    //auto end_Itd = thrust::copy_if(FItd, FItd + totfacets, FSth, Itd, pred);

    int valid = (int)(end_fx - fx);
    return valid;
}

// note: the following function requires dx, dy and dz to be normalized
__device__ float prismIntersectionDistance(float ox, float oy, float oz,
                                          float dx, float dy, float dz,
                                          float xmin, float xmax,
                                          float ymin, float ymax,
                                          float zmin, float zmax,
                                          float t) {

    const float INF = 1e30f;

    // X slab
    float tx1, tx2;
    if (fabsf(dx) < 1e-12f) {
        // Ray parallel to X planes: if origin outside slab -> no hit
        if (ox < xmin || ox > xmax) return 0.0f;
        tx1 = -INF;
        tx2 = INF;
    } else {
        tx1 = (xmin - ox) / dx;
        tx2 = (xmax - ox) / dx;
        if (tx1 > tx2) { float tmp = tx1; tx1 = tx2; tx2 = tmp; } // ensure entry <= exit
    }

    // Y slab
    float ty1, ty2;
    if (fabsf(dy) < 1e-12f) {
        if (oy < ymin || oy > ymax) return 0.0f;
        ty1 = -INF;
        ty2 = INF;
    } else {
        ty1 = (ymin - oy) / dy;
        ty2 = (ymax - oy) / dy;
        if (ty1 > ty2) { float tmp = ty1; ty1 = ty2; ty2 = tmp; }
    }

    // Z slab
    float tz1, tz2;
    if (fabsf(dz) < 1e-12f) {
        if (oz < zmin || oz > zmax) return 0.0f;
        tz1 = -INF;
        tz2 = INF;
    } else {
        tz1 = (zmin - oz) / dz;
        tz2 = (zmax - oz) / dz;
        if (tz1 > tz2) { float tmp = tz1; tz1 = tz2; tz2 = tmp; }
    }

    // Largest entry, smallest exit
    float t_entry = fmaxf(fmaxf(tx1, ty1), tz1);
    float t_exit  = fminf(fminf(tx2, ty2), tz2);

    // clamp by the length of the ray
    if (t_entry < 0.0f) t_entry = 0.0f;
    if (t_exit  > t)    t_exit  = t;

    // check if there is no intersection
    if (t_entry >= t_exit) {
        return 0.0f;
    }

    // return the length of the ray inside the prism
    return t_exit - t_entry;

}


__global__ void compTargetRays(float tx, float ty, float tz,
                               float* d_fx, float* d_fy, float* d_fz,
                               float* d_fnx, float* d_fny, float* d_fnz,
                               float* d_fux, float* d_fuy, float* d_fuz,
                               float* d_fvx, float* d_fvy, float* d_fvz,
                               float* d_Ttd, float* d_Tph, float* d_Tth,
                               int nfacets,
                               float* d_attXmin, float* d_attXmax,
                               float* d_attYmin, float* d_attYmax,
                               float* d_attZmin, float* d_attZmax,
                               float* d_alphas, float alpha2, int nAttenPrisms,
                               float* d_fRefrEI, float* d_fRefrEO) {
    
    // first get the distance betweem the source and all facets
    pointDistanceBulk(tx, ty, tz,
                      d_fx, d_fy, d_fz,
                      d_Ttd, nfacets);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfacets) {

        // get incident cartesian vector
        float Ix = (tx - d_fx[idx]) / d_Ttd[idx];
        float Iy = (ty - d_fy[idx]) / d_Ttd[idx];
        float Iz = (tz - d_fz[idx]) / d_Ttd[idx];

        // incident inclination
        d_Tth[idx] = fmaxf(-1.0f, fminf(1.0f, 
                            slowArcCos(dotProduct(-1*Ix, -1*Iy, -1*Iz, 
                            d_fnx[idx], d_fny[idx], d_fnz[idx]))));

        // incident azimuth
        d_Tph[idx] = slowAtan2(
            dotProduct(Ix, Iy, Iz, d_fux[idx], d_fuy[idx], d_fuz[idx]),
            dotProduct(Ix, Iy, Iz, d_fvx[idx], d_fvy[idx], d_fvz[idx])
        );

        // compute the attenuation distance through all prisms
        // first iterate over prisms
        float exponent = 0.0f;
        float total_atten_dist = 0.0f;
        
        for (int p=0; p<nAttenPrisms; p++) {

            // compute the distance in a given prism
            float atten_dist = prismIntersectionDistance(
                d_fx[idx], d_fy[idx], d_fz[idx],
                Ix, Iy, Iz,
                d_attXmin[p], d_attXmax[p],
                d_attYmin[p], d_attYmax[p],
                d_attZmin[p], d_attZmax[p],
                d_Ttd[idx]
            );

            // accumulate total attenuation distance and exponent
            total_atten_dist += atten_dist;
            exponent += d_alphas[p] * atten_dist;

        }

        // regions not in prisms also attenuate by the background alpha
        exponent += alpha2 * (d_Ttd[idx] - total_atten_dist);
        float atten = expf(-2.0 * exponent);

        // apply attenuation to ray weights in and out
        d_fRefrEI[idx] = atten;
        d_fRefrEO[idx] = atten;

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


__device__ float radarEq(float P, float G, float fs, float lam, float dist){
    
    // Pr = (Pt * G^2 * lam^2 * sigma) / ((4*pi)^3 * R^4)
    // note this is the radar equation using the RCS for a flat incident facet.
    float num = P * G * G * lam * lam * fs * fs * fs * fs;
    float denom = pow(4 * 3.14159, 3) * dist * dist * dist * dist; // (4*pi)^3
    return (num / denom);

}


__global__ void compReflectedEnergy(float* d_Itd, float* d_Ith, float* d_Iph,
                                    float* d_fRe, float* d_Rth, float* d_fRfrC,
                                    float P, float G, float fs, float lam, 
                                    float nu1, float nu2, float alpha1, 
                                    float ks, int polarization, int nfacets){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfacets) {

        // first get facet reradiation
        // we double inclination angle to as the center of the beam pattern is in the
        // exact opposite direction as the incident ray. 
        d_fRe[idx] = facetReradiation(d_Itd[idx], 2*d_Ith[idx], -1*d_Iph[idx], lam, fs);

        // losses from radar equation
        d_fRe[idx] = d_fRe[idx] * radarEq(P, G, fs, lam, d_Itd[idx]);

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
                                  float* d_Rth, 
                                  float* d_fx, float* d_fy, float* d_fz,
                                  float eps_1, float eps_2, int nfacets) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfacets) {

        // refracted inclination
        d_Rth[idx] = snellsLaw(d_Ith[idx], eps_1, eps_2);

    }    

}


__global__ void compRefrEnergyIn(
                    float* d_Rth, float* d_Itd, float* d_Iph,
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
        d_fRefrEI[id] *= facetReradiation(d_Ttd[id], delta_th, delta_ph, lam, fs);

        // refraction coefficient
        d_fRefrEI[id] = d_fRefrEI[id] * d_fRfrC[id];

        // signal attenuation
        //d_fRefrEI[id] = d_fRefrEI[id] * expf(-2 * alpha2 * d_Ttd[id]);

        // surface roughness losses
        float rough_loss = expf(-4*((ks*cosGPU(d_Rth[id]))*(ks*cosGPU(d_Rth[id]))));
        d_fRefrEI[id] = d_fRefrEI[id] * rough_loss;

        // total travel slant range
        d_fRfrSR[id] = d_Itd[id] + d_Ttd[id];

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

        d_fRefrEO[idx] *= facetReradiation(d_Itd[idx], RrTh, d_Tph[idx]-d_Iph[idx], lam, fs);

        // for transmission coefficient use refraction cofficient from before
        d_fRefrEO[idx] = d_fRefrEO[idx] * d_fRfrC[idx];

        // signal attenuation in subsurface
        //d_fRefrEO[idx] = d_fRefrEO[idx] * expf(-2.0f * alpha2 * d_Ttd[idx]);

        // surface roughness losses
        float rough_loss = expf(-4*((ks*cosGPU(d_Tth[idx]))*(ks*cosGPU(d_Tth[idx]))));
        d_fRefrEO[idx] = d_fRefrEO[idx] * rough_loss;

    }

}