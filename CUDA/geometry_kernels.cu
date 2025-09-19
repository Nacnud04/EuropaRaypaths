#include <cuComplex.h>
#include <math.h>

#include "elementary_kernels.cu"

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


__device__ cuFloatComplex facetReradiation(float dist, float th, float ph,  
                                  float lam, float fs){

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
    float sinc1 = sinc((fs / lam) * slowSin(ph) * slowCos(th));
    // np.sinc(((r) / lam) * np.sin(ph) * np.sin(th))
    float sinc2 = sinc((fs / lam) * slowSin(ph) * slowSin(th));

    // combine all together
    return cuCmulf(c_val, make_cuFloatComplex(sinc1 * sinc2, 0.0f));

}


__device__ float radarEq(float P, float G, float sigma, float lam, float dist, float fs){
    
    // Pr = (Pt * G^2 * lam^2 * sigma) / ((4*pi)^3 * R^4)
    float num = P * G * G * lam * lam * sigma;
    float denom = pow(4 * 3.14159, 3) * dist * dist * dist * dist; // (4*pi)^3
    return num / denom;

}


__global__ void compRefractedEnergy(float* d_Itd, float* d_Ith, float* d_Iph,
                                    cuFloatComplex* d_fRe,
                                    float P, float G, float sigma, float fs, float lam, 
                                    float nu1, float nu2, int nfacets){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfacets) {

        // first get facet reradiation
        d_fRe[idx] = facetReradiation(d_Itd[idx], d_Ith[idx], d_Iph[idx], lam, fs);

        // losses from radar equation
        d_fRe[idx] = cuCmulf(d_fRe[idx], make_cuFloatComplex(radarEq(P, G, sigma, lam, d_Itd[idx], fs), 0.0f));

        // just say theta2 = -0.35
        float theta2 = -0.35;

        // reflection coefficient
        float rho_h = (nu2 * slowCos(d_Ith[idx]) - nu1 * slowCos(d_Ith[idx])) /
                      (nu2 * slowCos(d_Ith[idx]) + nu1 * slowCos(d_Ith[idx]));
        d_fRe[idx] = cuCmulf(d_fRe[idx], make_cuFloatComplex(1 - rho_h * rho_h, 0.0f));

    }

}


__global__ void snellsLaw(float* Ith, float eps_1, float eps_2, int nfacets) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfacets) {

        float sin_Ith = slowSin(Ith[idx]);
        float sin_Rth = sqrtf(eps_1/eps_2) * sin_Ith;
        if (sin_Rth > 1.0f) {
            // total internal reflection set to something more than pi/2 so it is easy to identify
            Ith[idx] = 1e4;
        } else {
            Ith[idx] = slowArcSin(sin_Rth);
        }
    }
}