#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <cuComplex.h>

// custom kernels
#include "geometry_kernels.cu"
#include "facet_funcs.cu"
#include "radar_kernels.cu"

// timing
struct timeval t1, t2;
void startTimer() { gettimeofday(&t1, 0); }
void reportTime(const char* msg) {
    gettimeofday(&t2, 0);
    double elapsed = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e3;
    std::cout << msg << " took " << elapsed << " ms." << std::endl;
    gettimeofday(&t1, 0);
}

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        std::cerr << "CUDA ERROR IN: " << msg << "-> " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void reportNewAlloc(int bytes, const char* varName) {
    bytes /= 1e6;
    std::cout << "Allocating " << bytes << " MB on GPU for " << varName << std::endl;
}

void showMemAlloc()
{
    size_t fLEe_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&fLEe_byte, &total_byte);
    if (cudaSuccess != cuda_status) {
        std::cout << "Error: cudaMemGetInfo fails, " << cudaGetErrorString(cuda_status) << std::endl;
        exit(1);
    }
    float fLEe_db = (float)fLEe_byte;
    float total_db = (float)total_byte;
    float used_db = total_db - fLEe_db;
    std::cout << "GPU memory usage: used = " << used_db / 1024.0 / 1024.0 << " MB, fLEe = " << fLEe_db / 1024.0 / 1024.0 << " MB, total = " << total_db / 1024.0 / 1024.0 << " MB" << std::endl;
}

int main()
{

    // --- SOURCE ---

    float sx, sy, sz;
    sx = 0.0f; sy = 0.0f; sz = 10000.0f;

    float c     = 299792458.0f; // speed of light [m/s]
    float pi    = 3.14159;      // pi

    float P     = 100;          // Power [w]
    float f0    = 9e6;          // center fLEquency [Hz]
    //float B     = 1e6;          // bandwidth [Hz]
    float lam   = c / f0;       // wavelength [m]
    float k     = (2*pi)/lam;    // wave #
    float sigma = 1.0; 
    float Grefr = 75;           // subsurface gain [dB]
    float Grefl = 75;           // surface gain [dB]

    float rng_res = 300; // range resolution [m]

    int pol = 1; // vertical polarization

    // linear gain
    float Grefl_lin = pow(10, Grefl / 10.0f);
    float Grefr_lin = pow(10, Grefr / 10.0f);

    // --- TARGET ---

    float tx, ty, tz;
    tx = 0.0f; ty = 0.0f; tz = -1000.0f;

    // --- SURFACE PARAMS --- 
    
    float rms_h = 0.4;    // surface RMS height [m]
    float ks    = rms_h * k; // em roughness


    // --- SUBSURFACE PARAMS ---
    
    // permittivities
    float eps_0 = 8.85e-12;
    float eps_1 = 1;
    float eps_2 = 3.15;

    // velocities
    float c_1 = c / sqrt(eps_1);
    float c_2 = c / sqrt(eps_2);

    // impedances
    float nu0 = 376.7;
    float nu1 = nu0 / sqrt(eps_1);
    float nu2 = nu0 / sqrt(eps_2);

    // conductivies
    float sig1 = 0.0f;
    float sig2 = 1e-6f;

    // attenuation coefficients
    float eps_pp_1 = sig1 / (2 * pi * f0 * eps_0);
    float alpha1   = sqrt(1 + (eps_pp_1/eps_1)*(eps_pp_1/eps_1)) - 1;
          alpha1   = sqrt(0.5 * eps_1 * alpha1);
          alpha1   = (alpha1 * 2 * pi) / lam;

    float eps_pp_2 = sig2 / (2 * pi * f0 * eps_0);
    float alpha2   = sqrt(1 + (eps_pp_2/eps_2)*(eps_pp_2/eps_2)) - 1;
          alpha2   = sqrt(0.5 * eps_2 * alpha2);
          alpha2   = (alpha2 * 2 * pi) / lam;


    // --- FACET ARRAY ---

    // dimensions
    int nxfacets = 400;
    int nyfacets = 400;
    int nfacets  = nxfacets * nyfacets;

    // origin
    float x0 = -1250.0f; // x origin
    float y0 = -1250.0f; // y origin
    float z0 = 0.0f;     // z origin

    // facet size
    float fs = 5.0f;

    // allocate host memory for facet arrays
    float *h_fx = (float*)malloc(nfacets * sizeof(float));
    float *h_fy = (float*)malloc(nfacets * sizeof(float));
    float *h_fz = (float*)malloc(nfacets * sizeof(float));
    float *h_fnx = (float*)malloc(nfacets * sizeof(float));
    float *h_fny = (float*)malloc(nfacets * sizeof(float));
    float *h_fnz = (float*)malloc(nfacets * sizeof(float));
    float *h_fux = (float*)malloc(nfacets * sizeof(float));
    float *h_fuy = (float*)malloc(nfacets * sizeof(float));
    float *h_fuz = (float*)malloc(nfacets * sizeof(float));
    float *h_fvx = (float*)malloc(nfacets * sizeof(float));
    float *h_fvy = (float*)malloc(nfacets * sizeof(float));
    float *h_fvz = (float*)malloc(nfacets * sizeof(float));

    startTimer();

    // generate flat surface
    generateFlatSurface(h_fx, h_fy, h_fz, nxfacets, nyfacets, x0, y0, z0, fs);

    // compute facet normals
    generateFacetNormals(h_fx, h_fy, h_fz, 
                         h_fnx, h_fny, h_fnz, 
                         h_fux, h_fuy, h_fuz,
                         h_fvx, h_fvy, h_fvz,
                         nxfacets, nyfacets);

    reportTime("Generate facets and normals");

    // --- FACET FROM HOST -> DEVICE ---

    // facet coordinates
    float *d_fx, *d_fy, *d_fz;
    // facet normal
    float *d_fnx, *d_fny, *d_fnz;
    // facet tangents
    float *d_fux, *d_fuy, *d_fuz;
    float *d_fvx, *d_fvy, *d_fvz;

    reportNewAlloc(12 * nfacets * sizeof(float), "facets");

    // allocate
    cudaMalloc((void**)&d_fx, nfacets * sizeof(float));
    cudaMalloc((void**)&d_fy, nfacets * sizeof(float));
    cudaMalloc((void**)&d_fz, nfacets * sizeof(float));
    cudaMalloc((void**)&d_fnx, nfacets * sizeof(float));
    cudaMalloc((void**)&d_fny, nfacets * sizeof(float));
    cudaMalloc((void**)&d_fnz, nfacets * sizeof(float));
    cudaMalloc((void**)&d_fux, nfacets * sizeof(float));
    cudaMalloc((void**)&d_fuy, nfacets * sizeof(float));
    cudaMalloc((void**)&d_fuz, nfacets * sizeof(float));
    cudaMalloc((void**)&d_fvx, nfacets * sizeof(float));
    cudaMalloc((void**)&d_fvy, nfacets * sizeof(float));
    cudaMalloc((void**)&d_fvz, nfacets * sizeof(float));

    // copy host -> device
    cudaMemcpy(d_fx, h_fx, nfacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy, h_fy, nfacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fz, h_fz, nfacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fnx, h_fnx, nfacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fny, h_fny, nfacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fnz, h_fnz, nfacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fux, h_fux, nfacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fuy, h_fuy, nfacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fuz, h_fuz, nfacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fvx, h_fvx, nfacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fvy, h_fvy, nfacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fvz, h_fvz, nfacets * sizeof(float), cudaMemcpyHostToDevice);


    // --- INCIDENT RAY ARRAYS ---
    startTimer();
    float *d_Itd, *d_Iph, *d_Ith;

    // allocate
    reportNewAlloc(3 * nfacets * sizeof(float), "incident rays");
    cudaMalloc((void**)&d_Itd, nfacets * sizeof(float));
    cudaMalloc((void**)&d_Iph, nfacets * sizeof(float));
    cudaMalloc((void**)&d_Ith, nfacets * sizeof(float));


    // --- COMP INCIDENT RAYS ---
    
    int blockSize = 256;
    int numBlocks = (nfacets + blockSize - 1) / blockSize;
    compIncidentRays<<<numBlocks, blockSize>>>(sx, sy, sz,
                                               d_fx, d_fy, d_fz,
                                               d_fnx, d_fny, d_fnz,
                                               d_fux, d_fuy, d_fuz,
                                               d_fvx, d_fvy, d_fvz,
                                               d_Itd, d_Iph, d_Ith,
                                               nfacets);
    checkCUDAError("compIncidentRays kernel");


    // --- COMP REFRACTED RAYS ---
    
    // for refracted ray computation inclincation is computed via snells law
    // and the azumuth is the same as incidence
    float* d_Rth; float* d_Rtd;
    cudaMalloc((void**)&d_Rth, nfacets * sizeof(float));
    cudaMemset(d_Rth, 0, nfacets * sizeof(float));
    cudaMalloc((void**)&d_Rtd, nfacets * sizeof(float));
    cudaMemset(d_Rtd, 0, nfacets * sizeof(float));
    compRefractedRays<<<numBlocks, blockSize>>>(d_Ith, d_Iph, 
                                                d_Rth, d_Rtd,
                                                d_fx, d_fy, d_fz,
                                                tx, ty, tz,
                                                eps_1, eps_2, nfacets);
    checkCUDAError("compRefractedRays kernel");

    // array for refraction coefficients
    float* d_fRfrC;
    cudaMalloc((void**)&d_fRfrC, nfacets * sizeof(float));
    cudaMemset(d_fRfrC, 0, nfacets * sizeof(float));

    // array for refraction travel distance to target
    float* d_fRfrSR;
    cudaMalloc((void**)&d_fRfrSR, nfacets * sizeof(float));
    cudaMemset(d_fRfrSR, 0, nfacets * sizeof(float));


    // --- COMP REFLECTED ENERGY ---

    // reflected energy for each facet (default 0)
    cuFloatComplex* d_fReflE;
    reportNewAlloc(2 * nfacets * sizeof(cuFloatComplex), "energy arrays"); // this also accounts for refracted later
    cudaMalloc((void**)&d_fReflE, nfacets * sizeof(cuFloatComplex));
    cudaMemset(d_fReflE, 0, nfacets * sizeof(cuFloatComplex));

    compReflectedEnergy<<<numBlocks, blockSize>>>(d_Itd, d_Ith, d_Iph,
                                                  d_fReflE, d_Rth, d_fRfrC,
                                                  P, Grefl_lin, sigma, fs, lam,
                                                  nu1, nu2, alpha1, ks, 
                                                  pol, nfacets);
    checkCUDAError("compReflectedEnergy kernel");

    
    // --- CONSTRUCT REFLECTED SIGNAL ---

    // range window params
    float rst =  8000.0f;  // start range [m]
    float ren = 13000.0f;  // end range [m]
    int   nr  =  1000;      // number of range bins
    float dr  = (ren - rst) / (nr - 1); // range step [m]

    // refl signal array
    cuFloatComplex* d_refl_sig;
    cudaMalloc((void**)&d_refl_sig, nr * sizeof(cuFloatComplex));
    cudaMemset(d_refl_sig, 0, nr * sizeof(cuFloatComplex));

    reflRadarSignal<<<numBlocks, blockSize>>>(d_Itd, d_fReflE, d_refl_sig,
                                                   rst, dr, nr,
                                                   rng_res, lam, nfacets);
    checkCUDAError("constructRadarSignal kernel1");


    // --- CONSTRUCT REFRACTED WEIGHTS INWARDS ---

    // first get weights
    cuFloatComplex* d_fReflEI;
    cudaMalloc((void**)&d_fReflEI, nfacets * sizeof(cuFloatComplex));
    cudaMemset(d_fReflEI, 0, nfacets * sizeof(cuFloatComplex));
    compRefrEnergyIn<<<numBlocks, blockSize>>>(d_Itd, d_Iph,
                                             d_Rtd, d_Rth, d_fRfrC,
                                             d_fReflEI, d_fRfrSR,
                                             ks, nfacets, alpha2, c_1, c_2,
                                             fs, P, Grefr_lin, lam);
    checkCUDAError("compRefrEnergyIn kernel");


    // --- COMPUTE UPWARD TRANSMITTED RAYS ---

    cuFloatComplex* d_fReflEO;
    cudaMalloc((void**)&d_fReflEO, nfacets * sizeof(cuFloatComplex));
    cudaMemset(d_fReflEO, 0, nfacets * sizeof(cuFloatComplex));
    compRefrEnergyOut<<<numBlocks, blockSize>>>(d_Itd, d_Iph,
                                                d_Rtd, d_Rth, 
                                                d_fReflEO, d_fRfrC, 
                                                ks, nfacets, alpha1, alpha2, c_1, c_2,
                                                fs, P, lam);
    checkCUDAError("compRefrEnergyOut kernel");


    // now compute signal
    cuFloatComplex* d_refr_sig;
    cudaMalloc((void**)&d_refr_sig, nr * sizeof(cuFloatComplex));
    // zero the newly allocated refracted signal buffer (was incorrectly zeroing d_refl_sig)
    cudaMemset(d_refr_sig, 0, nr * sizeof(cuFloatComplex));
    refrRadarSignal<<<numBlocks, blockSize>>>(d_fRfrSR, d_fReflEI, d_fReflEO,
                                              d_refr_sig,
                                              rst, dr, nr,
                                              rng_res, lam, nfacets);
    checkCUDAError("refrRadarSignal kernel");

    reportTime("Constructing signals");

    // Wait for the GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // copy Itd to host and export as file
    cuFloatComplex* h_sig = (cuFloatComplex*)malloc(nr * sizeof(cuFloatComplex));
    cudaMemcpy(h_sig, d_refl_sig, nr * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    FILE* fItd = fopen("ReflSignal.txt", "w");
    //print example values
    for (int i = 0; i < nr; i++) {
        fprintf(fItd, "%f\n", cuCabsf(h_sig[i]));
    }
    fclose(fItd);

    h_sig = (cuFloatComplex*)malloc(nr * sizeof(cuFloatComplex));
    cudaMemcpy(h_sig, d_refr_sig, nr * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    fItd = fopen("RefrSignal.txt", "w");
    //print example values
    for (int i = 0; i < nr; i++) {
        fprintf(fItd, "%f\n", cuCabsf(h_sig[i]));
    }
    fclose(fItd);

    // copy Rtd to host and export as file
    float* h_Rth = (float*)malloc(nfacets * sizeof(float));
    cudaMemcpy(h_Rth, d_Itd, nfacets * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* fRtd = fopen("Rtd_cuda.txt", "w");
    //print example values
    for (int i = 0; i < nxfacets; i++) {
        fprintf(fRtd, "%f\n", h_Rth[i + i * 400]);
    }
    fclose(fRtd);

    return 0;
}