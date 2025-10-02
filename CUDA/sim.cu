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
#include "file_io.cu"

const float c     = 299792458.0f; // speed of light [m/s]
const float eps_0 = 8.85e-12;
const float nu0 = 376.7;

// timing
struct timeval t1, t2;
void startTimer() { gettimeofday(&t1, 0); }
void reportTime(const char* msg) {
    gettimeofday(&t2, 0);
    // report the amount of time in ms
    float dt = (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_usec - t1.tv_usec) * 1e-3;
    std::cout << msg << " took " << dt << " ms." << std::endl;
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

    SimulationParameters par;
    par = parseSimulationParameters("params.json");

    // --- FACET ARRAY ---

    // dimensions
    int nxfacets = 2000;
    int nyfacets = 400;
    int nfacets  = nxfacets * nyfacets;

    // origin
    float x0 = -5000.0f; // x origin
    float y0 = -1000.0f; // y origin
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

    // allocate incident rays
    float *d_Itd, *d_Iph, *d_Ith;
    cudaMalloc((void**)&d_Itd, nfacets * sizeof(float));
    cudaMalloc((void**)&d_Iph, nfacets * sizeof(float));
    cudaMalloc((void**)&d_Ith, nfacets * sizeof(float));

    // allocate refracted rays
    float* d_Rth; float* d_Rtd;
    cudaMalloc((void**)&d_Rth, nfacets * sizeof(float));
    cudaMemsetAsync(d_Rth, 0, nfacets * sizeof(float));
    cudaMalloc((void**)&d_Rtd, nfacets * sizeof(float));
    cudaMemsetAsync(d_Rtd, 0, nfacets * sizeof(float));

    // allocate refraction coefficients
    float* d_fRfrC;
    cudaMalloc((void**)&d_fRfrC, nfacets * sizeof(float));
    cudaMemsetAsync(d_fRfrC, 0, nfacets * sizeof(float));

    // array for refraction travel distance to target
    float* d_fRfrSR;
    cudaMalloc((void**)&d_fRfrSR, nfacets * sizeof(float));
    cudaMemsetAsync(d_fRfrSR, 0, nfacets * sizeof(float));

    // reflected energy for each facet (default 0)
    float* d_fReflE;
    cudaMalloc((void**)&d_fReflE, nfacets * sizeof(float));
    cudaMemsetAsync(d_fReflE, 0, nfacets * sizeof(float));

    // refl signal array
    cuFloatComplex* d_refl_sig;
    cudaMalloc((void**)&d_refl_sig, par.nr * sizeof(cuFloatComplex));
    cudaMemsetAsync(d_refl_sig, 0, par.nr * sizeof(cuFloatComplex));

    // forced ray
    float *d_Ttd, *d_Tph, *d_Tth;
    cudaMalloc((void**)&d_Ttd, nfacets * sizeof(float));
    cudaMalloc((void**)&d_Tph, nfacets * sizeof(float));
    cudaMalloc((void**)&d_Tth, nfacets * sizeof(float));

    // inwards refracted weights
    float* d_fReflEI;
    cudaMalloc((void**)&d_fReflEI, nfacets * sizeof(float));
    cudaMemsetAsync(d_fReflEI, 0, nfacets * sizeof(float));

    // upward transmitted
    float* d_fReflEO;
    cudaMalloc((void**)&d_fReflEO, nfacets * sizeof(float));
    cudaMemsetAsync(d_fReflEO, 0, nfacets * sizeof(float));

    // refracted signal
    cuFloatComplex* d_refr_sig;
    cudaMalloc((void**)&d_refr_sig, par.nr * sizeof(cuFloatComplex));
    cudaMemsetAsync(d_refr_sig, 0, par.nr * sizeof(cuFloatComplex));

    // joint signal
    cuFloatComplex* d_sig;
    cudaMalloc((void**)&d_sig, par.nr * sizeof(cuFloatComplex));
    cudaMemsetAsync(d_sig, 0, par.nr * sizeof(cuFloatComplex));


    // number of sources
    int ns  = 200;
    int dsx = 50;

    for (int is=0; is<ns; is++) {

        // update source position
        float sx = -5e3 + is * dsx;

        startTimer();

        // --- CLEAR PER-SOURCE ACCUMULATION BUFFERS ---
        // refl_sig and refr_sig are accumulated into with atomicAdd inside
        // the kernels. They must be cleared before each source so traces do
        // not accumulate across iterations.
        cudaMemset(d_refl_sig, 0, par.nr * sizeof(cuFloatComplex));
        cudaMemset(d_refr_sig, 0, par.nr * sizeof(cuFloatComplex));
        // d_sig is fully overwritten by combineRadarSignals, so clearing it is
        // optional; left out for slightly better performance.

        // --- COMP INCIDENT RAYS ---
        
        int blockSize = 256;
        int numBlocks = (nfacets + blockSize - 1) / blockSize;
        compIncidentRays<<<numBlocks, blockSize>>>(sx, par.sy, par.sz,
                                                d_fx, d_fy, d_fz,
                                                d_fnx, d_fny, d_fnz,
                                                d_fux, d_fuy, d_fuz,
                                                d_fvx, d_fvy, d_fvz,
                                                d_Itd, d_Iph, d_Ith,
                                                nfacets);
        checkCUDAError("compIncidentRays kernel");


        // --- COMP REFRACTED RAYS ---
                
        compRefractedRays<<<numBlocks, blockSize>>>(d_Ith, d_Iph, 
                                                    d_Rth, d_Rtd,
                                                    d_fx, d_fy, d_fz,
                                                    par.tx, par.ty, par.tz,
                                                    par.eps_1, par.eps_2, nfacets);
        checkCUDAError("compRefractedRays kernel");

        
        // --- COMP REFLECTED ENERGY ---

        compReflectedEnergy<<<numBlocks, blockSize>>>(d_Itd, d_Ith, d_Iph,
                                                    d_fReflE, d_Rth, d_fRfrC,
                                                    par.P, par.Grefl_lin, par.sigma, fs, par.lam,
                                                    par.nu_1, par.nu_2, par.alpha1, par.ks, 
                                                    par.pol, nfacets);
        checkCUDAError("compReflectedEnergy kernel");

        
        // --- CONSTRUCT REFLECTED SIGNAL ---

        // launch with shared memory for per-block accumulation (real+imag floats)
        reflRadarSignal<<<numBlocks, blockSize, 2 * REFL_TILE_NR * sizeof(float)>>>(d_Itd, d_fReflE, d_refl_sig,
                                par.rst, par.dr, par.nr,
                                par.rng_res, par.lam, nfacets);
        checkCUDAError("constructRadarSignal kernel1");


        // --- FORCED RAY TO TARGET COMP ---
        
        compTargetRays<<<numBlocks, blockSize>>>(par.tx, par.ty, par.tz,
                                                d_fx, d_fy, d_fz,
                                                d_fnx, d_fny, d_fnz,
                                                d_fux, d_fuy, d_fuz,
                                                d_fvx, d_fvy, d_fvz,
                                                d_Ttd, d_Tph, d_Tth,
                                                nfacets);
        checkCUDAError("compTargetRays kernel");


        // --- CONSTRUCT REFRACTED WEIGHTS INWARDS ---
        compRefrEnergyIn<<<numBlocks, blockSize>>>(d_Rtd, d_Rth, d_Itd, d_Iph,
                                                d_Ttd, d_Tth, d_Tph, d_fRfrC,
                                                d_fReflEI, d_fRfrSR,
                                                par.ks, nfacets, par.alpha2, par.c_1, par.c_2,
                                                fs, par.P, par.Grefr_lin, par.lam);
        checkCUDAError("compRefrEnergyIn kernel");


        // --- COMPUTE UPWARD TRANSMITTED RAYS ---

        compRefrEnergyOut<<<numBlocks, blockSize>>>(d_Itd, d_Iph,
                                                    d_Ttd, d_Tth, d_Tph, 
                                                    d_fReflEO, d_fRfrC, 
                                                    par.ks, nfacets, par.alpha1, par.alpha2, par.c_1, par.c_2,
                                                    fs, par.P, par.lam, par.eps_1, par.eps_2);
        checkCUDAError("compRefrEnergyOut kernel");
        
        // --- CONSTRUCT REFRACTED SIGNAL ---

        // launch with shared memory for per-block accumulation (real+imag floats)
        refrRadarSignal<<<numBlocks, blockSize, 2 * REFR_TILE_NR * sizeof(float)>>>(d_fRfrSR, d_Rtd, 
                        d_fReflEI, d_fReflEO,
                        d_refr_sig, 
                        par.rst, par.dr, par.nr, par.c, par.c_2,
                        par.rng_res, par.P, par.Grefr_lin, fs, par.lam, nfacets);
        checkCUDAError("refrRadarSignal kernel");


        // --- COMBINE INTO OUTPUT SIGNAL AND EXPORT ---

        combineRadarSignals<<<(par.nr + blockSize - 1) / blockSize, blockSize>>>(d_refl_sig, d_refr_sig, d_sig, par.nr);
        checkCUDAError("combineRadarSignals kernel");

        // Wait for the GPU to finish
        cudaDeviceSynchronize();

        // copy Itd to host and export as file
        char* filename = (char*)malloc(20 * sizeof(char));
        sprintf(filename, "rdrgrm/s%03d.txt", is);
        saveSignalToFile(filename, d_sig, par.nr);

        // overwrite progress printed to terminal
        printf("\rCompleted source %d of %d", is+1, ns);
        fflush(stdout);

    }

    cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
    cudaFree(d_fnx); cudaFree(d_fny); cudaFree(d_fnz);
    cudaFree(d_fux); cudaFree(d_fuy); cudaFree(d_fuz);
    cudaFree(d_fvx); cudaFree(d_fvy); cudaFree(d_fvz);
    cudaFree(d_Itd); cudaFree(d_Iph); cudaFree(d_Ith);
    cudaFree(d_Rth); cudaFree(d_Rtd);
    cudaFree(d_fRfrC);
    cudaFree(d_fRfrSR);
    cudaFree(d_fReflE);
    cudaFree(d_refl_sig);
    cudaFree(d_Ttd); cudaFree(d_Tph); cudaFree(d_Tth);
    cudaFree(d_fReflEI);
    cudaFree(d_fReflEO);
    cudaFree(d_refr_sig);
    cudaFree(d_sig);

    return 0;
}
