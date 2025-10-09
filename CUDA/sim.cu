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

    // --- LOAD FACETS ---

    // first open facet file
    const char* filename;
    filename = "facets.fct";

    // load file and count facets
    FILE *facetFile = fopen(filename, "r");
    const int totFacets = count_lines(facetFile);
    std::cout << "Found " << totFacets << " in " << filename << std::endl;

    // allocate host memory for facet arrays
    float *h_fx = (float*)malloc(totFacets * sizeof(float));
    float *h_fy = (float*)malloc(totFacets * sizeof(float));
    float *h_fz = (float*)malloc(totFacets * sizeof(float));
    float *h_fnx = (float*)malloc(totFacets * sizeof(float));
    float *h_fny = (float*)malloc(totFacets * sizeof(float));
    float *h_fnz = (float*)malloc(totFacets * sizeof(float));
    float *h_fux = (float*)malloc(totFacets * sizeof(float));
    float *h_fuy = (float*)malloc(totFacets * sizeof(float));
    float *h_fuz = (float*)malloc(totFacets * sizeof(float));
    float *h_fvx = (float*)malloc(totFacets * sizeof(float));
    float *h_fvy = (float*)malloc(totFacets * sizeof(float));
    float *h_fvz = (float*)malloc(totFacets * sizeof(float));

    startTimer();

    loadFacetFile(facetFile, totFacets,
                  h_fx, h_fy, h_fz, 
                  h_fnx, h_fny, h_fnz, 
                  h_fux, h_fuy, h_fuz,
                  h_fvx, h_fvy, h_fvz);

    reportTime("Loading facets and their bases");

    // estimate the number of illuminated facets
    float aperture = 7.5; // aperture in degrees
    int nfacets = nIlluminatedFacets(par.sz, 0, par.fs, aperture);
    std::cout << "Estimated that " << nfacets << " facets are illuminated by aperture" << std::endl;

    // temp change to stop illegal access
    nfacets = totFacets;

    // --- MOVE ALL FACETS FROM HOST TO GPU ---
    // ----- this alloc is for all facets -----

    // facet coordinates
    float *d_Ffx, *d_Ffy, *d_Ffz;
    // facet normal
    float *d_Ffnx, *d_Ffny, *d_Ffnz;
    // facet tangents
    float *d_Ffux, *d_Ffuy, *d_Ffuz;
    float *d_Ffvx, *d_Ffvy, *d_Ffvz;

    reportNewAlloc(12 * totFacets * sizeof(float), "all facets");

    // allocate
    cudaMalloc((void**)&d_Ffx, totFacets * sizeof(float));
    cudaMalloc((void**)&d_Ffy, totFacets * sizeof(float));
    cudaMalloc((void**)&d_Ffz, totFacets * sizeof(float));
    cudaMalloc((void**)&d_Ffnx, totFacets * sizeof(float));
    cudaMalloc((void**)&d_Ffny, totFacets * sizeof(float));
    cudaMalloc((void**)&d_Ffnz, totFacets * sizeof(float));
    cudaMalloc((void**)&d_Ffux, totFacets * sizeof(float));
    cudaMalloc((void**)&d_Ffuy, totFacets * sizeof(float));
    cudaMalloc((void**)&d_Ffuz, totFacets * sizeof(float));
    cudaMalloc((void**)&d_Ffvx, totFacets * sizeof(float));
    cudaMalloc((void**)&d_Ffvy, totFacets * sizeof(float));
    cudaMalloc((void**)&d_Ffvz, totFacets * sizeof(float));

    // copy host -> device
    cudaMemcpy(d_Ffx, h_fx, totFacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ffy, h_fy, totFacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ffz, h_fz, totFacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ffnx, h_fnx, totFacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ffny, h_fny, totFacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ffnz, h_fnz, totFacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ffux, h_fux, totFacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ffuy, h_fuy, totFacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ffuz, h_fuz, totFacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ffvx, h_fvx, totFacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ffvy, h_fvy, totFacets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ffvz, h_fvz, totFacets * sizeof(float), cudaMemcpyHostToDevice);

    // allocate incident rays
    float *d_FItd, *d_FIph, *d_FIth;
    cudaMalloc((void**)&d_FItd, totFacets * sizeof(float));
    cudaMalloc((void**)&d_FIph, totFacets * sizeof(float));
    cudaMalloc((void**)&d_FIth, totFacets * sizeof(float));


    // --- this alloc is for cropped facets ---

    // facet coordinates
    float *d_fx, *d_fy, *d_fz;
    // facet normal
    float *d_fnx, *d_fny, *d_fnz;
    // facet tangents
    float *d_fux, *d_fuy, *d_fuz;
    float *d_fvx, *d_fvy, *d_fvz;

    reportNewAlloc(29 * nfacets * sizeof(float), "illuminated facets");

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
        // compute incident rays for the full facet set (totFacets)
        int numBlocksAll = (totFacets + blockSize - 1) / blockSize;
        compIncidentRays<<<numBlocksAll, blockSize>>>(sx, par.sy, par.sz,
                            d_Ffx, d_Ffy, d_Ffz,
                            d_Ffnx, d_Ffny, d_Ffnz,
                            d_Ffux, d_Ffuy, d_Ffuz,
                            d_Ffvx, d_Ffvy, d_Ffvz,
                            d_FItd, d_FIph, d_FIth,
                            totFacets);
        checkCUDAError("compIncidentRays kernel");
        cudaDeviceSynchronize();

        // --- CROP TO ILLUMINATED ---
        // crop full facet arrays into the per-source cropped arrays and get
        // the number of valid (illuminated) facets returned by the function
        
        int valid_facets = cropByAperture(totFacets, nfacets, aperture,
                                        d_Ffx,  d_Ffy,  d_Ffz,
                                        d_Ffnx, d_Ffny, d_Ffnz,
                                        d_Ffux, d_Ffuy, d_Ffuz,
                                        d_Ffvx, d_Ffvy, d_Ffvz,
                                        d_FItd, d_FIph, d_FIth,
                                        d_fx,   d_fy,   d_fz,
                                        d_fnx,  d_fny,  d_fnz,
                                        d_fux,  d_fuy,  d_fuz,
                                        d_fvx,  d_fvy,  d_fvz,
                                        d_Itd,  d_Iph,  d_Ith);
        checkCUDAError("cropByAperture process");
        cudaDeviceSynchronize();

        // If no facets are illuminated, skip per-facet kernels to avoid
        // operating on uninitialised memory which can produce NaNs.
        if (valid_facets == 0) {
            std::cout << "No illuminated facets for source " << is << ", skipping." << std::endl;
            continue;
        }

        int numBlocks = (nfacets + blockSize - 1) / blockSize;

        // --- COMP REFRACTED RAYS ---
                
        compRefractedRays<<<numBlocks, blockSize>>>(d_Ith, d_Iph, 
                                                    d_Rth, d_Rtd,
                                                    d_fx, d_fy, d_fz,
                                                    par.tx, par.ty, par.tz,
                                                    par.eps_1, par.eps_2, valid_facets);
        checkCUDAError("compRefractedRays kernel");

        
        // --- COMP REFLECTED ENERGY ---

        compReflectedEnergy<<<numBlocks, blockSize>>>(d_Itd, d_Ith, d_Iph,
                                                    d_fReflE, d_Rth, d_fRfrC,
                                                    par.P, par.Grefl_lin, par.sigma, par.fs, par.lam,
                                                    par.nu_1, par.nu_2, par.alpha1, par.ks, 
                                                    par.pol, valid_facets);
        checkCUDAError("compReflectedEnergy kernel");

        
        // --- CONSTRUCT REFLECTED SIGNAL ---

        // launch with shared memory for per-block accumulation (real+imag floats)
        reflRadarSignal<<<numBlocks, blockSize, 2 * REFL_TILE_NR * sizeof(float)>>>(d_Itd, d_fReflE, d_refl_sig,
                                par.rst, par.dr, par.nr,
                                par.rng_res, par.lam, valid_facets);
        checkCUDAError("constructRadarSignal kernel1");


        // --- FORCED RAY TO TARGET COMP ---
        
        compTargetRays<<<numBlocks, blockSize>>>(par.tx, par.ty, par.tz,
                                                d_fx,  d_fy,  d_fz,
                                                d_fnx, d_fny, d_fnz,
                                                d_fux, d_fuy, d_fuz,
                                                d_fvx, d_fvy, d_fvz,
                                                d_Ttd, d_Tph, d_Tth,
                                                valid_facets);
        checkCUDAError("compTargetRays kernel");


        // --- CONSTRUCT REFRACTED WEIGHTS INWARDS ---
        compRefrEnergyIn<<<numBlocks, blockSize>>>(d_Rtd, d_Rth, d_Itd, d_Iph,
                                                d_Ttd, d_Tth, d_Tph, d_fRfrC,
                                                d_fReflEI, d_fRfrSR,
                                                par.ks, valid_facets, par.alpha2, par.c_1, par.c_2,
                                                par.fs, par.P, par.Grefr_lin, par.lam);
        checkCUDAError("compRefrEnergyIn kernel");


        // --- COMPUTE UPWARD TRANSMITTED RAYS ---

        compRefrEnergyOut<<<numBlocks, blockSize>>>(d_Itd, d_Iph,
                                                    d_Ttd, d_Tth, d_Tph, 
                                                    d_fReflEO, d_fRfrC, 
                                                    par.ks, valid_facets, par.alpha1, par.alpha2, par.c_1, par.c_2,
                                                    par.fs, par.P, par.lam, par.eps_1, par.eps_2);
        checkCUDAError("compRefrEnergyOut kernel");
        
        // --- CONSTRUCT REFRACTED SIGNAL ---

        // launch with shared memory for per-block accumulation (real+imag floats)
        refrRadarSignal<<<numBlocks, blockSize, 2 * REFR_TILE_NR * sizeof(float)>>>(d_fRfrSR, d_Rtd, 
                        d_fReflEI, d_fReflEO,
                        d_refr_sig, 
                        par.rst, par.dr, par.nr, par.c, par.c_2,
                        par.rng_res, par.P, par.Grefr_lin, par.fs, par.lam, valid_facets);
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

    cudaFree(d_Ffx); cudaFree(d_Ffy); cudaFree(d_Ffz);
    cudaFree(d_Ffnx); cudaFree(d_Ffny); cudaFree(d_Ffnz);
    cudaFree(d_Ffux); cudaFree(d_Ffuy); cudaFree(d_Ffuz);
    cudaFree(d_Ffvx); cudaFree(d_Ffvy); cudaFree(d_Ffvz);
    cudaFree(d_FItd); cudaFree(d_FIph); cudaFree(d_FIth);
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
