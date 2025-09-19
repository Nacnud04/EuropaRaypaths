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
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    if (cudaSuccess != cuda_status) {
        std::cout << "Error: cudaMemGetInfo fails, " << cudaGetErrorString(cuda_status) << std::endl;
        exit(1);
    }
    float free_db = (float)free_byte;
    float total_db = (float)total_byte;
    float used_db = total_db - free_db;
    std::cout << "GPU memory usage: used = " << used_db / 1024.0 / 1024.0 << " MB, free = " << free_db / 1024.0 / 1024.0 << " MB, total = " << total_db / 1024.0 / 1024.0 << " MB" << std::endl;
}

int main()
{

    // --- SOURCE ---

    float sx, sy, sz;
    sx = 0.0f; sy = 0.0f; sz = 10000.0f;

    float c     = 299792458.0f; // speed of light [m/s]
    float P     = 100;          // Power [w]
    float f0    = 9e6;          // center frequency [Hz]
    float B     = 1e6;          // bandwidth [Hz]
    float lam   = c / f0;       // wavelength [m]
    float sigma = 1.0; 
    float Grefl = 80;           // reflector gain [dB]

    float rng_res = 300; // range resolution [m]

    // linear reflector gain
    float Grefl_lin = pow(10, Grefl / 10.0f);

    // subsurface params
    float eps_1 = 1;
    float eps_2 = 3.15;

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


    // --- COMP REFLECTED ENERGY ---

    float nu0 = 376.7;
    float nu1 = nu0 / sqrt(eps_1);
    float nu2 = nu0 / sqrt(eps_2);

    // reflected energy for each facet (default 0)
    cuFloatComplex* d_fRe;
    reportNewAlloc(nfacets * sizeof(cuFloatComplex), "reflected energy");
    cudaMalloc((void**)&d_fRe, nfacets * sizeof(cuFloatComplex));
    cudaMemset(d_fRe, 0, nfacets * sizeof(cuFloatComplex));

    compRefractedEnergy<<<numBlocks, blockSize>>>(d_Itd, d_Ith, d_Iph,
                                                  d_fRe,
                                                  P, Grefl_lin, sigma, fs, lam,
                                                  nu1, nu2, nfacets);
    checkCUDAError("compRefractedEnergy kernel");

    
    // --- CONSTRUCT REFLECTED SIGNAL ---
    // range window params
    float rst =  8000.0f;  // start range [m]
    float ren = 12000.0f;  // end range [m]
    int   nr  =  1000;      // number of range bins
    float dr  = (ren - rst) / (nr - 1); // range step [m]

    // refl signal array
    cuFloatComplex* d_refl_sig;
    cudaMalloc((void**)&d_refl_sig, nr * sizeof(cuFloatComplex));
    cudaMemset(d_refl_sig, 0, nr * sizeof(cuFloatComplex));

    constructReflectedSignal<<<numBlocks, blockSize>>>(d_Itd, d_fRe, d_refl_sig,
                                                       rst, dr, nr,
                                                       rng_res, lam, c, 1, nfacets);
    checkCUDAError("constructReflectedSignal kernel");
    reportTime("Constructing reflection");

    // Wait for the GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // copy Itd to host and export as file
    cuFloatComplex* h_sig = (cuFloatComplex*)malloc(nr * sizeof(cuFloatComplex));
    cudaMemcpy(h_sig, d_refl_sig, nr * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    FILE* fItd = fopen("Ith_cuda.txt", "w");
    //print example values
    for (int i = 0; i < nr; i++) {
        fprintf(fItd, "%f\n", cuCabsf(h_sig[i]));
    }

    return 0;
}