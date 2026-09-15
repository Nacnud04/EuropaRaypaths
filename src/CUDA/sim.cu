/******************************************************************************
 * File:        sim.cu
 * Author:      Duncan Byrne
 * Institution: University of Colorado Boulder
 * Department:  Aerospace Engineering Sciences
 * Email:       duncan.byrne@colorado.edu
 * Date:        2025-11-07
 *
 * Description:
 *    Generates radargram over surface given a faceted surface and param file
 *
 * Compilation:
 *    nvcc sim.cu -O3 -lineinfo -o sim -lcufft
 * 
 * Run:
 *    ./sim [parameter_file.json] [facet_file.fct] [target_file.txt] [output_file_prefix]
 *
 * Notes:
 *    Requires cufft library. 
 *
 ******************************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <cuComplex.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

// custom kernels
#include "file_io.cu"
#include "geometry_kernels.cu"
#include "facet_funcs.cu"
#include "radar_kernels.cu"

// some macros 
#define uChar unsigned char

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
float reportTimeNum(){
    gettimeofday(&t2, 0);
    // report the amount of time in ms
    float dt = (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_usec - t1.tv_usec) * 1e-3;
    return dt;
}

void checkCUDAError(const char* msg)
{
    // Catch launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA LAUNCH ERROR in " << msg
                  << " : " << cudaGetErrorString(err) << std::endl;
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

int main(int argc, const char* argv[])
{

    std::cout << "Using parameter file: " << argv[1] << std::endl;
    std::cout << "Exporting to: " << argv[4] << std::endl;
    checkDirectoryExists(argv[4]);

    // Program-level timer: start
    struct timeval prog_t1, prog_t2;
    gettimeofday(&prog_t1, 0);

    SimulationParameters par;
    checkFileExists(argv[1]);
    par = parseSimulationParameters(argv[1]);


    // --- LOAD SOURCES ---

    bool sourceFileProvided = false;

    float  *h_sx,  *h_sy,  *h_sz;
    float *h_snx, *h_sny, *h_snz;

    if (par.source_path_file != "NONE") {

        sourceFileProvided = true;

        // first open the source file
        const char* source_filename;
        source_filename = par.source_path_file.c_str();
        std::cout << "Using source file: " << source_filename << std::endl;
        checkFileExists(source_filename);
        FILE *sourceFile = fopen(source_filename, "r");
        
        // report number of sources
        par.ns = count_lines(sourceFile);
        std::cout << "Found " << par.ns << " sources in " << source_filename << std::endl;
        
        // allocate memory for source arrays
        // source coordinates
        h_sx = (float*)malloc(par.ns * sizeof(float));
        h_sy = (float*)malloc(par.ns * sizeof(float));
        h_sz = (float*)malloc(par.ns * sizeof(float));

        // source normal
        h_snx = (float*)malloc(par.ns * sizeof(float));
        h_sny = (float*)malloc(par.ns * sizeof(float));
        h_snz = (float*)malloc(par.ns * sizeof(float));

        // load source positions into memory
        loadSourceFile(sourceFile, par.ns, 
                        h_sx, h_sy, h_sz, h_snx, h_sny, h_snz);

    }


    // --- LOAD GAIN PATTERN ---

    bool gainPatternProvided = false;

    float *h_gRefl; float *h_gRefr;

    if (par.gainFile != "NONE") {
    
        gainPatternProvided = true;

        // first open the gain file
        const char* gain_filename;
        gain_filename = par.gainFile.c_str();
        checkFileExists(gain_filename);
        FILE *gainFile = fopen(gain_filename, "r");

        if (count_lines(gainFile) != par.ns) {
            std::cerr << "ERROR: Gain pattern file has " << count_lines(gainFile) << " lines, but number of sources is " << par.ns << std::endl;
        }

        // allocate memory
        h_gRefl = (float*)malloc(par.ns * sizeof(float));
        h_gRefr = (float*)malloc(par.ns * sizeof(float));

        // fill memory
        loadGainFile(gainFile, par.ns, h_gRefl, h_gRefr);

    }


    // --- LOAD TARGETS ---

    // variables for aperture calculation
    float tvc_x, tvc_y, tvc_z;
    float tvc_mag;
    float th_target;

    // first open the target file
    const char* target_filename;
    target_filename = argv[3];
    std::cout << "Using target file: " << argv[3] << std::endl;
    checkFileExists(target_filename);
    FILE *targetFile = fopen(target_filename, "r");
    const int ntargets = count_lines(targetFile) - 1;
    // report number of targets
    std::cout << "Found " << ntargets << " targets in " << target_filename << std::endl;
    
    // allocate memory for target arrays
    // target coordinates
    float* h_tx = (float*)malloc(ntargets * sizeof(float));
    float* h_ty = (float*)malloc(ntargets * sizeof(float));
    float* h_tz = (float*)malloc(ntargets * sizeof(float));
    // target normal
    float* h_tnx = (float*)malloc(ntargets * sizeof(float));
    float* h_tny = (float*)malloc(ntargets * sizeof(float));
    float* h_tnz = (float*)malloc(ntargets * sizeof(float));
    // target type
    uChar* h_ttype = (uChar*)malloc(ntargets * sizeof(uChar));
    memset(h_ttype, 0, ntargets * sizeof(uChar));
    
    // load target positions into memory
    loadTargetFile(targetFile, ntargets, 
                    h_tx, h_ty, h_tz, h_tnx, h_tny, h_tnz, h_ttype);

    // create GPU target vars
    float* d_tx; float* d_ty; float* d_tz;
    bool* d_tInApt;

    // copy target positions over to GPU
    cudaMalloc((void**)&d_tx, ntargets * sizeof(float));
    cudaMemcpy(d_tx, h_tx, ntargets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_ty, ntargets * sizeof(float));
    cudaMemcpy(d_ty, h_ty, ntargets * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_tz, ntargets * sizeof(float));
    cudaMemcpy(d_tz, h_tz, ntargets * sizeof(float), cudaMemcpyHostToDevice);

    // make mask for targets within aperture
    bool* h_tInApt = (bool*)malloc(ntargets * sizeof(bool));
    memset(h_tInApt, 0, ntargets * sizeof(bool));
    cudaMalloc((void**)&d_tInApt, ntargets * sizeof(bool));
    cudaMemcpy(d_tInApt, h_tInApt, ntargets * sizeof(bool), cudaMemcpyHostToDevice);
                    
    // --- LOAD FACETS ---

    // first open facet file
    const char* filename;
    filename = argv[2];
    std::cout << "Using facet file:     " << argv[2] << std::endl;

    // load file and count facets
    checkFileExists(filename);
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
    int nfacets;
    if (!sourceFileProvided) {
        nfacets = nIlluminatedFacets(par.sz, 0, par.fs, par.aperture, par.buff);
    } else {
        // if a source file is provided, use the altitude parameter for estimation
        if (par.altitude != 0.0f) {
            nfacets = nIlluminatedFacets(par.altitude, 0, par.fs, par.aperture, par.buff);
        } else {
            std::cout << "WARNING: No altitude parameter provided; using total facet count" << std::endl;
            nfacets = totFacets;
        }
    }

    // you cannot have more facets illuminated than exist
    if (nfacets > totFacets) {
        nfacets = totFacets;
    }

    std::cout << "Estimated that " << nfacets << " facets are illuminated by par.aperture" << std::endl;

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

    // inclination from source array
    float *d_FSth;
    cudaMalloc((void**)&d_FSth, totFacets * sizeof(float));
    cudaMemsetAsync(d_FSth, 0, totFacets * sizeof(float));

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
    cudaMemsetAsync(d_Iph, 0, nfacets * sizeof(float));
    cudaMemsetAsync(d_Ith, 0, nfacets * sizeof(float));

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

    // forced ray to target
    float *d_Ttd, *d_Tph, *d_Tth;
    cudaMalloc((void**)&d_Ttd, nfacets * sizeof(float));
    cudaMalloc((void**)&d_Tph, nfacets * sizeof(float));
    cudaMalloc((void**)&d_Tth, nfacets * sizeof(float));

    // inclination relative to target normal
    float *d_TargetTh;
    cudaMalloc((void**)&d_TargetTh, nfacets * sizeof(float));

    // inwards refracted weights
    float* d_fRefrEI;
    cudaMalloc((void**)&d_fRefrEI, nfacets * sizeof(float));
    cudaMemsetAsync(d_fRefrEI, 0, nfacets * sizeof(float));

    // upward transmitted
    float* d_fRefrEO;
    cudaMalloc((void**)&d_fRefrEO, nfacets * sizeof(float));
    cudaMemsetAsync(d_fRefrEO, 0, nfacets * sizeof(float));

    cuFloatComplex* d_refr_phasor;
    cuFloatComplex* d_refl_phasor;
    cuFloatComplex* d_phasorTrace;
    short* d_refr_rbs; short* d_refl_rbs;
    float* d_refl_chirp; float* d_refr_chirp;

    // reflected and refracted phasors & range bins
    cudaMalloc((void**)&d_refr_phasor, 2 * nfacets * sizeof(cuFloatComplex));
    cudaMemsetAsync(d_refr_phasor, 0, 2 * nfacets * sizeof(cuFloatComplex));
    
    cudaMalloc((void**)&d_refr_rbs, 2 * nfacets * sizeof(short));
    cudaMemsetAsync(d_refr_rbs, 0, 2 * nfacets * sizeof(short));
    
    cudaMalloc((void**)&d_refl_phasor, 2 * nfacets * sizeof(cuFloatComplex));
    cudaMemsetAsync(d_refl_phasor, 0, 2 * nfacets * sizeof(cuFloatComplex));

    cudaMalloc((void**)&d_refl_rbs, 2 * nfacets * sizeof(short));
    cudaMemsetAsync(d_refl_rbs, 0, 2 * nfacets * sizeof(short));

    // phasor trace
    cudaMalloc((void**)&d_phasorTrace, par.nr * sizeof(cuFloatComplex));
    cudaMemsetAsync(d_phasorTrace, 0, par.nr * sizeof(cuFloatComplex));

    // create streams for target parallelization
    const int Nstreams = 8;
    cudaStream_t streams[Nstreams];

    cuFloatComplex* d_PTtarg[Nstreams];
    cuFloatComplex* d_Ptarg[Nstreams];
    cuFloatComplex* d_Psour[Nstreams];
    cuFloatComplex* d_refr_temp[Nstreams];

    cuFloatComplex* d_signalPad[Nstreams];
    cuFloatComplex* d_kernelPad[Nstreams];

    cufftHandle plans[Nstreams];

    int nrPad = 2 * par.nr - 1;

    for (int s = 0; s < Nstreams; s++) {
        cudaStreamCreate(&streams[s]);

        cudaMalloc(&d_PTtarg[s],    par.nr * sizeof(cuFloatComplex));
        cudaMalloc(&d_Ptarg[s],      par.nr * sizeof(cuFloatComplex));
        cudaMalloc(&d_Psour[s],      par.nr * sizeof(cuFloatComplex));
        cudaMalloc(&d_refr_temp[s],  par.nr * sizeof(cuFloatComplex));

        cudaMalloc(&d_signalPad[s], nrPad * sizeof(cuFloatComplex));
        cudaMalloc(&d_kernelPad[s], nrPad * sizeof(cuFloatComplex));

        cufftPlan1d(&plans[s], nrPad, CUFFT_C2C, 1);
        cufftSetStream(plans[s], streams[s]);
    }



    //cudaMemsetAsync(d_refr_temp, 0, par.nr * sizeof(cuFloatComplex));

    // tmp for phasor trace
    cuFloatComplex* d_PTTmp;
    cudaMalloc((void**)&d_PTTmp, par.nr * sizeof(cuFloatComplex));

    // tmp for surf phasor trace
    cuFloatComplex* d_PSurf;
    cudaMalloc((void**)&d_PSurf, par.nr * sizeof(cuFloatComplex));

    // refracted signal
    cuFloatComplex* d_refr_sig;
    cudaMalloc((void**)&d_refr_sig, par.nr * sizeof(cuFloatComplex));
    cudaMemsetAsync(d_refr_sig, 0, par.nr * sizeof(cuFloatComplex));

    // joint signal
    cuFloatComplex* d_sig;
    cudaMalloc((void**)&d_sig, par.nr * sizeof(cuFloatComplex));
    cudaMemsetAsync(d_sig, 0, par.nr * sizeof(cuFloatComplex));

    // --- ATTENUATION GEOMETRY SETUP ---

    int nAttenPrisms = 0;
    bool attenuationGeometry = false;
    if (par.atten_geom_path != "NONE") {
        attenuationGeometry = true;
    }
    
    // load file
    FILE *attenPrismFile;
    if (attenuationGeometry) {

        checkFileExists(par.atten_geom_path.c_str());
        attenPrismFile = fopen(par.atten_geom_path.c_str(), "r");
        nAttenPrisms = count_lines(attenPrismFile);

        // report number of attenuation prisms
        std::cout << "Found " << nAttenPrisms << " attenuation prisms in " << par.atten_geom_path << std::endl;
    }

    // set up arrays on CPU
    float* alphas    = (float*)malloc(nAttenPrisms * sizeof(float));
    float* h_attXmin = (float*)malloc(nAttenPrisms * sizeof(float));
    float* h_attXmax = (float*)malloc(nAttenPrisms * sizeof(float));
    float* h_attYmin = (float*)malloc(nAttenPrisms * sizeof(float));
    float* h_attYmax = (float*)malloc(nAttenPrisms * sizeof(float));
    float* h_attZmin = (float*)malloc(nAttenPrisms * sizeof(float));
    float* h_attZmax = (float*)malloc(nAttenPrisms * sizeof(float));

    // load attenuation prism file
    if (attenuationGeometry) {
        loadAttenPrismFile(attenPrismFile, nAttenPrisms,
                        alphas,
                        h_attXmin, h_attXmax,
                        h_attYmin, h_attYmax,
                        h_attZmin, h_attZmax, par);
    }

    // move to GPU
    float *d_attXmin, *d_attXmax;
    float *d_attYmin, *d_attYmax;
    float *d_attZmin, *d_attZmax; 
    float *d_alphas;
    cudaMalloc((void**)&d_attXmin, nAttenPrisms * sizeof(float));
    cudaMalloc((void**)&d_attXmax, nAttenPrisms * sizeof(float));
    cudaMalloc((void**)&d_attYmin, nAttenPrisms * sizeof(float));
    cudaMalloc((void**)&d_attYmax, nAttenPrisms * sizeof(float));
    cudaMalloc((void**)&d_attZmin, nAttenPrisms * sizeof(float));
    cudaMalloc((void**)&d_attZmax, nAttenPrisms * sizeof(float));
    cudaMalloc((void**)&d_alphas, nAttenPrisms * sizeof(float));
    cudaMemcpy(d_attXmin, h_attXmin, nAttenPrisms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attXmax, h_attXmax, nAttenPrisms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attYmin, h_attYmin, nAttenPrisms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attYmax, h_attYmax, nAttenPrisms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attZmin, h_attZmin, nAttenPrisms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attZmax, h_attZmax, nAttenPrisms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alphas, alphas, nAttenPrisms * sizeof(float), cudaMemcpyHostToDevice);

    // print to make sure stuff actually moved onto gpu
    if (nAttenPrisms > 0) {
        std::cout << "Attenuation geometry prisms on GPU:" << std::endl;
    }
    for (int i=0; i<nAttenPrisms; i++) {
        std::cout << " Prism " << i << ": "
                  << "Xmin=" << h_attXmin[i] << ", Xmax=" << h_attXmax[i] << ", "
                  << "Ymin=" << h_attYmin[i] << ", Ymax=" << h_attYmax[i] << ", "
                  << "Zmin=" << h_attZmin[i] << ", Zmax=" << h_attZmax[i] << std::endl;
    }

    float snx, sny, snz;
    float  sx,  sy,  sz;
    // use a fixed source normal if no source file is provided
    if (!sourceFileProvided) {
        snx = 0; sny = 0; snz = 1 * par.source_normal_multiplier;
        std::cout << "Using fixed source normal: (" << snx << ", " << sny << ", " << snz << ")" << std::endl;
        sy = par.sy;
        sz = par.sz;
    }

    
    // load in rx window position file if it exists
    bool rxWindowPositionFileProvided = false;
    float* rx_window_pos = nullptr;
    
    // if the file exists, load it
    if (par.rxWindowPositionFile != "NONE") {

        rxWindowPositionFileProvided = true;

        // allocate memory
        rx_window_pos = (float*)malloc(par.ns * sizeof(float));

        // load file
        const char* rxWindowPosFilename;
        rxWindowPosFilename = par.rxWindowPositionFile.c_str();
        std::cout << "Using rx window position file: " << rxWindowPosFilename << std::endl;
        checkFileExists(rxWindowPosFilename);
        FILE *sourceFile = fopen(rxWindowPosFilename, "r");

        // read in data to memory
        loadRxWindowPositions(sourceFile, par.ns, rx_window_pos);

    }
        
    float runtime = 0; // run time in seconds
    int blockSize = 256;

    cuFloatComplex* d_chirpComplex;
    cudaMalloc(&d_chirpComplex, sizeof(cuFloatComplex) * par.nr);

    // --- GENERATE CHIRP ---
    // We need a chirp for both the surface and the subsurface. As the surface requires a full 
    // bandwidth chirp, where as the subsurface requires a half bandwidth chirp.

    int paddedNr = 2 * par.nr;
    cudaMalloc((void**)&d_refl_chirp, paddedNr * sizeof(float));
    cudaMemsetAsync(d_refl_chirp, 0, paddedNr * sizeof(float));

    genCenteredChirpPadded<<<(paddedNr + blockSize - 1) / blockSize, blockSize>>>(d_refl_chirp, par.dr, par.nr, paddedNr, par.rng_res);
    checkCUDAError("genCenteredChirpPadded kernel");

    cudaMalloc((void**)&d_refr_chirp, paddedNr * sizeof(float));
    cudaMemsetAsync(d_refr_chirp, 0, paddedNr * sizeof(float));

    genCenteredChirpPadded<<<(paddedNr + blockSize - 1) / blockSize, blockSize>>>(d_refr_chirp, par.dr, par.nr, paddedNr, 2 * par.rng_res);
    checkCUDAError("genCenteredChirpPadded kernel");


    // Wipe write direcory
    // this happens at the last minute as in case we cancel a process before
    // the main simulation begins not all data is lost
    remove_s_txt_files(argv[4]);
    std::cout << "Number of range bins: " << par.nr << std::endl;

    // --- MAIN LOOP OVER SOURCES ---

    for (int is=0; is<par.ns; is++) {

       // first clear phasor buffers
        cudaMemsetAsync(d_PSurf, 0, par.nr * sizeof(cuFloatComplex));

        // update par.rst
        if (rxWindowPositionFileProvided) {
            par.rst = rx_window_pos[is];
        }

        // update gains
        if (gainPatternProvided) {
            par.Grefl_lin = powf(10.0f, h_gRefl[is]/10.0f);
            par.Grefr_lin = powf(10.0f, h_gRefr[is]/10.0f);
        }

        // if the source file is not provided, use linear solution
        if (!sourceFileProvided) {
            sx = par.sx0 + is * par.sdx;
        } else {
            sx  = h_sx[is];
            sy  = h_sy[is];
            sz  = h_sz[is];
            snx = h_snx[is] * par.source_normal_multiplier;
            sny = h_sny[is] * par.source_normal_multiplier;
            snz = h_snz[is] * par.source_normal_multiplier;
        }

        startTimer();

        // --- CLEAR PER-SOURCE ACCUMULATION BUFFERS ---
        // refl_sig and refr_sig are accumulated into with atomicAdd inside
        // the kernels. They must be cleared before each source so traces do
        // not accumulate across iterations.
        cudaMemset(d_refl_sig, 0, par.nr * sizeof(cuFloatComplex));
        cudaMemset(d_refr_sig, 0, par.nr * sizeof(cuFloatComplex));
        // d_sig is fully overwritten by combineRadarSignals, so clearing it is
        // optional; this is left out for slightly better performance.

        
        // --- GET INCLINATION OF EACH RAY RELATIVE TO SOURCE ---
        // compute incident rays for the full facet set (totFacets)
        int numBlocksAll = (totFacets + blockSize - 1) / blockSize;
        compSourceInclination<<<numBlocksAll, blockSize>>>(sx, sy, sz,
                            d_Ffx, d_Ffy, d_Ffz,
                            snx, sny, snz,
                            d_FItd, d_FSth, totFacets);
        checkCUDAError("compSourceInclination kernel");

        // --- CROP TO ILLUMINATED ---
        // crop full facet arrays into the per-source cropped arrays and get
        // the number of valid (illuminated) facets returned by the function
        int valid_facets = cropByAperture(totFacets, nfacets, par.aperture, d_FSth,
                                        d_Ffx,  d_Ffy,  d_Ffz,
                                        d_Ffnx, d_Ffny, d_Ffnz,
                                        d_Ffux, d_Ffuy, d_Ffuz,
                                        d_Ffvx, d_Ffvy, d_Ffvz,
                                        d_FItd,
                                        d_fx,   d_fy,   d_fz,
                                        d_fnx,  d_fny,  d_fnz,
                                        d_fux,  d_fuy,  d_fuz,
                                        d_fvx,  d_fvy,  d_fvz,
                                        d_Itd);
        checkCUDAError("cropByAperture process");
        cudaDeviceSynchronize();
        
        // If no facets are illuminated, skip per-facet kernels to avoid
        // operating on uninitialised memory which can produce NaNs.
        if (valid_facets == 0) {
            printf("\rNo illuminated facets for source %d, skipping.", is);
	    continue;
        }

        if (valid_facets > nfacets) {
            std::cout << "ERROR: valid_facets > nfacets. CHECK FACET SIZE OR ESTIMATE BUFFER!!" << std::endl;
        }

        int numBlocks = (nfacets + blockSize - 1) / blockSize;

        // --- COMP INCIDENT RAYS ---
        compIncidentRays<<<numBlocks, blockSize>>>(sx, sy, sz,
                            d_fx,  d_fy,   d_fz,
                            d_fnx, d_fny, d_fnz,
                            d_fux, d_fuy, d_fuz,
                            d_fvx, d_fvy, d_fvz,
                            d_Itd, d_Iph, d_Ith,
                            valid_facets);
        checkCUDAError("compIncidentRays kernel");
        cudaDeviceSynchronize();
        

        // --- COMP REFRACTED RAYS ---
                
        compRefractedRays<<<numBlocks, blockSize>>>(d_Ith, d_Iph, 
                                                    d_Rth, 
                                                    d_fx, d_fy, d_fz,
                                                    par.eps_1, par.eps_2, valid_facets);
        checkCUDAError("compRefractedRays kernel");

        
        // --- COMP REFLECTED ENERGY ---

        compReflectedEnergy<<<numBlocks, blockSize>>>(d_Itd, d_Ith, d_Iph,
                                                    d_fReflE, d_Rth, d_fRfrC,
                                                    par, valid_facets);
        checkCUDAError("compReflectedEnergy kernel");

        surfacePT<<<numBlocks, blockSize>>>(d_refl_phasor, d_refl_rbs, d_Ith, d_Iph, d_Itd,
                                            d_fReflE, par, valid_facets);
        cudaDeviceSynchronize();
        checkCUDAError("surfacePT kernel");
        genPhasorTrace(d_PSurf, d_refl_rbs, d_refl_phasor, 2 * valid_facets, par.nr);
        checkCUDAError("genPhasorTrace Reflected process");

        // if debug, write out surface phasor
        if (par.debug_surface) {
            debugSaveSignal(argv[4], "Psurf", is, 0, d_PSurf, par.nr, 0);
        }

        if (par.disable_surface) {
            cudaMemsetAsync(d_PSurf, 0, par.nr * sizeof(cuFloatComplex));
        }
        checkCUDAError("reflected signal squareComplex Kernel");

        // convolve with range compressed chirp
        convolvePhasorChirpLinear(d_PSurf, d_refl_chirp, d_refl_sig, par.nr, par, argv, is, 0);
        checkCUDAError("convolvePhasorChirpLinear Reflected process");

        // limit targets to just within aperture
        int tarBlocks = (ntargets + blockSize - 1) / blockSize;
        findTargetsInAperture<<<tarBlocks, blockSize>>>(d_tx, d_ty, d_tz,
                                                        sx, sy, sz,
                                                        snx, sny, snz,
                                                        d_tInApt, par.aperture, ntargets);
        checkCUDAError("findTargetsInAperture kernel");

        // copy target mask to host
        cudaMemcpy(h_tInApt, d_tInApt, ntargets * sizeof(bool), cudaMemcpyDeviceToHost);
        /*
        for (int it=0; it<ntargets; it++) {

            // check if target is within aperture
            if (!h_tInApt[it]) {
                continue;
            }

            cudaMemsetAsync(d_PTtarg, 0, par.nr * sizeof(cuFloatComplex));
            cudaMemsetAsync(d_Ptarg, 0, par.nr * sizeof(cuFloatComplex));
            cudaMemsetAsync(d_Psour, 0, par.nr * sizeof(cuFloatComplex));
            cudaMemsetAsync(d_PTTmp, 0, par.nr * sizeof(cuFloatComplex));

            // --- FORCED RAY TO TARGET COMP ---
            // this is also when we compute the attenuation
            compTargetRays<<<numBlocks, blockSize>>>(h_tx[it], h_ty[it], h_tz[it],
                                                    h_tnx[it], h_tny[it], h_tnz[it],
                                                    d_fx,  d_fy,  d_fz,
                                                    d_fnx, d_fny, d_fnz,
                                                    d_fux, d_fuy, d_fuz,
                                                    d_fvx, d_fvy, d_fvz,
                                                    d_Ttd, d_Tph, d_Tth,
                                                    d_TargetTh,
                                                    valid_facets, 
                                                    d_attXmin, d_attXmax,
                                                    d_attYmin, d_attYmax,
                                                    d_attZmin, d_attZmax,
                                                    d_alphas, par.alpha2, nAttenPrisms,
                                                    d_fRefrEI, d_fRefrEO);
            checkCUDAError("compTargetRays kernel");

            // --- CONSTRUCT REFRACTED WEIGHTS INWARDS ---
            compRefrEnergyIn<<<numBlocks, blockSize>>>(d_Rth, d_Itd, d_Iph,
                                                    d_Ttd, d_Tth, d_Tph, d_fRfrC,
                                                    d_fRefrEI, d_fRfrSR,
                                                    par, valid_facets);
            checkCUDAError("compRefrEnergyIn kernel");
            
            // --- CALCULATE POWER AT TARGET ---
            // this is the inward phasor trace
            accumulateTarget<<<numBlocks, blockSize>>>(d_refr_phasor, d_refr_rbs, 
                                                        d_Ith, d_Iph, d_Itd,
                                                        d_Tth, d_Tph, d_Ttd, d_Rth,
                                                        d_TargetTh, d_fRefrEI, d_fRfrSR,
                                                        d_fx, d_fy, d_fz,
                                                        par, valid_facets);
            cudaDeviceSynchronize();
            checkCUDAError("accumulateTarget kernel");
            genPhasorTrace(d_PTtarg, d_refr_rbs, d_refr_phasor, 2 * valid_facets, par.nr);
            checkCUDAError("genPhasorTrace Refracted process 1");
            convolvePhasorChirpLinear(d_PTtarg, d_refr_chirp, d_Ptarg, par.nr, par, argv, is, it);

            // --- COMPUTE UPWARD TRANSMITTED RAYS ---
            compRefrEnergyOut<<<numBlocks, blockSize>>>(d_Itd, d_Iph,
                                                        d_Ttd, d_Tth, d_Tph, 
                                                        d_fRefrEO, d_fRfrC, 
                                                        par, valid_facets);
            checkCUDAError("compRefrEnergyOut kernel");
            
            // --- CALCULATE OUTWARD PHASOR TRACE ---
            radiateTarget<<<numBlocks, blockSize>>>(d_refr_phasor, d_refr_rbs,
                                                    d_Ith, d_Iph, d_Itd,
                                                    d_Tth, d_Tph, d_Ttd, d_Rth,
                                                    d_TargetTh, d_fRefrEO, d_fRfrSR,
                                                    par, valid_facets);
            cudaDeviceSynchronize();
            checkCUDAError("radiateTarget kernel");
            genPhasorTrace(d_Psour, d_refr_rbs, d_refr_phasor, 2 * valid_facets, par.nr);
            checkCUDAError("genPhasorTrace Refracted process 2");

            
            // --- CONVOLVE INTO FULL PHASOR TRACE ---
            //convolveComplex(d_Psour, d_Ptarg, d_PTTmp, par.nr);
            convolveComplex(d_Psour, d_Ptarg, d_refr_temp, par, argv, is, it);
            checkCUDAError("convolveComplexSquare kernel");
            cudaDeviceSynchronize();


            // accumulate this target's contribution into the running sum
            addComplexArrays<<<(par.nr + blockSize - 1) / blockSize, blockSize>>>(d_refr_sig, d_refr_temp, par.nr);
            checkCUDAError("addComplexArrays accumulate refracted target");

            if (par.debug_surface) {
                debugSaveSignal(argv[4], "Ptarg", is, it, d_Ptarg, par.nr, 1);
                debugSaveSignal(argv[4], "Psour", is, it, d_Psour, par.nr, 1);
                debugSaveSignal(argv[4], "PTTmp", is, it, d_refr_temp, par.nr, 1);
            }
            
        }*/

        for (int it = 0; it < ntargets; it++) {
            if (!h_tInApt[it]) continue;

            int s = it % Nstreams;
            cudaStream_t st = streams[s];

            cudaMemsetAsync(d_PTtarg[s],   0, par.nr * sizeof(cuFloatComplex), st);
            cudaMemsetAsync(d_Ptarg[s],    0, par.nr * sizeof(cuFloatComplex), st);
            cudaMemsetAsync(d_Psour[s],    0, par.nr * sizeof(cuFloatComplex), st);
            cudaMemsetAsync(d_refr_temp[s],0, par.nr * sizeof(cuFloatComplex), st);

            compTargetRays<<<numBlocks, blockSize, 0, st>>>(h_tx[it], h_ty[it], h_tz[it],
                                                            h_tnx[it], h_tny[it], h_tnz[it],
                                                            d_fx, d_fy, d_fz,
                                                            d_fnx, d_fny, d_fnz,
                                                            d_fux, d_fuy, d_fuz,
                                                            d_fvx, d_fvy, d_fvz,
                                                            d_Ttd, d_Tph, d_Tth,
                                                            d_TargetTh,
                                                            valid_facets,
                                                            d_attXmin, d_attXmax,
                                                            d_attYmin, d_attYmax,
                                                            d_attZmin, d_attZmax,
                                                            d_alphas, par.alpha2, nAttenPrisms,
                                                            d_fRefrEI, d_fRefrEO);

            compRefrEnergyIn<<<numBlocks, blockSize, 0, st>>>(d_Rth, d_Itd, d_Iph,
                                                            d_Ttd, d_Tth, d_Tph, d_fRfrC,
                                                            d_fRefrEI, d_fRfrSR,
                                                            par, valid_facets);

            accumulateTarget<<<numBlocks, blockSize, 0, st>>>(d_refr_phasor, d_refr_rbs,
                                                            d_Ith, d_Iph, d_Itd,
                                                            d_Tth, d_Tph, d_Ttd, d_Rth,
                                                            d_TargetTh, d_fRefrEI, d_fRfrSR,
                                                            d_fx, d_fy, d_fz,
                                                            par, valid_facets);

            genPhasorTraceAsync(d_PTtarg[s], d_refr_rbs, d_refr_phasor,
                                2 * valid_facets, par.nr, st);

            convolvePhasorChirpLinearAsync(d_PTtarg[s], d_refr_chirp, d_Ptarg[s],
                                        par.nr, par, argv, is, it, st);

            compRefrEnergyOut<<<numBlocks, blockSize, 0, st>>>(d_Itd, d_Iph,
                                                            d_Ttd, d_Tth, d_Tph,
                                                            d_fRefrEO, d_fRfrC,
                                                            par, valid_facets);

            radiateTarget<<<numBlocks, blockSize, 0, st>>>(d_refr_phasor, d_refr_rbs,
                                                        d_Ith, d_Iph, d_Itd,
                                                        d_Tth, d_Tph, d_Ttd, d_Rth,
                                                        d_TargetTh, d_fRefrEO, d_fRfrSR,
                                                        par, valid_facets);

            genPhasorTraceAsync(d_Psour[s], d_refr_rbs, d_refr_phasor,
                                2 * valid_facets, par.nr, st);

            convolveComplexAsync(d_Psour[s], d_Ptarg[s], d_refr_temp[s],
                                d_signalPad[s], d_kernelPad[s],
                                plans[s], par, st);

            addComplexArrays<<<(par.nr + blockSize - 1) / blockSize, blockSize, 0, st>>>(
                d_refr_sig, d_refr_temp[s], par.nr);
        }

        for (int s = 0; s < Nstreams; s++) {
            cudaStreamSynchronize(streams[s]);
        }


        
        // --- COMBINE INTO OUTPUT SIGNAL AND EXPORT ---
        combineRadarSignals<<<(par.nr + blockSize - 1) / blockSize, blockSize>>>(d_refl_sig, d_refr_sig, d_sig, par.nr);
        checkCUDAError("combineRadarSignals kernel");

        // Wait for the GPU to finish
        cudaDeviceSynchronize();

        char* filename = (char*)malloc(64 * sizeof(char));
        sprintf(filename, "%s/s%06d.txt", argv[4], is);
        saveSignalToFile(filename, d_sig, par.nr);
        free(filename);

        // overwrite progress printed to terminal
        runtime += reportTimeNum();
        float time_remain = (((par.ns-is))*((runtime*1e-3)/is));
        int min_remain = time_remain / 60;
        float sec_remain = time_remain - min_remain * 60;
        printf("\rCompleted source %d of %d in %5.1f ms. Remaining: %d min %.1f sec         ", is+1, par.ns, reportTimeNum(), min_remain, sec_remain);
        fflush(stdout);

    }

    // Program-level timer: end and report summary
    gettimeofday(&prog_t2, 0);
    float total_ms = (prog_t2.tv_sec - prog_t1.tv_sec) * 1e3f + (prog_t2.tv_usec - prog_t1.tv_usec) * 1e-3f;
    float total_s = total_ms / 1e3f;
    float total_min = total_s / 60.0f;
    std::cout << std::endl;
    std::cout << "=== Run summary ===" << std::endl;
    std::cout << "Total runtime: " << (int)total_min << " min " << (int)(total_s - (int)total_min * 60) << " seconds " << (int)(total_ms - ((int)total_s)*1e3f) << " milliseconds" << std::endl;
    if (par.ns > 0) {
        float per_source_ms = total_ms / (float)par.ns;
        float per_source_s = per_source_ms / 1e3f;
        std::cout << "Time per source (trace): " << per_source_ms << " ms (" << per_source_s << " s)" << std::endl;
        float traces_per_sec = ((float)par.ns) / total_s;
        std::cout << "Throughput: " << traces_per_sec << " traces/sec" << std::endl;
    } else {
        std::cout << "No sources (par.ns == 0), cannot compute per-source statistics." << std::endl;
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
    cudaFree(d_fRefrEI);
    cudaFree(d_fRefrEO);
    cudaFree(d_refr_sig);
    cudaFree(d_sig);
    cudaFree(d_refr_temp);

    return 0;
}
