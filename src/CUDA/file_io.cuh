#pragma once

#include <string>
#include <cuComplex.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sys/stat.h>
#include <stdexcept>
#include <string>
#include <dirent.h>
#include <unistd.h>

#define uChar unsigned char

// roughness models
enum RoughnessModel {
    SPECULAR = 0,
    COHERENT_SPECULAR = 1,
    SIMPLE_BACKSCATTER = 2
};


struct SimulationParameters {

    float c = 299792458.0f; // speed of light in m/s
    float nu0 = 376.7f;
    float eps_0 = 8.85e-12;
    float pi = 3.14159265f;

    // source geometry
    std::string source_path_file; // source path file
    float source_normal_multiplier = 1.0f; // multiplier to source normal vector
    float altitude;
    
    // there can also be evenly spaced sources over X 
    float sy, sz;   // Source position
    float sx0, sdx;
    int ns;

    // source params
    float f0;           // Radar centerfrequency
    float B;            // Bandwidth
    float P;            // Transmitted power
    float Grefr;        // Subsurface gain
    float Grefl;        // Surface gain
    float rng_res;      // Range resolution
    float aperture;     // Source aperture
    std::string gainFile; // file containing gain pattern (if exists)

    // computed params
    float lam;
    float k;
    int pol = 0;
    float Grefr_lin;    // Subsurface gain (linear)
    float Grefl_lin;    // Surface gain (linear)
    
    // surface parameters
    float rms_h;        // surface rms height
    float ks;
    float buff;         // buffer for estimated facet count

    // material parameters
    float eps_1;        // Permittivity of medium 1 
    float eps_2;        // Permittivity of medium 2 
    float sig_1;        // Conductivity in medium 1 
    float sig_2;        // Conductivity in medium 2

    // computed params
    float c_1; // speed in medium 1
    float c_2; // speed in medium 2
    float nu_1; // impedance in medium 1
    float nu_2; // impedance in medium
    float alpha1;
    float alpha2;

    // target parameters
    int rerad_funct;
    float target_size;  // Size of facet-like targets
    float tx, ty, tz;   // Target position

    // range window params
    float rst, dr;
    int nr;
    float smpl;         // Sampling rate (Hz)
    std::string rxWindowPositionFile;

    // facet parameters
    float fs;

    // roughness model
    RoughnessModel roughness_model;

    // subsurface attenuation geometry file
    std::string atten_geom_path;

    // processing parameters
    bool convolution;
    bool convolution_linear;
    bool specular;
    bool lossless;
    bool incoherent = false;
    bool disable_surface = false;
    bool fresnel = false;

    // debug parameters
    bool debug_surface = false;

};



// FUNCTIONS

__host__ SimulationParameters parseSimulationParameters(const std::string& filename);

__host__ void checkFileExists(const char* filename);

__host__ void checkDirectoryExists(const char* path);

__host__ void saveSignalToFile(const char* filename, cuFloatComplex* d_sig, int nr);

__host__ void saveFloatsToFile(const char* filename, float* d_sig, int nr);

int count_lines(FILE* file, int counter_start = 1);

__host__ void loadFacetFile(FILE * file, const int totfacets,
                            float* h_fx, float* h_fy, float* h_fz,
                            float* h_fnx, float* h_fny, float* h_fnz,
                            float* h_fux, float* h_fuy, float* h_fuz,
                            float* h_fvx, float* h_fvy, float* h_fvz);

__host__ void loadTargetFile(FILE* file, const int nt,
                            float* h_tx, float* h_ty, float* h_tz,
                            float* h_tnx, float* h_tny, float* h_tnz,
                            uChar* h_ttype);

__host__ void loadSourceFile(FILE* file, const int ns,
                             float* h_sx, float* h_sy, float* h_sz,
                             float* h_snx, float* h_sny, float* h_snz);

__host__ void loadAttenPrismFile(FILE* file, const int nPrisms,
                                 float* h_alphas, 
                                 float* h_attXmin, float* h_attXmax,
                                 float* h_attYmin, float* h_attYmax,
                                 float* h_attZmin, float* h_attZmax,
                                 SimulationParameters par);

__host__ void loadRxWindowPositions(FILE* file,
                                    const int ns,
                                    float* h_rx_window_positions);

__host__ void remove_s_txt_files(const std::string& dir);

__host__ void loadGainFile(FILE* file, const int ns,
                           float* h_gRefl, float* h_gRefr);

void debugSaveSignal(const char* outdir,
                     const char* name,
                     int is, int it,
                     cuFloatComplex* data,
                     int nr,
                     int include_t);