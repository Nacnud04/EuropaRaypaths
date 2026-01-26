/******************************************************************************
 * File:        file_io.cu
 * Author:      Duncan Byrne
 * Institution: Univeristy of Colorado Boulder
 * Department:  Aerospace Engineering Sciences
 * Email:       duncan.byrne@colorado.edu
 * Date:        2025-11-07
 *
 * Description:
 *    File providing functions to read simulation parameters from JSON and load facets
 *
 * Contents:
 *    - SimulationParameters struct: Holds all simulation parameters
 *    - parseSimulationParameters: Reads parameters from a JSON file
 *    - saveSignalToFile: saves a complex signal from device to a text file
 *    - loadFacetFile: loads facet data from a text file into host arrays
 *    - checkFileExists: checks if a file exists. if not throws an exception
 *    - checkDirectoryExists: checks if a directory exists and isnt a file
 *
 * Usage:
 *    #include "file_io.cu"
 * 
 * Notes:
 *
 ******************************************************************************/

#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sys/stat.h>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

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
    float tx, ty, tz;   // Target position

    // range window params
    float rst, dr;
    int nr;
    float smpl;         // Sampling rate (Hz)
    std::string rxWindowPositionFile;

    // facet parameters
    float fs;

    // subsurface attenuation geometry file
    std::string atten_geom_path;

    // processing parameters
    bool convolution;
    bool convolution_linear;

};

__host__ SimulationParameters parseSimulationParameters(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open JSON file: " + filename);
    }

    json j;
    file >> j;

    SimulationParameters params;

    // load source geometry
    params.source_path_file = j.value("source_path_file", std::string("NONE"));

    if (params.source_path_file == "NONE") {
        std::cout << "No source geometry file found, using linear source spacing along x." << std::endl;
        params.sy = j["sy"];
        params.sz = j["sz"];
        params.sx0 = j["sx0"];
        params.sdx = j["sdx"];
        params.ns  = j["ns"];
    }

    // multiplier to source normal vector (if exists)
    params.source_normal_multiplier = j.value("source_normal_multiplier", 1.0f);

    // source altitude (used for facet memory allocation)
    params.altitude = j.value("altitude", 0.0f);

    // source function params
    params.f0 = j["frequency"];
    params.B = j["bandwidth"];
    params.P = j["power"];
    params.Grefr = j["subsurface_gain"];
    params.Grefl = j["surface_gain"];
    //params.rng_res = j["range_resolution"];
    params.aperture = j["aperture"];

    // compute the range resolution from bandwidth
    params.rng_res = params.c / params.B;

    params.lam = params.c / params.f0;
    params.k = 2 * params.pi / params.lam;
    params.Grefr_lin = pow(10, params.Grefr / 10.f);
    params.Grefl_lin = pow(10, params.Grefl / 10.f);

    // print frequency and wavelength
    std::cout << "Frequency: " << params.f0 / 1e6 << " MHz, Wavelength: " << params.lam << " m" << std::endl;

    params.rms_h = j["rms_height"];
    params.ks = params.k * params.rms_h;
    params.buff = j["buff"];

    params.eps_1 = j["eps_1"];
    params.eps_2 = j["eps_2"];
    params.sig_1 = j["sig_1"];
    params.sig_2 = j["sig_2"];

    params.c_1 = params.c / sqrt(params.eps_1);
    params.c_2 = params.c / sqrt(params.eps_2);
    params.nu_1 = params.nu0 / sqrt(params.eps_1);
    params.nu_2 = params.nu0 / sqrt(params.eps_2);

    //print speeds
    std::cout << "Speed in medium 1: " << params.c_1 << " m/s" << std::endl;
    std::cout << "Speed in medium 2: " << params.c_2 << " m/s" << std::endl;

    float eps_pp = params.sig_1 / (2 * 3.14159 * params.f0 * params.eps_0);
    float alpha   = sqrt(1 + (eps_pp/params.eps_1)*(eps_pp/params.eps_1)) - 1;
          alpha   = sqrt(0.5 * params.eps_1 * alpha);
          alpha   = (alpha * 2 * 3.14159) / params.lam;
    params.alpha1  = alpha;

    eps_pp = params.sig_2 / (2 * 3.14159 * params.f0 * params.eps_0);
    alpha   = sqrt(1 + (eps_pp/params.eps_2)*(eps_pp/params.eps_2)) - 1;
    alpha   = sqrt(0.5 * params.eps_2 * alpha);
    alpha   = (alpha * 2 * 3.14159) / params.lam;
    params.alpha2  = alpha;

    params.smpl = j["rx_sample_rate"];

    params.rxWindowPositionFile = j.value("rx_window_position_file", std::string("NONE"));

    if (params.rxWindowPositionFile == "NONE") {
        std::cout << "No Rx window position file found, using static offset." << std::endl;
        params.rst = float(j["rx_window_offset_m"]);
    }

    params.nr  = (float(j["rx_window_m"]) / 299792480.0f) * params.smpl;
    params.dr  = float(j["rx_window_m"]) / float(params.nr);

    // report sampling window
    std::cout << "Sampling window: " << params.rst << " to " << float(j["rx_window_offset_m"]) + float(j["rx_window_m"]) << " m, ";
    std::cout << params.nr << " samples at " << params.smpl << " Hz" << std::endl;

    // load facet params
    params.fs = j["fs"];
    std::cout << "Facet size: " << params.fs << std::endl;

    // target reradiation function
    params.rerad_funct = j["rerad_funct"];

    // attenuation geometry file
    params.atten_geom_path = j.value("attenuation_geometry_file", std::string("NONE"));

    // processing parameters
    params.convolution = j["convolution"];
    params.convolution_linear = j["convolution_linear"];
    
    return params;
}


__host__ void checkFileExists(const char* filename)
{
    struct stat sb;

    if (stat(filename, &sb) != 0) {
        throw std::invalid_argument(std::string("File not found: ") + filename);
    }

    if (!S_ISREG(sb.st_mode)) {
        throw std::invalid_argument(std::string("Not a regular file: ") + filename);
    }
}


__host__ void checkDirectoryExists(const char* path)
{
    struct stat sb;

    if (stat(path, &sb) != 0) {
        throw std::invalid_argument(
            std::string("Directory not found: ") + path
        );
    }

    if (!S_ISDIR(sb.st_mode)) {
        throw std::invalid_argument(
            std::string("Path is not a directory: ") + path
        );
    }
}


__host__ void saveSignalToFile(const char* filename, cuFloatComplex* d_sig, int nr) {

    // copy Itd to host and export as file
    cuFloatComplex* h_sig = (cuFloatComplex*)malloc(nr * sizeof(cuFloatComplex));
    cudaMemcpy(h_sig, d_sig, nr * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    FILE* fItd = fopen(filename, "w");
    // export real and imag parts
    for (int i = 0; i < nr; i++) {
        fprintf(fItd, "%e %e\n", cuCrealf(h_sig[i]), cuCimagf(h_sig[i]));
    }
    fclose(fItd);

}

#define BUF_SIZE 65536

int count_lines(FILE* file, int counter_start = 1)
{
    char buf[BUF_SIZE];
    int counter = counter_start;
    for(;;)
    {
        size_t res = fread(buf, 1, BUF_SIZE, file);
        if (ferror(file))
            return -1;

        int i;
        for(i = 0; i < res; i++)
            if (buf[i] == '\n')
                counter++;

        if (feof(file))
            break;
    }
    rewind(file);

    return counter;
}

__host__ void loadFacetFile(FILE * file, const int totfacets,
                            float* h_fx, float* h_fy, float* h_fz,
                            float* h_fnx, float* h_fny, float* h_fnz,
                            float* h_fux, float* h_fuy, float* h_fuz,
                            float* h_fvx, float* h_fvy, float* h_fvz) {

    // go through line by line and load the facet into memory
    char line[256];
    int i = 0;
    while (fgets(line, sizeof(line), file)) {

        // parse line
        sscanf(line, "%f,%f,%f:%f,%f,%f:%f,%f,%f:%f,%f,%f",
               &h_fx[i],   &h_fy[i],  &h_fz[i], &h_fnx[i], &h_fny[i], &h_fnz[i],
               &h_fux[i], &h_fuy[i], &h_fuz[i], &h_fvx[i], &h_fvy[i], &h_fvz[i]);
        i++;

        // somehow if we are out of memory, break before segfault
        if (i > totfacets) {
            std::cout << "File has more facets than memory allocated - stopping read." << std::endl;
            break;
        }

    }

}


__host__ void loadTargetFile(FILE* file, const int nt,
                            float* h_tx, float* h_ty, float* h_tz,
                            float* h_tnx, float* h_tny, float* h_tnz) {

    // go through line by line and load the targets into memory
    char line[256];
    int i = 0;
    while (fgets(line, sizeof(line), file)) {

        // detect if line has 3 or 6 entries
        int comma_count = 0;
        for (char* p = line; *p != '\0'; p++) {
            if (*p == ',') {
                comma_count++;
            }
        }

        // if only 3 entries, set normal vector to default as upward Z
        if (comma_count == 2) {
            sscanf(line, "%f,%f,%f",
                &h_tx[i],   &h_ty[i],  &h_tz[i]);
            h_tnx[i] = 0.0f;
            h_tny[i] = 0.0f;
            h_tnz[i] = 1.0f;
        }
        else if (comma_count == 5) {
            sscanf(line, "%f,%f,%f,%f,%f,%f",
                &h_tx[i],   &h_ty[i],  &h_tz[i],
                &h_tnx[i], &h_tny[i], &h_tnz[i]);
        }
        else {
            std::cerr << "Error: Target file line number=" << i+1 << " has incorrect number of entries." << std::endl;
            break;
        }

        // somehow if we are out of memory, break before segfault
        if (i > nt) {
            std::cout << "File has more targets than memory allocated - stopping read." << std::endl;
            break;
        }

        i++;

    }

}


__host__ void loadSourceFile(FILE* file, const int ns,
                             float* h_sx, float* h_sy, float* h_sz,
                             float* h_snx, float* h_sny, float* h_snz) {

    // go through line by line and load the sources into memory
    char line[256];
    int i = 0;
    while (fgets(line, sizeof(line), file)) {

        // detect if line has 3 or 6 entries
        int comma_count = 0;
        for (char* p = line; *p != '\0'; p++) {
            if (*p == ',') {
                comma_count++;
            }
        }

        // if only 3 entries, set normal vector to default as upward Z
        if (comma_count == 2) {
            sscanf(line, "%f,%f,%f",
                &h_sx[i],   &h_sy[i],  &h_sz[i]);
            h_snx[i] = 0.0f;
            h_sny[i] = 0.0f;
            h_snz[i] = 1.0f;
        }
        else if (comma_count == 5) {
            sscanf(line, "%f,%f,%f,%f,%f,%f",
                &h_sx[i],   &h_sy[i],  &h_sz[i],
                &h_snx[i], &h_sny[i], &h_snz[i]);
        }
        else {
            std::cerr << "Error: Source file line number=" << i+1 << " has incorrect number of entries." << std::endl;
            break;
        }

        // somehow if we are out of memory, break before segfault
        if (i > ns) {
            std::cout << "File has more sources than memory allocated - stopping read." << std::endl;
            break;
        }

        i++;

    }

}


__host__ void loadAttenPrismFile(FILE* file, const int nPrisms,
                                 float* h_alphas, 
                                 float* h_attXmin, float* h_attXmax,
                                 float* h_attYmin, float* h_attYmax,
                                 float* h_attZmin, float* h_attZmax,
                                 SimulationParameters par) {

    // go through line by line and load attenuation geometry into memory
    char line[256];

    // conductivity for file read
    float conductivity;
    float eps_pp;

    int i = 0;
    while (fgets(line, sizeof(line), file)) {

        // parse line
        sscanf(line, "%f,%f,%f,%f,%f,%f,%f",
               &conductivity,
               &h_attXmin[i], &h_attYmin[i],
               &h_attZmin[i], &h_attXmax[i],
               &h_attYmax[i], &h_attZmax[i]);

        // turn the conductivity into the alpha value
        eps_pp  = conductivity / (2 * 3.14159 * par.f0 * par.eps_0);
        h_alphas[i]   = sqrt(1 + (eps_pp/par.eps_2)*(eps_pp/par.eps_2)) - 1;
        h_alphas[i]   = sqrt(0.5 * par.eps_2 * h_alphas[i]);
        h_alphas[i]   = (h_alphas[i] * 2 * 3.14159) / par.lam;

        i++;

        // somehow if we are out of memory, break before segfault
        if (i >= nPrisms) {
            std::cout << "File has more attenuation prisms than memory allocated - stopping read." << std::endl;
            break;
        }

    }

}                            


__host__ void loadRxWindowPositions(FILE* file,
                                    const int ns,
                                    float* h_rx_window_positions)
{
    char line[256];
    int i = 0;

    while (fgets(line, sizeof(line), file)) {

        // too many entries
        if (i >= ns) {
            std::cerr << "Error: rx_window_positions file has MORE entries than sources (ns = "
                      << ns << ")" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (sscanf(line, "%f", &h_rx_window_positions[i]) != 1) {
            std::cerr << "Error: Invalid float on line " << i + 1 << std::endl;
            std::exit(EXIT_FAILURE);
        }

        i++;
    }

    // too few entries
    if (i != ns) {
        std::cerr << "Error: rx_window_positions file has "
                  << i << " entries, but expected exactly "
                  << ns << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


__host__ void remove_s_txt_files(const std::filesystem::path& dir)
{
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        std::cerr << "Invalid directory\n";
        return;
    }

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file())
            continue;

        const auto& path = entry.path();
        const std::string filename = path.filename().string();

        // Match: s*.txt
        if (filename.size() >= 5 &&
            filename[0] == 's' &&
            path.extension() == ".txt")
        {
            std::filesystem::remove(path);
            std::cout << "Clearing write directory... Removed: " << path << "         \r";
        }
    }

    std::cout << "\n" << std::endl;
}