#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

struct SimulationParameters {

    float c = 299792480.0f; // speed of light in m/s
    float nu0 = 376.7f;
    float eps_0 = 8.85e-12;

    // source parameters
    float sx, sy, sz;   // Source position
    float f0;           // Radar centerfrequency
    float B;            // Bandwidth
    float P;            // Transmitted power
    float Grefr;        // Subsurface gain
    float Grefl;        // Surface gain
    float rng_res;      // Range resolution
    // computed params
    float lam;
    float k;
    int pol = 0;
    float Grefr_lin;    // Subsurface gain (linear)
    float Grefl_lin;    // Surface gain (linear)
    
    // surface parameters
    float sigma;        // sigma
    float rms_h;        // surface rms height
    float ks;

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
    float tx, ty, tz;   // Target position

    // range window params
    float rst, ren, dr;
    int nr;
    float smpl;         // Sampling rate (Hz)

};

__host__ SimulationParameters parseSimulationParameters(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open JSON file: " + filename);
    }

    json j;
    file >> j;

    SimulationParameters params;
    params.sx = j["sx"];
    params.sy = j["sy"];
    params.sz = j["sz"];
    params.f0 = j["frequency"];
    params.B = j["bandwidth"];
    params.P = j["power"];
    params.Grefr = j["subsurface_gain"];
    params.Grefl = j["surface_gain"];
    params.rng_res = j["range_resolution"];

    params.lam = params.c / params.f0;
    params.k = 2 * 3.14159 / params.lam;
    params.Grefr_lin = pow(10, params.Grefr / 10.f);
    params.Grefl_lin = pow(10, params.Grefl / 10.f);

    params.sigma = j["sigma"];
    params.rms_h = j["rms_height"];
    params.ks = params.k * params.rms_h;

    params.eps_1 = j["eps_1"];
    params.eps_2 = j["eps_2"];
    params.sig_1 = j["sig_1"];
    params.sig_2 = j["sig_2"];

    params.c_1 = params.c / sqrt(params.eps_1);
    params.c_2 = params.c / sqrt(params.eps_2);
    params.nu_1 = params.nu0 / sqrt(params.eps_1);
    params.nu_2 = params.nu0 / sqrt(params.eps_2);

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

    params.tx = j["tx"];
    params.ty = j["ty"];
    params.tz = j["tz"];

    params.smpl = j["rx_sample_rate"];

    params.rst = j["rx_window_offset_m"];
    params.ren = float(j["rx_window_offset_m"]) + float(j["rx_window_m"]);
    params.nr  = (float(j["rx_window_m"]) / 299792480.0f) * params.smpl;
    params.dr  = float(j["rx_window_m"]) / float(params.nr);

    // report sampling window
    std::cout << "Sampling window: " << params.rst << " to " << params.ren << " m, ";
    std::cout << params.nr << " samples at " << params.smpl << " Hz" << std::endl;

    return params;
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