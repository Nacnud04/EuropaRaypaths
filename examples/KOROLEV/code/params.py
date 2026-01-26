import sys
import pickle
import numpy as np
import json

params = {

    # radar parameters
    "power": 100,             # Transmitter power [W]
    "frequency": 20e6,        # Radar frequency [Hz]
    "bandwidth": 10e6,        # Radar bandwidth [Hz]
    "surface_gain": 20,       # Antenna gain [dB]
    "subsurface_gain": 0,    # Subsurface antenna gain [dB]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)
    "aperture": 1,            # aperture (from nadir->edge) [deg]

    # receive window parameters
    "rx_window_m":  7.5e3,         # receive window length [m]
    "rx_window_offset_m": 300e3,  # receive window offset [m]
    "rx_sample_rate": 40e6,       # receive sample rate [Hz]
    "rx_window_position_file": "data/rx_window_positions.txt",

    # surface parameters
    "rms_height": 0.4,       # surface roughness [m]
    "buff": 1.5,             # buffer for facet estimate

    # atmosphere/subsurface parameters
    "eps_1": 1.0,            # permittivity of medium 1 
    "eps_2": 3.15,           # permittivity of medium 2
    "sig_1": 0.0,            # conductivity of medium 1 [S/m]
    "sig_2": 1e-6,           # conductivity of medium 2 [S/m]
    "mu_1": 1.0,             # permeability of medium 1
    "mu_2": 1.0,             # permeability of medium 2

    # source parameters 
    "source_path_file": "data/Observation/s_00554201_srcs.txt",

    # facet params
    "fs": 463.114129,              # facet size [m]

    # target params
    "rerad_funct": 2,  # 0.5-degree boxcar

    # processing parameters (BOOLEAN)
    "convolution": True,   # use convolution-based processing
    "convolution_linear": True,  # use linear convolution instead of circular

}

with open("data/params.json", "w") as f:
    json.dump(params, f, indent=4)

with open("data/params.pkl", 'wb') as hdl:
    pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)


