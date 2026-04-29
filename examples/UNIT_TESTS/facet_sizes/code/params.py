import sys
import pickle
import numpy as np
import json

sys.path.append("../../../archive/src")
from terrain import Terrain

params = {

    # radar parameters
    "power": 100,             # Transmitter power [W]
    "frequency": 60e6,         # Radar frequency [Hz]
    "bandwidth": 10e6,         # Radar bandwidth [Hz]
    "surface_gain": 7.3,       # Antenna gain [dB]
    "subsurface_gain": 7.3,   # Subsurface antenna gain [dB]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)
    "aperture": 20,           # aperture (from nadir->edge) [deg]

    # receive window parameters
    "rx_window_offset_m":  199.75e3,         # receive window length [m]
    "rx_window_m": 0.75e3,  # receive window offset [m]
    "rx_sample_rate": 60e6,       # receive sample rate [Hz]

    # surface parameters
    "rms_height": 0.4,       # surface roughness [m]
    "buff": 1.1,             # buffer for facet estimate

    # atmosphere/subsurface parameters
    "eps_1": 1.0,            # permittivity of medium 1 
    "eps_2": 3.15,           # permittivity of medium 2
    "sig_1": 0.0,            # conductivity of medium 1 [S/m]
    "sig_2": 1e-6,           # conductivity of medium 2 [S/m]
    "mu_1": 1.0,             # permeability of medium 1
    "mu_2": 1.0,             # permeability of medium 2

    # source parameters 
    "sy": 0,                # source y location       [m]
    "sz": 200e3,             # source z location       [m]
    "sdx": 10,              # source x discretization [m]
    "sx0": 0,            # source x origin         [m]
    "ns": 1,             # source count            [.]

    # facet array params
    "ox": -5e3,
    "oy": -5e3,
    "oz": 0,

    # target params
    "rerad_funct": 2,  # 1-degree boxcar

    # processing parameters (BOOLEAN)
    "convolution": True,   # use convolution-based processing
    "convolution_linear": True,  # use linear convolution instead of circular
    "specular": False,     # use specular computation methods for specific circumstances only
    "lossless": True,      # simulate without loss (spreading not included)

}

fss = (25, 50, 100, 150, 250)

for i, fs in enumerate(fss):

    params["fs"] = fs
    params["nx"] = int(10e3 / fs)
    params["ny"] = int(10e3 / fs)

    with open(f"inputs/params{i}.json", "w") as f:
        json.dump(params, f, indent=4)

    with open(f"inputs/params{i}.pkl", 'wb') as hdl:
        pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

    xmin, xmax = params["ox"], params["ox"]+params["nx"]*params["fs"]
    ymin, ymax = params["oy"], params["oy"]+params["ny"]*params["fs"]

    terrain = Terrain(xmin, xmax, ymin, ymax, params["fs"])
    terrain.gen_flat(0)

    # write output facet data
    terrain.export(f"inputs/facets{i}.fct")
