import sys
import pickle
import numpy as np
import json

params = {

    # radar parameters
    "power": 100,             # Transmitter power [W]
    "frequency": 9e6,         # Radar frequency [Hz]
    "bandwidth": 1e6,         # Radar bandwidth [Hz]
    "surface_gain": 55,       # Antenna gain [dB]
    "subsurface_gain": 65,   # Subsurface antenna gain [dB]
    "range_resolution": 300,  # range resolution [m]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)
    "aperture": 7,           # aperture (from nadir->edge) [deg]

    # receive window parameters
    "rx_window_m":  10e3,         # receive window length [m]
    "rx_window_offset_m": 7.5e3,  # receive window offset [m]
    "rx_sample_rate": 48e6,       # receive sample rate [Hz]

    # surface parameters
    "sigma": 1,              # sigma [?]
    "rms_height": 0.4,       # surface roughness [m]
    "buff": 1.5,             # buffer for facet estimate

    # atmosphere/subsurface parameters
    "eps_1": 1.0,            # permittivity of medium 1 
    "eps_2": 3.15,           # permittivity of medium 2
    "sig_1": 0.0,            # conductivity of medium 1 [S/m]
    "sig_2": 1e-6,           # conductivity of medium 2 [S/m]
    "mu_1": 1.0,             # permeability of medium 1
    "mu_2": 1.0,             # permeability of medium 2

    # target parameters
    "tx": 0,                # target x location [m]
    "ty": 0,                # target y location [m]
    "tz": -3000,            # target z location [m]

    # source parameters 
    "sy": 0,                # source y location       [m]
    "sz": 10e3,#25e3,       # source z location       [m]
    "sdx": 10,              # source x discretization [m]
    "sx0": -5e3,            # source x origin         [m]
    "ns": 1000,              # source count            [.]

    # facet array params
    "ox": -5e3,
    "oy": -1e3,
    "oz": 0,
    "fs": 5,
    "nx": 2000,
    "ny": 400,

    # target function
    "rerad_funct": 1, # 1-degree boxcar

    # processing parameters (BOOLEAN)
    "convolution": True,   # use convolution-based processing
    "convolution_linear": False,  # use linear convolution instead of circular

}

modes = ["NoConv", "CircConv", "LinConv"]
convs = [ (False, False), (True, False), (True, True) ]

for mode, (conv, conv_lin) in zip(modes, convs):
    params["convolution"] = conv
    params["convolution_linear"] = conv_lin
    with open(f"inputs/{mode}.json", "w") as f:
        json.dump(params, f, indent=4)
    with open(f"inputs/{mode}.pkl", 'wb') as hdl:
        pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

# --- MAKE FACET FILE
sys.path.append("../../../archive/src")
from terrain import Terrain

xmin, xmax = params["ox"], params["ox"]+params["nx"]*params["fs"]
ymin, ymax = params["oy"], params["oy"]+params["ny"]*params["fs"]

terrain = Terrain(xmin, xmax, ymin, ymax, params["fs"])
terrain.gen_flat(0)
terrain.export("inputs/facets.fct")

# --- MAKE TARGET FILE
target_filename = "inputs/targets.txt"
with open(target_filename, 'w') as f:
    f.write(f"{params['tx']},{params['ty']},{params['tz']}")
