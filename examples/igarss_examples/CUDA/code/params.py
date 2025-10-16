import sys
import pickle
import numpy as np
import json

params = {

    # radar parameters
    "power": 100,             # Transmitter power [W]
    "frequency": 9e6,         # Radar frequency [Hz]
    "bandwidth": 1e6,         # Radar bandwidth [Hz]
    "surface_gain": 75,       # Antenna gain [dB]
    "subsurface_gain": 98,   # Subsurface antenna gain [dB]
    "range_resolution": 300,  # range resolution [m]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)
    #"aperture": 5,           # aperture (from nadir->edge) [deg]
    "aperture": 8,           # aperture (from nadir->edge) [deg]

    # receive window parameters
    "rx_window_m":  10e3,          # receive window length [m]
    "rx_window_offset_m": 22.5e3,  # receive window offset [m]
    "rx_sample_rate": 48e6,        # receive sample rate [Hz]

    # surface parameters
    "sigma": 1,              # sigma [?]
    "rms_height": 0.4,       # surface roughness [m]
    "buff": 1.05,            # buffer on estimated facet count [.]

    # atmosphere/subsurface parameters
    "eps_1": 1.0,            # permittivity of medium 1 
    "eps_2": 3.15,           # permittivity of medium 2
    "sig_1": 0.0,            # conductivity of medium 1 [S/m]
    "sig_2": 1e-6,           # conductivity of medium 2 [S/m]
    "mu_1": 1.0,             # permeability of medium 1
    "mu_2": 1.0,             # permeability of medium 2

    # target parameters
    "tx": 5e3,                # target x location [m]
    "ty": 0,                # target y location [m]
    "tz": -1.5e3,            # target z location [m]

    # source parameters 
    "sy": 0,                # source y location       [m]
    "sz": 25e3,             # source z location       [m]
    "sdx": 10,              # source x discretization [m]
    "sx0": -10e3,           # source x origin         [m]
    "ns": 3000,             # source count            [.]

    # facet array params
    "ox": -11e3,
    "oy": -1.1e3,
    "oz": 0,
    "fs": 5,
    "nx": 6400,
    "ny": 440,

}


with open("params/params.json", "w") as f:
    json.dump(params, f, indent=4)

with open("params/params.pkl", 'wb') as hdl:
    pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

# --- MAKE FACET FILE
sys.path.append("../../../src")
from terrain import Terrain

xmin, xmax = params["ox"], params["ox"]+params["nx"]*params["fs"]
ymin, ymax = params["oy"], params["oy"]+params["ny"]*params["fs"]

# gen flat
terrain = Terrain(xmin, xmax, ymin, ymax, params["fs"])
terrain.gen_flat(0)
terrain.export("facets/flat.fct")

# gen double ridge
amp       = 300     # amplitude [m]
peak_dist = 6e3     # peak distance [m]
ridge_wid = 4e3     # ridge width [m]
x_offset  = 5e3     # x offset [m]
terrain.double_ridge(amp, amp, peak_dist, ridge_wid, x_offset)
terrain.export("facets/ridge.fct")