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
    "subsurface_gain": 10,    # Subsurface antenna gain [dB]
    "range_resolution": 300,  # range resolution [m]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)
    "aperture": 2,            # aperture (from nadir->edge) [deg]

    # receive window parameters
    "rx_window_m":  25e3,         # receive window length [m]
    "rx_window_offset_m": 260e3,  # receive window offset [m]
    "rx_sample_rate": 40e6,       # receive sample rate [Hz]
    "rx_window_position_file": "inputs/rx_window_positions.txt",

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
    "sy": 0,                # source y location       [m]
    "sz": 270e3,            # source z location       [m]
    "sdx": 25,              # source x discretization [m]
    "sx0": -50e3,           # source x origin         [m]
    "ns": 4000,             # source count            [.]

    # facet params
    "fs": 125,              # facet size [m]

    # target params
    "rerad_funct": 1,  # 1-degree boxcar

    # processing parameters (BOOLEAN)
    "convolution": True,   # use convolution-based processing
    "convolution_linear": True,  # use linear convolution instead of circular

}

with open("inputs/params.json", "w") as f:
    json.dump(params, f, indent=4)

with open("inputs/params.pkl", 'wb') as hdl:
    pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

# --- MAKE FACET FILE ---
sys.path.append("../../../archive/src")
from terrain import Terrain

# facet array params
ox = -50e3
oy = -10e3
oz = 0
nx = 800
ny = 160

xmin, xmax = ox, ox+nx*params['fs']
ymin, ymax = oy, oy+ny*params['fs']

terrain = Terrain(xmin, xmax, ymin, ymax, params["fs"])
terrain.gen_flat(0)

# write output facet data
terrain.export("inputs/facets.fct")


# --- MAKE TARGET FILE ---
tpos = np.array([[0, 0, -3e3]])
with open("inputs/targets.txt", 'w') as f:
    for i in range(tpos.shape[0]):
        f.write(f"{tpos[i,0]},{tpos[i,1]},{tpos[i,2]}")


# --- CREATE RX OPENING WINDOW FILE ---
rx_window_positions = params["rx_window_offset_m"] - 5e3 * (np.arange(params["ns"])/params['ns'])
with open("inputs/rx_window_positions.txt", 'w') as f:
    for pos in rx_window_positions:
        f.write(f"{pos}\n")
