import pickle, json, sys
import numpy as np

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
    "rx_window_m": 0.75e3,  # receive window offset [m]
    "rx_sample_rate": 60e6,       # receive sample rate [Hz]
    "rx_window_position_file": "inputs/rx_window_positions.txt",

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
    "ns": 2000,               # source count            [.]
    "source_path_file": "inputs/source_path.txt",

    # facet array params
    "ox": -5e3,
    "oy": -5e3,
    "oz": 0,
    "fs": 100,
    "nx": 100,
    "ny": 100,

    # target params
    "rerad_funct": 2,  # 1-degree boxcar

    # processing parameters (BOOLEAN)
    "convolution": True,   # use convolution-based processing
    "convolution_linear": True,  # use linear convolution instead of circular
    "specular": False,     # use specular computation methods for specific circumstances only
    "lossless": True,      # simulate without loss (spreading not included)
    "incoherent": False,

    # enable debug to see surface response phasor trace
    "debug_surface": False,

}

# --- SAVE PARAMS ---

with open("inputs/co_params.json", "w") as f:
    json.dump(params, f, indent=4)

with open("inputs/co_params.pkl", 'wb') as hdl:
    pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

params['incoherent'] = True

with open("inputs/inco_params.json", "w") as f:
    json.dump(params, f, indent=4)

with open("inputs/inco_params.pkl", 'wb') as hdl:
    pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

# --- MAKE FACETS ---

xmin, xmax = params["ox"], params["ox"]+params["nx"]*params["fs"]
ymin, ymax = params["oy"], params["oy"]+params["ny"]*params["fs"]

terrain = Terrain(xmin, xmax, ymin, ymax, params["fs"])
terrain.gen_flat(0)

# write output facet data
terrain.export(f"inputs/facets.fct")

# --- MAKE SOURCE PATH ---

maxZ = 200e3 # maximum altitude of source path [m]
minZ =  50e3 # minimum altitude of source path [m]

sz = np.linspace(minZ, maxZ, params['ns'])
sy = np.zeros_like(sz)
sx = np.zeros_like(sz)

with open("inputs/source_path.txt", 'w') as f:
    for i in range(params['ns']):
        if i == params['ns'] - 1:
            f.write(f"{sx[i]},{sy[i]},{sz[i]}")
        else:
            f.write(f"{sx[i]},{sy[i]},{sz[i]}\n")

# --- RX OPENING WINDOW FILE ---

rx_window_positions = sz - 250

with open("inputs/rx_window_positions.txt", 'w') as f:
    for pos in rx_window_positions:
        f.write(f"{pos}\n")