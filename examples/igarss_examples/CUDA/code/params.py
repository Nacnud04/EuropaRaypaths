import sys, os
import pickle
import numpy as np
import json
import pandas as pd

import matplotlib

os.environ["PATH"] += os.pathsep + '/usr/share/texlive/texmf-dist/tex/xelatex'

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

params = {

    # radar parameters
    "power": 100,             # Transmitter power [W]
    "frequency": 9e6,         # Radar frequency [Hz]
    "bandwidth": 1e6,         # Radar bandwidth [Hz]
    "surface_gain": 53,       # Antenna gain [dB]
    "subsurface_gain": 63,   # Subsurface antenna gain [dB]
    "range_resolution": 300,  # range resolution [m]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)
    #"aperture": 5,           # aperture (from nadir->edge) [deg]
    "aperture": 8,           # aperture (from nadir->edge) [deg]

    # receive window parameters
    "rx_window_m": 7e3,            # receive window length [m]
    "rx_window_offset_m": 22.5e3,  # receive window offset [m]
    "rx_sample_rate": 48e6,        # receive sample rate [Hz]

    # surface parameters
    "rms_height": 0.4,       # surface roughness [m]
    "buff": 1.05,            # buffer on estimated facet count [.]

    # atmosphere/subsurface parameters
    "eps_1": 1.0,            # permittivity of medium 1 
    "eps_2": 3.15,           # permittivity of medium 2
    "sig_1": 0.0,            # conductivity of medium 1 [S/m]
    "sig_2": 1e-6,           # conductivity of medium 2 [S/m]
    "mu_1": 1.0,             # permeability of medium 1
    "mu_2": 1.0,             # permeability of medium 2

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

    # target params
    "rerad_funct": 2,  # 1-degree boxcar

    # processing parameters
    "convolution": True,   # use convolution-based processing
    "convolution_linear": True,  # use linear convolution instead of circular

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

# --- GENERATE w/ GIVEN DEM GEOMETRY

# make track 60 km instead now
params["ox"] = -30e3 
params["nx"] = 12000
xmin, xmax = params["ox"], params["ox"]+params["nx"]*params["fs"]

# increase source count
params['sx0'] = -30e3
params['ns']  = 6000

# move target to new center
params['tx'] = 0

# export parameter file
with open("params/dem.json", "w") as f:
    json.dump(params, f, indent=4)

with open("params/dem.pkl", 'wb') as hdl:
    pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

# load CSV
DEM = pd.read_csv("facets/dem_profile.csv")

# create terrain object
terrain = Terrain(xmin, xmax, ymin, ymax, params['fs'])
terrain.gen_from_provided(DEM['Along Track (km)']*1e3, DEM['Surface Height (m)'])
terrain.show_profile('x', 0, savefig="figures/dem_profile.pgf", shape=(3,1))
terrain.export("facets/dem.fct")


# --- GENERATE TARGET FILE ------------------------------------------

# first create deep point target
txs, tys, tzs = [5e3], [0], [-1.5e3]

# now generate layer
#tspace = 200
#for x in np.arange(params['sx0'], -3e3, tspace):
#    txs.append(x)
#    tys.append(0)
#    tzs.append(-1000)

#for x in np.arange(12.5e3, 20e3, tspace):
#    txs.append(x)
#    tys.append(0)
#    tzs.append(-1000)

# export to file
with open("params/layer.txt", "w") as f:
    for i, (x, y, z) in enumerate(zip(txs, tys, tzs)):
        if i > 0:
            f.write(f"\n{x}, {y}, {z}")
        else:
            f.write(f"{x}, {y}, {z}")
