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
    "subsurface_gain": 100,   # Subsurface antenna gain [dB]
    "range_resolution": 300,  # range resolution [m]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)
    "aperture": 10,           # aperture (from nadir->edge) [deg]

    # receive window parameters
    "rx_window_m":  10e3,         # receive window length [m]
    "rx_window_offset_m": 7.5e3,  # receive window offset [m]
    "rx_sample_rate": 48e6,       # receive sample rate [Hz]

    # surface parameters
    "sigma": 1,              # sigma [?]
    "rms_height": 0.4,       # surface roughness [m]

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
    "sdx": 20,              # source x discretization [m]
    "sx0": -5e3,            # source x origin         [m]
    "ns": 500,              # source count            [.]

    # facet array params
    "ox": -5e3,
    "oy": -1e3,
    "oz": 0,
    "fs": 5,
    "nx": 2000,
    "ny": 400,

}

with open("params.json", "w") as f:
    json.dump(params, f, indent=4)

with open("params.pkl", 'wb') as hdl:
    pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

# --- MAKE FACET FILE
sys.path.append("../src")
from terrain import Terrain

xmin, xmax = params["ox"], params["ox"]+params["nx"]*params["fs"]
ymin, ymax = params["oy"], params["oy"]+params["ny"]*params["fs"]

terrain = Terrain(xmin, xmax, ymin, ymax, params["fs"])
terrain.gen_flat(0)
# make double ridge
#amp       = 225     # amplitude [m]
#peak_dist = 6e3     # peak distance [m]
#ridge_wid = 4e3     # ridge width [m]
#x_offset  = 0       # x offset [m]
#terrain.double_ridge(amp, amp, peak_dist, ridge_wid, x_offset)

# write output facet data
xx, yy, zz = terrain.XX, terrain.YY, terrain.zs
norms      = np.reshape(terrain.normals, (xx.shape[0]*xx.shape[1], 3))
uvecs      = np.reshape(terrain.uvectors, (xx.shape[0]*xx.shape[1], 3))
vvecs      = np.reshape(terrain.vvectors, (xx.shape[0]*xx.shape[1], 3))

with open("facets.fct", 'w') as f:
    for x, y, z, (nx, ny, nz), (ux, uy, uz), (vx, vy, vz) in zip(xx.flatten(), yy.flatten(), zz.flatten(), norms, uvecs, vvecs):
        f.write(f"{x},{y},{z}:{nx},{ny},{nz}:{ux},{uy},{uz}:{vx},{vy},{vz}\n")