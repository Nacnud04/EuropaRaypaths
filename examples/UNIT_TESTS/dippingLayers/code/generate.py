import sys, pickle, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.append("../../../archive/src")
from terrain import Terrain



# --- PARAMETER FILE GENERATION -------------------------------------

params = {

    # radar parameters
    "power": 100,             # Transmitter power [W]
    "frequency": 9e6,         # Radar frequency [Hz]
    "bandwidth": 1e6,         # Radar bandwidth [Hz]
    "surface_gain": 55,       # Antenna gain [dB]
    "subsurface_gain": 62,    # Subsurface antenna gain [dB]
    "range_resolution": 300,  # range resolution [m]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)
    "aperture": 12,            # aperture (from nadir->edge) [deg]

    # receive window parameters
    "rx_window_m":  10e3,         # receive window length [m]
    "rx_window_offset_m": 7.5e3,  # receive window offset [m]
    "rx_sample_rate": 48e6,       # receive sample rate [Hz]

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
    "sy": 0,                 # source y location       [m]
    "sz": 10e3,              # source z location       [m]
    "sdx": 10,               # source x discretization [m]
    "sx0": -5e3,             # source x origin         [m]
    "ns": 1000,              # source count            [.]

    # facet array params
    "ox": -5e3,
    "oy": -1e3,
    "oz": 0,
    "fs": 5,
    "nx": 2000,
    "ny": 400,

    # target params
    "rerad_funct": 1,  # 1-degree boxcar

    # attenuation geometry file (not used here)
    #"attenuation_geometry_file":"params/halfspace.txt",

    # processing parameters (BOOLEAN)
    "convolution": True,   # use convolution-based processing
    "convolution_linear": True,  # use linear convolution instead of circular

}

# export parameters as pickle and json
with open("params/params.json", "w") as f:
    json.dump(params, f, indent=4)
with open("params/params.pkl", 'wb') as hdl:
    pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)


# --- FACET FILE GENERATION -----------------------------------------

# get bounds
xmin, xmax = params["ox"], params["ox"]+params["nx"]*params["fs"]
ymin, ymax = params["oy"], params["oy"]+params["ny"]*params["fs"]

# generate terrain object
terrain = Terrain(xmin, xmax, ymin, ymax, params["fs"])
terrain.gen_flat(0)

# export as file
terrain.export("facets/facets.fct")


# --- GENERATE TARGET FILE ------------------------------------------

tdx = 50
xspace = np.arange(xmin, xmax, tdx)
l1zmin = -3e3
l1zmax = -2.5e3
zspace = np.linspace(l1zmin, l1zmax, len(xspace))

# generate normals for each target
inclination = np.arctan((l1zmax - l1zmin) / (xmax - xmin))
tnxspace = np.ones_like(xspace) * np.sin(inclination)
tnyspace = np.zeros_like(xspace)
tnzspace = np.ones_like(xspace) * np.cos(inclination)

# y positions spaced every 500 m from -2000 to +2000
#y_offsets = np.arange(-500, 501, 250)
y_offsets = [0]

# export to file
with open("params/targets.txt", "w") as f:
    first = True
    for y in y_offsets:
        for x, z, nx, ny, nz in zip(xspace, zspace, tnxspace, tnyspace, tnzspace):
            line = f"{x}, {y}, {z}, {nx}, {ny}, {nz}"
            if first:
                f.write(line)
                first = False
            else:
                f.write("\n" + line)

