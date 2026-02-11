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
    "surface_gain": 25,       # Antenna gain [dB]
    "subsurface_gain": 50,    # Subsurface antenna gain [dB]
    "range_resolution": 300,  # range resolution [m]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)
    "aperture": 7,            # aperture (from nadir->edge) [deg]

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

    # attenuation geometry file (NOT REQUIRED)
    "attenuation_geometry_file":"params/halfspace.txt",

    # processing parameters (BOOLEAN)
    "convolution": True,   # use convolution-based processing
    "convolution_linear": True,  # use linear convolution instead of circular

}

# export parameters as pickle and json
with open("params/halfspace.json", "w") as f:
    json.dump(params, f, indent=4)
with open("params/halfspace.pkl", 'wb') as hdl:
    pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

params["attenuation_geometry_file"] = "params/window.txt"

# export parameters as pickle and json
with open("params/window.json", "w") as f:
    json.dump(params, f, indent=4)
with open("params/window.pkl", 'wb') as hdl:
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

# first create deep point target
txs, tys, tzs = [0], [0], [0]
txs[0] = 0
tys[0] = 0
tzs[0] = -1500

# now generate layer
tspace = 300
for x in np.arange(params['sx0'], params['sx0']+params['sdx']*params['ns'], tspace):
    txs.append(x)
    tys.append(0)
    tzs.append(-3000)

# export to file
with open("params/targets.txt", "w") as f:
    for i, (x, y, z) in enumerate(zip(txs, tys, tzs)):
        if i > 0:
            f.write(f"\n{x}, {y}, {z}")
        else:
            f.write(f"{x}, {y}, {z}")


# ---- GENERATE ATTENUATION PLOTS -------------------------------------

def gen_attenuation_plot(conductivity, ex1_xmins, ex1_xmaxs, ex1_ymins, ex1_ymaxs, ex1_zmins, ex1_zmaxs, name):

    cmap = plt.get_cmap('inferno')
    norm = plt.Normalize(vmin=params['sig_1'], vmax=conductivity)

    fig, ax = plt.subplots(figsize=(6, 4))

    # set xlim and ylim in km
    ax.set_ylim((params['rx_window_offset_m']+params['rx_window_m'])/1e3, params['rx_window_offset_m']/1e3)
    ax.set_xlim(params['sx0']/1e3, (params['sx0']+params['sdx']*params['ns'])/1e3)

    # plot atmosphere conductivity
    rect_x = params['sx0'] / 1e3
    rect_y = params['sz'] / 1e3
    rect_w = (params['sdx'] * params['ns']) / 1e3
    rect_h = -1*params['rx_window_m'] / 1e3
    ax.add_patch(Rectangle((rect_x, rect_y), rect_w, rect_h,
                        color=cmap(norm(params['sig_1'])), zorder=0))

    # plot subsurface conductivity
    rect_x = params['sx0'] / 1e3
    rect_y = params['sz'] / 1e3
    rect_w = (params['sdx'] * params['ns']) / 1e3
    rect_h = params['rx_window_m'] / 1e3
    ax.add_patch(Rectangle((rect_x, rect_y), rect_w, rect_h,
                        color=cmap(norm(params['sig_2'])), zorder=0))

    # plot conductivity rectangles
    for xmin, xmax, ymin, ymax, zmin, zmax in zip(ex1_xmins, ex1_xmaxs, ex1_ymins, ex1_ymaxs, ex1_zmins, ex1_zmaxs):
        rect_x = xmin / 1e3
        rect_y = (params['sz'] + zmax) / 1e3
        rect_w = (xmax - xmin) / 1e3
        rect_h = (zmax - zmin) / 1e3
        ax.add_patch(Rectangle((rect_x, rect_y), rect_w, rect_h,
                            color=cmap(norm(conductivity)), zorder=1))

    plt.savefig(f"plots/{name}.png")

# --- GENERATE ATTENUATION GEOMETRIES -------------------------------

conductivity = 1e-5

ex1_xmins = [-5e3];   ex1_xmaxs = [0]
ex1_ymins = [-2.5e3]; ex1_ymaxs = [2.5e3]
ex1_zmins = [-7e3];   ex1_zmaxs = [0]

with open("params/halfspace.txt", 'w') as f:
    for i, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(zip(ex1_xmins, ex1_xmaxs, ex1_ymins, ex1_ymaxs, ex1_zmins, ex1_zmaxs)):
        if i > 0:
            f.write(f"\n{conductivity}, {xmin}, {ymin}, {zmin}, {xmax}, {ymax}, {zmax}")
        else:
            f.write(f"{conductivity}, {xmin}, {ymin}, {zmin}, {xmax}, {ymax}, {zmax}")

#gen_attenuation_plot(conductivity, ex1_xmins, ex1_xmaxs, ex1_ymins, ex1_ymaxs, ex1_zmins, ex1_zmaxs, "halfspace")

ex1_xmins = [-5e3, 1e3];   ex1_xmaxs = [-1e3, 5e3]
ex1_ymins = [-2.5e3, -2.5e3]; ex1_ymaxs = [2.5e3, 2.5e3]
ex1_zmins = [-0.5e3, -0.5e3];   ex1_zmaxs = [0, 0]

with open("params/window.txt", 'w') as f:
    for i, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(zip(ex1_xmins, ex1_xmaxs, ex1_ymins, ex1_ymaxs, ex1_zmins, ex1_zmaxs)):
        if i > 0:
            f.write(f"\n{conductivity}, {xmin}, {ymin}, {zmin}, {xmax}, {ymax}, {zmax}")
        else:
            f.write(f"{conductivity}, {xmin}, {ymin}, {zmin}, {xmax}, {ymax}, {zmax}")

#gen_attenuation_plot(conductivity, ex1_xmins, ex1_xmaxs, ex1_ymins, ex1_ymaxs, ex1_zmins, ex1_zmaxs, "window")
