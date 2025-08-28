import sys
sys.path.append("../../src")

from terrain import *
from surface import *
from source import *
from simulator import *

# --- GENERATE TERRAIN OBJECT ---

# terrain params
xmin, xmax = 0, 100
ymin, ymax = -5, 5
fs = 0.01
dims = (600, 600)

# make terrain object
terrain = Terrain(xmin, xmax, ymin, ymax, fs)

# generate sinusoid and normals
terrain.double_ridge(2, 2, 30, 20, 50)
#terrain.gen_flat(0)

# --- SET UP SOURCES AND TARGET ---

f0 = 5e9
B  = 6e9

ss = source_linspace('x', 10, 90, 0, 100, 250, f0, B)

# target location
t = (50, 0, -2)   # target location (x, y, z)

# show terrain
terrain.show_2d_heatmap(show=False, savefig="terrain.png", ss=ss, t=t)
terrain.show_profile('x', 0, savefig="doubleridgeprofile.png")

# --- LOOK AT A CHIRP ---

ss[0].plot(savefig="chirp.png")

# --- CHANGE SIM PARS ---

Rr = 299792458 / (2 * B)

par = {
    "rx_window_offset": 90,   # [m]
    "rx_window_m"     : 20,   # [m]
    "sampling"        : 15e9,  # [Hz]
    "range_resolution": Rr,  # [m]
    "lambda"          : 0.06, # [m]
    "aspect"          : 2000,   # [.]
    "surf_gain"       : 98,
}

rdrgrm, ts = run_sim_terrain(terrain, dims, ss, t, reflect=True, polarization='h',
                             xmin=10, xmax=90, gpu=True, savefig="rdrgrm-doubleridge.png", 
                             rough=False, nsmpl=1000, par=par)