import sys
sys.path.append("../../src")

from terrain import *
from surface import *
from source import *
from simulator import *

import imageio.v3 as iio
import matplotlib.cm as cm

# --- GENERATE TERRAIN OBJECT ---

# terrain params
xmin, xmax = -11.25e3, 11.25e3
ymin, ymax = -1.25e3, 1.25e3
fs = 5
dims = (500, 500)

# make terrain object
terrain = Terrain(xmin, xmax, ymin, ymax, fs)

# generate sinusoid and normals
terrain.sinusoid(300, 'x', 2e3, 0)

# --- SET UP SOURCES AND TARGET ---

ss = source_linspace('x', -10e3, 10e3, 0, 5e3, 250, 9e6, 1e6)

# target location
t = (0, 0, -1000)   # target location (x, y, z)

# show terrain
terrain.show_2d_heatmap(show=False, savefig="terrain.png", ss=ss, t=t)

# --- SIMULATE ---

par = {
    "rx_window_offset":2e3,
    "rx_window_m":10e3,
    "surf_gain": 95
}

rdrgrm, ts = run_sim_terrain(terrain, dims, ss, t, reflect=True, polarization='v', 
                            xmax=10, gpu=True, savefig="sinusoid_terrain.png", par=par)
