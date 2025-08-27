# imports
import sys, copy
sys.path.append("../../src")

import numpy as np
import matplotlib.pyplot as plt

from surface import *
from source import *
from simulator import *
from focus import *

# --- GENERATE SURFACES ---


surfflat = Surface(origin=(-1.25e3, -1.25e3), dims=(500, 500), fs=5, overlap=0)
surfflat.gen_flat(10)

surfsym = Surface(origin=(-1.25e3, -1.25e3), dims=(500, 500), fs=5, overlap=0)

gap = 1000

hs = repeating_gaussian(surfsym.x, 130, 1000/2.5, gap, xoffset=-gap)
surfsym.arr_along_axis(hs, axis=1)

surfoffx = Surface(origin=(-1.25e3, -1.25e3), dims=(500, 500), fs=5, overlap=0)
hoffs = repeating_slant_gaussian(surfoffx.x, 130, 1000/2.5, 250/2.5, gap, xoffset=-gap)
surfoffx.arr_along_axis(hoffs, axis=1)

surfs = [surfflat, surfsym, surfoffx]

# --- GENERATE SOURCES AND TARGET ---

ss = source_linspace('x', -10e3, 10e3, 0, 5e3, 250, 9e6, 1e6)

# target location
t = (0, 0, -100)   # target location (x, y, z)

# --- SIMULATE FOR SYMMETRIC DUNES ---

# change in parameter matrix
pars = {
    "rx_window_offset":2e3,
}

names = ("flat_surface.png", "symmetric_dune.png", "offsetx_dune.png")

for surf, name in zip(surfs, names):

    surf.show_2d_heatmap(ss=ss, t=t, savefig=f"surf-{name}", show=False)
    #surf.show_normals()
    surf.get_profile(axis=1, save=f"prof-{name}", show=False)

    rdrgrm, time = run_sim_ms(surf, ss, t, reflect=False, polarization='v', 
                            sltrng=False, xmax=10, gpu=True, savefig=name, 
                            par=pars)
