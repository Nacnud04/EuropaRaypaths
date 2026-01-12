import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob
import sys
import pickle

# focusing helper
sys.path.append("../../archive/src")
from focus import est_slant_range

sys.path.append("../../src/PYTHON")
import output_handling as oh
import simple_focusing as sf
import rdr_plots       as rp

paramss = ("halfspace", "window")
directs = ("rdr_halfspace", "rdr_window")

for param, direct in zip(paramss, directs):

    params = oh.load_params(f"params/{param}.pkl", f"params/targets.txt")

    # we don't want to focus at the first target, we want to focus at a
    # specific location so we override
    params["tx"] = 0
    params["ty"] = 0
    params["tz"] = -1500

    rdr = oh.compile_rdrgrm(direct, params)
    Nr, Na = rdr.shape

    focused = sf.focus_window(rdr, params, win_center=Na//2, win_width=Na//5, scale=50)

    rp.plot_rdr_attenuation_prof(rdr, focused, f"params/{param}.txt", params, savefig=f"plots/{param}.png")