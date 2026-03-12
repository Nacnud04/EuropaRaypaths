import numpy as np
import matplotlib.pyplot as plt
import pickle, sys

sys.path.append("../../../src/PYTHON")
import rdr_plots       as rp

directory = "rdrgrms/"
unfocused_files = ["NoConv_raw.npz", "LinConv_raw.npz"]
focused_files   = ["NoConv_focused.npz", "LinConv_focused.npz"]
names           = ["Non-Convolution", "Linear Convolution"]

with open("inputs/NoConv.pkl", 'rb') as hdl:
    params = pickle.load(hdl)

rp.TGRS_rdrgrm_conv(params, directory, focused_files, unfocused_files, names)
