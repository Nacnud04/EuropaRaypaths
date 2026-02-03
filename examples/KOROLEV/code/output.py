import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../src/PYTHON")
import kutil           as ku
import simple_focusing as sf
import rdr_plots       as rp
import output_handling as oh
import unit_convs      as uc

params = oh.load_params("data/params.pkl", "data/Subsurface/KOR_T.txt")
params['ns'] = 2000

rdrgrm = oh.compile_rdrgrm("rdrgrm", params, rx_win_file="data/rx_window_positions.npy")
print(f"Radargram shape: {rdrgrm.shape}")

np.save("output/rdrgrm.npy", rdrgrm)

# TMP LOAD ORBIT
DIRECTORY = "data/Observation"
OBS       = "00554201"

# load in dataset
geometry = ku.load_sharad_orbit(DIRECTORY, OBS)

# restrict to part of radargram within interest
mincol = 750
maxcol = 1200
geometry = geometry[(geometry['COL'] > mincol) * (geometry['COL'] < maxcol)]

# convert the satellite location into x, y, z
sat_x, sat_y, sat_z = ku.planetocentric_to_cartesian(geometry['SRAD'], geometry['LAT'], geometry['LON'])

# interpolate
sat_x, sat_y, sat_z = uc.interpolate_sources(params['ns'], sat_x, sat_y, sat_z)

# convert from KM into M
sat_x, sat_y, sat_z = uc.km_to_m(sat_x, sat_y, sat_z)

params['spacing'] = uc.estimate_spacing(sat_x, sat_y, sat_z)
params['altitude'] = 1e3 * np.mean(geometry['SRAD']-geometry['MRAD'])

focused = sf.focus_rdrgrm(rdrgrm, params, st=150, en=650)

np.save("output/focused.npy", focused)
