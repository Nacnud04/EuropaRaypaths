import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append("../../src/PYTHON")
import rdr_plots       as rp
import unit_convs      as uc
import output_handling as oh
import rdr_plots       as rp
import kutil           as ku

# Load Co-SHARPS downsampled orbit:
DIRECTORY = "data/Observation/rdrgrm"
OBS       = "00554201"

# load geometries (simulation source PKL, full track aeroid, full track MOLA)
geometry = ku.load_sharad_orbit_PKL(DIRECTORY, OBS)
aeroid = ku.load_sharad_orbit_AEROID(DIRECTORY, OBS)
mola = ku.load_sharad_orbit_MOLA(DIRECTORY, OBS)['TOPO']

# load and correct radargram
rdrgrm = ku.load_COSHARPS_rdrgrm(DIRECTORY, OBS)
NoOffset = ku.correct_rdrgrm(rdrgrm, aeroid)

# FIRST LOAD REAL DATA 
sharad_data_path = "data/Observation/rdr-cosharps/r_0554201_001_ss19_700_a.dat"

data = ku.load_SHARAD_RDR(sharad_data_path, st=18000, en=30000, latmin=70.768, latmax=74.2075)
#ku.plot_SHARAD_RDR(data, geometry)

par = oh.load_params("data/params.pkl", "data/Subsurface/KOR_T.txt")
par['ns'] = 2000
par['spacing'] = 225.758820 # spacing between sources in [m]

rdrgrm = np.load("output/rdrgrm.npy")
focused = np.load("output/focused.npy")
rx_win = np.load("data/rx_window_positions.npy")

rdr_db = uc.lin_to_db(np.abs(rdrgrm))
foc_db = uc.lin_to_db(np.abs(focused))

# what traces in the geometry file did we successfully simulate?
sim_st, sim_en = oh.get_simulation_range("rdrgrm")
sim_en += 1

# what range of traces in the real data do we want to interpolate?
trc_st = 750
trc_en = 1200

# what slantrange do we want to interpolate?
rea_rng_st = 314e3
rea_rng_en = 319e3

foc_intrp = oh.interpolate_rdrgrm(geometry, rx_win, foc_db,
                                  sim_st=sim_st, sim_en=sim_en,
                                  xst=trc_st, xen=trc_en,
                                  rea_rng_st=rea_rng_st, rea_rng_en=rea_rng_en)

correction = 9675
ymin, ymax = ku.corrected_ymin_ymax(rdrgrm, aeroid, trc_st, trc_en, correction)

# load in crater interior
korolev_interior = ku.import_korolev_interior("data/Subsurface/")
trc, depth = ku.clean_korolev_interior(korolev_interior, aeroid)

plotpar = {
    'obs': OBS,
    'savefig': "figures/KOROLEV.png",
    'trc_st': trc_st,
    'trc_en': trc_en,
    'rea_rng_st': rea_rng_st,
    'rea_rng_en': rea_rng_en,
    'ymin': ymin,
    'ymax': ymax,
    'rea_min': np.min(NoOffset),
    'rea_max': 0.001,
    'syn_min': 0,
    'syn_max': 8,
    "trc":trc,
    "depth":depth,
}

# plot
rp.plot_SHARAD_comparison(NoOffset, foc_intrp, geometry, aeroid, mola, plotpar)


# generate final plot
geometry = ku.load_sharad_orbit_PKL(DIRECTORY, OBS)
rp.plot_unfoc_foc(rdr_db, foc_db, rx_win, OBS, geometry=geometry, 
                    rdrmin=-15, rdrmax=5, 
                    focmin=plotpar['syn_min'], focmax=plotpar['syn_max'],
                    ymax=319)