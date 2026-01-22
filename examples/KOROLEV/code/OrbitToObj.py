import sys
import numpy as np
import pandas as pd

sys.path.append("../../src/PYTHON")
import kutil as ku
import output_handling as oh
import unit_convs as uc

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
sat_x, sat_y, sat_z = uc.interpolate_sources(2000, sat_x, sat_y, sat_z)

# convert from KM into M
sat_x, sat_y, sat_z = uc.km_to_m(sat_x, sat_y, sat_z)

# compute a normal vector for each observation
n_hat = ku.sharad_normal(sat_x, sat_y, sat_z, nmult=1)

# export as source file and obj
ku.sources_norms_to_file(DIRECTORY, OBS, sat_x, sat_y, sat_z, n_hat[:, 0], n_hat[:, 1], n_hat[:, 2]) 
ku.sources_norms_to_obj(DIRECTORY, OBS, sat_x, sat_y, sat_z, n_hat[:, 0], n_hat[:, 1], n_hat[:, 2])

