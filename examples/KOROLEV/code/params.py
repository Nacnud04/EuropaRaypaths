import sys
import pickle
import numpy as np
import json

sys.path.append("../../src/PYTHON")
import kutil           as ku
import output_handling as oh
import unit_convs as uc

params = {

    # radar parameters
    "power": 100,             # Transmitter power [W]
    "frequency": 20e6,        # Radar frequency [Hz]
    "bandwidth": 10e6,        # Radar bandwidth [Hz]
    "surface_gain": 20,       # Antenna gain [dB]
    "subsurface_gain": 35,    # Subsurface antenna gain [dB]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)
    "aperture": 1,            # aperture (from nadir->edge) [deg]

    # receive window parameters
    "rx_window_m":  7.5e3,         # receive window length [m]
    "rx_window_offset_m": 300e3,  # receive window offset [m]
    "rx_sample_rate": 40e6,       # receive sample rate [Hz]
    "rx_window_position_file": "data/rx_window_positions.txt",

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
    "source_path_file": "data/Observation/s_00554201_srcs.txt",

    # facet params
    "altitude": 316e3,           # approximate altitude [m]
    "fs": 115.66,              # facet size [m]

    # target params
    "rerad_funct": 2,  # 3-degree boxcar

    # processing parameters (BOOLEAN)
    "convolution": True,   # use convolution-based processing
    "convolution_linear": True,  # use linear convolution instead of circular

}

with open("data/params.json", "w") as f:
    json.dump(params, f, indent=4)

with open("data/params.pkl", 'wb') as hdl:
    pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

# --- EXPORT RX OPENING WINDOW ---

NS = 2000
sharad_data_path = "data/Observation/rdr-cosharps/r_0554201_001_ss19_700_a.dat"
data = ku.load_SHARAD_RDR(sharad_data_path, st=18000, en=30000, latmin=70.768, latmax=74.2075)
ku.rxOpenWindow(data, "data/rx_window_positions", NS)

# estimate the effective PRF
duration = data['EPHEMERIS_TIME'][-1] - data['EPHEMERIS_TIME'][0]
print(f"Duration of observation is {duration} seconds")
prf = NS / duration
print(f"Therefore the PRF is {prf} Hz")
prf_data = len(data) / duration
print(f"The input (RDR) data has an approximate prf of {prf_data} Hz")

# --- GENERATE SOURCE FILE ---

DIRECTORY = "data/Observation/rdrgrm"
OBS       = "00554201"

# load in dataset
geometry = ku.load_sharad_orbit_MOLA(DIRECTORY, OBS)

# restrict to part of radargram within interest
mincol = 750
maxcol = 1200
geometry = geometry[(geometry['COL'] > mincol) * (geometry['COL'] < maxcol)]

prf_data = len(geometry) / duration
print(f"The input (Co-SHARPS) data has an approximate prf of {prf_data} Hz")

geometry = uc.upsample_df(NS, geometry)

# save geometry as a pickle
with open(f'data/Observation/rdrgrm/s_{OBS}_sources.pkl', 'wb') as file:
    pickle.dump(geometry, file)

# convert the satellite location into x, y, z
sat_x, sat_y, sat_z = ku.planetocentric_to_cartesian(geometry['SRAD'], geometry['LAT'], geometry['LON'])

# interpolate
sat_x, sat_y, sat_z = uc.interpolate_sources(NS, sat_x, sat_y, sat_z)

# convert from KM into M
sat_x, sat_y, sat_z = uc.km_to_m(sat_x, sat_y, sat_z)

# compute a normal vector for each observation
n_hat = ku.sharad_normal(sat_x, sat_y, sat_z, nmult=1)


# export as source file and obj
ku.sources_norms_to_file(DIRECTORY, OBS, sat_x, sat_y, sat_z, n_hat[:, 0], n_hat[:, 1], n_hat[:, 2]) 
ku.sources_norms_to_obj(DIRECTORY, OBS, sat_x, sat_y, sat_z, n_hat[:, 0], n_hat[:, 1], n_hat[:, 2])

# MAKE SUBSURFACE TARGETS

# make some test subsurface targets directly beneath source
tar_x, tar_y, tar_z = ku.planetocentric_to_cartesian(geometry['SRAD']-317, geometry['LAT'], geometry['LON'])
tar_x = tar_x[500:1500]
tar_y = tar_y[500:1500]
tar_z = tar_z[500:1500]

# convert from KM into M
tar_x, tar_y, tar_z = uc.km_to_m(tar_x, tar_y, tar_z)

# compute a normal vector for each observation
#n_hat = ku.sharad_normal(tar_x, tar_y, tar_z, nmult=1)


# export as source file and obj
ku.target_norms_to_obj("data/Subsurface", "KOR_T", tar_x, tar_y, tar_z, n_hat[:, 0], n_hat[:, 1], n_hat[:, 2], norms=False)
ku.target_norms_to_file("data/Subsurface", "KOR_T", tar_x, tar_y, tar_z, n_hat[:, 0], n_hat[:, 1], n_hat[:, 2])


"""
# MAKE BASIC SIMPLE SURFACE
# make some test subsurface targets directly beneath source
tar_x, tar_y, tar_z = ku.planetocentric_to_cartesian(geometry['SRAD']-315, geometry['LAT'], geometry['LON'])

# convert from KM into M
tar_x, tar_y, tar_z = uc.km_to_m(tar_x, tar_y, tar_z)

# compute a normal vector for each observation
n_hat = ku.sharad_normal(tar_x, tar_y, tar_z, nmult=1)

# create a basis
ref = np.zeros_like(n_hat)
ref[:, 2] = 1.0  # z-axis

# if n is too close to z, switch to x-axis
ref_mask = np.abs(n_hat[:, 2]) > 0.9
ref[ref_mask] = np.array([1.0, 0.0, 0.0])

# first tangent: u_hat = normalize(ref * n)
u_hat = np.cross(ref, n_hat)
u_hat /= np.linalg.norm(u_hat, axis=1, keepdims=True)

# second tangent: v_hat = n * u_hat
v_hat = np.cross(n_hat, u_hat)

# export as source file and obj
#ku.target_norms_to_file("data/Subsurface", "KOR_F", tar_x, tar_y, tar_z, n_hat[:, 0], n_hat[:, 1], n_hat[:, 2])
with open("data/Subsurface/KOR_F.fct", 'w') as f:
    i = 0
    for x, y, z, nx, ny, nz, ux, uy, uz, vx, vy, vz in zip(tar_x, tar_y, tar_z, n_hat[:, 0], n_hat[:, 1], n_hat[:, 2], u_hat[:, 0], u_hat[:, 1], u_hat[:, 2], v_hat[:, 0], v_hat[:, 1], v_hat[:, 2]):
        if i != len(tar_x) - 1:
            f.write(f"{x:.6f},{y:.6f},{z:.6f}:{ux:.6f},{uy:.6f},{uz:.6f}:{vx:.6f},{vy:.6f},{vz:.6f}\n")
        else:
            f.write(f"{x:.6f},{y:.6f},{z:.6f}:{ux:.6f},{uy:.6f},{uz:.6f}:{vx:.6f},{vy:.6f},{vz:.6f}")
        i += 1
    print(f"Exported facet data to: data/Subsurface/KOR_F.fct")
"""