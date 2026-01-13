import sys
import pickle
import numpy as np
import json

import matplotlib.pyplot as plt

params = {

    # radar parameters
    "power": 100,             # Transmitter power [W]
    "frequency": 9e6,         # Radar frequency [Hz]
    "bandwidth": 1e6,         # Radar bandwidth [Hz]
    "surface_gain": 55,       # Antenna gain [dB]
    "subsurface_gain": 70,   # Subsurface antenna gain [dB]
    "range_resolution": 300,  # range resolution [m]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)
    "aperture": 7,           # aperture (from nadir->edge) [deg]

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

    # actual source path
    "source_path_file": "inputs/source_path.txt",

    # source altitude (use max altitude)
    "altitude": 10e3,        # source altitude above surface [m]
    # (3691.066-3378.617)*1e3 <- this is for MARS.

    # facet array params
    "fs": 5,

    # target params
    "rerad_funct": 1,  # 1-degree boxcar

    # processing parameters (BOOLEAN)
    "convolution": True,   # use convolution-based processing
    "convolution_linear": True,  # use linear convolution instead of circular

}

with open("inputs/params.json", "w") as f:
    json.dump(params, f, indent=4)

with open("inputs/params.pkl", 'wb') as hdl:
    pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

# --- MAKE FACET FILE
sys.path.append("../../../archive/src")
from terrain import Terrain

ox = -5e3
oy = -1e3
oz = 0
nx = 2000
ny = 400

xmin, xmax = ox, ox+nx*params["fs"]
ymin, ymax = oy, oy+ny*params["fs"]

terrain = Terrain(xmin, xmax, ymin, ymax, params["fs"])
terrain.gen_flat(0)

# extract facets and normals from terrain object
xx, yy, zz = terrain.XX, terrain.YY, terrain.zs
norms      = np.reshape(terrain.normals, (xx.shape[0]*xx.shape[1], 3))
uvecs      = np.reshape(terrain.uvectors, (xx.shape[0]*xx.shape[1], 3))
vvecs      = np.reshape(terrain.vvectors, (xx.shape[0]*xx.shape[1], 3))

# now we rotate by a 45 degree inclination angle about the x axis
# (this is a rotation matrix for a 45 degree rotation about the x-axis)
theta = np.radians(45)
rot = np.array([[1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]])

norms = norms @ rot.T
uvecs = uvecs @ rot.T
vvecs = vvecs @ rot.T

# now rotate facet positions
pts = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
pts = pts @ rot.T

# write output facet data
filename = "inputs/facets.fct"
with open(filename, 'w') as f:
    i = 0
    for (x, y, z), (nx, ny, nz), (ux, uy, uz), (vx, vy, vz) in zip(pts, norms, uvecs, vvecs):
        if i % 100 == 0:
            print(f"Writing out {filename}: {round((100*(i+1))/pts.shape[0], 2)} %", end="    \r")
        f.write(f"{x},{y},{z}:{nx},{ny},{nz}:{ux},{uy},{uz}:{vx},{vy},{vz}\n")
        i += 1

# --- MAKE SOURCE PATH ---

# source parameters 
sy = 0                # source y location       [m]
sz = 10e3             # source z location       [m]
sdx = 10              # source x discretization [m]
sx0 = -5e3            # source x origin         [m]
ns = 1000             # source count            [.]

sx = np.linspace(sx0, sx0 + sdx*(ns-1), ns)
sy = sy * np.ones(ns)
sz = sz * np.ones(ns)
spos = np.vstack((sx, sy, sz)).T
spos = spos @ rot.T

snx = 0 * np.ones(ns)
sny = 0 * np.ones(ns)
snz = 1 * np.ones(ns)
snorms = np.vstack((snx, sny, snz)).T
snorms = snorms @ rot.T

# export
with open("inputs/source_path.txt", 'w') as f:
    for i in range(ns):
        f.write(f"{spos[i,0]},{spos[i,1]},{spos[i,2]},{snorms[i,0]},{snorms[i,1]},{snorms[i,2]}\n")

# --- MAKE TARGET FILE ---
# we give the target the same normal as the source
tpos = np.array([[0, 0, -3e3]])
tpos = tpos @ rot.T
with open("inputs/targets.txt", 'w') as f:
    for i in range(tpos.shape[0]):
        f.write(f"{tpos[i,0]},{tpos[i,1]},{tpos[i,2]},{snorms[i,0]},{snorms[i,1]},{snorms[i,2]}")

fig, ax = plt.subplots(figsize=(12, 12))
ax.scatter(spos[:,1], spos[:,2], c='b', s=1, label="Source Path")
ax.scatter(pts[:,1], pts[:,2], c='k', s=1, label="Facets")
ax.scatter(tpos[0,1], tpos[0,2], c='r', s=10, label="Target")
ax.set_xlabel("Y (m)")
ax.set_ylabel("Z (m)")
ax.set_aspect('equal')
plt.legend()
plt.savefig("figures/source_path.png")
plt.close()
