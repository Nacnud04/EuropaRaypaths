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

rdrgrm = oh.compile_rdrgrm("rdrgrm", params)
print(f"Radargram shape: {rdrgrm.shape}")

np.save("output/rdrgrm.npy", rdrgrm)

print(sys.argv)

if len(sys.argv) > 1:
    if sys.argv[1] == "plot":

        print("Plotting only no focusing...")

        rdr_db = uc.lin_to_db(np.abs(rdrgrm))

        fig, ax = plt.subplots(1, 1, figsize=(4, 12))
        im = ax.imshow(rdr_db, vmin=-20, vmax=-4, cmap="viridis", aspect=3)
        plt.colorbar(im)
        plt.savefig("figures/rdrgrm.png")
        plt.close()
        
        sys.exit()

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

dx = sat_x - np.roll(sat_x, 1)
dy = sat_y - np.roll(sat_y, 1)
dz = sat_z - np.roll(sat_z, 1)
d = np.sqrt(dx**2 + dy**2 + dz**2)
spacing = np.nanmedian(d)
print(f"Spacing={spacing} m")
params['spacing'] = spacing
params['altitude'] = 1e3 * np.mean(geometry['SRAD']-geometry['MRAD'])

focused = sf.focus_rdrgrm(rdrgrm, params)

np.save("output/focused.npy", focused)
