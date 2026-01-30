import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("../../src/PYTHON")
import rdr_plots       as rp
import unit_convs      as uc
import output_handling as oh
import rdr_plots       as rp
import kutil           as ku

# Load Co-SHARPS downsampled orbit:
DIRECTORY = "data/Observation/rdrgrm"
OBS       = "00554201"
geometry = ku.load_sharad_orbit_PKL(DIRECTORY, OBS)
# THE MARS RADIUS IS THE AEROID??????
aeroid = ku.load_sharad_orbit_AEROID(DIRECTORY, OBS)
RX_WIN_OPEN = aeroid['MRAD']*1e3#aeroid['SRAD']*1e3 - aeroid['MRAD']*1e3

mola = ku.load_sharad_orbit_MOLA(DIRECTORY, OBS)['TOPO']

c = 299792458
RNG_BIN_INT = 0.0375e-6
rollpar = aeroid['AEROID']-aeroid['SRAD']*1e3
roll = (rollpar - np.min(rollpar)) // (RNG_BIN_INT * c / 2)

rdrgrm = ku.load_COSHARPS_rdrgrm(DIRECTORY, OBS)

NoOffset = np.zeros_like(rdrgrm)
for i, shift in enumerate(roll):
    NoOffset[:, i] = np.roll(rdrgrm[:, i], -1*shift)

#extent = [0, rdrgrm.shape[1], 
#          np.min(RX_WIN_OPEN)+rdrgrm.shape[0]*RNG_BIN_INT*c/2,
#          np.min(RX_WIN_OPEN)]

RX_WIN_OPEN -= 3396e3

MRAD = 3396e3
srad = np.max(aeroid['SRAD'][750:1200]) * 1e3

ymin = np.min(RX_WIN_OPEN)-rdrgrm.shape[0]*RNG_BIN_INT*c/2
ymax = np.min(RX_WIN_OPEN)

# correction factor
correction = 9650
ymin = srad - (ymin + correction + MRAD)
ymax = srad - (ymax + correction + MRAD)

# correct for spacecraft altitude
extent_real = [0, rdrgrm.shape[1], 
         ymin, ymax]

# FIRST LOAD REAL DATA 
sharad_data_path = "data/Observation/rdr-cosharps/r_0554201_001_ss19_700_a.dat"

data = ku.load_SHARAD_RDR(sharad_data_path, st=18000, en=30000, latmin=70.768, latmax=74.2075)
ku.plot_SHARAD_RDR(data, geometry)

par = oh.load_params("data/params.pkl", "data/Subsurface/KOR_T.txt")
par['ns'] = 2000
par['spacing'] = 225.758820 # spacing between sources in [m]

rdrgrm = np.load("output/rdrgrm.npy")
focused = np.load("output/focused.npy")
rx_win = np.load("data/rx_window_positions.npy")

Nr, Na = rdrgrm.shape

rdr_db = uc.lin_to_db(np.abs(rdrgrm))

extent_syn = [
    0, Na,
    np.min(rx_win) + 7.5e3, np.min(rx_win),
]

#print(srad - mola[750:1200])
#print(geometry['SRAD'] geometry['TOPO'])

mola_interp = np.interp(np.linspace(0, 1, 2000), np.linspace(0, 1, 1200-750), aeroid['SRAD'][750:1200] * 1e3 - mola[750:1200])

fig, ax = plt.subplots(2, 1, figsize=(11, 15))
ax[0].imshow(NoOffset, vmax=0.01, extent=extent_real, aspect=0.1)
ax[0].plot(aeroid['SRAD'][750:1200]*1e3 - mola, color="red", linewidth=1)
ax[0].set_xlim(750, 1200)
ax[0].set_ylim(318e3, 314e3)
ax[1].imshow(uc.lin_to_db(np.abs(focused)), extent=extent_syn, vmin=-5, vmax=10, aspect=0.35)
ax[1].plot(mola_interp[55:1550], color="red", linewidth=1)
ax[1].set_ylim(318e3, 314e3)
plt.savefig("tmp.png")
plt.close()

sys.exit()

fig, ax = plt.subplots(1, 1, figsize=(4, 12))
im = ax.imshow(rdr_db, vmin=-20, vmax=-4, cmap="viridis", aspect=3)
plt.colorbar(im)
plt.savefig("figures/rdrgrm.png")
plt.close()

# generate final plot
geometry = ku.load_sharad_orbit_PKL(DIRECTORY, OBS)

extent = [
    0, Na, #(Na * par['spacing'])/1e3,
    np.min(rx_win)/1e3 + 7.5, np.min(rx_win)/1e3,
]

fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharey=True)

# rdrgrms
im1 = ax[0].imshow(rdr_db, vmin=-20, vmax=-4, cmap="viridis", aspect=250, extent=extent)
plt.colorbar(im1, label="dB", shrink=0.9)
im2 = ax[1].imshow(uc.lin_to_db(np.abs(focused)), extent=extent, vmin=-5, vmax=10, aspect=250)
plt.colorbar(im2, label="dB", shrink=0.9)

# labels
for a in ax: a.set_ylabel("Range [km]")
ax[1].set_xlabel("Echo Count")
ax[0].text(
    -0.16, 0.5, "Unfocused",
    transform=ax[0].transAxes,
    rotation=90,
    va="center",
    ha="center",
    fontsize=12,
    fontweight="bold"
)
ax[1].text(
    -0.16, 0.5, "Focused",
    transform=ax[1].transAxes,
    rotation=90,
    va="center",
    ha="center",
    fontsize=12,
    fontweight="bold"
)
plt.suptitle("SHARAD 0554201", fontsize=18, fontweight="bold")

# cropping
plt.ylim(318, 314.15)

# crop geometry to just what successfully simulated
# upsample the geometry range
srange = np.interp(np.linspace(0, 1, 2000), np.linspace(0, 1, len(geometry)), geometry["SRANGE"]/1e3)
for a in ax:
   a.plot(srange[55:1550], color="red", linewidth=1) 

# export
plt.tight_layout(rect=(0.08, 0, 1, 1))
plt.savefig("figures/output.png")
plt.close()

