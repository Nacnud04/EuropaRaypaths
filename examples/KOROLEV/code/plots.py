import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("../../src/PYTHON")
import rdr_plots       as rp
import unit_convs      as uc
import output_handling as oh
import rdr_plots       as rp

par = oh.load_params("data/params.pkl", "data/Subsurface/KOR_T.txt")
par['ns'] = 2000
par['spacing'] = 225.758820 # spacing between sources in [m]

rdrgrm = np.load("output/rdrgrm.npy")
focused = np.load("output/focused.npy")

Nr, Na = rdrgrm.shape

rdr_db = uc.lin_to_db(np.abs(rdrgrm))

fig, ax = plt.subplots(1, 1, figsize=(4, 12))
im = ax.imshow(rdr_db, vmin=-20, vmax=-4, cmap="viridis", aspect=3)
plt.colorbar(im)
plt.savefig("figures/rdrgrm.png")
plt.close()

# generate final plot
extent = [
    0, (Na * par['spacing'])/1e3,
    (par['rx_window_offset_m']+par['rx_window_m'])/1e3, (par['rx_window_offset_m'])/1e3,
]

fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharey=True)

# rdrgrms
im1 = ax[0].imshow(rdr_db, vmin=-20, vmax=-4, cmap="viridis", aspect=50, extent=extent)
plt.colorbar(im1, label="dB", shrink=0.9)
im2 = ax[1].imshow(uc.lin_to_db(np.abs(focused)), extent=extent, vmin=-5, vmax=10, aspect=50)
plt.colorbar(im2, label="dB", shrink=0.9)

# labels
for a in ax: a.set_ylabel("Range [km]")
ax[1].set_xlabel("Azimuth [km]")
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
plt.ylim(316.8, 312.5)

# export
plt.tight_layout(rect=(0.08, 0, 1, 1))
plt.savefig("figures/output.png")
plt.close()

