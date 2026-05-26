import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("../../src/PYTHON")
import unit_convs      as uc
import kutil           as ku

DIRECTORY = "data/Observation/rdrgrm"
OBS       = "00554201"
aeroid = ku.load_sharad_orbit_AEROID(DIRECTORY, OBS)
rdrgrm = ku.load_COSHARPS_rdrgrm(DIRECTORY, OBS)
NoOffset = ku.correct_rdrgrm(rdrgrm, aeroid)
st = 750; en = 1175
NoOffset = NoOffset[2000:2600, st:en]
#plt.imshow(NoOffset, vmin=np.min(NoOffset), vmax=0.001)
#plt.show()

# --- PANEL 0 ---
#focused = np.load("output/focused.npy")

# convert to dB
#focused_db = uc.lin_to_db(np.abs(focused))

# mask away the subsurface
#focused_db[488:, :] = np.min(focused_db)
focused_db = uc.lin_to_db(NoOffset)

# gen extent
rx_win = np.loadtxt("data/rx_window_positions.txt")
win_mean = np.mean(rx_win)
alt = 316e3
fs = 40e6
extent = [st, en, (win_mean + (focused_db.shape[0] * (299792458 / fs))), win_mean]
extent[2] /= 299.792458
extent[3] /= 299.792458

xs = np.arange(st, en)

# normalize for plotting
norm_foc_rdr = focused_db - np.max(focused_db)

fig, ax = plt.subplots(4, sharex=True, figsize=(9, 12), constrained_layout=True)
im = ax[0].imshow(norm_foc_rdr, aspect="auto", extent=extent, vmin=-30, vmax=-5)
#ax[0].set_ylim(1060, 1050)
ax[0].set_ylabel("Range [us]", fontsize=10)
fig.colorbar(im, ax=ax[0], label="Normalized Power [dB]", pad=0.01)

# --- PANEL 1 ---
# get the maximum surface power per column
surface_max = np.nanmax(focused_db, axis=0)
psrf_lin = uc.db_to_lin(surface_max)
sMax = np.nanmax(surface_max)
sMax_lin = np.nanmax(psrf_lin)
surface_max -= sMax

sMaxLoc = np.argmax(surface_max)
print(f"Maximum surface power is {sMax:.2f} dB at trace {sMaxLoc}")

ax[1].plot(xs, surface_max, color="red")
ax[1].set_ylabel(r"$P-P_{max}$ [dB]", fontsize=10)

# --- PANEL 2 ---
# the ratio of power is equivalent to the ratio of squared areas
# so the square root of the power ratio is the ratio of areas
# (assuming everything else is constant)
area_ratio = np.sqrt(uc.db_to_lin(surface_max))
ax[2].fill_between(xs, np.ones_like(area_ratio), area_ratio, color="red", alpha=0.2)
ax[2].plot(xs, area_ratio, color="red")
ax[2].axhline(1, color="black", linewidth=2)
ax[2].set_ylabel(r"$A_c\;/\;A_{c,max}$", fontsize=10)

# --- PANEL 3 ---
# assuming everything else is constant, the ratio of power is  
# equivalent to the ratio of reflection coefficients.
refl_ratio = uc.db_to_lin(surface_max)
ax[3].fill_between(xs, np.ones_like(refl_ratio), refl_ratio, color="red", alpha=0.2)
ax[3].plot(xs, refl_ratio, color="red")
ax[3].axhline(1, color="black", linewidth=2)
ax[3].set_ylabel(r"$|\Gamma^2|\;/\;|\Gamma^2|_{max}$", fontsize=10)
ax[3].set_xlabel("Trace number", fontsize=10)

# add subplot labels
labels = ["a", "b", "c", "d"]
for i in range(4):
    ax[i].text(0.02, 0.05, labels[i], fontsize=18, fontweight="bold", transform=ax[i].transAxes, color="black", 
                bbox=dict(
                facecolor="white",   # Background color
                edgecolor="white",  # Border color
                boxstyle="round,pad=0.3",
                alpha=0.5))

plt.savefig("figures/PSS_SHARAD.png", dpi=300)
plt.show()

