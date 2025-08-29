import sys
sys.path.append("../../src")

from terrain import *
from surface import *
from source import *
from simulator import *
from focus import *

# --- GENERATE TERRAIN OBJECT ---

# terrain params
xmin, xmax = 0, 100
ymin, ymax = -5, 5
fs = 0.01
dims = (600, 600)

# make terrain object
terrain = Terrain(xmin, xmax, ymin, ymax, fs)

# generate sinusoid and normals
terrain.double_ridge(2, 2, 30, 20, 50)
#terrain.gen_flat(0)

# --- SET UP SOURCES AND TARGET ---

f0 = 5e9
B  = 6e9
N  = 250
stx, enx = 10, 90

ss = source_linspace('x', stx, enx, 0, 100, N, f0, B)

# target location
t = (50, 0, -2)   # target location (x, y, z)

# show terrain
terrain.show_2d_heatmap(show=False, savefig="terrain.png", ss=ss, t=t)
terrain.show_profile('x', 0, savefig="doubleridgeprofile.png")

# --- LOOK AT A CHIRP ---

ss[0].plot(savefig="chirp.png")

# --- CHANGE SIM PARS ---

c        = 299792458   # speed of light [m/s]
lam      = 0.06        # wavelength [m]
rng_st   = 96          # starting range [m]
rng_en   = 108         # ending range [m]
rng_res  = c / (2 * B) # range resolution [m]
sampl    = 25e9        # sampling rate [Hz]
dt       = 1 / sampl   # time interval [s]
dt_m     = dt * c      # space interval [m]

# how many range bins?
rb = int(((rng_en - rng_st) / c) / (1 / sampl))

par = {
    "rx_window_offset": rng_st,   # [m]
    "rx_window_m"     : rng_en-rng_st,   # [m]
    "sampling"        : sampl,  # [Hz]
    "range_resolution": rng_res,  # [m]
    "lambda"          : 0.06, # [m]
    "aspect"          : 2000,   # [.]
    "surf_gain"       : 96,
}

rdrgrm, ts = run_sim_terrain(terrain, dims, ss, t, reflect=True, polarization='h',
                             xmin=10, xmax=90, gpu=True, savefig="rdrgrm-doubleridge.png", 
                             rough=False, nsmpl=rb, par=par)

# --- ATTEMPT TO FOCUS --

sx = np.linspace(stx, enx, N)
sz = 100
c1 = 299792458
c2 = c1 / np.sqrt(3.15)
slantrange_est = est_slant_range(sx, sz, t[0], t[2], c1, c2)
slantrange_time = 2 * 10**6 * slantrange_est / c1

slant_rb = ((slantrange_est - rng_st) // dt_m).astype(int)

k = (2 * np.pi) / par['lambda']

match_filter = np.exp(-2j * k * slantrange_est)

focused = focus_jit(rdrgrm, slant_rb, match_filter, rb)

extent = (10, 90, 2*(rng_en/c)*10**6, 2*(rng_st/c)*10**6)
fig, ax = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, dpi=300, sharex=True)

im0 = ax[0].imshow(np.abs(rdrgrm), cmap="gray", aspect=par['aspect']/3, extent=extent)
im1 = ax[1].imshow(np.abs(focused), cmap="gray", aspect=par['aspect']/3, extent=extent)

labels = ["(a) Unfocused", "(b) Focused"]
for a, label in zip(ax, labels):
    a.set_ylabel("Range [µs]", fontsize=11)
    a.tick_params(axis="both", which="major", labelsize=9, direction="in")
    a.tick_params(axis="both", which="minor", direction="in")
    a.text(0.02, 0.95, label, transform=a.transAxes, fontsize=11,
           fontweight="bold", va="top", ha="left", color="black",
           bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

ax[1].set_xlabel("Azimuth [m]", fontsize=11)

cbar = fig.colorbar(im1, ax=ax, orientation="vertical", fraction=0.025, pad=0.04)
cbar.set_label("Power [W]", fontsize=11)
cbar.ax.tick_params(labelsize=9)

plt.savefig("focused.png", dpi=300, bbox_inches="tight")
plt.close()
