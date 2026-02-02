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
correction = 9675
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
foc_db = uc.lin_to_db(np.abs(focused))

# interpolate to match
sim_st, sim_en = 55, 1628
geom_crop = geometry.iloc[sim_st:sim_en]
intrp_st, intrp_en = geom_crop.index[0], geom_crop.index[-1]+1

sltrng = 4e3
samples = int(sltrng // (RNG_BIN_INT*c/2))
foc_rng_st = np.min(rx_win)
foc_rng_en = np.min(rx_win) + 7.5e3
foc_rng    = np.linspace(foc_rng_st, foc_rng_en, focused.shape[0])
rea_rng_st = 314e3
rea_rng_en = 318e3
rea_rng    = np.linspace(rea_rng_st, rea_rng_en, samples)

# interpolate (iterate over traces)
xst = 750; xen = 1200
foc_intrp = np.zeros((samples, xen - xst))
for t in range(xen - xst):

    t_full = t + 750
    
    # stack traces
    trace_ids = [i-intrp_st for i, c in geom_crop.iterrows() if int(c['COL']) == t_full]
    trace_stack = np.zeros_like(foc_db[:,0])
    if len(trace_ids) == 0:
        print(f"Warning: No traces found for trace {t_full}")
    for tid in trace_ids:
        clean = foc_db[:, tid]
        clean[np.isnan(clean)] = np.nanmin(clean)
        trace_stack += clean
    trace_stack /= len(trace_ids)

    # interpolate
    foc_intrp[:, t] = np.interp(rea_rng, foc_rng, trace_stack)

# clean up infs and nans
foc_intrp[np.isinf(foc_intrp)] = np.nanmin(foc_intrp[np.isfinite(foc_intrp)])
foc_intrp[np.isnan(foc_intrp)] = np.nanmin(foc_intrp[np.isfinite(foc_intrp)])

extent_syn = [
    xst, xen,
    rea_rng_en, rea_rng_st,
]

mola_interp = np.interp(np.linspace(0, 1, 2000), np.linspace(0, 1, 1200-750), aeroid['SRAD'][750:1200] * 1e3 - mola[750:1200])

# crop no offset
rbCROP_st = int((rea_rng_st - ymax) // (RNG_BIN_INT*c/2))
rbCROP_en = int((rea_rng_en - ymax) // (RNG_BIN_INT*c/2))
NoOffset_crop = NoOffset[rbCROP_st:rbCROP_en-1, xst:xen]

# adjust if necessary
if NoOffset_crop.shape[1] > foc_intrp.shape[1]:
    NoOffset_crop = NoOffset_crop[:, :foc_intrp.shape[1]]
if foc_intrp.shape[0] > NoOffset_crop.shape[0]:
    foc_intrp = foc_intrp[:NoOffset_crop.shape[0], :]

# scale rdr images
rea_scl = uc.scale_range(NoOffset_crop, np.min(NoOffset_crop), 0.001)
syn_min, syn_max = 35, 45
foc_scl = uc.scale_range(foc_intrp, syn_min, syn_max)

print(np.min(rea_scl), np.max(rea_scl))
print(np.min(foc_scl), np.max(foc_scl))

# rgb color scale
rea_clr = np.array([0xFF, 0x00, 0x80])
foc_clr = np.array([0x00, 0xFF, 0x80])

# create rgb images
rgb = np.zeros((rea_scl.shape[0], rea_scl.shape[1], 3))

for i in range(3):
    rgb[:, :, i] += rea_scl * rea_clr[i]
    rgb[:, :, i] += foc_scl * foc_clr[i]

# blue channel can wrap over so lets fix that
rgb[:, :, 2][rgb[:, :, 2] > 255] = 255

# export
plt.imsave("figures/rdr_comparison.png", rgb.astype(np.uint8))



fig, ax = plt.subplots(figsize=(8, 9))

extent_syn = [
    np.max(aeroid['LAT']),
    np.min(aeroid["LAT"]),
    rea_rng_en/1e3,
    rea_rng_st/1e3,
]

im = ax.imshow(rgb.astype(np.uint8), extent=extent_syn, aspect="auto")

# --- COLORBARS -----------------------------------------------------

divider = make_axes_locatable(ax)

# Place both bars close to the image
cax1 = divider.append_axes("right", size="3%", pad=0.05)
cax2 = divider.append_axes("right", size="3%", pad=0.4)

# Build exact colormaps
rea_clr = np.array([255, 0, 128]) / 255
foc_clr = np.array([0, 255, 128]) / 255

cmap_rea = mcolors.LinearSegmentedColormap.from_list("rea_map", [(0,0,0), rea_clr])
cmap_foc = mcolors.LinearSegmentedColormap.from_list("foc_map", [(0,0,0), foc_clr])

mappable_rea = plt.cm.ScalarMappable(
    norm=plt.Normalize(vmin=np.min(rea_scl), vmax=np.max(rea_scl)),
    cmap=cmap_rea
)
mappable_rea.set_array([])

mappable_foc = plt.cm.ScalarMappable(
    norm=plt.Normalize(vmin=np.min(foc_scl), vmax=np.max(foc_scl)),
    cmap=cmap_foc
)
mappable_foc.set_array([])

# Create colorbars
cbar1 = fig.colorbar(mappable_foc, cax=cax1)
cbar1.set_label("Synthetic Power [dB]", rotation=270, labelpad=-6)

cbar2 = fig.colorbar(mappable_rea, cax=cax2)
cbar2.set_label("Real Power [?]", rotation=270, labelpad=-6)

ticklocs = [0, 0.25, 0.75, 1.0]
cbar1.set_ticks(ticklocs)
cbar1.set_ticklabels([f"{t * (syn_max - syn_min) + syn_min:.1f}" for t in ticklocs], fontsize=8)

cbar2.set_ticks(ticklocs)
cbar2.set_ticklabels([f"{t:.1f}" for t in ticklocs], fontsize=8)

# --- LABELS --------------------------------------------------------

ax.set_xlabel("Latitude [deg]")
ax.set_ylabel("Range [km]")
ax.set_title("SHARAD Obs. 0554201", fontsize=18, fontweight="bold")

plt.savefig("figures/KOROLEV.png", dpi=300, bbox_inches="tight")
plt.close()




# generate final plot
geometry = ku.load_sharad_orbit_PKL(DIRECTORY, OBS)

extent = [
    0, Na,
    np.min(rx_win)/1e3 + 7.5, np.min(rx_win)/1e3,
]

fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharey=True)

# rdrgrms
im1 = ax[0].imshow(rdr_db, vmin=22, vmax=35, cmap="viridis", aspect=250, extent=extent)
plt.colorbar(im1, label="dB", shrink=0.9)
im2 = ax[1].imshow(uc.lin_to_db(np.abs(focused)), extent=extent, vmin=35, vmax=45, aspect=250)
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

