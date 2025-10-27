import numpy as np
import glob, sys, pickle, os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

os.environ["PATH"] += os.pathsep + '/usr/share/texlive/texmf-dist/tex/xelatex'

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


# try to focus the radargram
sys.path.append("../../../src")
from focus import est_slant_range
from util import mag_to_db

# load parameters from pickle
with open("params/dem.pkl", 'rb') as hdl:
    params = pickle.load(hdl)

params['aspect'] = 0.5

lbl="dem"

filenames = glob.glob(f"rdrgrm/{lbl}/s*.txt")

# sort filenames to ensure correct order
filenames.sort()

rdrgrm = []

for i, f in enumerate(filenames):
    if i < params['ns']:
        arr = np.loadtxt(f).T
        trc = arr[0] + 1j * arr[1]
        if np.isnan(trc).any():
            print(f"Found NaN in file: {f}")
        rdrgrm.append(trc)

rdrgrm = np.array(rdrgrm).T

sx = params['sx0'] + params['sdx'] * np.arange(params['ns']) # source x locations [m]
sy = params['sy']                          # source y locations [m]
sz = params['sz']                          # source z locations [m]

c1 = 299792458
c2 = c1 / np.sqrt(params["eps_2"])

rb = int((params["rx_window_m"] / c1) / (1 / params["rx_sample_rate"]))
dm = c1 / params["rx_sample_rate"]

sltrng   = est_slant_range(sx, sz, params["tx"], params["tz"], c1, c2, trim=False)
sltrng_t = 2 * 10**6 * sltrng / c1  # in microseconds

# compute sample-bin indices (meters -> samples)
slt_rb = ((sltrng - params["rx_window_offset_m"]) // dm).astype(int)

k = (2 * np.pi) / (c1 / params["frequency"])
match_filter = np.exp(-2j * k * sltrng)

def lin_to_db(x):
    return 10 * np.log10(x)

lst = [lin_to_db(np.abs(rdrgrm)), np.angle(rdrgrm)]
names = ["rdrgrmAbs.png", "rdrgrmPhase.png"]
cmaps = ["viridis", "twilight"]
cbar_labels = ["Power [dB]", "Phase [rad]"]
vmin  = [-42, None]

for arr, name, cmap, cbar_label, v in zip(lst, names, cmaps, cbar_labels, vmin):
    plt.imshow(arr, aspect='auto', cmap=cmap, interpolation='nearest', vmin=v,
            extent=[params["sx0"]/1e3, (params["sx0"] + params["sdx"] * params["ns"])/1e3, 2*(params["rx_window_offset_m"] + params["rx_window_m"])/299.792458, 2*params["rx_window_offset_m"]/299.792458])
    plt.colorbar(label=cbar_label)
    plt.xlabel("Azimuth [km]")
    plt.ylabel("Range [us]")
    plt.savefig(f"figures/{lbl}-{name}")
    plt.close()


Nr, Na = rdrgrm.shape

# some buffer on each side for which we don't roll or apply filter
buffer = 2000

# --- 2. Range Cell Migration Correction (RCMC) ---
shift_amounts = slt_rb - np.min(slt_rb)

# get rid of shift outside of center region
shift_amounts[:buffer] = 0
shift_amounts[-buffer:] = 0

rolled_matrix = np.array([
    np.roll(rdrgrm[:, i], -int(shift_amounts[i]))
    for i in range(rdrgrm.shape[1])
]).T

# --- 3. Azimuth matched filter ---
k = (2 * np.pi) / (c1 / params["frequency"])
match_filter = np.exp(-2j * k * sltrng)

# --- 1. Azimuth FFT ---
fft_len = int(2 * Na)
pad = fft_len - Na

# Pad only at the end
rolled_matrix = np.pad(rolled_matrix, ((0, 0), (0, pad)), mode='constant')
match_filter = np.pad(match_filter, (0, pad), mode='constant')

# 0 match filter outside of center region
match_filter[:buffer] = 0
match_filter[-buffer:] = 0

# FFT along azimuth
az_fft = np.fft.fft(rolled_matrix, axis=1, n=fft_len)
H_az = np.fft.fft(match_filter, n=fft_len)

# Apply filter
focused_freq = az_fft * H_az[np.newaxis, :]

# --- 4. Inverse FFT ---
focused = np.fft.ifft(focused_freq, axis=1)

# Crop to original azimuth size (centered)
start = pad // 2
end = start + Na
focused = focused[:, start:end]

params['aspect'] *= 2


plt.imshow(lin_to_db(np.abs(focused)), aspect='auto', vmin=-10, 
        extent=[-5, 5, 2*(params["rx_window_offset_m"] + params["rx_window_m"])/299.792458, 2*params["rx_window_offset_m"]/299.792458])
plt.colorbar(label='Power [dB]')
plt.xlabel("Azimuth [km]")
plt.ylabel("Range [us]")
plt.savefig(f"figures/{lbl}-focused.png")
plt.close()

extent = (-30, 30, 2*((params['rx_window_offset_m'] + params['rx_window_m'])/c1)*10**6,
        2*(params['rx_window_offset_m']/c1)*10**6)

fig, ax = plt.subplots(3, 1, figsize=(4, 6), constrained_layout=True, dpi=300)

im0 = ax[0].imshow(mag_to_db(np.abs(rdrgrm)), cmap="viridis",
                aspect=params['aspect']*0.75, extent=extent,
                vmin=-30, vmax=-15)

im1 = ax[1].imshow(mag_to_db(np.abs(focused)), cmap="viridis",
                aspect=params['aspect']*0.75, extent=extent,
                vmin=-10, vmax=8)

# zoomed panel
s1, s2 = 0.49/0.7, 0.59/0.7
zoomed = np.abs(focused[int(s1*rb):int(s2*rb),
                        int(0.4*focused.shape[1]):int(0.6*focused.shape[1])])
zoomed_extent = (-30 + 60 * 0.4, -30 + 60 * 0.6,
                2*((params['rx_window_offset_m'] + s2 * params['rx_window_m'])/c1)*10**6,
                2*((params['rx_window_offset_m'] + s1 * params['rx_window_m'])/c1)*10**6)
im3 = ax[2].imshow(mag_to_db(zoomed), cmap="viridis",
                aspect=params['aspect']*0.75, extent=zoomed_extent,
                vmin=-10, vmax=8)

# shrink bottom plot to fit
pos_top = ax[0].get_position()
pos_bottom = ax[2].get_position()
new_width = pos_top.width
new_x = 0.05 + pos_bottom.x0 + (pos_bottom.width - new_width) / 2  # center it
ax[2].set_position([new_x, pos_bottom.y0, new_width, pos_bottom.height])

# add rectangle
rect = Rectangle((zoomed_extent[0], zoomed_extent[2]),
                zoomed_extent[1]-zoomed_extent[0],
                zoomed_extent[3]-zoomed_extent[2],
                linewidth=1, edgecolor="red", facecolor="none")
ax[1].add_patch(rect)

# labels and text
labels = ["(a)", "(b) Focused", "(c) Focused"]
fontsizes = (11, 11, 9)
for a, label, fs in zip(ax, labels, fontsizes):
    a.set_ylabel("Range [µs]", fontsize=11)
    a.tick_params(axis="both", which="major", labelsize=9, direction="out")
    a.tick_params(axis="both", which="minor", direction="out")
    a.text(0.02, 0.95, label, transform=a.transAxes, fontsize=fs,
        fontweight="bold", va="top", ha="left", color="black",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

ax[2].set_xlabel("Azimuth [km]", fontsize=11)

# colorbars
ims = [im0, im1, im3]
for a, im in zip(ax, ims):
    cax = inset_axes(a, width="3%", height="100%",
                    loc='center right', borderpad=-2)  # negative pad pushes it outward
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Power [dB]", fontsize=8, labelpad=2)

plt.savefig(f"figures/{lbl}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"figures/{lbl}.pgf", dpi=300, bbox_inches="tight")
plt.close()