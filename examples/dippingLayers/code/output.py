import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob, sys, pickle

# focusing helper
sys.path.append("../../src")
from focus import est_slant_range

# load parameters
with open("params/params.pkl", "rb") as h:
    params = pickle.load(h)

params["tx"] = 0
params["ty"] = 0
params["tz"] = -1500

# load radargram files
files = sorted(glob.glob("rdrgrm/s*.txt"))
rdr = []

for i, f in enumerate(files):
    if i >= params["ns"]:
        break
    arr = np.loadtxt(f).T
    rdr.append(arr[0] + 1j * arr[1])

rdr = np.array(rdr).T

# source locations
sx = params["sx0"] + params["sdx"] * np.arange(params["ns"])
sy = params["sy"]
sz = params["sz"]

# propagation speeds
c0 = 299_792_458
c_med = c0 / np.sqrt(params["eps_2"])

# window and sample geometry
rx_win = params["rx_window_m"]
rx_off = params["rx_window_offset_m"]
sample_rate = params["rx_sample_rate"]

rb = int((rx_win / c0) / (1 / sample_rate))
dm = c0 / sample_rate

# slant range and matched filter terms
sl = est_slant_range(sx, sz, params["tx"], params["tz"], c0, c_med)
sl_us = 2e6 * sl / c0  # microseconds
sl_rb = ((sl - rx_off) // dm).astype(int)

k = (2 * np.pi) / (c0 / params["frequency"])
mf = np.exp(-2j * k * sl)

def lin_to_db(x):
    return 10 * np.log10(x)

# quick diagnostic plots
plots = [
    (lin_to_db(np.abs(rdr)), "rdrgrmAbs.png",   "viridis",  "Power [dB]", -17),
    (np.angle(rdr),          "rdrgrmPhase.png", "twilight", "Phase [rad]", None),
]

extent = [
    -5, 5,
    (rx_off + rx_win) / 1e3,
    rx_off / 1e3
]

for arr, name, cmap, cbar, vmin in plots:
    plt.imshow(arr, aspect="auto", cmap=cmap, interpolation="nearest",
               vmin=vmin, vmax=-4, extent=extent)
    plt.colorbar(label=cbar)
    plt.xlabel("Azimuth [km]")
    plt.ylabel("Range [km]")
    plt.savefig(f"plots/{name}")
    plt.close()

Nr, Na = rdr.shape

# range cell migration correction
shifts = sl_rb - sl_rb.min()
rdr_rcmc = np.column_stack([
    np.roll(rdr[:, i], -int(shifts[i]))
    for i in range(Na)
])

# -------------------------------
# Selective azimuth focusing window
# -------------------------------

# choose window center and width (in azimuth samples)
# example: 20% wide window centered in the middle
az_center = Na // 2
az_width = Na // 5    # adjust as needed

az_start = max(0, az_center - az_width // 2)
az_end   = min(Na, az_center + az_width // 2)

# extract windowed azimuth data
rdr_win = rdr_rcmc[:, az_start:az_end]
Na_win = rdr_win.shape[1]

# -------------------------------
# Apply Range - Doppler focusing ONLY to the window
# -------------------------------

fft_len = 2 * Na_win
pad = fft_len - Na_win

# pad data and matched filter to FFT length
rdr_pad = np.pad(rdr_win, ((0, 0), (0, pad)))
mf_pad = np.pad(mf[az_start:az_end], (0, pad))

az_fft = np.fft.fft(rdr_pad, axis=1, n=fft_len)
H = np.fft.fft(mf_pad, fft_len)

focused_freq = az_fft * H[np.newaxis, :]
focused_win = np.fft.ifft(focused_freq, axis=1)

# crop back to window size
start = pad // 2
end = start + Na_win
focused_win = focused_win[:, start:end]

# -------------------------------
# Combine focused window with unfocused remainder
# -------------------------------

focused = 50*rdr.copy()
focused[:, az_start:az_end] = focused_win

# final plot
range_extent = [
    -5, 5,
    2 * (rx_off + rx_win) / 299.792458,
    2 * rx_off / 299.792458
]

plt.imshow(lin_to_db(np.abs(focused)), aspect="auto", vmin=0,
           extent=range_extent)
plt.colorbar(label="Power [dB]")
plt.xlabel("Azimuth [km]")
plt.ylabel("Range [us]")
plt.savefig("plots/focused.png")
plt.close()
