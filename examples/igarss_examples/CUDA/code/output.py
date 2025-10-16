import numpy as np
import matplotlib.pyplot as plt
import glob, sys, pickle

# load parameters from pickle
with open("params/params.pkl", 'rb') as hdl:
    params = pickle.load(hdl)

filenames = glob.glob("rdrgrm/s*.txt")

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

# try to focus the radargram
sys.path.append("../../../src")
from focus import est_slant_range

sx = params['sx0'] + params['sdx'] * np.arange(params['ns']) # source x locations [m]
sy = params['sy']                          # source y locations [m]
sz = params['sz']                          # source z locations [m]

c1 = 299792458
c2 = c1 / np.sqrt(params["eps_2"])

rb = int((params["rx_window_m"] / c1) / (1 / params["rx_sample_rate"]))
dm = c1 / params["rx_sample_rate"]

sltrng   = est_slant_range(sx, sz, params["tx"], params["tz"], c1, c2, trim=True)
sltrng_t = 2 * 10**6 * sltrng / c1  # in microseconds

# compute sample-bin indices (meters -> samples)
slt_rb = ((sltrng - params["rx_window_offset_m"]) // dm).astype(int)

k = (2 * np.pi) / (c1 / params["frequency"])
match_filter = np.exp(-2j * k * sltrng)

def lin_to_db(x):
    return 10 * np.log10(x)

lst = [lin_to_db(np.abs(rdrgrm)), np.angle(rdrgrm)]
names = ["figures/rdrgrmAbs.png", "figures/rdrgrmPhase.png"]
cmaps = ["viridis", "twilight"]
cbar_labels = ["Power [dB]", "Phase [rad]"]
vmin  = [-42, None]

for arr, name, cmap, cbar_label, v in zip(lst, names, cmaps, cbar_labels, vmin):
    plt.imshow(arr, aspect='auto', cmap=cmap, interpolation='nearest', vmin=v,
            extent=[params["sx0"]/1e3, (params["sx0"] + params["sdx"] * params["ns"])/1e3, 2*(params["rx_window_offset_m"] + params["rx_window_m"])/299.792458, 2*params["rx_window_offset_m"]/299.792458])
    plt.colorbar(label=cbar_label)
    plt.xlabel("Azimuth [km]")
    plt.ylabel("Range [us]")
    plt.savefig(name)
    plt.close()


Nr, Na = rdrgrm.shape

# --- 2. Range Cell Migration Correction (RCMC) ---
shift_amounts = slt_rb - np.min(slt_rb)
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


plt.imshow(lin_to_db(np.abs(focused)), aspect='auto', vmin=-10, 
           extent=[-5, 5, 2*(params["rx_window_offset_m"] + params["rx_window_m"])/299.792458, 2*params["rx_window_offset_m"]/299.792458])
plt.colorbar(label='Power [dB]')
plt.xlabel("Azimuth [km]")
plt.ylabel("Range [us]")
plt.savefig("figures/focused.png")
plt.close()
