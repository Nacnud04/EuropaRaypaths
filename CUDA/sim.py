import numpy as np
import matplotlib.pyplot as plt
import json
import glob, sys

params = {

    # radar parameters
    "power": 100,             # Transmitter power [W]
    "frequency": 9e6,         # Radar frequency [Hz]
    "bandwidth": 1e6,         # Radar bandwidth [Hz]
    "surface_gain": 75,       # Antenna gain [dB]
    "subsurface_gain": 95,   # Subsurface antenna gain [dB]
    "range_resolution": 300,  # range resolution [m]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)

    # receive window parameters
    "rx_window_m":  10e3,         # receive window length [m]
    "rx_window_offset_m": 7.5e3,  # receive window offset [m]
    "rx_sample_rate": 48e6,       # receive sample rate [Hz]

    # surface parameters
    "sigma": 1,              # sigma [?]
    "rms_height": 0.4,       # surface roughness [m]

    # atmosphere/subsurface parameters
    "eps_1": 1.0,            # permittivity of medium 1 
    "eps_2": 3.15,           # permittivity of medium 2
    "sig_1": 0.0,            # conductivity of medium 1 [S/m]
    "sig_2": 1e-6,           # conductivity of medium 2 [S/m]
    "mu_1": 1.0,             # permeability of medium 1
    "mu_2": 1.0,             # permeability of medium 2

    # target parameters
    "tx": 0,                # target x location [m]
    "ty": 0,                # target y location [m]
    "tz": -1000,            # target z location [m]

    # source parameters
    "sx": 0,                # source x location [m]
    "sy": 0,                # source y location [m]
    "sz": 10e3,#25e3,             # source z location [m]

    # facet array params
    "ox": -5e3,
    "oy": -1e3,
    "oz": 0,
    "fs": 5,
    "nx": 2000,
    "ny": 400,

}

with open("params.json", "w") as f:
    json.dump(params, f, indent=4)


# --- MAKE FACET FILE
sys.path.append("../src")
from terrain import Terrain

xmin, xmax = params["ox"], params["ox"]+params["nx"]*params["fs"]
ymin, ymax = params["oy"], params["oy"]+params["ny"]*params["fs"]

terrain = Terrain(xmin, xmax, ymin, ymax, params["fs"])
terrain.gen_flat(0)

# write output facet data
xx, yy, zz = terrain.XX, terrain.YY, terrain.zs
norms      = np.reshape(terrain.normals, (xx.shape[0]*xx.shape[1], 3))
uvecs      = np.reshape(terrain.uvectors, (xx.shape[0]*xx.shape[1], 3))
vvecs      = np.reshape(terrain.vvectors, (xx.shape[0]*xx.shape[1], 3))

with open("facets.fct", 'w') as f:
    for x, y, z, (nx, ny, nz), (ux, uy, uz), (vx, vy, vz) in zip(xx.flatten(), yy.flatten(), zz.flatten(), norms, uvecs, vvecs):
        f.write(f"{x},{y},{z}:{nx},{ny},{nz}:{ux},{uy},{uz}:{vx},{vy},{vz}\n")
    


# --- END MAKE FACET FILE


filenames = glob.glob("rdrgrm/s*.txt")

# sort filenames to ensure correct order
filenames.sort()

rdrgrm = []

for f in filenames:
    print(f"Loading in {f}", end="       \r")
    arr = np.loadtxt(f).T
    rdrgrm.append(arr[0] + 1j * arr[1])

rdrgrm = np.array(rdrgrm).T

# try to focus the radargram
sys.path.append("../src")
from focus import *

sx = -5e3 + 50 * np.arange(200) # source x locations [m]
sy = 0                          # source y locations [m]
sz = 10e3                       # source z locations [m]

c1 = 299792458
c2 = c1 / np.sqrt(params["eps_2"])

rb = int((params["rx_window_m"] / c1) / (1 / params["rx_sample_rate"]))
dm = c1 / params["rx_sample_rate"]

sltrng   = est_slant_range(sx, sz, params["tx"], params["tz"], c1, c2)
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
vmin  = [-20, None]

for arr, name, cmap, cbar_label, v in zip(lst, names, cmaps, cbar_labels, vmin):
    plt.imshow(arr, aspect='auto', cmap=cmap, interpolation='nearest', vmin=v,
            extent=[-5, 5, 2*(params["rx_window_offset_m"] + params["rx_window_m"])/299.792458, 2*params["rx_window_offset_m"]/299.792458])
    plt.plot(sx/1e3, sltrng_t, color='red', linewidth=1)
    plt.colorbar(label=cbar_label)
    plt.xlabel("Azimuth [km]")
    plt.ylabel("Range [us]")
    plt.savefig(name)
    plt.close()

#focused = focus_jit(rdrgrm, slt_rb, match_filter, rb)


Nr, Na = rdrgrm.shape

# --- 2. Range Cell Migration Correction (RCMC) ---
shift_amounts = slt_rb - np.min(slt_rb)
rolled_matrix = np.array([
    np.roll(rdrgrm[:, i], -int(shift_amounts[i]))
    for i in range(rdrgrm.shape[1])
]).T

#plt.imshow(np.abs(rolled_matrix), aspect=0.2)
#plt.show()

# --- 3. Azimuth matched filter ---
k = (2 * np.pi) / (c1 / params["frequency"])
match_filter = np.exp(-2j * k * sltrng)

#plt.plot(sltrng)
#plt.show()

#fig, ax = plt.subplots(2)
#ax[0].plot(np.real(match_filter))
#ax[0].plot(np.real(match_filter*rolled_matrix[np.min(slt_rb), :])/np.max(np.real(match_filter*rolled_matrix[np.min(slt_rb), :])))
#ax[1].plot(np.imag(match_filter))
#ax[1].plot(np.imag(match_filter*rolled_matrix[np.min(slt_rb), :])/np.max(np.imag(match_filter*rolled_matrix[np.min(slt_rb), :])))
#plt.show()

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


plt.imshow(lin_to_db(np.abs(focused)), aspect='auto', vmin=0, 
           extent=[-5, 5, 2*(params["rx_window_offset_m"] + params["rx_window_m"])/299.792458, 2*params["rx_window_offset_m"]/299.792458])
plt.colorbar(label='Power [dB]')
plt.xlabel("Azimuth [km]")
plt.ylabel("Range [us]")
plt.savefig("focused.png")
plt.close()


"""
x = 2 * np.linspace(params["rx_window_offset_m"], params["rx_window_offset_m"] + params["rx_window_m"], len(arr[0])) / 299.792458

plt.plot(x, arr[0], color='blue', label='Real', linewidth=1)
plt.plot(x, arr[1], color='red', label='Imaginary', linewidth=1)
plt.plot(x, np.abs(arr[0] + 1j * arr[1]), color='black', label='Magnitude', linewidth=1)
plt.grid()
plt.xlabel("2-Way Time [us]")
plt.ylabel("Power [W]")
plt.legend()
plt.savefig(f.replace(".txt", ".png"))
plt.close()
"""
