import numpy as np
import matplotlib.pyplot as plt
import json
import glob

params = {

    # radar parameters
    "power": 100,             # Transmitter power [W]
    "frequency": 9e6,         # Radar frequency [Hz]
    "bandwidth": 1e6,         # Radar bandwidth [Hz]
    "surface_gain": 75,       # Antenna gain [dB]
    "subsurface_gain": 100,   # Subsurface antenna gain [dB]
    "range_resolution": 300,  # range resolution [m]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)

    # receive window parameters
    "rx_window_m":  10e3,        # receive window length [m]
    "rx_window_offset_m": 7.5e3, # receive window offset [m]
    "rx_sample_rate": 38e6,      # receive sample rate [Hz]

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
    "sz": 10000,            # source z location [m]

}

with open("params.json", "w") as f:
    json.dump(params, f, indent=4)

filenames = glob.glob("rdrgrm/s*.txt")

# sort filenames to ensure correct order
filenames.sort()

rdrgrm = []

for f in filenames:
    print(f"Loading in {f}", end="       \r")
    arr = np.loadtxt(f).T
    rdrgrm.append(arr[0] + 1j * arr[1])

rdrgrm = np.array(rdrgrm).T
plt.imshow(np.abs(rdrgrm), aspect='auto', 
           extent=[2*params["rx_window_offset_m"]/299.792458, 
                   2*(params["rx_window_offset_m"] + params["rx_window_m"])/299.792458, -1000, 1000])
plt.colorbar(label='Power [W]')
plt.xlabel("2-Way Time [us]")
plt.ylabel("Range [m]")
plt.savefig("rdrgrm.png")
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
