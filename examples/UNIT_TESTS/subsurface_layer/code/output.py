import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sys.path.append("../../../src/PYTHON")
import output_handling as oh

params = oh.load_params("inputs/params.pkl", "inputs/layer.txt")

h = np.linspace(100e3, 25e3, 100)

# get analytic result
lam = 299792458 / params["frequency"]
G   = 10 ** (params["surface_gain"] / 10)
P_t = params["power"]

P_r = (P_t * G**2 * lam**2) / ((4 * np.pi)**2 * (2*h)**2)

# load radargram
rdrgrm = oh.compile_rdrgrm("rdrgrm", params)
plt.imshow(np.abs(rdrgrm)**2)
plt.show()

P_num = np.zeros_like(h)

for i, alt in enumerate(h):
    f = f"rdrgrm/s{i:06d}.txt"
    arr = np.loadtxt(f).T
    sig = arr[0] + 1j * arr[1]
    P_num[i] = np.max(np.abs(sig)**2)
    plt.plot(np.abs(sig)**2)
    plt.show()
    plt.plot(np.abs(sig[:100])**2)
    plt.show()

error = np.abs(P_num - P_r) / P_r * 100

fig, ax = plt.subplots(2, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

# plot Pmax vs altitude
ax[0].plot(h/1e3, P_num*1e9, label="Numerical", color="black", linewidth=1)
ax[0].plot(h/1e3, P_r*1e9, color="red", linestyle="--", label="Analytic", linewidth=1)
ax[0].set_ylabel("Maximum Power [nW]")
ax[0].legend()
ax[0].set_title("Coherent Surface with Infinite Area")

# plot error
ax[1].plot(h/1e3, error, color="blue", label="Error (%)", linewidth=1)
ax[1].set_xlabel("Altitude [km]")
ax[1].set_ylabel("Error (%)")
ax[1].set_ylim(0, 5)

plt.xlim(h[-1]/1e3, h[0]/1e3)

plt.show()
