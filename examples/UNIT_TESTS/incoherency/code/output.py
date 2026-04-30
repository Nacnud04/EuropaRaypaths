import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sys.path.append("../../../src/PYTHON")
import output_handling as oh

params = oh.load_params("inputs/params.pkl", "inputs/targets.txt")

coRDR = oh.compile_rdrgrm("coRDR", params)
incoRDR = oh.compile_rdrgrm("incoRDR", params)

def extract_peak_power(rdr):
    peak_row = np.argmax(np.abs(rdr), axis=0)
    return np.abs(rdr[peak_row, np.arange(rdr.shape[1])])**2

co_PP = extract_peak_power(coRDR)
inco_PP = extract_peak_power(incoRDR)

# rolling mean of peak powers
window_size = 50
inco_rolling = np.convolve(inco_PP, np.ones(window_size)/window_size, mode='same')

alts = np.loadtxt("inputs/source_path.txt", delimiter=",")[:,2]

def coherent_curve(r, A):
    return A / r**2
def incoherent_curve(r, A, dr):
    return A / (r+dr)**3

co_popt, pcov = curve_fit(coherent_curve, alts, co_PP)
inco_popt, pcov = curve_fit(incoherent_curve, alts, inco_rolling)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

alts /= 1e3

# first coherent
ax[0].plot(alts, co_PP, color="black", label="Synthetic Peak Power")
ax[0].plot(alts, coherent_curve(alts*1e3, *co_popt), color="red", label=f"{co_popt[0]:.3e}/R^2", linestyle="--")
ax[0].set_title("Coherent Surface")
ax[0].set_ylabel("Peak Power [W]")

ax[1].plot(alts, inco_PP, color="black", label="Synthetic Peak Power")
ax[1].plot(alts, inco_rolling, color="blue", label="Rolling Mean")
ax[1].plot(alts, incoherent_curve(alts*1e3, *inco_popt), color="red", label=f"{inco_popt[0]:.3e}/(R+{inco_popt[1]:.3e})^3", linestyle="--")
ax[1].set_title("Incoherent Surface")
ax[1].set_ylabel("Peak Power [W]")

for a in ax: a.set_xlabel("Altitude [km]")
for a in ax: a.legend()

plt.savefig("figures/FalloffComparison.png", dpi=300)
plt.close()
