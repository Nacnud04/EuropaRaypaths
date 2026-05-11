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

# --- Does this match Haynes? ---

c = 299792458
f = 60e6
G = 10**(7.3/10)
refl = 1
R = alts*1e3
P_t = 100

lam = c / f
A = 10e3*10e3

# first for coherent
component = ((refl**2 * A**2) / ((4 * np.pi)**2 * lam**2)) * (1/R**4)
coh_analytic_Pr =  component * (P_t * G**2 * lam**2)

plt.plot(R/1e3, co_PP / coh_analytic_Pr)
plt.title(r"Simulation$\times\left(\frac{P_tG_tG_r\Gamma^2A^2}{(4\pi)^2R^4}\right)^{-1}$")
plt.ylabel("Simulator / Analytic")
plt.xlabel("Range [km]")
plt.subplots_adjust(top=0.85)
plt.show()

# then for incoherent
sig_0 = 1
component = ((sig_0 * A) / ((4 * np.pi)**3)) * (1/R**4)
incoh_analytic_Pr =  component * (P_t * G**2 * lam**2)
plt.plot(R/1e3, inco_rolling / incoh_analytic_Pr)
plt.title(r"Simulation$\times\left(\frac{P_tG_tG_r\sigma_0A}{(4\pi)^3R^4}\right)^{-1}$")
plt.ylabel("Simulator / Analytic")
plt.xlabel("Range [km]")
plt.subplots_adjust(top=0.85)
plt.show()