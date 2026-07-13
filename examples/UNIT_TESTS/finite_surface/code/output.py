import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sys.path.append("../../../src/PYTHON")
import output_handling as oh

params = oh.load_params("inputs/co_params.pkl", "inputs/targets.txt")

coRDR = oh.compile_rdrgrm("coRDR", params)

def extract_peak_power(rdr):
    peak_row = np.argmax(np.abs(rdr), axis=0)
    return np.abs(rdr[peak_row, np.arange(rdr.shape[1])])**2

co_PP = extract_peak_power(coRDR)

inco_PPs = []
for i in range(10):
    incoRDR = oh.compile_rdrgrm(f"incoRDR/rdr{i}", params)
    inco_PPs.append(extract_peak_power(incoRDR))
inco_PP = np.nanmean(inco_PPs, axis=0)

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


# --- Does this match Haynes? ---

c = 299792458
refl = 1
R = alts*1e3

A = (0.25e3)**2

# --- COMPARE COHERENT ---

lam = c / params["frequency"]
G   = 10 ** (params["surface_gain"] / 10)
P_t = params["power"]

coh_analytic_Pr = (P_t * G**2 * lam**2 * A**2) / ((4 * np.pi)**2 * lam**2 * R**4)

error = np.abs(co_PP - coh_analytic_Pr) / coh_analytic_Pr * 100

# write out as csv
df = pd.DataFrame({"ALT":R,"NUM_POW":co_PP,"ANA_POW":coh_analytic_Pr,"ERR":error})
df.to_csv("figures/finite_coherent.csv", index=False)

plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(2, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

# plot Pmax vs altitude
ax[0].plot(R/1e3, co_PP*1e9, label="Numerical", color="black", linewidth=1)
ax[0].plot(R/1e3, coh_analytic_Pr*1e9, color="red", linestyle="--", label="Analytic", linewidth=1)
ax[0].set_ylabel("Maximum Power [nW]")
ax[0].legend()
ax[0].set_title("Coherent Surface with Fixed Area")

# plot error
ax[1].plot(R/1e3, error, color="blue", label="Error (%)", linewidth=1)
ax[1].set_xlabel("Altitude [km]")
ax[1].set_ylabel("Error (%)")
ax[1].set_ylim(0, 5)

plt.xlim(R[-1]/1e3, R[0]/1e3)

plt.savefig("figures/FixedCoherentAreaFalloffComparison.png", dpi=300)
plt.close()

# --- COMPARE INCOHERENT ---

inv_r4 = 1 / R**4

# solve for C in least squares sense
C = np.sum(inco_PP * inv_r4) / np.sum(inv_r4**2)
fit = C * inv_r4

# solve for sigma_0
# C = (P*G*G*lam^2*sigma_0*A) / ((4*pi)^3)
sigma_0 = (C * (4 * np.pi)**3) / (P_t * G**2 * lam**2 * A)

label_poly = f"Analytic: $\sigma_0$ = {sigma_0:.2f}"

error = np.abs(inco_PP - fit) / fit * 100

# write out as csv
df = pd.DataFrame({"ALT":R,"NUM_POW":inco_PP,"ANA_POW":fit,"SIG0":np.ones_like(inco_PP)*sigma_0,"ERR":error})
df.to_csv("figures/finite_incoherent.csv", index=False)

fig, ax = plt.subplots(2, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

# plot Pmax vs altitude
ax[0].plot(R/1e3, inco_PP*1e9, label="Numerical", color="black", linewidth=1)
ax[0].plot(R/1e3, fit*1e9, color="red", linestyle="--", label=label_poly, linewidth=1)
ax[0].set_ylabel("Maximum Power [nW]")
ax[0].legend()
ax[0].set_title("Incoherent Surface with Fixed Area")

# plot error
ax[1].plot(R/1e3, error, color="blue", label="Error (%)", linewidth=1)
ax[1].set_xlabel("Altitude [km]")
ax[1].set_ylabel("Error (%)")
ax[1].set_ylim(0, 50)

plt.xlim(R[-1]/1e3, R[0]/1e3)

plt.savefig("figures/FixedIncoherentAreaFalloffComparison.png", dpi=300)
plt.close()


