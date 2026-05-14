import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../../../src/PYTHON")
import output_handling as oh

par = oh.load_params("inputs/params0.pkl", "inputs/targets.txt")

ns = 100
alts = np.linspace(350e3, 150e3, ns)

# --- FIRST DO FLAT ---

# load coherent simulation outputs
P_num = []

for i in range(ns):

    f = f"flatRDR/trc{i}/s000000.txt"
    arr = np.loadtxt(f).T
    sig = arr[0] + 1j * arr[1]

    P_num.append(np.max(np.abs(sig)**2))

P_num = np.array(P_num)

# get analytic result
P_t = par['power']
G   = 10 ** (par["surface_gain"] / 10)
lam = par['lam']
P_r = (P_t * G**2 * lam**2) / ((4 * np.pi)**2 * (alts)**2)

# find error
error = np.abs(P_num - P_r) / P_r * 100
print(f"Error Average: {np.mean(error):.2f} %")

fig, ax = plt.subplots(2, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

# plot Pmax vs altitude
ax[0].plot(alts/1e3, P_num*1e9, label="Numerical", color="black", linewidth=1)
ax[0].plot(alts/1e3, P_r*1e9, color="red", linestyle="--", label="Analytic", linewidth=1)
ax[0].set_ylabel("Maximum Power [nW]")
ax[0].legend()
ax[0].set_title("Coherent, Flat Fresnel Zone Surface")

# plot error
ax[1].plot(alts/1e3, error, color="blue", label="Error (%)", linewidth=1)
ax[1].set_xlabel("Altitude [km]")
ax[1].set_ylabel("Error (%)")
ax[1].set_ylim(0, 5)

plt.xlim(alts[-1]/1e3, alts[0]/1e3)

plt.savefig("figures/FlatFresnelZoneComparison.png", dpi=300)
plt.show()

# --- NOW DO FLAT, INCOHERENT ---

#P_num = []

#for i in range(ns):

#    f = f"inco_flatRDR/trc{i}/s000000.txt"
#    arr = np.loadtxt(f).T
#    sig = arr[0] + 1j * arr[1]

#    P_num.append(np.max(np.abs(sig)**2))

#P_num = np.array(P_num)

P_num = []
for i in range(10):
    P_num.append(np.load(f"inco_flatRDR/Pmax{i}.npy"))
P_num = np.array(P_num)

P_num = np.nanmean(P_num, axis=0)

# compute 1/x^3 term
inv_r3 = 1 / alts**3

# solve for C in least squares sense
C = np.sum(P_num * inv_r3) / np.sum(inv_r3**2)

# analytic fit
fit = C * inv_r3

# solve for sigma_0:
# C = (P*G*G*lam^3*sigma_0)/(2^7*pi^2)
sigma_0 = (C * 2**7 * np.pi**2) / (P_t * G**2 * lam**3)

label_poly = f"Analytic: $\sigma_0$ = {sigma_0:.2f}"

error = np.abs(P_num - fit) / fit * 100
print(f"Error Average: {np.mean(error):.2f} %")

# plot
fig, ax = plt.subplots(2, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

# plot Pmax vs altitude
ax[0].plot(alts/1e3, P_num*1e9, label="Numerical", color="black", linewidth=1)
ax[0].plot(alts/1e3, fit*1e9, color="red", linestyle="--", label=label_poly, linewidth=1)
ax[0].set_ylabel("Maximum Power [nW]")
ax[0].legend()
ax[0].set_title("Incoherent, Flat Fresnel Zone Surface")

# plot error
ax[1].plot(alts/1e3, error, color="blue", label="Error (%)", linewidth=1)
ax[1].set_xlabel("Altitude [km]")
ax[1].set_ylabel("Error (%)")
ax[1].set_ylim(0, 50)

plt.xlim(alts[-1]/1e3, alts[0]/1e3)

plt.savefig("figures/FlatFresnelZoneIncoherentComparison.png", dpi=300)
plt.show()

# --- THEN DO CONVEX ---

# load coherent simulation outputs
P_num = []

for i in range(ns):

    f = f"convexRDR/trc{i}/s000000.txt"
    arr = np.loadtxt(f).T
    sig = arr[0] + 1j * arr[1]

    P_num.append(np.max(np.abs(sig)**2))

P_num = np.array(P_num)

# solve analytic
r_convex = alts
factor = r_convex**2 / (r_convex + alts)**2
P_r *= factor

# find error
error = np.abs(P_num - P_r) / P_r * 100
print(f"Error Average: {np.mean(error):.2f} %")

fig, ax = plt.subplots(2, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

# plot Pmax vs altitude
ax[0].plot(alts/1e3, P_num*1e9, label="Numerical", color="black", linewidth=1)
ax[0].plot(alts/1e3, P_r*1e9, color="red", linestyle="--", label="Analytic", linewidth=1)
ax[0].set_ylabel("Maximum Power [nW]")
ax[0].legend()
ax[0].set_title("Coherent, Spherical Fresnel Zone Surface")

# plot error
ax[1].plot(alts/1e3, error, color="blue", label="Error (%)", linewidth=1)
ax[1].set_xlabel("Altitude [km]")
ax[1].set_ylabel("Error (%)")
ax[1].set_ylim(0, 5)

plt.xlim(alts[-1]/1e3, alts[0]/1e3)

plt.savefig("figures/SphericalFresnelZoneComparison.png", dpi=300)
plt.show()