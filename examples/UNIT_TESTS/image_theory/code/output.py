import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sys.path.append("../../../src/PYTHON")
import output_handling as oh

params = oh.load_params("inputs/params.pkl", "inputs/targets.txt")

h = np.linspace(200e3, 25e3, 500)

# get analytic result
lam = 299792458 / params["frequency"]
G   = 10 ** (params["surface_gain"] / 10)
P_t = params["power"]

P_r = (P_t * G**2 * lam**2) / ((4 * np.pi)**2 * (2*h)**2)

P_num = np.zeros_like(h)

for i, alt in enumerate(h):
    f = f"coRDR/rdr20m/s{i:06d}.txt"
    arr = np.loadtxt(f).T
    sig = arr[0] + 1j * arr[1]
    P_num[i] = np.max(np.abs(sig)**2)

error = np.abs(P_num - P_r) / P_r * 100

# get fresnel zone radius at 200 km altitude
rF = np.sqrt(lam * h[0])
print(f"Fresnel zone radius at {h[0]/1e3:.0f} km: {rF:.2f} m")
# approach 2
rF = np.sqrt((h[0] + lam / 4)**2 - h[0]**2)
print(f"Fresnel zone radius at {h[0]/1e3:.0f} km (approach 2): {rF:.2f} m")

plt.rcParams.update({'font.size': 14})

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

plt.savefig("figures/ImageMethodComparison.png", dpi=300)
plt.close()

# now see how those big error spikes change with facet size

errors = []

for j in range(10):
    P_num = np.zeros_like(h)
    for i, alt in enumerate(h):
        f = f"coRDR/rdr{j}/s{i:06d}.txt"
        arr = np.loadtxt(f).T
        sig = arr[0] + 1j * arr[1]
        P_num[i] = np.max(np.abs(sig)**2)
    error = np.abs(P_num - P_r) / P_r * 100
    errors.append(error)
    print(j)
    
errors = np.array(errors)

extent = (200, 25, 100, 10)

fig, ax = plt.subplots(constrained_layout=True)

im = ax.imshow(
    errors,
    aspect="auto",
    extent=extent,
    cmap="magma",
    vmax=7.5
)

plt.colorbar(im, label="Error [%]")

# Primary axes labels
ax.set_xlabel("Altitude [km]")
ax.set_ylabel("Facet size [m]")

# --- Secondary X-axis: altitude in wavelengths ---
def km_to_lam(x):
    return (x * 1000) / lam

def lam_to_km(x):
    return (x * lam) / 1000

secax_x = ax.secondary_xaxis("top", functions=(km_to_lam, lam_to_km))
secax_x.set_xlabel("Altitude [λ]")

# --- Secondary Y-axis: facet size in wavelengths ---
def m_to_lam(y):
    return y / lam

def lam_to_m(y):
    return y * lam

secax_y = ax.secondary_yaxis("right", functions=(m_to_lam, lam_to_m))
secax_y.set_ylabel("Facet size [λ]")

plt.savefig("figures/ErrorFSvsALT.png")
plt.close()
