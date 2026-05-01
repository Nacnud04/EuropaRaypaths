import numpy as np
import matplotlib.pyplot as plt

origin = np.linspace(-10e3, -0.25e3, 100)

# load in all incoherent simulation outputs
i_Pmaxs = []
for i in range(50):
    i_Pmaxs.append(np.load(f"incoh_outputs/Pmax{i}.npy"))
i_Pmaxs = np.array(i_Pmaxs)

# load in areas
areas = np.load("incoh_outputs/areas.npy")

i_Pmaxs = np.nanmean(i_Pmaxs, axis=0)

# load coherent simulation outputs
c_Pmaxs = []

for i, area in enumerate(areas):

    f = f"coh_outputs/trc{i}/Psurf_s000000.txt"
    arr = np.loadtxt(f).T
    sig = arr[0] + 1j * arr[1]

    c_Pmaxs.append(np.max(np.abs(sig)**2))

    print(f"area = {area:.0f} m², max abs={np.max(np.abs(sig))**2:.3e}")

# fit an A^2 curve
c_coeffs = np.polyfit(np.array(areas)/(1e3*1e3), c_Pmaxs, 2)
c_poly = np.poly1d(c_coeffs)

# fit an A^1 curve
i_coeffs = np.polyfit(np.array(areas)/(1e3*1e3), i_Pmaxs, 1)
i_poly = np.poly1d(i_coeffs)

# -- PLOT ---
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

ax[0].plot(np.array(areas)/(1e3*1e3), c_Pmaxs, color="black", label="Simulation")
ax[0].plot(np.array(areas)/(1e3*1e3), c_poly(np.array(areas)/(1e3*1e3)), color="red", 
           linestyle="--", 
           label=f"P = {c_coeffs[0]:.2e} * A^2 + \n       {c_coeffs[1]:.2e} * A + \n        {c_coeffs[2]:.2e}")

ax[1].plot(np.array(areas)/(1e3*1e3), i_Pmaxs, color="black", label="Simulation")
ax[1].plot(np.array(areas)/(1e3*1e3), i_poly(np.array(areas)/(1e3*1e3)), color="red", 
           linestyle="--", 
           label=f"P = {i_coeffs[0]:.2e} * A + \n       {i_coeffs[1]:.2e}")

ax[0].set_xlabel("Area [km^2]")
ax[0].set_ylabel("Max Power [W]")
ax[0].legend()
ax[0].set_title(f"Coherent Facets")

ax[1].set_xlabel("Area [km^2]")
ax[1].set_ylabel("Max Power [W]")
ax[1].legend()
ax[1].set_title(f"Incoherent Facets\nMean max power over 50 instances")

plt.savefig("figures/CoherentAreaScaling.png", dpi=300)
plt.show()