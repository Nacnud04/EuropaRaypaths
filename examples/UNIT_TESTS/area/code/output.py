import numpy as np
import matplotlib.pyplot as plt



origin = np.linspace(-10e3, -0.25e3, 100)

Pmax = []
areas = []

for i, o in enumerate(origin):

    area = (np.abs(o) * 2) ** 2

    f = f"outputs/trc{i}/Psurf_s000000.txt"
    arr = np.loadtxt(f).T
    sig = arr[0] + 1j * arr[1]

    Pmax.append(np.max(np.abs(sig)**2))
    areas.append(area)

    print(f"area = {area:.0f} m², max abs={np.max(np.abs(sig))**2:.3e}")

# fit an A^2 curve
coeffs = np.polyfit(np.array(areas)/(1e3*1e3), Pmax, 2)
poly = np.poly1d(coeffs)

# fit an A^1 curve
coeffs = np.polyfit(np.array(areas)/(1e3*1e3), Pmax, 1)
poly = np.poly1d(coeffs)

plt.plot(np.array(areas)/(1e3*1e3), Pmax, color="black", label="Simulation")
#plt.plot(np.array(areas)/(1e3*1e3), poly(np.array(areas)/(1e3*1e3)), color="red", linestyle="--", label=f"Fit: P = {coeffs[0]:.2e} * A^2 + {coeffs[1]:.2e} * A + {coeffs[2]:.2e}")
plt.plot(np.array(areas)/(1e3*1e3), poly(np.array(areas)/(1e3*1e3)), color="red", linestyle="--", label=f"Fit: P = {coeffs[0]:.2e} * A + {coeffs[1]:.2e}")
plt.xlabel("Area [km^2]")
plt.ylabel("Max Power [W]")
plt.legend()
plt.title("Incoherent Facets")
plt.savefig("figures/CoherentAreaScaling.png", dpi=300)
plt.show()