import sys
import numpy as np
import matplotlib.pyplot as plt



origin = np.linspace(-10e3, -0.25e3, 100)

Pmax = []
areas = []

for i, o in enumerate(origin):

    area = (np.abs(o) * 2) ** 2

    f = f"incoh_outputs/trc{i}/Psurf_s000000.txt"
    arr = np.loadtxt(f).T
    sig = arr[0] + 1j * arr[1]

    Pmax.append(np.max(np.abs(sig)**2))
    areas.append(area)

    print(f"area = {area:.0f} m², max abs={np.max(np.abs(sig))**2:.3e}")

np.save("incoh_outputs/areas.npy", areas)
np.save(f"incoh_outputs/Pmax{sys.argv[1]}.npy", Pmax)