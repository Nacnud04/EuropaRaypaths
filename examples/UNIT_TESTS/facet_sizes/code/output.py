import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

fss = (25, 50, 100, 150, 250)

for i, fs in enumerate(fss):

    f = f"trc{i}/Psurf_s000000.txt"
    arr = np.loadtxt(f).T
    sig = arr[0] + 1j * arr[1]

    print(f"facet size = {fs} m, max abs={np.max(np.abs(sig)):.3e}")

    plt.plot(np.arange(len(sig)) * (1/60e6) * 299792458, np.abs(sig), linewidth=1, label=f"{fs} m", linestyle="--")

plt.xlim(200, 450)
plt.xlabel("Range (m)")
plt.title("Phasor Trace Comparison")
plt.legend()
plt.savefig("figures/TraceComparison.png", dpi=300)
plt.close()