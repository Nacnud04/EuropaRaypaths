import sys
import numpy as np
import matplotlib.pyplot as plt

Pmax = []
areas = []
ns = 100

for i in range(ns):

    f = f"inco_flatRDR/trc{i}/s000000.txt"
    arr = np.loadtxt(f).T
    sig = arr[0] + 1j * arr[1]

    Pmax.append(np.max(np.abs(sig)**2))

np.save(f"inco_flatRDR/Pmax{sys.argv[1]}.npy", Pmax)
