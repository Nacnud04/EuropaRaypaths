import numpy as np
import matplotlib.pyplot as plt

filenames = ["ReflSignal.txt", "RefrSignal.txt"]

for f in filenames:
 
    arr = np.loadtxt(f)
    x = np.linspace(8000, 13000, 1000)

    plt.plot(x, arr)
    plt.savefig(f.replace(".txt", ".png"))
    plt.close()
