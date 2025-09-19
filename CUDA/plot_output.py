import numpy as np
import matplotlib.pyplot as plt

arr = np.loadtxt("Ith_cuda.txt")
x = np.linspace(8000, 12000, 1000)

plt.plot(x, arr)
plt.savefig("pythonplot.png")
plt.close()
