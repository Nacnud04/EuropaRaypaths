import numpy as np
import matplotlib.pyplot as plt

def load_complex(filename):

    tdat = np.loadtxt(filename)
    return tdat[:, 0] + 1j * tdat[:, 1]

def amp_to_ax(ax, data, dt, abs=False):
    if abs:
        ax.plot(np.arange(len(data)) * dt, np.abs(data), color="black", linewidth=1, label="abs")
    else:
        ax.plot(np.arange(len(data)) * dt, np.real(data), color="red", linewidth=1, label="real")
        ax.plot(np.arange(len(data)) * dt, np.imag(data), color="blue", linewidth=1, label="imag")
    ax.legend()

def ph_to_ax(ax, data, dt):
    ax.plot(np.arange(len(data)) * dt, np.angle(data) * 180 / np.pi, color="purple", linewidth=1)
    ax.set_ylim(-180, 180)
