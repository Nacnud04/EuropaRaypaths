import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append("../../../.")
from util import *

rdrgrms = np.load("variable_fs.npy")
pathtms = np.load("pathtime_fs.npy")

fss = (20, 10, 5, 3, 2, 1)

fig, ax = plt.subplots(3, 2, figsize=(6.5, 6))
ax = ax.flatten()
for i, rdr in enumerate(rdrgrms):
    ax[i].imshow(np.abs(rdr), cmap="grey", aspect=0.07)
    ax[i].set_title(f"Facet size: {fss[i]:03d} m")
plt.tight_layout()
plt.savefig("VariableFsUnfocused.png")
plt.show()

# quick and dirty focusing (grab phase history from high resolution sim)
hr = np.abs(rdrgrms[-1, :, :])

f0  = 9e6             # center frequency [Hz]
dt  = 1 / (8 * f0)    # time delta to avoid aliasing [s]
start = 166.8e-6      # [s]
end   = 177e-6        # [s]

# path times to range bin
focus_rb = (pathtms - start) // dt

# find travel time at each azumith location
tt_rb = np.argmax(hr, axis=0)

# offset by minimum time
tt_rb -= np.min(tt_rb)

# =============================================================================
# FOCUSING FUNCTIONS

def focus_pix_alt(rdr, t, T, rngbins, k, dt, frb)

def focus_pix(rdr, t, T, rngbins, k, dt):

    c = 299_792_458  # speed of light in m/s

    # Compute range offsets relative to center
    rng = np.arange(len(rngbins)) - (len(rngbins) // 2) + T

    # Filter valid range values
    valid_mask = (rng >= 0) & (rng < rdr.shape[1])
    rng = rng[valid_mask]
    rngbins_shifted = rngbins[valid_mask] + t

    # Further filter to make sure rngbins are within bounds
    valid_mask = (rngbins_shifted >= 0) & (rngbins_shifted < rdr.shape[0])
    rng = rng[valid_mask]
    rngbins_shifted = rngbins_shifted[valid_mask]

    # Compute range time and phase correction
    rngtime = rngbins_shifted * dt * c
    exp = np.conjugate(np.exp(2j * k * rngtime))

    return np.sum(rdr[rngbins_shifted.astype(int), rng.astype(int)] * exp)

def focus(rdr, rngbn, f0):
    
    lam =  299792458 / f0
    k = (2 * np.pi) / lam
    focused = np.zeros_like(rdr)
    for t in range(rdr.shape[0]):
        print(f"Focusing... {t+1}/{rdr.shape[0]}", end="    \r")
        for T in range(rdr.shape[1]):
            focused[t, T] = focus_pix(rdr, t, T, rngbn, k, dt )

    return focused

fig, ax = plt.subplots(3, 2, figsize=(6.5, 6))
ax = ax.flatten()
for i, rdr in enumerate(rdrgrms):
    focused = focus(rdr, tt_rb, f0)
    ax[i].imshow(np.abs(focused), cmap="grey", aspect=0.07)
    ax[i].set_title(f"Facet size: {fss[i]:03d} m")
plt.tight_layout()
plt.savefig("VariableFsFocused.png")
plt.show()