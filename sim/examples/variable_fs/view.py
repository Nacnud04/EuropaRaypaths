import numpy as np
import matplotlib.pyplot as plt
import sys, os
from numba import njit, prange #type: ignore
sys.path.append("../../../.")
from util import *

rdrgrms = np.load("variable_fs.npy")
sltrngs = np.load("sltrange_fs.npy")
offset_rng = sltrngs[-1] - np.argmin(sltrngs[-1])
slt_rng = sltrngs[-1]

fss = (20, 10, 5, 3, 2, 1)

fig, ax = plt.subplots(3, 2, figsize=(6.5, 6))
ax = ax.flatten()
extent = (-10, 20, 25, -5)
for i, rdr in enumerate(rdrgrms):
    ax[i].imshow(np.abs(rdr), cmap="grey", aspect=0.45, extent=extent)
    ax[i].set_title(f"Facet size: {fss[i]:03d} m")
    ax[i].set_xlabel("Distance [km]")
    ax[i].set_ylabel("Depth [km]")
ax[-1].axis("off")
plt.tight_layout()
plt.savefig("VariableFsUnfocused.png")
plt.show()

# quick and dirty focusing (grab phase history from high resolution sim)
hr = np.abs(rdrgrms[-1, :, :])

c   = 299792458
f0  = 9e6             # center frequency [Hz]
start = (20e3*2) / c  # [s]
end   = (50e3*2) / c  # [s]
rb    = 4803
dt    = (end - start) / rb


# find travel time at each azumith location
tt_rb = np.argmax(np.abs(hr), axis=0)

match_filter = np.conjugate(rdrgrms[-1][tt_rb, range(100)])

# =============================================================================
# FOCUSING FUNCTIONS


def focus_pix(rdr, t, T, rngbins, k, dt, c, slt_rng, plot=False):
    
    # t is fast time and T is slow time

    # get vallid offsets from pixel to compute
    rng = np.arange(len(rngbins)) - (len(rngbins) // 2) + T
    valid_mask = (rng >= 0) & (rng < rdr.shape[1])

    # --- GET VALUES ALONG ESTIMATED HYPERBOLA ---
    # turn offsets into indicies
    hyper_bins = ((slt_rng[valid_mask] - 20e3) // (dt * c)).astype(np.int32)
    hyper_bins -= np.min(hyper_bins)
    hyper_bins *= 2
    hyper_trcs = rng[valid_mask].astype(np.int32)

    # offset bins by pixel location and windoow
    hyper_bins += t
    valid_bins = (hyper_bins > 0) * (hyper_bins < rb)
    hyper_trcs = hyper_trcs[valid_bins]
    hyper_bins = hyper_bins[valid_bins] 

    # --- MULTIPLY VALUES BY CONJUGATE AND SUM ---
    focused_pix = rdr[hyper_bins, hyper_trcs] * np.exp(-2j * k * slt_rng[valid_mask][valid_bins])

    if plot:
        plt.imshow(np.abs(rdr), aspect=7e-3)
        plt.plot(hyper_trcs, hyper_bins, color="red", linewidth=1)
        plt.show()

        plt.imshow(np.abs(rdr), extent=(0, 99, 50e3, 20e3), aspect=2.5e-3)
        plt.plot(slt_rng, color="red", linewidth=1)
        plt.show()

        plt.plot(np.abs(rdr[hyper_bins, hyper_trcs]), color="black", linewidth=1)
        plt.plot(np.real(rdr[hyper_bins, hyper_trcs]), color="blue", linewidth=1)
        plt.plot(np.imag(rdr[hyper_bins, hyper_trcs]), color="red", linewidth=1)
        plt.title("Signal amplitude over hyperbola")
        plt.show()

        plt.plot(np.abs(focused_pix), color="black", linewidth=1)
        plt.plot(np.real(focused_pix), color="blue", linewidth=1)
        plt.plot(np.imag(focused_pix), color="red", linewidth=1)
        plt.title("Match filtered amplitude over hyperbola")
        plt.show()



    return np.sum(focused_pix)


def focus_pix2(rdr, t, T, rngbins, k, dt, c, slt_rng, plot=False):
    
    # t is fast time and T is slow time

    # get vallid offsets from pixel to compute
    rng = np.arange(len(rngbins)) - (len(rngbins) // 2) + T
    valid_mask = (rng >= 0) & (rng < rdr.shape[1])

    # --- GET VALUES ALONG ESTIMATED HYPERBOLA ---
    # turn offsets into indicies
    hyper_bins = (tt_rb[valid_mask]).astype(np.int32)
    hyper_bins -= np.min(hyper_bins)
    hyper_trcs = rng[valid_mask].astype(np.int32)

    # offset bins by pixel location and windoow
    hyper_bins += t
    valid_bins = (hyper_bins > 0) * (hyper_bins < rb)
    hyper_trcs = hyper_trcs[valid_bins]
    hyper_bins = hyper_bins[valid_bins] 

    # --- MULTIPLY VALUES BY CONJUGATE AND SUM ---
    focused_pix = rdr[hyper_bins, hyper_trcs] * match_filter[valid_mask][valid_bins]

    if plot:
        plt.imshow(np.abs(rdr), aspect=7e-3)
        plt.plot(hyper_trcs, hyper_bins, color="red", linewidth=1)
        plt.show()

        plt.imshow(np.abs(rdr), extent=(0, 99, 50e3, 20e3), aspect=2.5e-3)
        plt.plot(slt_rng, color="red", linewidth=1)
        plt.show()

        plt.plot(np.abs(rdr[hyper_bins, hyper_trcs]), color="black", linewidth=1)
        plt.plot(np.real(rdr[hyper_bins, hyper_trcs]), color="blue", linewidth=1)
        plt.plot(np.imag(rdr[hyper_bins, hyper_trcs]), color="red", linewidth=1)
        plt.title("Signal amplitude over hyperbola")
        plt.show()

        plt.plot(np.abs(focused_pix), color="black", linewidth=1)
        plt.plot(np.real(focused_pix), color="blue", linewidth=1)
        plt.plot(np.imag(focused_pix), color="red", linewidth=1)
        plt.title("Match filtered amplitude over hyperbola")
        plt.show()



    return np.sum(focused_pix)


def focus(rdr, rngbn, f0, dt):
    k = (2 * np.pi) / 33.3
    focused = np.zeros_like(rdr, dtype=np.complex128)

    for t in range(rdr.shape[0]):
        print(f"focusing... {t}/{rdr.shape[0]}", end="      \r")
        for T in range(rdr.shape[1]):
            focused[t, T] = focus_pix2(rdr, t, T, rngbn, k, dt, c, slt_rng)

    return focused

"""
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
"""

focus_pix(rdrgrms[-1], 959, 50, tt_rb, (2 * np.pi) / 33.3, dt, c, slt_rng, plot=True)
focus_pix2(rdrgrms[-1], 959, 50, tt_rb, (2 * np.pi) / 33.3, dt, c, slt_rng, plot=True)

fig, ax = plt.subplots(3, 2, figsize=(6.5, 6))
ax = ax.flatten()
for i, rdr in enumerate(rdrgrms):
    focused = focus(rdr, tt_rb, f0, dt)
    max_idx = np.argmax(np.abs(focused))
    y_max, x_max = np.unravel_index(max_idx, focused.shape)
    print(f"Point target found at rb: {y_max} and trc: {x_max}")
    ax[i].imshow(np.abs(focused), cmap="grey", aspect=0.45, extent=extent)
    ax[i].set_title(f"Facet size: {fss[i]:03d} m")
    ax[i].set_xlabel("Distance [km]")
    ax[i].set_ylabel("Depth [km]")
ax[-1].axis("off")
plt.tight_layout()
plt.savefig("VariableFsFocused.png")
plt.show()
