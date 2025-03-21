import numpy as np
import matplotlib.pyplot as plt

rdrgrms = np.load("variable_fs.npy")

fss = (20, 10, 5, 3, 2)

fig, ax = plt.subplots(3, 2, figsize=(6.5, 6))
ax = ax.flatten()
for i, rdr in enumerate(rdrgrms):
    ax[i].imshow(np.abs(rdr), cmap="grey", aspect=0.07)
    ax[i].set_title(f"Facet size: {fss[i]:03d} m")
ax[-1].axis("off")
plt.tight_layout()
plt.savefig("VariableFsUnfocused.png")
plt.show()

# quick and dirty focusing (grab phase history from high resolution sim)
hr = np.abs(rdrgrms[-1, :, :])

f0  = 9e6             # center frequency [Hz]
dt  = 1 / (8 * f0)    # time delta to avoid aliasing [s]
start = 166.8e-6      # [s]
end   = 177e-6        # [s]

# find travel time at each azumith location
tt_rb = np.argmax(hr, axis=0)

# offset by minimum time
tt_rb -= np.min(tt_rb)

# =============================================================================
# FOCUSING FUNCTIONS

def focus_pix(rdr, t, T, rngbins, k, dt=dt):

    c = 299792458
    
    # compute the range which is within the radargram
    rng = np.arange(len(rngbins)) - (len(rngbins) // 2) + T
    
    # compute the value of the range bins in the image
    rngbins = rngbins[rng >= 0] + t
    
    # crop to the region in the image
    rng = rng[rng >= 0]
    rngbins = rngbins[rng < rdr.shape[1]]
    rng = rng[rng < rdr.shape[1]]
    rng = rng[rngbins < rdr.shape[0]]
    rngbins = rngbins[rngbins < rdr.shape[0]]

    # rangebins to time for exponent calculation
    rngtime = rngbins * dt * c
    exp = np.conjugate(np.exp(2j * k * rngtime))

    return np.sum(rdr[rngbins.astype(int), rng.astype(int)] * exp)

def focus(rdr, rngbn, f0):
    
    lam =  299792458 / f0
    k = (2 * np.pi) / lam
    focused = np.zeros_like(rdr)
    for t in range(rdr.shape[0]):
        print(f"Focusing... {t+1}/{rdr.shape[0]}", end="    \r")
        for T in range(rdr.shape[1]):
            focused[t, T] = focus_pix(rdr, t, T, rngbn, k)

    return focused

fig, ax = plt.subplots(3, 2, figsize=(6.5, 6))
ax = ax.flatten()
for i, rdr in enumerate(rdrgrms):
    focused = focus(rdr, tt_rb, f0)
    ax[i].imshow(np.abs(focused), cmap="grey", aspect=0.07)
    ax[i].set_title(f"Facet size: {fss[i]:03d} m")
ax[-1].axis("off")
plt.tight_layout()
plt.savefig("VariableFsFocused.png")
plt.show()