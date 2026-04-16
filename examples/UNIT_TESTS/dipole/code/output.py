import sys

sys.path.append("../../../src/PYTHON")
import simple_focusing as sf
import rdr_plots       as rp
import output_handling as oh

from dipole_src import *

import numpy as np
import matplotlib.pyplot as plt

dt = 1 / 18 # sample rate
fig, ax = plt.subplots(3, 2, sharex=True, figsize=(10, 8))
tdat = load_complex("rdrgrm/Ptarg_s000050_t00.txt")
amp_to_ax(ax[0,0], tdat, dt, abs=True)
ph_to_ax(ax[0,1], tdat, dt)
ax[0,0].set_title("Inward Phasor Trace", fontsize=10)
sdat = load_complex("rdrgrm/Psour_s000050_t00.txt")
amp_to_ax(ax[1,0], sdat, dt, abs=True)
ph_to_ax(ax[1,1], sdat, dt)
ax[1,0].set_title("Outward Phasor Trace", fontsize=10)
conv = np.convolve(tdat, sdat, mode='full')[::2]
ax[2,0].plot(np.arange(len(conv)) * dt, np.abs(conv), color="black", linewidth=1, label="Python")
ax[2,1].plot(np.arange(len(conv)) * dt, np.angle(conv) * 180 / np.pi, color="black", linewidth=1, label="Python")
fdat = load_complex("rdrgrm/PTTmp_s000050_t00.txt")
ax[2,0].plot(np.arange(len(fdat)) * dt, np.abs(fdat), color="red", linewidth=1, label="CUDA FFT", linestyle=":")
ax[2,1].plot(np.arange(len(fdat)) * dt, np.angle(fdat) * 180 / np.pi, color="red", linewidth=1, label="CUDA FFT", linestyle=":")
ax[2,0].legend()
ax[2,1].legend()
ax[2,0].set_title("Convolution of Inward and Outward Traces", fontsize=10)
ax[2,0].set_xlabel("Time [us]")
for a in ax[:,0]: a.set_ylabel("Power [W]")
plt.xlim(20, 30)
plt.savefig("figures/Ptarg.png")
plt.close()

for i in range(46, 55):
    fdat = load_complex(f"rdrgrm/PTTmp_s0000{i}_t00.txt")
    plt.plot(np.arange(len(fdat)) * dt, np.abs(fdat), linewidth=1, label=i)
plt.xlim(25.5, 27)
plt.legend()
plt.savefig("figures/PTtmp.png")
plt.close()

surf_dat = load_complex("rdrgrm/Psurf_s000050.txt")
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(np.arange(len(surf_dat)) * dt, np.abs(surf_dat), color="black", linewidth=1, label="Surface")
ax.plot(np.arange(len(fdat)) * dt, np.abs(fdat), color="red", linewidth=1, label="Subsurface")
ax.set_xlabel("Time [us]")
ax.set_ylabel("Power [w]")
plt.savefig("figures/PTComparison.png")
plt.close()

params = oh.load_params("inputs/params.pkl", "inputs/targets.txt")

rdrgrm = oh.compile_rdrgrm("rdrgrm", params)

rp.simple_rdrgrm(rdrgrm, params, "figures/rdrgrm.png", title="Very basic radargram", vmin=-30)

focused = sf.full_focus_at_center(rdrgrm, params)
#focused[:500] = rdrgrm[:500]

rp.simple_rdrgrm(focused, params, "figures/focused.png", title="Very basic focused radargram", vmin=0)

params['aspect'] = 'auto'
rp.TGRS_rdrgrm_focused(rdrgrm, focused, params, "figures/TGRS_BASIC", vminrdr=-10, vminfoc=10)

# calc focusing gain
import numpy as np
targ_loc = np.unravel_index(np.argmax(np.abs(focused)), focused.shape)
unfoc_pow = 10*np.log10(np.abs(rdrgrm[targ_loc]))
foc_pow   = 10*np.log10(np.abs(focused[targ_loc]))

print(f"Unfocused power at target: {unfoc_pow:.2f} dB")
print(f"Focused power at target: {foc_pow:.2f} dB")
print(f"Focusing gain: {foc_pow - unfoc_pow:.2f} dB")
