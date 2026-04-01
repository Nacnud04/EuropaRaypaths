import sys

sys.path.append("../../../src/PYTHON")
import simple_focusing as sf
import rdr_plots       as rp
import output_handling as oh

import numpy as np
import matplotlib.pyplot as plt

dt = 1 / 18 # sample rate
fig, ax = plt.subplots(3, sharex=True, figsize=(10, 6))
tdat = np.loadtxt("rdrgrm/Ptarg_s000050_t00.txt")
ax[0].plot(np.arange(len(tdat)) * dt, tdat, linewidth=1)
ax[0].set_title("Inward Phasor Trace")
sdat = np.loadtxt("rdrgrm/Psour_s000050_t00.txt")
ax[1].plot(np.arange(len(sdat)) * dt, sdat, linewidth=1)
ax[1].set_title("Outward Phasor Trace")
conv = np.convolve(tdat, sdat, mode='full')[::2]
ax[2].plot(np.arange(len(conv)) * dt,conv, linewidth=1)
ax[2].set_title("Convolution of Inward and Outward Traces")
ax[2].set_xlabel("Time [us]")
for a in ax: a.set_ylabel("Power [W]")
plt.savefig("figures/Ptarg.png")
plt.close()

params = oh.load_params("inputs/params.pkl", "inputs/targets.txt")

rdrgrm = oh.compile_rdrgrm("rdrgrm", params)

rp.simple_rdrgrm(rdrgrm, params, "figures/rdrgrm.png", title="Very basic radargram", vmin=-10)

focused = sf.full_focus_at_center(rdrgrm, params)
#focused[:500] = rdrgrm[:500]

rp.simple_rdrgrm(focused, params, "figures/focused.png", title="Very basic focused radargram", vmin=10)

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
