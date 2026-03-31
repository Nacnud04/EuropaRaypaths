import sys

sys.path.append("../../../src/PYTHON")
import simple_focusing as sf
import rdr_plots       as rp
import output_handling as oh

import numpy as np
import matplotlib.pyplot as plt
tdat = np.loadtxt("rdrgrm/Ptarg_s000050_t00.txt")
plt.plot(tdat)
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
