import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../src/PYTHON")
import simple_focusing as sf
import rdr_plots       as rp
import output_handling as oh
import unit_convs      as uc

params = oh.load_params("inputs/params.pkl", "inputs/targets.txt")

rdrgrm = oh.compile_rdrgrm("radargram", params)
print(f"Radargram shape: {rdrgrm.shape}")

rp.simple_rdrgrm(rdrgrm, params, "figures/radargram.png", linspace=True, figsize=(9, 4))

np.save("radargram/radargram.npy", rdrgrm)

# prepare to focus
params['spacing'] = params['sdx']
"""
match_filter = sf.focus_rdrgrm(rdrgrm, params, return_match_filter=True)

plt.plot(np.real(match_filter), label="Real part")
plt.plot(np.imag(match_filter), label="Imaginary part")
plt.legend()
plt.show()

shifts = sf.focus_rdrgrm(rdrgrm, params, return_shifts=True)

plt.imshow(uc.lin_to_db(np.abs(rdrgrm)), aspect="auto")
plt.plot(np.arange(len(shifts))+1620, 52+shifts, color="red", linewidth=1)
plt.show()

"""
focused = sf.focus_rdrgrm(rdrgrm, params)

rp.simple_rdrgrm(focused, params, "figures/focused.png", linspace=True, figsize=(9, 4))

np.save("radargram/focused.npy", focused)