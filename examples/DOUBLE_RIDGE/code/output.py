import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../src/PYTHON")
import output_handling as oh

params = oh.load_params("inputs/params.pkl", "inputs/targets.txt")

rdrgrm = oh.compile_rdrgrm("radargram", params)
print(f"Radargram shape: {rdrgrm.shape}")

rdrgrm_dB = 20 * np.log10(np.abs(rdrgrm))
plt.imshow(rdrgrm_dB, aspect="auto")
plt.colorbar(label="Power [dB]")
plt.show()
#rp.simple_rdrgrm(rdrgrm, params, "figures/radargram.png", linspace=True, figsize=(9, 4))

np.save("radargram/radargram.npy", rdrgrm)
