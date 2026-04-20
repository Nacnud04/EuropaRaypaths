import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../src/PYTHON")
import kutil           as ku
import simple_focusing as sf
import rdr_plots       as rp
import output_handling as oh
import unit_convs      as uc

params = oh.load_params("inputs/params.pkl", "inputs/targets.txt")

rdrgrm = oh.compile_rdrgrm("radargram", params)
print(f"Radargram shape: {rdrgrm.shape}")

rp.simple_rdrgrm(rdrgrm, params, "figures/radargram.png", linspace=True, figsize=(9, 3))

np.save("radargram/radargram.npy", rdrgrm)