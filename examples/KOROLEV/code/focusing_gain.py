import numpy as np

import matplotlib.pyplot as plt

rdrgrm_subsurf = np.load("output/subsurf-rdrgrm.npy")
focused_subsurf = np.load("output/subsurf-focused.npy")

# clip out bad target area
trc_max = 1600
rdrgrm_subsurf[:,trc_max:] = 0
focused_subsurf[:,trc_max:] = 0

rdrgrm_surf = np.load("output/rdrgrm.npy")
focused_surf = np.load("output/focused.npy")

# get absolute power
rdrgrm_subsurf  = np.abs(rdrgrm_subsurf)
focused_subsurf = np.abs(focused_subsurf)
rdrgrm_surf     = np.abs(rdrgrm_surf)
focused_surf    = np.abs(focused_surf)

# turn into a focusing gain for both surface and subsurface
surface_gain = focused_surf / rdrgrm_surf
subsurf_gain = focused_subsurf / rdrgrm_subsurf

# convert to dB
surface_gain = 10 * np.log10(surface_gain)
subsurf_gain = 10 * np.log10(subsurf_gain)

plt.imshow(surface_gain, vmin=10, vmax=20)
plt.colorbar()
plt.show()

plt.imshow(subsurf_gain, vmin=-10, vmax=20)
plt.colorbar()
plt.show()
