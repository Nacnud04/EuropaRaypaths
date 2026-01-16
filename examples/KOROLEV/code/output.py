import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../src/PYTHON")
import simple_focusing as sf
import rdr_plots       as rp
import output_handling as oh
import unit_convs      as uc

params = oh.load_params("data/params.pkl", "data/Subsurface/KOR_T.txt")
params['ns'] = 449

rdrgrm = oh.compile_rdrgrm("rdrgrm", params)
rdr_db = uc.lin_to_db(np.abs(rdrgrm))


fig, ax = plt.subplots(1, 1, figsize=(4, 12))
im = ax.imshow(rdr_db, vmin=-10, vmax=5, cmap="viridis")
plt.colorbar(im)
plt.savefig("figures/rdrgrm.png")
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(4, 6))
im = ax.imshow(rdr_db, vmin=-10, vmax=5, cmap="viridis")
plt.colorbar(im)
ax.set_ylim(2400, 1750)
plt.savefig("figures/rdrgrmZoomed.png")
plt.close()

#rp.simple_rdrgrm(rdrgrm, params, "figures/rdrgrm.png", title="Korolev Crater")#, vmin=-60, vmax=-45)

#focused = sf.full_focus_at_center(rdrgrm, params)
#focused = sf.focus_window(rdrgrm, params, int(params['ns']/2), win_width=600)

#rp.simple_rdrgrm(focused, params, "figures/focused.png", title="High altitude focused radargram")#, vmin=-40)
