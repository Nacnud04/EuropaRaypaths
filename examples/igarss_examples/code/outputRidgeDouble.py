import sys

sys.path.append("../../src/PYTHON")
import output_handling as oh
import simple_focusing as sf
import rdr_plots       as rp

sys.path.append("../../archive/src")
from terrain import Terrain


params = oh.load_params("params/params.pkl", "params/target.txt")
params['aspect'] = 0.5 * 0.75

# generate terrain again for the profile
xmin, xmax = params["ox"], params["ox"]+params["nx"]*params["fs"]
ymin, ymax = params["oy"], params["oy"]+params["ny"]*params["fs"]
terrain = Terrain(xmin, xmax, ymin, ymax, params['fs'])
amp       = 300     # amplitude [m]
peak_dist = 6e3     # peak distance [m]
ridge_wid = 4e3     # ridge width [m]
x_offset  = 5e3     # x offset [m]
terrain.double_ridge(amp, amp, peak_dist, ridge_wid, x_offset)

lbl="ridge"
rdrgrm = oh.compile_rdrgrm(f"rdrgrm/{lbl}", params)

rp.simple_rdrgrm(rdrgrm, params, f"figures/{lbl}-rdrgrm.png", title="Double Ridge Radargram", vmin=-42)

focused = sf.full_focus_at_center(rdrgrm, params)

rp.simple_rdrgrm(focused, params, f"figures/{lbl}-focused.png", title="Double Ridge Radargram Focused", vmin=-10)

rp.IGARSS2026_rdrgrm_focused_profile(rdrgrm, focused, terrain, params, 0.4, 0.6, (0.47/0.7), (0.57/0.7),
                             f"figures/{lbl}", vminrdr=-20, vminfoc=0)

