import sys
import pandas as pd

sys.path.append("../../../src/PYTHON")
import output_handling as oh
import simple_focusing as sf
import rdr_plots       as rp

sys.path.append("../../../archive/src")
from terrain import Terrain


params = oh.load_params("params/dem.pkl", "params/dem_target.txt")
params['aspect'] = 0.75

# generate terrain again for the profile
xmin, xmax = params["ox"], params["ox"]+params["nx"]*params["fs"]
ymin, ymax = params["oy"], params["oy"]+params["ny"]*params["fs"]
# import DEM csv
DEM = pd.read_csv("facets/dem_profile.csv")
terrain = Terrain(xmin, xmax, ymin, ymax, params['fs'])
terrain.gen_from_provided(DEM['Along Track (km)']*1e3, DEM['Surface Height (m)'])

lbl="dem"
rdrgrm = oh.compile_rdrgrm(f"rdrgrm/{lbl}", params)

rp.simple_rdrgrm(rdrgrm, params, f"figures/{lbl}-rdrgrm.png", title="DEM Ridge Radargram", vmin=-42)

focused = sf.focus_middle_only(rdrgrm, params, 2000)

rp.simple_rdrgrm(focused, params, f"figures/{lbl}-focused.png", title="DEM Radargram Focused", vmin=-10)

rp.IGARSS2026_rdrgrm_focused_profile(rdrgrm, focused, terrain, params, 0.4, 0.6, (0.49/0.7), (0.59/0.7),
                             f"figures/{lbl}", vminrdr=-20, vminfoc=0)
