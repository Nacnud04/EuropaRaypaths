import sys

sys.path.append("../../../src/PYTHON")
import output_handling as oh
import simple_focusing as sf
import rdr_plots       as rp


params = oh.load_params("params/params.pkl", "params/target.txt")
params['aspect'] = 0.5 * 0.75

lbl = "flat"
rdrgrm = oh.compile_rdrgrm(f"rdrgrm/{lbl}", params)

rp.simple_rdrgrm(rdrgrm, params, f"figures/{lbl}-rdrgrm.png", title="Flat Surface Radargram", vmin=-42)

focused = sf.full_focus_at_center(rdrgrm, params)

rp.simple_rdrgrm(focused, params, f"figures/{lbl}-focused.png", title="Flat Surface Radargram Focused", vmin=-10)

rp.IGARSS2026_rdrgrm_focused(rdrgrm, focused, params, 0.4, 0.6, (0.47/0.7), (0.57/0.7),
                             f"figures/{lbl}", vminrdr=-20, vminfoc=0)
