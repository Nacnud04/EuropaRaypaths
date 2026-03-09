import sys

sys.path.append("../../../src/PYTHON")
import simple_focusing as sf
import rdr_plots       as rp
import output_handling as oh

params = oh.load_params("inputs/params.pkl", "inputs/targets.txt")

rdrgrm = oh.compile_rdrgrm("rdrgrm", params)

rp.simple_rdrgrm(rdrgrm, params, "figures/rdrgrm.png", title="Very basic radargram", vmin=-10)

focused = sf.full_focus_at_center(rdrgrm, params)
#focused[:500] = rdrgrm[:500]

rp.simple_rdrgrm(focused, params, "figures/focused.png", title="Very basic focused radargram", vmin=10)

params['aspect'] = 'auto'
rp.TGRS_rdrgrm_focused(rdrgrm, focused, params, "figures/TGRS_BASIC", vminrdr=-10, vminfoc=10)