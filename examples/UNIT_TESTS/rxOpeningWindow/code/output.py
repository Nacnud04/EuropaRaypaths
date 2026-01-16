import sys

sys.path.append("../../../src/PYTHON")
import simple_focusing as sf
import rdr_plots       as rp
import output_handling as oh

params = oh.load_params("inputs/params.pkl", "inputs/targets.txt")

rdrgrm = oh.compile_rdrgrm("rdrgrm", params)

rp.simple_rdrgrm(rdrgrm, params, "figures/rdrgrm.png", title="High altitude radargram")#, vmin=-60, vmax=-45)

#focused = sf.full_focus_at_center(rdrgrm, params)
focused = sf.focus_window(rdrgrm, params, int(params['ns']/2), win_width=600)

rp.simple_rdrgrm(focused, params, "figures/focused.png", title="High altitude focused radargram")#, vmin=-40)
