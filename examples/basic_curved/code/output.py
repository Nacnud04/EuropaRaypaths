import sys

sys.path.append("../../src/PYTHON")
import simple_focusing as sf
import rdr_plots       as rp
import output_handling as oh

params = oh.load_params("inputs/params.pkl", "inputs/targets.txt", source_path="inputs/source_path.txt")
params['sx0'] = -5e3
params['sdx'] = 10
params['sz']  = 10e3

rdrgrm = oh.compile_rdrgrm("rdrgrm", params)

rp.simple_rdrgrm(rdrgrm, params, "figures/rdrgrm.png", title="Sinusoidal Source Path", vmin=-20)

focused = sf.full_focus_at_center(rdrgrm, params)

rp.simple_rdrgrm(focused, params, "figures/focused.png", title="Sinusoidal Source Path", vmin=0)
