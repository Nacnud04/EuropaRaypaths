import sys

sys.path.append("../../../src/PYTHON")
import rdr_plots       as rp
import output_handling as oh

params = oh.load_params("inputs/params.pkl", "inputs/targets.txt")

rdrgrm = oh.compile_rdrgrm("rdrgrm", params)

rp.simple_rdrgrm(rdrgrm, params, "figures/rdrgrm.png", title="", linspace=False)
