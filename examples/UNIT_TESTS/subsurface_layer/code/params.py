import sys
import numpy as np

sys.path.append("../../../src/PYTHON")
import simple_surfaces as ss
import param_gen       as pg
"""
domainpar = {
    "ox": -2e3,
    "oy": -2e3,
    "oz": 0,
    "fs": 20*1.5,
    "nx": int(200/1.5),
    "ny": int(200/1.5),
}
"""
domainpar = {
    "ox": -2e3,
    "oy": -2e3,
    "oz": 0,
    "fs": 20,
    "nx": 200,
    "ny": 200,
}

recpar = {
    "rx_window_m": 1.25e3,         # receive window size [m]
    "rx_sample_rate": 60e6,       # receive sample rate [Hz]
    "rx_window_position_file": "inputs/rx_window_positions.txt",
}

sourcepar = {
    "ns": 25,               # source count            [.]
    "source_path_file": "inputs/source_path.txt",
    "aperture": 70,
}

otherpar = {
    "lossless": True,
    "debug_surface": False,
    "disable_surface": True,
}

depth = 20e3

# first export basic single example stuff
params = pg.gen_params("REASON_VHF", "vacuum", domainpar, recpar, sourcepar, par=otherpar)
params["rx_window_position_file"] = f"inputs/rx_window_positions.txt"
pg.export_params(params, f"params")

ss.make_surface(params, "flat", f"inputs/facets.fct")
ss.make_target_array(params, "flat", f"inputs/layer.txt", zoffset=-1*depth)

# --- MAKE SOURCE PATH ---

maxZ = 20e3 # maximum altitude of source path [m]
minZ = 30e3 # minimum altitude of source path [m]

params['sz'] = minZ
Fr = ss.calc_fresnel(params)
print(f"Fresnel zone radius for maximum altitude of {minZ} m is {Fr:.2f} m")
sz = pg.vert_source_path(params, minZ, maxZ, "source_path")

# --- RX OPENING WINDOW FILE ---
pg.track_Z_rxwin(sz, -1 * depth * np.sqrt(params["eps_2"]) + 500, f"rx_window_positions")