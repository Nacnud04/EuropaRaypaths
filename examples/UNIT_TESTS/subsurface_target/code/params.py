import sys
import numpy as np

sys.path.append("../../../src/PYTHON")
import simple_surfaces as ss
import param_gen       as pg

fct = 3

domainpar = {
    "ox": -2e3,
    "oy": -2e3,
    "oz": 0,
    "fs": 10 / fct,
    "nx": int(400 * fct),
    "ny": int(400 * fct),
}

recpar = {
    "rx_window_m": 1.25e3,         # receive window size [m]
    "rx_sample_rate": 30e6,       # receive sample rate [Hz]
    "rx_window_position_file": "inputs/rx_window_positions.txt",
}

sourcepar = {
    "ns": 500,               # source count            [.]
    "source_path_file": "inputs/source_path.txt",
    "aperture": 70,
}

otherpar = {
    "lossless": True,
    "debug_surface": True,
    "disable_surface": True,
}

tdepths = (-5e3, -2.5e3, -1e3, -0.5e3, -0.25e3)

# first export basic single example stuff
params = pg.gen_params("REASON_VHF", "planetary_ice", domainpar, recpar, sourcepar, par=otherpar)
for depth in tdepths:
    params["rx_window_position_file"] = f"inputs/rx_window_positions_{abs(int(depth)):04d}.txt"
    pg.export_params(params, f"params_{abs(int(depth)):04d}")

#ss.make_surface(params, "flat", f"inputs/facets.fct")
#ss.make_target_array(params, "flat", f"inputs/layer.txt", zoffset=-0.5e3)

# --- MAKE SOURCE PATH ---

maxZ = 19e3 # maximum altitude of source path [m]
minZ = 100e3 # minimum altitude of source path [m]

params['sz'] = minZ
Fr = ss.calc_fresnel(params)
print(f"Fresnel zone radius for maximum altitude of {minZ} m is {Fr:.2f} m")
sz = pg.vert_source_path(params, minZ, maxZ, "source_path")

# --- RX OPENING WINDOW FILE ---
for depth in tdepths:
    pg.track_Z_rxwin(sz, depth * np.sqrt(3.15) + 500, f"rx_window_positions_{abs(int(depth)):04d}")
