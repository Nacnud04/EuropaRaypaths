import sys
import numpy as np

sys.path.append("../../../src/PYTHON")
import simple_surfaces as ss
import param_gen       as pg

domainpar = {
    "ox": -10e3,
    "oy": -10e3,
    "oz": 0,
    "fs": 20,
    "nx": 1000,
    "ny": 1000,
}

recpar = {
    "rx_window_m": 0.75e3,         # receive window size [m]
    "rx_sample_rate": 120e6,       # receive sample rate [Hz]
    "rx_window_position_file": "inputs/rx_window_positions.txt",
}

sourcepar = {
    "ns": 500,               # source count            [.]
    "source_path_file": "inputs/source_path.txt",
    "aperture": 70,
}

otherpar = {
    "lossless": True,
}

# first export basic single example stuff
params = pg.gen_params("REASON_VHF", "planetary_ice", domainpar, recpar, sourcepar, par=otherpar)
pg.export_params(params, f"co_params")
print(f"Expored coherent parameters")
ss.make_surface(params, "flat", f"inputs/facets.fct")

side_len = 2*abs(domainpar['ox'])
fs_range = np.linspace(10, 100, 25).astype(int)
nfs      = (side_len / fs_range).astype(int)

for i, (fs, nf) in enumerate(zip(fs_range, nfs)):

    params['fs'] = int(fs)
    params['nx'] = int(nf)
    params['ny'] = int(nf)
    pg.export_params(params, f"co_params{i}")

    ss.make_surface(params, "flat", f"inputs/facets{i}.fct")

# --- MAKE SOURCE PATH ---

maxZ = 25e3 # maximum altitude of source path [m]
minZ = 200e3 # minimum altitude of source path [m]

sz = pg.vert_source_path(params, minZ, maxZ, "source_path")

# --- RX OPENING WINDOW FILE ---

pg.track_Z_rxwin(sz, 250, "rx_window_positions")
