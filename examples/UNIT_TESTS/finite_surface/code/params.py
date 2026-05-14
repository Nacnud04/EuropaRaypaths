import sys

sys.path.append("../../../src/PYTHON")
import simple_surfaces as ss
import param_gen       as pg

domainpar = {
    "ox": -0.125e3,
    "oy": -0.125e3,
    "oz": 0,
    "fs": 5,
    "nx": 50,
    "ny": 50,
}

recpar = {
    "rx_window_position_file": "inputs/rx_window_positions.txt",
    "rx_window_m": 0.75e3,         # receive window size [m]
    "rx_sample_rate": 120e6,       # receive sample rate [Hz]
}

sourcepar = {
    "sy": 0,                # source y location       [m]
    "sz": None,             # source z location       [m]
    "sdx": 10,              # source x discretization [m]
    "sx0": 0,               # source x origin         [m]
    "ns": 1000,             # source count            [.]
    "aperture": 70,
    "source_path_file": "inputs/source_path.txt",
}

otherpar = {
    "lossless": True,
}

params = pg.gen_params("REASON_VHF", "planetary_ice", domainpar, recpar, sourcepar, par=otherpar)

pg.export_params(params, f"co_params")

params['incoherent'] = True
pg.export_params(params, f"inco_params")

# --- make source path and window ---

maxZ = 50e3 # maximum altitude of source path [m]
minZ = 200e3 # minimum altitude of source path [m]

szs = pg.vert_source_path(sourcepar, minZ, maxZ, "source_path")
pg.track_Z_rxwin(szs, 250, "rx_window_positions")

# --- make surface ---
# make sure surface is smaller than fresnel zone
# set altitude to minZ to find minimum fresnel zone radius
params['sz'] = minZ 
Fr = ss.calc_fresnel(params)
if abs(params['ox']) > Fr:
    print(f"WARNING: Surface radius ({abs(params['ox']):.2f} m) greater than fresnel zone ({Fr:.2f} m)")

ss.make_surface(params, "flat", f"inputs/facets.fct")

