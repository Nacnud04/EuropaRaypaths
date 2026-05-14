import sys

sys.path.append("../../../src/PYTHON")
import simple_surfaces as ss
import param_gen       as pg

domainpar = {
    "ox": -1e3,
    "oy": -1e3,
    "oz": 0,
    "fs": 10,
    "nx": 200,
    "ny": 200,
}

recpar = {
    "rx_window_offset_m": None,
    "rx_window_m": 0.75e3,         # receive window size [m]
    "rx_sample_rate": 120e6,       # receive sample rate [Hz]
}

sourcepar = {
    "sy": 0,                # source y location       [m]
    "sz": None,             # source z location       [m]
    "sdx": 10,              # source x discretization [m]
    "sx0": 0,            # source x origin         [m]
    "ns": 1,             # source count            [.]
    "aperture": 70,
}

otherpar = {
    "lossless": True,
}

params = pg.gen_params("REASON_VHF", "planetary_ice", domainpar, recpar, sourcepar, par=otherpar)

# --- MAKE SOURCE PATH ---

maxZ = 150e3 # maximum altitude of source path [m]
minZ = 350e3 # minimum altitude of source path [m]

sp_par = {"ns" : 100}
szs = pg.vert_source_path(sp_par, minZ, maxZ, "source_path")

# --- MAKE FACETS and WRITE PARAMS ---

for i, sz in enumerate(szs): 
    params['incoherent'] = False
    params['sz'] = sz
    params['altitude'] = sz
    params['rx_window_offset_m'] = sz - 250
    pg.export_params(params, f"params{i}")
    params['incoherent'] = True
    pg.export_params(params, f"inc-params{i}")
    ss.make_surface(params, "fresnel", f"inputs/facets{i}.fct")
    ss.make_surface(params, "fresnel-convex", f"inputs/facets-conv-{i}.fct")
