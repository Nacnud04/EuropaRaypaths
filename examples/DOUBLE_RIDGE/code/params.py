import sys, pickle, json
sys.path.append("../../src/PYTHON")
import param_gen       as pg
import numpy as np

domainpar = {
    "ox": -10e3,
    "oy": -10e3,
    "oz": 0,
    "fs": 5,
    "nx": 400,
    "ny": 400,
}

recpar = {
    "rx_window_m": (4e3 + 1e3) * np.sqrt(3.15),         # receive window size [m]
    "rx_sample_rate": 20e6,       # receive sample rate [Hz]
    "rx_window_offset_m": 24e3,
}

sourcepar = {
    "sy":                       0,   # source y location       [m]
    "sz":                    25e3,   # source z location       [m]
    "sdx":                     20,   # source x discretization [m]
    "sx0":                   -10e3,   # source x origin         [m]
    "ns":                    1000,   # source count            [.]
    "aperture":                70,   # processing aperture     [deg]
}

otherpar = {
    "lossless": False,
    "disable_surface": False
}

params = pg.gen_params("REASON_VHF", "planetary_ice", domainpar, recpar, sourcepar, par=otherpar)
pg.export_params(params, f"params")
