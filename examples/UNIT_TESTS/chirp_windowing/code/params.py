import sys

sys.path.append("../../../src/PYTHON")
import param_gen       as pg

domainpar = {
    "ox": -10e3,
    "oy": -10e3,
    "oz": 0,
    "fs": 5,
    "nx": 1000,
    "ny": 1000,
}

recpar = {
    "rx_window_m":           2e3,    # receive window size [m]
    "rx_sample_rate":      120e6,    # receive sample rate [Hz]
    "rx_window_offset_m":   99e3,
}

sourcepar = {
    "sy":                       0,   # source y location       [m]
    "sz":                   100e3,   # source z location       [m]
    "sdx":                      0,   # source x discretization [m]
    "sx0":                      0,   # source x origin         [m]
    "ns":                       1,   # source count            [.]
    "aperture":                70,
}

otherpar = {
    "lossless": True,
    "chirp_window": "none"
}

window_types = ("none", "blackman", "nuttall")
for window in window_types:

    otherpar['chirp_window'] = window

    # first export basic single example stuff
    params = pg.gen_params("REASON_VHF", "vacuum", domainpar, recpar, sourcepar, par=otherpar)
    pg.export_params(params, window)
    print(f"Expored window: {window}")


