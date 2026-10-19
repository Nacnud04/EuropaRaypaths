import sys, pickle, json
sys.path.append("../../src/PYTHON")
import param_gen       as pg

"""pars = {

    # REASON VHF radar parameters
    "power":                  5.5,   # Transmitter power   [W]
    "frequency":              60e6,  # Center frequency    [Hz]
    "bandwidth":              10e6,  # Radar bandwidth     [Hz]
    "surface_gain":           7.3,   # Antenna gain        [dBi]
    "subsurface_gain":        7.3,   # Antenna gain        [dBi]
    "polarization":          "HH",   # Polarization (dont matter)
    "aperture":                 3,   # Processing aperture [deg]

    # source parameters 
    "sy":                       0,   # source y location       [m]
    "sz":                    25e3,   # source z location       [m]
    "sdx":                     25,   # source x discretization [m]
    "sx0":                 -3.5e3,   # source x origin         [m]
    "ns":                    2000,   # source count            [.]

    # receive window parameters
    "rx_window_offset_m": 24.75e3,   # Rx opening window delay [m]
    "rx_window_m":         0.75e3,   # Rx window length        [m]
    "rx_sample_rate":        60e6,   # Rx sample rate          [Hz]

    # surface parameters
    "rms_height":               0,   # Facet roughness         [m]
    "buff":                   1.1,   # Buffer for memory alloc [.]
    "altitude":              25e3,   # Approx. alt (for focus) [m]
    "fs":                      20,   # Facet size              [m]

    # atmosphere/subsurface parameters
    "eps_1":                  1.0,   # permittivity of medium 1 
    "eps_2":                 3.15,   # permittivity of medium 2
    "sig_1":                  0.0,   # conductivity of medium 1 [S/m]
    "sig_2":                 1e-6,   # conductivity of medium 2 [S/m]
    "mu_1":                   1.0,   # permeability of medium 1
    "mu_2":                   1.0,   # permeability of medium 2

    # processing parameters (BOOLEAN)
    "lossless": True,      # simulate without loss (spreading not included)
}"""

domainpar = {
    "ox": -10e3,
    "oy": -10e3,
    "oz": 0,
    "fs": 50,
    "nx": 400,
    "ny": 400,
}

recpar = {
    "rx_window_m": 3e3,         # receive window size [m]
    "rx_sample_rate": 30e6,       # receive sample rate [Hz]
    "rx_window_offset_m": 24e3,
}

sourcepar = {
    "sy":                       0,   # source y location       [m]
    "sz":                    25e3,   # source z location       [m]
    "sdx":                     10,   # source x discretization [m]
    "sx0":                   -5e3,   # source x origin         [m]
    "ns":                    1000,   # source count            [.]
    "aperture":                70,   # processing aperture     [deg]
}

otherpar = {
    "lossless": True,
}

params = pg.gen_params("REASON_VHF", "planetary_ice", domainpar, recpar, sourcepar, par=otherpar)
pg.export_params(params, f"params")

# save params
#with open("inputs/params.json", "w") as f:
#    json.dump(pars, f, indent=4)

#with open("inputs/params.pkl", 'wb') as hdl:
#    pickle.dump(pars, hdl, protocol=pickle.HIGHEST_PROTOCOL)