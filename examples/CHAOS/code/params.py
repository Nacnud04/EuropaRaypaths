import sys, pickle, json
import numpy as np

pars = {

    # REASON VHF radar parameters
    "power":                  5.5,   # Transmitter power   [W]
    "frequency":              60e6,  # Center frequency    [Hz]
    "bandwidth":              10e6,  # Radar bandwidth     [Hz]
    "surface_gain":           7.3,   # Antenna gain        [dBi]
    "subsurface_gain":        7.3,   # Antenna gain        [dBi]
    "polarization":          "HH",   # Polarization (dont matter)
    "aperture":                 2,   # Processing aperture [deg]

    # source parameters 
    "sy":                   2.5e3,   # source y location       [m]
    "sz":                    25e3,   # source z location       [m]
    "sdx":                     50,   # source x discretization [m]
    "sx0":                      0,   # source x origin         [m]
    "ns":                    1000,   # source count            [.]

    # receive window parameters
    "rx_window_offset_m": 24.75e3,   # Rx opening window delay [m]
    "rx_window_m":         0.75e3,   # Rx window length        [m]
    "rx_sample_rate":        60e6,   # Rx sample rate          [Hz]

    # surface parameters
    "rms_height":             0.4,   # Facet roughness         [m]
    "buff":                   1.1,   # Buffer for memory alloc [.]
    "altitude":              25e3,   # Approx. alt (for focus) [m]
    "fs":                      50,   # Facet size              [m]

    # atmosphere/subsurface parameters
    "eps_1":                  1.0,   # permittivity of medium 1 
    "eps_2":                 3.15,   # permittivity of medium 2
    "sig_1":                  0.0,   # conductivity of medium 1 [S/m]
    "sig_2":                 1e-6,   # conductivity of medium 2 [S/m]
    "mu_1":                   1.0,   # permeability of medium 1
    "mu_2":                   1.0,   # permeability of medium 2

    # target params (do I need this?)
    "rerad_funct":              1,   # 3-degree boxcar

    # processing parameters (BOOLEAN)
    "convolution": True,   # use convolution-based processing
    "convolution_linear": True,  # use linear convolution instead of circular
    "specular": False,     # use specular computation methods for specific circumstances only
    "lossless": False,      # simulate without loss (spreading not included)

}

# save params
with open("inputs/params.json", "w") as f:
    json.dump(pars, f, indent=4)

with open("inputs/params.pkl", 'wb') as hdl:
    pickle.dump(pars, hdl, protocol=pickle.HIGHEST_PROTOCOL)