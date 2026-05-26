import pickle, json
import numpy as np

def export_params(params, filename, directory="inputs"):

    # check for nulls
    for k in params.keys():
        if params[k] is None:
            raise ValueError(f"Found null for key [{k}] when exporting parameter dict")

    with open(f"{directory}/{filename}.json", "w") as f:
        json.dump(params, f, indent=4)

    with open(f"{directory}/{filename}.pkl", 'wb') as hdl:
        pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

def gen_params(platform, media, domainpar, recpar, sourcepar, par=None):

    pars = {}
    
    # first start with unchanging parameters
    # atmosphere
    pars["eps_1"] = 1.0
    pars["sig_1"] = 0.0
    pars["mu_1"]  = 1.0
    pars["mu_2"]  = 1.0

    # processing parameters
    pars["convolution"] = True
    pars["convolution_linear"] = True
    pars["specular"] = False
    pars["lossless"] = False
    pars["incoherent"] = False

    # debug pars
    pars["debug_surface"] = False

    # default target behavior (doesn't matter if non-specular)
    pars["rerad_funct"] = 2 # 1 degree boxcar

    # low level params
    pars["buff"] = 1.1

    # add instrument parameters
    if platform == "REASON_VHF":
        pars["power"] = 100
        pars["frequency"] = 60e6
        pars["bandwidth"] = 10e6
        pars["surface_gain"] = 7.3
        pars["subsurface_gain"] = pars["surface_gain"]
        pars["polarization"] = "HH"
    else:
        raise ValueError(f"platform {platform} not recognized.")
    
    # add media parameters
    if media == "planetary_ice":
        pars["eps_2"] = 3.15
        pars["sig_2"] = 1e-6
        pars["rms_height"] = 0.4
    elif media == "vacuum":
        pars['eps_2'] = 1.0
        pars['sig_2'] = 0
        pars['rms_height'] = 0
    else:
        raise ValueError(f"media {media} not recognized.")
    
    # add domain parameters:
    if "fs" not in domainpar.keys():
        raise ValueError(f"Facet size (fs) missing from input dictionary domainpar")
    pars['fs'] = domainpar['fs']
    if "ox" in domainpar.keys():
        print(f"Parameter generation detected simple surface type domain inputs. Adding to parameter dict.")
        for k in domainpar.keys():
            if k not in pars.keys():
                pars[k] = domainpar[k]

    # add receive window parameters
    rec_keys = ["rx_window_m", "rx_sample_rate"]
    for k in rec_keys:
        if k not in recpar.keys():
            raise ValueError(f"key {k} neccessary in recpar which has keys {recpar.keys()}")
        pars[k] = recpar[k]
    if "rx_window_position_file" in recpar.keys():
        pars["rx_window_position_file"] = recpar["rx_window_position_file"]
    if "rx_window_offset_m" in recpar.keys():
        pars["rx_window_offset_m"] = recpar["rx_window_offset_m"]
    if "rx_window_position_file" not in recpar.keys() and "rx_window_offset_m" not in recpar.keys():
        raise ValueError(f"input dictionary recpar needs either a rx window position file or a static rx window offset")

    # add source params
    src_keys = ["ns", "aperture"]
    for k in src_keys:
        if k not in sourcepar.keys():
            raise ValueError(f"key {k} necessary in sourcepar which has keys {sourcepar.keys()}")
        pars[k] = sourcepar[k]
    if "source_path_file" in sourcepar.keys():
        pars['source_path_file'] = sourcepar['source_path_file']
    else:
        linear_keys = ['sx0', 'sy', 'sz', 'sdx']
        for lk in linear_keys:
            pars[lk] = sourcepar[lk]

    # update with custom optional parameters
    if par is not None:
        for k in par.keys():
            pars[k] = par[k]

    # some additional parameters for ease of calculation later
    pars['lam'] = 299792458 / pars['frequency']
    pars['k']   = (2 * np.pi) / pars['lam']

    if 'sz' in pars.keys():
        pars['altitude'] = pars['sz']

    return pars

def vert_source_path(params, minZ, maxZ, filename, direct="inputs"):

    sz = np.linspace(minZ, maxZ, params['ns'])
    sy = np.zeros_like(sz)
    sx = np.zeros_like(sz)

    with open(f"{direct}/{filename}.txt", 'w') as f:
        for i in range(params['ns']):
            if i == params['ns'] - 1:
                f.write(f"{sx[i]},{sy[i]},{sz[i]}")
            else:
                f.write(f"{sx[i]},{sy[i]},{sz[i]}\n")

    return sz

def track_Z_rxwin(sz, ahead, filename, direct="inputs"):

    rx_window_positions = sz - ahead

    with open(f"{direct}/{filename}.txt", 'w') as f:
        for pos in rx_window_positions:
            f.write(f"{pos}\n")