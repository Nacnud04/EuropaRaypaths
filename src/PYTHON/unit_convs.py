import numpy as np


def lin_to_db(x):
    return 10 * np.log10(x)

def c2(params, c1=299792458):
    return c1 / np.sqrt(params['eps_2'])

def slantrange_to_twoway_us(sltrng, c1=299792458):
    return 2 * 10**6 * sltrng / c1

def slantrange_to_rangebin(sltrng, params, c1=299792458):
    dm = c1 / params["rx_sample_rate"]
    return ((sltrng - params["rx_window_offset_m"]) // dm).astype(int)

def wav_k(params, c1=299792458):
    return (2 * np.pi) / (c1 / params['frequency'])

def match_filter(sltrng, params, c1=299792458):
    k = wav_k(params)
    return np.exp(-2j * k * sltrng)

def m_to_km(*inpts):
    return [inpt/1e3 for inpt in inpts]

def km_to_m(*inpts):
    return [inpt*1e3 for inpt in inpts]

def dLat_to_m(lat_delta, radius):
    return np.radians(np.abs(lat_delta)) * radius

def interpolate_sources(ns, sat_x, sat_y, sat_z):

    interp_rng = np.linspace(0, 1, ns)
    orig_rng = np.linspace(0, 1, len(sat_x))

    sats = []
    for sat in (sat_x, sat_y, sat_z):
        sat = np.interp(interp_rng, orig_rng, sat)
        sats.append(sat)
        
    return sats

def upsample(n, data):

    interp_rng = np.linspace(0, 1, n)
    orig_rng = np.linspace(0, 1, len(data))

    return np.interp(interp_rng, orig_rng, data)

def upsample_df(n, df):
    interp_rng = np.linspace(0, 1, n)
    orig_rng = np.linspace(0, 1, len(df))

    numeric_df = df.select_dtypes(include=[np.number])

    return numeric_df.apply(lambda col: np.interp(interp_rng, orig_rng, col))

def scale_range(data, vmin, vmax):

    dat_clip = np.clip(data, vmin, vmax)
    scaled = (dat_clip - vmin) / (vmax - vmin)

    return scaled
