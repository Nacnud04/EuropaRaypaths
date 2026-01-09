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
