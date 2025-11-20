import numpy as np
import matplotlib.pyplot as plt
import glob, sys, pickle

sys.path.append("../../src")
from focus import est_slant_range

names = ["NoConv", "CircConv", "LinConv"]

for n in names:

    with open(f"{n}.pkl", 'rb') as hdl:
        params = pickle.load(hdl)

    filenames = glob.glob(f"{n}/s*.txt")

    # sort filenames to ensure correct order
    filenames.sort()

    rdrgrm = []

    for i, f in enumerate(filenames):
        if i < params['ns']:
            arr = np.loadtxt(f).T
            rdrgrm.append(arr[0] + 1j * arr[1])

    rdrgrm = np.array(rdrgrm).T
    print(rdrgrm.shape)

    # try to focus the radargram

    sx = params['sx0'] + params['sdx'] * np.arange(params['ns']) # source x locations [m]
    sy = params['sy']                          # source y locations [m]
    sz = params['sz']                       # source z locations [m]

    c1 = 299792458
    c2 = c1 / np.sqrt(params["eps_2"])

    rb = int((params["rx_window_m"] / c1) / (1 / params["rx_sample_rate"]))
    dm = c1 / params["rx_sample_rate"]

    sltrng   = est_slant_range(sx, sz, params["tx"], params["tz"], c1, c2)
    sltrng_t = 2 * 10**6 * sltrng / c1  # in microseconds

    # compute sample-bin indices (meters -> samples)
    slt_rb = ((sltrng - params["rx_window_offset_m"]) // dm).astype(int)

    k = (2 * np.pi) / (c1 / params["frequency"])
    match_filter = np.exp(-2j * k * sltrng)

    def lin_to_db(x):
        return 10 * np.log10(x)

    lst = [lin_to_db(np.abs(rdrgrm)), np.angle(rdrgrm)]
    
    np.savez(f"rdrgrms/{n}_raw.npz", rdrgrm=lin_to_db(np.abs(rdrgrm)))

    Nr, Na = rdrgrm.shape

    # --- 2. Range Cell Migration Correction (RCMC) ---
    shift_amounts = slt_rb - np.min(slt_rb)
    rolled_matrix = np.array([
        np.roll(rdrgrm[:, i], -int(shift_amounts[i]))
        for i in range(rdrgrm.shape[1])
    ]).T

    # --- 3. Azimuth matched filter ---
    k = (2 * np.pi) / (c1 / params["frequency"])
    match_filter = np.exp(-2j * k * sltrng)

    # --- 1. Azimuth FFT ---
    fft_len = int(2 * Na)
    pad = fft_len - Na

    # Pad only at the end
    rolled_matrix = np.pad(rolled_matrix, ((0, 0), (0, pad)), mode='constant')
    match_filter = np.pad(match_filter, (0, pad), mode='constant')

    # FFT along azimuth
    az_fft = np.fft.fft(rolled_matrix, axis=1, n=fft_len)
    H_az = np.fft.fft(match_filter, n=fft_len)

    # Apply filter
    focused_freq = az_fft * H_az[np.newaxis, :]

    # --- 4. Inverse FFT ---
    focused = np.fft.ifft(focused_freq, axis=1)

    # Crop to original azimuth size (centered)
    start = pad // 2
    end = start + Na
    focused = focused[:, start:end]

    np.savez(f"rdrgrms/{n}_focused.npz", focused=lin_to_db(np.abs(focused)))

