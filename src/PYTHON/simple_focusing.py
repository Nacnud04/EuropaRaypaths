import numpy as np

from scipy.optimize import root_scalar

import unit_convs as uc

import matplotlib.pyplot as plt


def snell_intersection(xA, zA, xB, zB, v1, v2):
    """
    Finds the x-coordinate of the point where a ray from A to B intersects the interface at z = 0.
    The ray travels from medium 1 (v1) to medium 2 (v2), obeying Snell's Law.
    
    Parameters:
    xA, zA : coordinates of point A (in medium 1)
    xB, zB : coordinates of point B (in medium 2)
    v1    : velocity in medium 1
    v2    : velocity in medium 2
    
    Returns:
    x : x-coordinate of intersection point on interface (z = 0)
    """

    def f(x):
        # Derivative of travel time with respect to x
        num1 = (x - xA)
        den1 = np.sqrt((x - xA)**2 + zA**2)
        num2 = (xB - x)
        den2 = np.sqrt((xB - x)**2 + zB**2)
        return num1 / (v1 * den1) - num2 / (v2 * den2)

    # Choose a reasonable bracket for root finding
    x_min = min(xA, xB) - abs(xB - xA)
    x_max = max(xA, xB) + abs(xB - xA)

    # Solve f(x) = 0 using root-finding
    result = root_scalar(f, bracket=[x_min, x_max], method='brentq')

    if result.converged:
        return result.root
    else:
        raise RuntimeError("Root finding did not converge.")



def est_slant_range(sx, sz, tx, tz, c1, c2, trim=True, clutter=False):

    # cast lists into numpy arrays
    for s in (sx, sz):
        if type(s) == list:
            s = np.array(s)

    sltrng_ests = []
    # if target is above the surface there is no need for snell intersection
    if tz > 0 or clutter == True:
        sltrng_ests = np.sqrt((sx-tx)**2+(sz-tz)**2)
        return sltrng_ests
    
    # find the surface intersection location
    for x in sx:
        # estimate surface intersections
        ix = snell_intersection(x, sz, tx, tz, c1, c2)
        sltrng_ests.append(np.sqrt((x - ix)**2 + sz**2) + np.sqrt((ix - tx)**2 + tz**2) * (c1/c2))
    sltrng_ests = np.array(sltrng_ests)
    if not trim:
        return sltrng_ests

    # now center the estimate to adjust for target azumith offset
    trc_min = np.argmin(sltrng_ests)
    if trc_min < len(sltrng_ests) // 2:
        newsltrng = sltrng_ests[:-2 * int(len(sltrng_ests) // 2 - trc_min)]
    elif trc_min >= len(sltrng_ests) // 2:
        newsltrng = sltrng_ests[2 * int(trc_min - len(sltrng_ests) // 2):]
    sltrng_ests = newsltrng
    return sltrng_ests


def full_focus_at_center(rdrgrm, par, c1=299792458, sx_linspace=True):

    if sx_linspace:
        sx = par['sx0'] + par['sdx'] * np.arange(par['ns'])
    else:
        raise NotImplementedError("Irregular source spacing is not yet supported.")

    Nr, Na = rdrgrm.shape

    # speed of light in subsurface
    c2 = uc.c2(par, c1=c1)

    # get the slant range
    sltrng    = est_slant_range(sx, par['sz'], par['tx'], par['tz'], c1, c2)
    sltrng_t  = uc.slantrange_to_twoway_us(sltrng, c1=c1)
    sltrng_rb = uc.slantrange_to_rangebin(sltrng, par, c1=c1)

    # turn into matched filter
    match_filter = uc.match_filter(sltrng, par, c1=c1)

    # range cell migration correction
    shift_amounts = sltrng_rb - np.min(sltrng_rb)
    rolled_matrix = np.array([
        np.roll(rdrgrm[:, i], -int(shift_amounts[i]))
        for i in range(rdrgrm.shape[1])
    ]).T

    # fft along azimuth
    fft_len = int(2 * Na)
    pad = fft_len - Na

    # pad at end
    rolled_matrix = np.pad(rolled_matrix, ((0, 0), (0, pad)), mode='constant')
    match_filter = np.pad(match_filter, (0, pad), mode='constant')

    # actually do fft
    az_fft = np.fft.fft(rolled_matrix, axis=1, n=fft_len)
    H_az = np.fft.fft(match_filter, n=fft_len)

    # apply matched filter
    focused_freq = az_fft * H_az[np.newaxis, :]

    # IFFT to turn into the focused image
    focused = np.fft.ifft(focused_freq, axis=1)

    # crop to original size
    start = pad // 2
    end = start + Na
    focused = focused[:, start:end]

    return focused


def focus_middle_only(rdrgrm, par, buffer, c1=299792458, sx_linspace=True):

    if sx_linspace:
        sx = par['sx0'] + par['sdx'] * np.arange(par['ns'])
    else:
        raise NotImplementedError("Irregular source spacing is not yet supported.")

    Nr, Na = rdrgrm.shape

    # speed of light in subsurface
    c2 = uc.c2(par, c1=c1)

    # get the slant range
    sltrng    = est_slant_range(sx, par['sz'], par['tx'], par['tz'], c1, c2)
    sltrng_t  = uc.slantrange_to_twoway_us(sltrng, c1=c1)
    sltrng_rb = uc.slantrange_to_rangebin(sltrng, par, c1=c1)

    # turn into matched filter
    match_filter = uc.match_filter(sltrng, par, c1=c1)

    # range cell migration correction
    shift_amounts = sltrng_rb - np.min(sltrng_rb)
    
    # 0 shift outside of center region
    shift_amounts[:buffer] = 0
    shift_amounts[-buffer:] = 0
    
    # roll radargram
    rolled_matrix = np.array([
        np.roll(rdrgrm[:, i], -int(shift_amounts[i]))
        for i in range(rdrgrm.shape[1])
    ]).T

    # fft along azimuth
    fft_len = int(2 * Na)
    pad = fft_len - Na

    # pad at end
    rolled_matrix = np.pad(rolled_matrix, ((0, 0), (0, pad)), mode='constant')
    match_filter = np.pad(match_filter, (0, pad), mode='constant')
    match_filter[:buffer] = 0
    match_filter[-buffer:] = 0

    # actually do fft
    az_fft = np.fft.fft(rolled_matrix, axis=1, n=fft_len)
    H_az = np.fft.fft(match_filter, n=fft_len)

    # apply matched filter
    focused_freq = az_fft * H_az[np.newaxis, :]

    # IFFT to turn into the focused image
    focused = np.fft.ifft(focused_freq, axis=1)

    # crop to original size
    start = pad // 2
    end = start + Na
    focused = focused[:, start:end]

    return focused


def focus_window(rdrgrm, par, win_center, win_width, c1=299792458, sx_linspace=True, scale=50):

    if sx_linspace:
        sx = par['sx0'] + par['sdx'] * np.arange(par['ns'])
    else:
        raise NotImplementedError("Irregular source spacing is not yet supported.")

    Nr, Na = rdrgrm.shape

    # speed of light in subsurface
    c2 = uc.c2(par, c1=c1)

    # get the slant range
    sltrng    = est_slant_range(sx, par['sz'], par['tx'], par['tz'], c1, c2)
    sltrng_rb = uc.slantrange_to_rangebin(sltrng, par, c1=c1)

    # turn into matched filter
    match_filter = uc.match_filter(sltrng, par, c1=c1)

    # range cell migration correction
    shift_amounts = sltrng_rb - np.min(sltrng_rb)
    rdr_rcmc = np.array([
        np.roll(rdrgrm[:, i], -int(shift_amounts[i]))
        for i in range(rdrgrm.shape[1])
    ]).T

    # turn window into azimuth start and end
    az_start = max(0, win_center - win_width // 2)
    az_end   = min(Na, win_center + win_width // 2)

    # extract windowed azimuth data
    rdr_win = rdr_rcmc[:, az_start:az_end]
    Na_win = rdr_win.shape[1]

    # do range doppler focusing ONLY to the window
    fft_len = 2 * Na_win
    pad = fft_len - Na_win

    # pad data and matched filter to FFT length
    rdr_pad = np.pad(rdr_win, ((0, 0), (0, pad)))
    mf_pad = np.pad(match_filter[az_start:az_end], (0, pad))

    az_fft = np.fft.fft(rdr_pad, axis=1, n=fft_len)
    H = np.fft.fft(mf_pad, fft_len)

    focused_freq = az_fft * H[np.newaxis, :]
    focused_win = np.fft.ifft(focused_freq, axis=1)

    # crop back to window size
    start = pad // 2
    end = start + Na_win
    focused_win = focused_win[:, start:end]

    # combine with the unfocused portion
    focused = scale*rdrgrm.copy()
    focused[:, az_start:az_end] = focused_win

    return focused


# function to focus entire image with a given aperture
def focus_rdrgrm(rdrgrm, par, st=None, en=None):

    # NOTE: This function requires some additional parameters. Such as:
    # spacing: approximate spacing between sources
    # altitude: approximate altitude of the source in meters

    # Nr = # of range bins
    # Na = # of azimuth traces
    Nr, Na = rdrgrm.shape 

    rdr_f = np.zeros_like(rdrgrm)

    # first we need to iterate over each row. 
    for i in range(Nr):

        if st and en:
            if i < st or i > en:
                continue

        # get the range assuming no media change
        rng = par['rx_window_m'] * (i / Nr) + par['rx_window_offset_m']

        # given satellite aperture how wide should half the SAR aperture be?
        apt_m = rng * np.sin(np.radians(par['aperture']))
        # convert that into approximate number of traces
        apt_smpl = int(2*apt_m / par["spacing"])
        apt_hsmpl = int(apt_m / par["spacing"])

        # compute a slant range which would work for anything at a given range bin
        # NOTE: This should eventually be made to work for varying surface topography
        sx = np.linspace(-1*apt_m, 1*apt_m, apt_smpl)
        sz = par["altitude"]
        tx = 0
        tz = par["altitude"] - rng
        sltrng = est_slant_range(sx, sz, tx, tz, par["eps_1"], par["eps_2"], trim=False, clutter=True)
        sltrng_rb = uc.slantrange_to_rangebin(sltrng, par)

        # turn slant range into matched filter
        mth_filt = uc.match_filter(sltrng, par)

        # iterate over every single trace
        for j in range(Na):

            # get window of data within aperture
            az_st = max(0, j-apt_hsmpl)
            az_en = min(Na, j+apt_hsmpl)
            window = rdrgrm[:, az_st:az_en]

            # range correction
            # NOTE: to be proper I should crop the slantrange array to work for edges
            shift_amounts = sltrng_rb - i
            rdr_rcmc = np.array([
                np.roll(window[:, k], -int(shift_amounts[k]))
                for k in range(window.shape[1])
            ]).T

            # get single row we care about and do fft
            fft_len = int(2*apt_smpl)
            pad = fft_len - (az_en - az_st)

            # first we need to pad things
            row_pad = np.pad(rdr_rcmc[i,:], (0, pad), mode="constant")

            # now we pad the matched filter
            filt_pad = np.pad(mth_filt, (0, pad), mode="constant")

            # do fft
            az_fft = np.fft.fft(row_pad, n=fft_len)
            ft_fft = np.fft.fft(filt_pad, n=fft_len)

            # apply matched filter
            focused_freq = az_fft * ft_fft

            # IFFT to focused image
            focused_row = np.fft.ifft(focused_freq)

            # get the pixel we care about
            rdr_f[i, j] = focused_row[(j - az_st) + pad // 2]

        print(f"{i}/{Nr}", end="       \r")

    return rdr_f
