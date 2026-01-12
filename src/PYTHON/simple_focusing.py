import numpy as np

from scipy.optimize import root_scalar

import unit_convs as uc


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



def est_slant_range(sx, sz, tx, tz, c1, c2, trim=True):

    # cast lists into numpy arrays
    for s in (sx, sz):
        if type(s) == list:
            s = np.array(s)
    
    # find the surface intersection location
    sltrng_ests = []
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
