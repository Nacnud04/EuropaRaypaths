import numpy as np
import numba as nb # type: ignore
from scipy.optimize import root_scalar

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
    
    sltrng_ests = []
    for x in sx:
        # estimate surface intersections
        ix = snell_intersection(x, sz, tx, tz, c1, c2)
        sltrng_ests.append(np.sqrt((x - ix)**2 + sz**2) + np.sqrt((ix - tx)**2 + tz**2) * (c1/c2))
        #sltrng_ests.append(np.sqrt((x - ix)**2 + sz**2) + np.sqrt((ix - tx)**2 + tz**2))
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

@nb.njit
def focus_pix_jit(rdr, t, T, bins, filter, rb):

    n_cols = len(bins)
    half_cols = n_cols // 2
    focused_pix = 0.0 + 0.0j

    for i in range(n_cols):
        trc = T + i - half_cols  # Corrected: azimuth index relative to T
        if trc < 0 or trc >= rdr.shape[1]:
            continue

        bin_idx = int(bins[i])
        bin_idx -= np.min(bins)  # Still needed for JIT compile compatibility
        bin_idx += t

        if bin_idx <= 0 or bin_idx >= rb:
            continue

        amp = rdr[bin_idx, trc]
        filt = filter[i]
        focused_pix += amp * filt

    return focused_pix



@nb.njit(parallel=True)
def focus_jit(rdr, bins, match_filter, rb):

    """
    Focus entire radar image.
    NOTE: This uses the same matched filter and range bin history for all.
          This causes distortions if attempting to focus things very far apart.
          This is because as the target increases in depth the hyperbolic response
          ever so slightly changes which is not accounted for.
    """

    rows, cols = rdr.shape
    focused = np.zeros((rows, cols), dtype=np.complex128)

    for t in nb.prange(rows):
        for T in range(cols):
            focused[t, T] = focus_pix_jit(rdr, t, T, bins, match_filter, rb)

    return focused