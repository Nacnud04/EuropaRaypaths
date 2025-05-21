import math
import numpy as np

def cart_to_sp(coord, vec=False):
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Parameters:
    coord (tuple): coordinate
    
    Returns:
    tuple: (r, theta, phi) where
        r is the radius
        theta is the polar angle (in radians)
        phi is the azimuthal angle (in radians)
    """
    if vec:
        r = np.sqrt(coord[0,:,:]**2 + coord[1,:,:]**2 + coord[2,:,:]**2)
        phi = np.arccos(coord[2,:,:] / r)
        phi[r == 0] = 0
        theta = np.arctan2(coord[1,:,:], coord[0,:,:])
        return np.stack((r, theta, phi))
    else:
        x, y, z = coord
        r = math.sqrt(x**2 + y**2 + z**2)
        phi = math.acos(z / r) if r != 0 else 0
        theta = math.atan2(y, x)
        return np.array((r, theta, phi))

def sp_to_cart(coord):
    r, th, ph = coord
    
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)

    return np.array((x, y, z))

def dbi_to_db(dbi):
    return dbi - 2.15

def db_to_mag(db):
    return 10 ** (db / 20)

def dbi_to_mag(dbi):
    return db_to_mag(dbi_to_db(dbi))

def mag_to_db(mag):
    return 20 * np.log10(mag)

def db_to_dbi(db):
    return db + 2.15

def mag_to_dbi(mag):
    return db_to_dbi(mag_to_db(mag))

def radar_eq(P, G, s, lam, R):
    return (P * (G**2) * s * (lam**2)) / ((4*np.pi)**3 * (R**4))

def normalize_vectors(arr):

    magnitudes = np.linalg.norm(arr, axis=0)
    magnitudes[magnitudes == 0] = 1
    arr_normalized = arr / magnitudes

    return np.copy(arr_normalized)

def comp_refracted_vectorized(surf_norms, rp_to_facet, vel1, vel2, rev=False):
    # if computing the reverse refraction reverse the relative vector
    if rev:
        # get inbound raypath relative to facet direction
        relative = rp_to_facet - surf_norms
        relative *= -1
    else:
        # get inbound raypath relative to facet direction
        relative = rp_to_facet + surf_norms
        relative[2, :, :] *= -1

    # convert to spherical coordinates
    inbound = cart_to_sp(relative, vec=True)
    inbound[2, :, :] = np.abs(inbound[2, :, :] - np.pi)

    # NOTE: I am not sure if for a reversed raypath this should be vel1/vel2 instead.
    # snells law to find new phi value
    k = (vel2 / vel1) * np.sin(inbound[2, :, :])

    # set new phi value
    inbound[2, :, :] = np.pi - np.arcsin(k)

    # where no energy makes it set as none
    for i in range(3):
        inbound[i, :, :][np.abs(k) > 1] = None

    return inbound

def fast_dot_product(A, B):

    if len(B.shape) == 1:
        return np.einsum('ijk,i->jk', A, B)
    else:
        return np.einsum('ijk,ijk->jk', A, B)