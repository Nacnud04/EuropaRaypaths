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
        r = np.sqrt(coord[:,:,0]**2 + coord[:,:,1]**2 + coord[:,:,2]**2)
        phi = np.arccos(coord[:,:,2] / r)
        phi[r == 0] = 0
        theta = np.arctan2(coord[:,:,1], coord[:,:,0])
        return np.stack((r, theta, phi), axis=2)
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

def radar_eq(P, G, s, lam, R):
    return (P * (G**2) * s * (lam**2)) / ((4*np.pi)**3 * (R**4))