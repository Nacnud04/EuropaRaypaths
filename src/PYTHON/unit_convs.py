import numpy as np
from scipy.interpolate import interp1d

c = 299792458

def lin_to_db(x, warning=False):
    if np.any(x <= 0):
        if warning:
            print("Warning: Found non-positive values in input to lin_to_db")
        x = np.maximum(x, np.min(x[x>0]))  # Avoid log(0)
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

def estimate_spacing(sat_x, sat_y, sat_z):

    dx = sat_x - np.roll(sat_x, 1)
    dy = sat_y - np.roll(sat_y, 1)
    dz = sat_z - np.roll(sat_z, 1)
    d = np.sqrt(dx**2 + dy**2 + dz**2)
    spacing = np.nanmedian(d)

    return spacing

def planetocentric_to_cartesian(rad, lat, lon):

    x = rad * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = rad * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z = rad * np.sin(np.radians(lat))

    return x, y, z

def normalize(x, y, z, nmult=1, parts=False):

    n_hat = np.stack((x, y, z), axis=-1) * nmult
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)

    if parts:
        return n_hat[:, 0], n_hat[:, 1], n_hat[:, 2]

    return n_hat

def trc_depth_2_facets(trc, depth, aeroid, upsample=5, min_depth=None):

    tx_all, ty_all, tz_all = [], [], []
    tnx_all, tny_all, tnz_all = [], [], []

    for trc_layer, depth_layer in zip(trc, depth):

        trc_layer = np.array(trc_layer)
        depth_layer = np.array(depth_layer)

        # sort by trace index
        sort_idx = np.argsort(trc_layer)
        trc_layer = trc_layer[sort_idx]
        depth_layer = depth_layer[sort_idx]

        if min_depth is not None:
            mask = depth_layer >= min_depth
            if np.sum(mask) < 2: continue # need at least 2 point for interpolation
            trc_layer = trc_layer[mask]
            depth_layer = depth_layer[mask]

        # create dense trace sampling
        trc_dense = np.linspace(trc_layer.min(),
                                trc_layer.max(),
                                len(trc_layer) * upsample)

        # interpolate depth
        depth_interp = interp1d(trc_layer, depth_layer, kind='linear')
        depth_dense = depth_interp(trc_dense)

        # interp lat lon
        lat_interp = interp1d(trc_layer, aeroid['LAT'][trc_layer])
        lon_interp = interp1d(trc_layer, aeroid['LON'][trc_layer])
        srad_interp = interp1d(trc_layer, aeroid['SRAD'][trc_layer])

        lats = lat_interp(trc_dense)
        lons = lon_interp(trc_dense)
        radii = srad_interp(trc_dense) - (depth_dense + 315.4)

        # move to cartesian
        tx, ty, tz = planetocentric_to_cartesian(radii*1e3, lats, lons)
        tnx, tny, tnz = normalize(tx, ty, tz, parts=True)

        tx_all.append(tx)
        ty_all.append(ty)
        tz_all.append(tz)
        tnx_all.append(tnx)
        tny_all.append(tny)
        tnz_all.append(tnz)

    return (np.concatenate(tx_all),
            np.concatenate(ty_all),
            np.concatenate(tz_all),
            np.concatenate(tnx_all),
            np.concatenate(tny_all),
            np.concatenate(tnz_all))