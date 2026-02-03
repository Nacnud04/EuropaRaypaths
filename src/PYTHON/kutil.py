"""

kutil.py
Utilities for managing data relevant to the KOROLEV crater example

Author:      Duncan Byrne
Institution: University of Colorado Boulder
Department:  Aerospace Engineering Science
Date:        2026-01-13
Contact:     duncan.byrne@colorado.edu

"""

import numpy  as np
import pandas as pd
import tifffile, pickle

import unit_convs as uc
import output_handling as oh
import matplotlib.pyplot as plt

# load sharad orbit from file
def load_sharad_orbit(DIRECTORY, OBS):
   
    filename = f"{DIRECTORY}/s_{OBS}_geom.tab"
    col_names = ["COL", "TIME", "LAT", "LON", "MRAD", "SRAD", "RVEL", "TVEL", "SZA", "PHSE"]
    geometry = pd.read_csv(filename, header=None, names=col_names)
    print(f"Found {len(geometry)} observations in {filename}")
    
    altitude = geometry['SRAD'] - geometry['MRAD']
    print(f"Average altitude of {np.mean(altitude):3.2f} km")

    return geometry

def load_sharad_orbit_MOLA(DIRECTORY, OBS):

    filename = f"{DIRECTORY}/s_{OBS}_geom_MOLA.csv"
    col_names = ["COL", "TIME", "LAT", "LON", "MRAD", "SRAD", "RVEL", "TVEL", "SZA", "PHSE", "TOPO"]
    geometry = pd.read_csv(filename, header=None, names=col_names, skiprows=1)
    print(f"Found {len(geometry)} observations in {filename}")
    
    altitude = geometry['SRAD'] - geometry['MRAD']
    print(f"Average altitude of {np.mean(altitude):3.2f} km")

    geometry["TOPO"] += 3396e3
    
    # range from spacecraft to surface
    geometry["SRANGE"] = geometry['SRAD'] * 1e3 - geometry["TOPO"]

    return geometry

def load_sharad_orbit_AEROID(DIRECTORY, OBS):

    filename = f"{DIRECTORY}/s_{OBS}_geom_AEROID.csv"
    col_names = ["COL", "TIME", "LAT", "LON", "MRAD", "SRAD", "RVEL", "TVEL", "SZA", "PHSE", "AEROID"]
    geometry = pd.read_csv(filename, header=None, names=col_names, skiprows=1)
    print(f"Found {len(geometry)} observations in {filename}")
    
    altitude = geometry['SRAD'] - geometry['MRAD']
    print(f"Average altitude of {np.mean(altitude):3.2f} km")

    geometry["AEROID"] += 3396e3

    return geometry


def load_sharad_orbit_PKL(DIRECTORY, OBS):

    filename = f'{DIRECTORY}/s_{OBS}_sources.pkl'
    with open(filename, 'rb') as file:
        geometry = pickle.load(file)

    return geometry


def load_COSHARPS_rdrgrm(DIRECTORY, OBS):

    lines = 3600
    samples = 2656
    dtype = np.float32

    img = np.fromfile(f'{DIRECTORY}/s_{OBS}_rgram.img', dtype=dtype)

    img = img.reshape((lines, samples))

    return img


# convert from planetocentric to martian cartesian
def planetocentric_to_cartesian(rad, lat, lon):

    x = rad * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = rad * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z = rad * np.sin(np.radians(lat))

    return x, y, z


# create normal vector based on geometry for sharad
def sharad_normal(sat_x, sat_y, sat_z, nmult=1):
    
    # oddly since we are in spherical and we want the normal vector to
    # be in the direction perpindicular to the "sphere" of mars our
    # normal vector is simply the normalized position vector

    n_hat = np.stack((sat_x, sat_y, sat_z), axis=-1) * nmult
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)

    return n_hat


# source exporting (with normals)
def sources_norms_to_file(DIRECTORY, OBS, sx, sy, sz, snx, sny, snz):

    i = 0

    source_file = f"{DIRECTORY}/s_{OBS}_srcs.txt"
    f = open(source_file, 'w')

    for x, y, z, nx, ny, nz in zip(sx, sy, sz, snx, sny, snz):

        if i != len(sx) - 1:
            f.write(f"{x},{y},{z},{nx},{ny},{nz}\n")

        else:
            f.write(f"{x},{y},{z},{nx},{ny},{nz}")

        i += 1

    f.close()

    print(f"Exported {i} sources to {source_file}")


# source exporting (without normals)
def sources_to_file(DIRECTORY, OBS, sx, sy, sz):

    i = 0

    source_file = f"{DIRECTORY}/s_{OBS}_srcs.txt"
    f = open(source_file, 'w')

    for x, y, z in zip(sx, sy, sz):

        if i != len(sat_x) - 1:
            f.write(f"{x},{y},{z}\n")

        else:
            f.write(f"{x},{y},{z}")

        i += 1

    f.close()

    print(f"Exported {i} sources to {source_file}")


# source exporting to obj (with normals)
def sources_norms_to_obj(DIRECTORY, OBS, sx, sy, sz, snx, sny, snz, nscale=0.5e3):

    obj_file = f"{DIRECTORY}/s_{OBS}_srcs.obj"

    with open(obj_file, 'w') as f:

        f.write("# Transit of SHARAD over KOROLEV crater\n")

        i = 0

        for x, y, z, nx, ny, nz in zip(sx, sy, sz, snx, sny, snz):
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            f.write(f"v {x+nx*nscale:.6f} {y+ny*nscale:.6f} {z+nz*nscale:.6f}\n")
            f.write(f"l {i+1} {i+2}\n")

            i += 2

        print(f"Exported {i} verticies to {obj_file}")


# source exporting to obj (without normals)
def sources_to_obj(DIRECTORY, OBS, sx, sy, sz, nscale=0.5e3):

    obj_file = f"{DIRECTORY}/s_{OBS}_srcs.obj"

    with open(obj_file, 'w') as f:

        f.write("# Transit of SHARAD over KOROLEV crater\n")

        i = 0

        for x, y, z in zip(sx, sy, sz):
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        print(f"Exported {i} verticies to {obj_file}")


# load tiff of cropped mola
def load_cropped_mola_tif(filename):

    # load tiff
    with tifffile.TiffFile(filename) as tif:
        page = tif.pages[0]
        data = page.asarray()  # raw Int16 image

        tags = page.tags
        model_tiepoint = tags["ModelTiepointTag"].value
        model_pixelscale = tags["ModelPixelScaleTag"].value

    # find tiff origin
    tpnt_lat = model_tiepoint[4]
    tpnt_lon = model_tiepoint[3]
    print(f"Longitude tiepoint: {tpnt_lon:6.3f}")
    print(f"Latitude  tiepoint: {tpnt_lat:6.3f}")

    # get pixel scale
    lat_scale, lon_scale = model_pixelscale[:2]
    lat_scale *= -1
    lat_scale_rad = np.radians(lat_scale)
    lon_scale_rad = np.radians(lon_scale)
    print(f"Latitude scale: {lat_scale}")
    print(f"Longitude scale: {lon_scale}")

    # get image coverage/dimensions
    rows, columns = data.shape
    print(f"TIFF Shape: {rows}, {columns}")
    lonmin, lonmax = tpnt_lon, tpnt_lon + columns * lon_scale
    latmax, latmin = tpnt_lat, tpnt_lat + rows * lat_scale
    lats = np.arange(rows) * lat_scale + tpnt_lat
    lons = np.arange(columns) * lon_scale + tpnt_lon
    llons, llats = np.meshgrid(lons, lats)

    extent = (lonmin, lonmax, latmin, latmax)

    # move values into dictionary
    tiff_par = {

        "rows": rows, "cols": columns,

        "tpnt_lat": tpnt_lat, "tpnt_lon": tpnt_lon,
        "scl_lat": lat_scale, "scl_lon": lon_scale,
        "scl_latr": lat_scale_rad, "scl_lonr": lon_scale_rad,

        "lonmin":lonmin, "lonmax":lonmax,
        "latmin":latmin, "latmax":latmax,

        "lats":lats, "lons": lons,

        "llats":llats,  "llons": llons,

        "extent":extent
    }

    return data, tiff_par


rdr_dtype = np.dtype([
    ("SCET_BLOCK_WHOLE", "<u4"),
    ("SCET_BLOCK_FRAC", "<u2"),
    ("TLM_COUNTER", "<u4"),
    ("FMT_LENGTH", "<u2"),
    ("SCET_OST_WHOLE", "<u4"),
    ("SCET_OST_FRAC", "<u2"),
    ("OST_LINE_NUMBER", "<u1"),
    ("PULSE_REPETITION_INTERVAL", "<u1"),
    ("PHASE_COMPENSATION_TYPE", "<u1"),
    ("DATA_TAKE_LENGTH", "<u4"),
    ("OPERATIVE_MODE", "<u1"),
    ("MANUAL_GAIN_CONTROL", "<u1"),
    ("COMPRESSION_SELECTION", "<u1"),
    ("CLOSED_LOOP_TRACKING", "<u1"),
    ("TRACKING_DATA_STORAGE", "<u1"),
    ("TRACKING_PRE_SUMMING", "<u1"),
    ("TRACKING_LOGIC_SELECTION", "<u1"),
    ("THRESHOLD_LOGIC_SELECTION", "<u1"),
    ("SAMPLE_NUMBER", "<u1"),
    ("ALPHA_BETA", "<u1"),
    ("REFERENCE_BIT", "<u1"),
    ("THRESHOLD", "<u1"),
    ("THRESHOLD_INCREMENT", "<u1"),
    ("INITIAL_ECHO_VALUE", "<u1"),
    ("EXPECTED_ECHO_SHIFT", "<u1"),
    ("WINDOW_LEFT_SHIFT", "<u1"),
    ("WINDOW_RIGHT_SHIFT", "<u1"),
    ("DATA_BLOCK_ID", "<u4"),
    ("SCIENCE_DATA_SOURCE_COUNTER", "<u2"),
    ("SCIENTIFIC_DATA_TYPE", "<u1"),
    ("SEGMENTATION_FLAG", "<u1"),
    ("DMA_ERROR", "<u1"),
    ("TC_OVERRUN", "<u1"),
    ("FIFO_FULL", "<u1"),
    ("TEST", "<u1"),
    ("DATA_BLOCK_FIRST_PRI", "<u4"),
    ("TIME_DATA_BLOCK_WHOLE", "<u4"),
    ("TIME_DATA_BLOCK_FRAC", "<u2"),
    ("SDI_BIT_FIELD", "<u2"),
    ("TIME_N", "<f4"),
    ("RADIUS_N", "<f4"),
    ("TANGENTIAL_VELOCITY_N", "<f4"),
    ("RADIAL_VELOCITY_N", "<f4"),
    ("TLP", "<f4"),
    ("TIME_WPF", "<f4"),
    ("DELTA_TIME", "<f4"),
    ("TLP_INTERPOLATE", "<f4"),
    ("RADIUS_INTERPOLATE", "<f4"),
    ("TANGENTIAL_VELOCITY_INTERPOLATE", "<f4"),
    ("RADIAL_VELOCITY_INTERPOLATE", "<f4"),
    ("END_TLP", "<f4"),
    ("S_COEFFS", "<f4", (8,)),
    ("C_COEFFS", "<f4", (7,)),
    ("SLOPE", "<f4"),
    ("TOPOGRAPHY", "<f4"),
    ("PHASE_COMPENSATION_STEP", "<f4"),
    ("RECEIVE_WINDOW_OPENING_TIME", "<f4"),
    ("ANTENNA_RELATIVE_GAIN", "<f4"),
    ("ECHO_SAMPLES_REAL", "<f4", (667,)),
    ("ECHO_SAMPLES_IMAG", "<f4", (667,)),
    ("N_PRE", "<u2"),
    ("BLOCK_NR", "<u2"),
    ("BLOCK_ROWS", "<u2"),
    ("DOPPLER_BW", "<f4"),
    ("DOPPLER_CENTROID", "<f4"),
    ("AZ_TIME_SPACING", "<f4"),
    ("AZ_RES", "<f4"),
    ("T_INT", "<f4"),
    ("AVG_TAN_VELOCITY", "<f4"),
    ("RANGE_SHIFT", "<i2"),
    ("EPHEMERIS_TIME", "<f8"),
    ("GEOMETRY_EPOCH", "S23"),
    ("SOLAR_LONGITUDE", "<f8"),
    ("ORBIT_NUMBER", "<i4"),
    ("MARS_SC_POSITION_VECTOR", "<f8", (3,)),
    ("SPACECRAFT_ALTITUDE", "<f8"),
    ("SUB_SC_EAST_LONGITUDE", "<f8"),
    ("SUB_SC_PLANETOCENTRIC_LATITUDE", "<f8"),
    ("SUB_SC_PLANETOGRAPHIC_LATITUDE", "<f8"),
    ("MARS_SC_VELOCITY_VECTOR", "<f8", (3,)),
    ("MARS_SC_RADIAL_VELOCITY", "<f8"),
    ("MARS_SC_TANGENTIAL_VELOCITY", "<f8"),
    ("LOCAL_TRUE_SOLAR_TIME", "<f8"),
    ("SOLAR_ZENITH_ANGLE", "<f8"),
    ("SC_PITCH_ANGLE", "<f8"),
    ("SC_YAW_ANGLE", "<f8"),
    ("SC_ROLL_ANGLE", "<f8"),
    ("MRO_SAMX_INNER_GIMBAL_ANGLE", "<f8"),
    ("MRO_SAMX_OUTER_GIMBAL_ANGLE", "<f8"),
    ("MRO_SAPX_INNER_GIMBAL_ANGLE", "<f8"),
    ("MRO_SAPX_OUTER_GIMBAL_ANGLE", "<f8"),
    ("MRO_HGA_INNER_GIMBAL_ANGLE", "<f8"),
    ("MRO_HGA_OUTER_GIMBAL_ANGLE", "<f8"),
    ("DES_TEMP", "<f4"),
    ("DES_5V", "<f4"),
    ("DES_12V", "<f4"),
    ("DES_2V5", "<f4"),
    ("RX_TEMP", "<f4"),
    ("TX_TEMP", "<f4"),
    ("TX_LEV", "<f4"),
    ("TX_CURR", "<f4"),
    ("QUALITY_CODE", "<u1"),
])

RADARGRAM_RETURN_INTERVAL = {
    "0554201":312041,
}

ADC_SAMP_INT = 0.0375e-6
RDR_RETR_INT = 0.075e-6
RNG_BIN_INT  = 0.075e-6
c = 299792458

def load_SHARAD_RDR(filename, st=None, en=None, latmin=None, latmax=None):

    data = np.fromfile(filename, dtype=rdr_dtype)
    
    # now that we have read in the data we need to find the exact radius and distance
    # from the spacecraft that the RADARGRAM RETURN INTERVAL cooresponds to
    #return_interval_mars_radius = (RADARGRAM_RETURN_INTERVAL["0554201"] * RNG_BIN_INT * c) / 2

    if st and en:

        data = data[st:en]

    if latmin and latmax:
    
        data = data[
            (data['SUB_SC_PLANETOCENTRIC_LATITUDE'] > latmin) *\
            (data['SUB_SC_PLANETOCENTRIC_LATITUDE'] < latmax)
        ]

    return data

def rxOpenWindow(data, filename, ns):

    rx_win = data['RECEIVE_WINDOW_OPENING_TIME'] * (3/80e6) - 11.98e-6 + 1428e-6
    rx_win_dist = rx_win * c / 2
    rx_win_upsample = uc.upsample(ns, rx_win_dist)

    # save as npy
    np.save(f"{filename}.npy", rx_win_upsample)

    with open(f"{filename}.txt", 'w') as f:
        for pos in rx_win_upsample:
            f.write(f"{round(pos)}\n")
    
def plot_SHARAD_RDR(data, geometry):

    rx_win = data['RECEIVE_WINDOW_OPENING_TIME'] * (3/80e6) - 11.98e-6 + 1428e-6
    rx_win_dist = rx_win * c / 2
    min_rx_win = np.min(rx_win_dist)
    rx_win_rb_offset = (rx_win_dist - np.max(rx_win_dist)) // (RNG_BIN_INT * c / 2)

    rdrgrm = data["ECHO_SAMPLES_REAL"] + 1j * data["ECHO_SAMPLES_IMAG"]

    range_shift = (data["RANGE_SHIFT"] * RDR_RETR_INT) / 2
    range_shift_rb = range_shift // RNG_BIN_INT    

    # correct radargram
    NoOffset = np.zeros_like(rdrgrm)
    for i, shift in enumerate(range_shift_rb):
        NoOffset[i, :] = np.roll(rdrgrm[i, :], -1*shift)# - rx_win_rb_offset[i])

    ymin = min_rx_win / 1e3
    ymax = min_rx_win / 1e3 + (667 * RNG_BIN_INT * c / 2) / 1e3
    extent = [0, NoOffset.shape[0], ymax, ymin]
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    im1 = ax.imshow(np.abs(NoOffset).T, aspect=7e2, vmax=8, vmin=0, extent=extent)

    # interpolate mola topography
    geometry_intp = geometry.sort_values('LAT')
    srange = np.interp(data['SUB_SC_PLANETOCENTRIC_LATITUDE'], geometry_intp['LAT'], geometry_intp["SRANGE"]/1e3)
    plt.plot(srange, color="red", linewidth=1)

    plt.title("SHARAD 0554201 - Raw")
    plt.colorbar(im1)
    plt.savefig("figures/0554201-NoOffset.png")
    plt.close()


def read_COSHARPS_GEOM(filepath, latmin=None, latmax=None):

    cols = ["COL", "TIME", "LAT", "LON", "MRAD", "SRAD", "RV", "TV", "SZA", "PHASE"]

    cosharps = pd.read_csv(filepath, header=None, names=cols)

    cosharps = cosharps[(cosharps["LAT"]>70.768)*(cosharps["LAT"]<74.2075)]

    plt.plot(cosharps["MRAD"])
    plt.savefig("tmp.png")
    plt.show()


def correct_rdrgrm(rdrgrm, aeroid):

    rollpar = aeroid['AEROID'] - aeroid['SRAD'] * 1e3
    roll = (rollpar - np.min(rollpar)) // (ADC_SAMP_INT * c / 2)

    return oh.roll_rdrgrm(rdrgrm, roll)


def corrected_ymin_ymax(rdrgrm, aeroid, trc_st, trc_en, correction):

    RX_WIN_OPEN = aeroid['MRAD']*1e3 - 3396e3
    srad = np.max(aeroid['SRAD'][trc_st:trc_en]) * 1e3

    ymin = np.min(RX_WIN_OPEN) - rdrgrm.shape[0]*ADC_SAMP_INT*c/2
    ymax = np.min(RX_WIN_OPEN)

    # account for correction factor
    MRAD = 3396e3
    ymin = srad - (ymin + correction + MRAD)
    ymax = srad - (ymax + correction + MRAD)

    return ymin, ymax 