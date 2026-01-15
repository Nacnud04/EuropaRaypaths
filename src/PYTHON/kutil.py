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
import tifffile

# load sharad orbit from file
def load_sharad_orbit(DIRECTORY, OBS):
   
    filename = f"{DIRECTORY}/s_{OBS}_geom.tab"
    col_names = ["COL", "TIME", "LAT", "LON", "MRAD", "SRAD", "RVEL", "TVEL", "SZA", "PHSE"]
    geometry = pd.read_csv(filename, header=None, names=col_names)
    print(f"Found {len(geometry)} observations in {filename}")

    return geometry


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
