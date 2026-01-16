import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors

import sys
sys.path.append("/storage/School/GradSchool/Repostories/EuropaRaypaths/src/PYTHON")
import kutil as ku
import unit_convs as uc

# path to mars radius DEM
path = r"data/MOLA/RCropped.tif"

# mars radius offset
MARS_RADIUS = 3396000

data, tpar = ku.load_cropped_mola_tif(path)

# compute the resolution in meters along latitude to get the square side of each pixel
resolution = uc.dLat_to_m(tpar['scl_lat'], MARS_RADIUS)#np.radians(np.abs(tpar['scl_lat'])) * MARS_RADIUS
print(f"Latitude resolution: {resolution} m")

# --- VECTORIZED NORMAL COMPUTATION ---
# get radius from the center of mars across all points
data_Rs = data.astype(float) + MARS_RADIUS
data_Rs[data==0] = None

# start with latitude (inclination)
# all operations are done purely to the interior and then copied to the edges
# compute third side of the triangle
r_p1 = np.roll(data_Rs,  1, axis=0)
r_n1 = np.roll(data_Rs, -1, axis=0)
a = np.sqrt(r_p1**2 + r_n1**2 - 2 * r_p1 * r_n1 * np.cos(np.abs(tpar['scl_latr'])))

# get basal angle
th_R1 = np.degrees(np.arcsin(np.clip((r_p1 * np.sin(np.abs(tpar['scl_latr'])))/a, a_min=None, a_max=1)))

# compute the final incliantion of each dem pixel
# note that we need to turn latitude of base of triangle into angle from North (spherical coordinates)
# hence 90 - lat
# we also need the sign of the R1 modifier angle off of the normal
sign  = np.ones_like(r_p1)
sign[r_n1 > r_p1] *= -1
dem_inc = 90 - (tpar['llats'] + (th_R1 - 90) * sign)
# NOTE: The inclination values output are in degrees, where positive values are due to position on the mars
#       sphere, whereas negative values are actually due to terrain variations. This is from different
#       angles dominating. For the positive values the latitude of the "triangle" dominates whereas for 
#       negative values the R1 angle dominates (approx same radaii vs different radaii).

# now do longitude (azimuthal angle)
# geographic variation in this should be somewhat extreme due to being near the poles
# get radaii and project into 2d plane at given latitude
r_p1 = np.roll(data_Rs,  1, axis=1)
r_n1 = np.roll(data_Rs, -1, axis=1)
r_p1 = r_p1 * np.cos(np.radians(tpar['llats'])) # for conventional spherical coordinates this would be sin but we are in planetary coordinates 
r_n1 = r_n1 * np.cos(np.radians(tpar['llats']))

# compute third side of the triangle
a = np.sqrt(r_p1**2 + r_n1**2 - 2 * r_p1 * r_n1 * np.cos(tpar['scl_lonr']))

# get "basal" angle (here it is the angle from meridian, aka longitude)
# first we need the sign of R1 as it is the modifier angle off of the normal
sign  = -1 * np.ones_like(r_p1)
sign[r_n1 > r_p1] *= -1
th_R1 = np.degrees(np.arcsin(np.clip((r_p1 * np.sin(tpar['scl_lonr']))/a, a_min=-1, a_max=1)))

# compute the final azimuthal angle of each dem pixel
dem_azi = tpar['llons'] + (th_R1 - 90) * sign

# fill in the edges for inc and azi
for fill_mat in (dem_inc, dem_azi):
    edges = np.isnan(fill_mat) * (data != 0)
    edge_indicies = np.nonzero(edges)

    # go edge by edge and find a value to fill in with
    for r, c in zip(edge_indicies[0], edge_indicies[1]):
        filled = False
    
        for offr in (0, -1, 1):
            for offc in (1, -1, 0):
        
                # skip same index as we know it is filled
                if offr == 0 and offc == 0:
                    continue
                # make sure things don't go out of bounds
                if r+offr >= tpar['rows'] or c+offc >= tpar['cols'] or r+offr < 0 or c+offc < 0:
                    continue
                if not np.isnan(fill_mat[r+offr, c+offc]):
                    fill_mat[r, c] = fill_mat[r+offr, c+offc]
                    filled = True
                if filled == True:
                    break
            
            if filled == True:
                break
# --- END VECTOR NORMAL COMPUTATION --- 

# plot vector normals in spherical
fig, ax = plt.subplots(2, figsize=(8, 6))
im1 = ax[0].imshow(dem_inc, extent=tpar['extent'])
fig.colorbar(im1, ax=ax[0])
ax[0].set_title("Inclination angle from North [deg]")
im2 = ax[1].imshow(dem_azi, extent=tpar['extent'], vmin=145, vmax=175)
fig.colorbar(im2, ax=ax[1])
ax[1].set_title("Azimuthal angle from Meridian [deg]")
for a in ax: a.set_xlabel("Longitude [deg]"); a.set_ylabel("Latitude [deg]")
plt.tight_layout()
plt.savefig("figures/KOR_F_SPNORMS.png")
plt.close()

# output array to house all facet coordinates
fxs, fys, fzs = np.array([]), np.array([]), np.array([])
frs = np.array([])

# output arrays for orthonormals
fnxs, fnys, fnzs = np.array([]), np.array([]), np.array([])
fuxs, fuys, fuzs = np.array([]), np.array([]), np.array([])
fvxs, fvys, fvzs = np.array([]), np.array([]), np.array([])

# begin iteration over latitudes
for i_lat in range(tpar['rows']):

    lat = tpar['tpnt_lat'] + tpar['scl_lat']* i_lat
    lat_rad = np.radians(lat)

    r_lat = MARS_RADIUS * np.cos(lat_rad)

    row = data[i_lat, :]
    idx = np.nonzero(row)[0]
    if len(idx) < 2:
        continue

    row_lons = tpar['tpnt_lon'] + tpar['scl_lon'] * idx
    row_rads = row[idx].astype(float) + MARS_RADIUS

    # Convert longitude to meters
    lon_rad = np.radians(row_lons)
    arc_m = r_lat * lon_rad

    # Uniform spacing in meters
    arc_min, arc_max = arc_m[0], arc_m[-1]
    lon_meters = arc_max - arc_min
    lon_facets = max(2, int(lon_meters / resolution))

    arc_uniform = np.linspace(arc_min, arc_max, lon_facets)

    # Interpolate radius in meter-space
    facet_radius = np.interp(arc_uniform, arc_m, row_rads)

    # Convert back to longitude
    facet_lons_rad = arc_uniform / r_lat

    # Spherical * Cartesian conversion (PLANETOCENTRIC)
    facet_xs = facet_radius * np.cos(lat_rad) * np.cos(facet_lons_rad)
    facet_ys = facet_radius * np.cos(lat_rad) * np.sin(facet_lons_rad)
    facet_zs = facet_radius * np.sin(lat_rad)

    fxs = np.append(fxs, facet_xs)
    fys = np.append(fys, facet_ys)
    fzs = np.append(fzs, facet_zs)
    frs = np.append(frs, facet_radius)

    # copy and interpolate normal vectors
    row_inc = np.radians(dem_inc[i_lat, :])
    row_azi = np.radians(dem_azi[i_lat, :])

    # interpolate
    facet_inc = np.interp(arc_uniform, arc_m, row_inc[idx])
    facet_azi = np.interp(arc_uniform, arc_m, row_azi[idx])

    # move to cartesian (calculated inclination is not planetocentric!)
    facet_nxs = np.sin(facet_inc) * np.cos(facet_azi)
    facet_nys = np.sin(facet_inc) * np.sin(facet_azi)
    facet_nzs = np.cos(facet_inc)

    fnxs = np.append(fnxs, facet_nxs)
    fnys = np.append(fnys, facet_nys)
    fnzs = np.append(fnzs, facet_nzs)

    # stack normal vectors: shape (N, 3)
    n_hat = np.stack((facet_nxs, facet_nys, facet_nzs), axis=-1)

    # choose a reference vector that is not parallel to n
    # (this avoids numerical issues near the poles)
    ref = np.zeros_like(n_hat)
    ref[:, 2] = 1.0  # z-axis

    # if n is too close to z, switch to x-axis
    mask = np.abs(n_hat[:, 2]) > 0.9
    ref[mask] = np.array([1.0, 0.0, 0.0])

    # first tangent: u_hat = normalize(ref * n)
    u_hat = np.cross(ref, n_hat)
    u_hat /= np.linalg.norm(u_hat, axis=1, keepdims=True)

    # second tangent: v_hat = n * u_hat
    v_hat = np.cross(n_hat, u_hat)

    # u_hat, v_hat, n_hat now form a right-handed orthonormal basis
   
    fuxs = np.append(fuxs, u_hat[:,0])
    fuys = np.append(fuys, u_hat[:,1])
    fuzs = np.append(fuzs, u_hat[:,2])

    fvxs = np.append(fvxs, v_hat[:,0])
    fvys = np.append(fvys, v_hat[:,1])
    fvzs = np.append(fvzs, v_hat[:,2])

fig, ax = plt.subplots(2, 2, figsize=(14, 8))
ax[0, 0].scatter(fxs, fys, s=1, c=frs)
ax[0, 0].set_xlabel("x"); ax[0, 0].set_ylabel("y")
ax[0, 0].set_title("X v Y")
ax[1, 0].scatter(fxs, fzs, s=1, c=frs)
ax[1, 0].set_xlabel("x"); ax[1, 0].set_ylabel("z")
ax[1, 0].set_title("X v Z")
ax[0, 1].scatter(fys, fzs, s=1, c=frs)
ax[0, 1].set_xlabel("y"); ax[0, 1].set_ylabel("z")
ax[0, 1].set_title("Y v Z)")
ax[1, 1].set_axis_off()
plt.savefig("figures/KOR_F_CART.png")
plt.close()

# export into facet file
facet_file = "data/MOLA/KOR_F.fct"

with open(facet_file, 'w') as f:
    i = 0
    for x, y, z, nx, ny, nz, ux, uy, uz, vx, vy, vz in zip(fxs, fys, fzs, fnxs, fnys, fnzs, fuxs, fuys, fuzs, fvxs, fvys, fvzs):
        if i != len(fxs) - 1:
            f.write(f"{x:.6f},{y:.6f},{z:.6f}:{ux:.6f},{uy:.6f},{uz:.6f}:{vx:.6f},{vy:.6f},{vz:.6f}\n")
        else:
            f.write(f"{x:.6f},{y:.6f},{z:.6f}:{ux:.6f},{uy:.6f},{uz:.6f}:{vx:.6f},{vy:.6f},{vz:.6f}")
        i += 1
    print(f"Exported facet data to: {facet_file}")

# generate a target file
# place the target 1 km deep (ish b/c straight along axis) in the center of the ice mound
# use the same normal as the facet above the target ("upwards")
facet_above_target = 34138
tx, ty, tz = fxs[facet_above_target], fys[facet_above_target], fzs[facet_above_target] - 1e3
tnx, tny, tnz = [n[facet_above_target] for n in (fnxs, fnys, fnzs)]
# export
target_file = "data/Subsurface/KOR_T.txt"
with open(target_file, 'w') as f:
    i = 0
    f.write(f"{tx},{ty},{tz},{-1*tnx},{-1*tny},{-1*tnz}")


def export_obj_points_colored(filename, xs, ys, zs, values, nxs, nys, nzs, cmap_name="magma", vmin=None, vmax=None, nscale=0.5e3):
    # Normalize values to [0, 1]
    if vmin and vmax:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    cmap = matplotlib.colormaps.get_cmap(cmap_name)

    print(f"Exporting to: {filename}")

    with open(filename, "w") as f:
        f.write("# Point cloud OBJ with vertex colors\n")
        i = 0
        for x, y, z, v, nx, ny, nz in zip(xs, ys, zs, values, nxs, nys, nzs):
            r, g, b, _ = cmap(norm(v))
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
            f.write(f"v {x+nx*nscale:.6f} {y+ny*nscale:.6f} {z+nz*nscale:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
            f.write(f"l {i+1} {i+2}\n")
            i += 2
            print(f"Exporting... {i}/{len(xs)*2}", end="      \r")


export_obj_points_colored(
    "data/MOLA/KOR_F.obj",
    fxs, fys, fzs,
    frs,
    fnxs, fnys, fnzs,
    cmap_name="magma"
)
