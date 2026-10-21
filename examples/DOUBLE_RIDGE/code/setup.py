"""
 * File:        setup.py
 * Author:      Duncan Byrne
 * Institution: University of Colorado Boulder
 * Department:  Aerospace Engineering Sciences
 * Email:       duncan.byrne@colorado.edu
 * Date:        2026-10-15
 *
 * Description:
 *    Generates simulation domain for double ridge example. 
 *    Sets up surface facets, and subsurface fissure. 
 * 
 * Notes:
 *    None yet. 
 *
"""

# imports 
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from gstools import SRF, TPLStable


# === surface parameters === 

# SEE: https://doi.org/10.1016/j.icarus.2020.113669
hurst  = 0.43
length_scl = 1000
chaos_variance   = 0.66**2

# matrix mean height
height_mean      = 0.0

# random seeds
rough_seed       = 2002


# == simulation discretization ===

# simulation domain size [m]
domain_size_x    = 20e3
domain_size_y    = 2e3
domain           = (domain_size_x, domain_size_y)

# simulation origin [m]
origin_x         = -1 * domain_size_x / 2
origin_y         = -1 * domain_size_y / 2
origin           = (origin_x, origin_y)

# facet size [m]
facet_size       = 5.0
Nfct_x           = int(domain_size_x / facet_size)
Nfct_y           = int(domain_size_y / facet_size)
Nfct             = Nfct_x * Nfct_y

# arrays
xs               = np.linspace(origin_x, origin_x + domain_size_x, Nfct_x)
ys               = np.linspace(origin_y, origin_y + domain_size_y, Nfct_y)
XX, YY           = np.meshgrid(xs, ys)
ZZ               = np.zeros_like(XX)


# --- GENERATE FRACTAL ROUGHNESS ---

noise = TPLStable(dim=2, hurst=hurst, len_scale=length_scl, var=chaos_variance/100)

nSml = SRF(noise, mean=0.0, seed=rough_seed)
surface_rough = nSml((ys, xs), mesh_type="structured")


# --- TURN INTO HEIGHT FIELD ---

# first start with matrix
ZZ = height_mean + surface_rough * 0


# --- ADD DOUBLE RIDGE ---

ridge_height = 500.0          # m
ridge_sep    = 1750.0         # m peak-to-peak spacing
ridge_width  = 750.0         # m, approximate FWHM

# Gaussian sigma from FWHM
sigma = ridge_width / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# ridge centers, symmetric about origin
x1 = -ridge_sep / 2.0
x2 =  ridge_sep / 2.0

# two ridges extending along y
ridge_profile = (
    ridge_height * np.exp(-0.5 * ((XX - x1) / sigma)**2) +
    ridge_height * np.exp(-0.5 * ((XX - x2) / sigma)**2)
)

ZZ = ZZ + ridge_profile * 0

#plt.imshow(ZZ, extent=(origin_x, origin_x + domain_size_x, origin_y, origin_y + domain_size_y), origin="lower", cmap="terrain")
#plt.colorbar(label="Height [m]")
#plt.show()


# --- MAKE SUBSURFACE FISSURE ---

target_size = 200

frac_TPL = TPLStable(dim=1, hurst=0.5, len_scale=4000, var=50000)
nSml = SRF(frac_TPL, mean=0.0, seed=rough_seed+1)


frac_depths = np.arange(0, -4000, -1 * target_size)
fracture_path = nSml(frac_depths, mesh_type="structured")

# move start to center of double ridge
fracture_path -= fracture_path[0]

# x(z) profile
dx_dz = np.diff(fracture_path) / np.diff(frac_depths)

# 3D normals for a y-invariant fracture surface:
# tangent in x-z is (dx/dz, 1), so a normal is (1, 0, -dx/dz)
norms = np.sqrt(1 + dx_dz**2)

n_hat = np.zeros((len(dx_dz), 3))
n_hat[:, 0] = 1.0 / norms      # x
n_hat[:, 1] = 0.0              # y
n_hat[:, 2] = -dx_dz / norms   # z

# midpoint locations of each facet
x_mid = 0.5 * (fracture_path[:-1] + fracture_path[1:])
z_mid = 0.5 * (frac_depths[:-1] + frac_depths[1:])
"""
# draw short normal "hairs" in x-z projection
hair_len = 200.0  # meters

fig, ax = plt.subplots()

ax.plot(xs, ZZ[Nfct_y//2, :], color="black", label="Surface profile")
ax.plot(fracture_path, frac_depths, color="blue", label="Subsurface fracture")

for xm, zm, nx, nz in zip(x_mid, z_mid, n_hat[:, 0], n_hat[:, 2]):
    ax.plot(
        [xm - 0.5 * hair_len * nx, xm + 0.5 * hair_len * nx],
        [zm - 0.5 * hair_len * nz, zm + 0.5 * hair_len * nz],
        color="red", linewidth=0.8
    )

ax.set_xlabel("X [m]")
ax.set_ylabel("Height [m]")
ax.set_aspect("equal", adjustable="box")
ax.legend()
plt.show()
"""

# export surface profile
profile_to_save = np.stack([xs, ZZ[Nfct_y//2, :]])
np.save("inputs/profile.npy", profile_to_save)

# export fracture info
def export_fracture_plane(
    fracture_path,
    frac_depths,
    n_hat,
    y_values,
    filename="fracture_plane.txt",
):
    # Facet midpoints in x-z
    x_mid = 0.5 * (fracture_path[:-1] + fracture_path[1:])
    z_mid = 0.5 * (frac_depths[:-1] + frac_depths[1:])

    # Repeat along y
    x_rep = np.tile(x_mid, len(y_values))
    z_rep = np.tile(z_mid, len(y_values))
    y_rep = np.repeat(y_values, len(x_mid))

    nx_rep = np.tile(n_hat[:, 0], len(y_values))
    ny_rep = np.tile(n_hat[:, 1], len(y_values))
    nz_rep = np.tile(n_hat[:, 2], len(y_values))

    out = np.column_stack([
        x_rep,
        y_rep,
        z_rep,
        nx_rep,
        ny_rep,
        nz_rep,
        np.ones(len(x_rep), dtype=int),
    ])

    np.savetxt(
        filename,
        out,
        fmt="%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%d",
        comments="",
    )


#y_values = np.arange(-10e3, 10e3, target_size)
y_values = np.arange(-1e3, 1e3, target_size)
export_fracture_plane(
    fracture_path=fracture_path,
    frac_depths=frac_depths,
    n_hat=n_hat,
    y_values=y_values,
    filename="inputs/fracture_plane.txt",
)



# --- COMPUTE ORTHONORMAL BASIS ---

# first compute surface normals via central differences
dZdx = np.zeros_like(ZZ)
dZdy = np.zeros_like(ZZ)
dZdx[:, 1:-1] = (ZZ[:, 2:] - ZZ[:, :-2]) / (2 * facet_size)
dZdy[1:-1, :] = (ZZ[2:, :] - ZZ[:-2, :]) / (2 * facet_size)

# fill in edges with copy
dZdx[:, 0] = dZdx[:, 1]
dZdx[:, -1] = dZdx[:, -2]
dZdy[0, :] = dZdy[1, :]
dZdy[-1, :] = dZdy[-2, :]

# compute normal vectors
norms = np.sqrt(dZdx**2 + dZdy**2 + 1)
n_hat = np.zeros((Nfct_y, Nfct_x, 3))
n_hat[:, :, 0] = -dZdx / norms
n_hat[:, :, 1] = -dZdy / norms
n_hat[:, :, 2] = 1 / norms

# generate other two basis using reference vector (pointing in x direction)
ref = np.zeros_like(n_hat)
ref[:, :, 0] = 1

# if n_hat is nearly parallel with ref, switch to y direction
parallel_mask = np.abs(n_hat[:, :, 0]) > 0.9
ref[parallel_mask, :] = [0, 1, 0]

# get first tangent with cross product
u_hat = np.cross(ref, n_hat)
u_hat /= np.linalg.norm(u_hat, axis=2, keepdims=True)

# get second tangent with cross product
v_hat = np.cross(n_hat, u_hat)

# flatten into output arrays
fxs = XX.flatten()
fys = YY.flatten()
fzs = ZZ.flatten()
fnxs = n_hat[:,:,0].flatten()
fnys = n_hat[:,:,1].flatten()
fnzs = n_hat[:,:,2].flatten()
fuxs = u_hat[:,:,0].flatten()
fuys = u_hat[:,:,1].flatten()
fuzs = u_hat[:,:,2].flatten()
fvxs = v_hat[:,:,0].flatten()
fvys = v_hat[:,:,1].flatten()
fvzs = v_hat[:,:,2].flatten()

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
            if i % 66 == 0:
                print(f"Writing out facet data ... {i}/{len(xs)*2}", end="      \r")


export_obj_points_colored(
    "figures/ridge_facets.obj",
    fxs, fys, fzs,
    fzs,
    fnxs, fnys, fnzs,
    cmap_name="magma", nscale=80
)


# export into facet file
facet_file = "inputs/facets.fct"

with open(facet_file, 'w') as f:
    i = 0
    for x, y, z, nx, ny, nz, ux, uy, uz, vx, vy, vz in zip(fxs, fys, fzs, fnxs, fnys, fnzs, fuxs, fuys, fuzs, fvxs, fvys, fvzs):
        if i != len(fxs) - 1:
            f.write(f"{x:.6f},{y:.6f},{z:.6f}:{ux:.6f},{uy:.6f},{uz:.6f}:{vx:.6f},{vy:.6f},{vz:.6f}\n")
        else:
            f.write(f"{x:.6f},{y:.6f},{z:.6f}:{ux:.6f},{uy:.6f},{uz:.6f}:{vx:.6f},{vy:.6f},{vz:.6f}")
        i += 1
    print(f"Exported facet data to: {facet_file}")


# --- EXPORT TO OBJ FILE ---

print(XX.shape)

filename = "ridge_terrain.obj"
with open(filename, 'w') as f:
    f.write("# Ridge terrain simulation domain\n")
    f.write(f"# {Nfct_x} x {Nfct_y} facets\n")
    f.write("\n")
    # write vertices
    for j in range(Nfct_y):
        for i in range(Nfct_x):
            print(f"Writing vertex {j*Nfct_x + i + 1:5d}/{Nfct}...", end="\r")
            f.write(f"v {XX[j, i]} {YY[j, i]} {ZZ[j, i]}\n")
    print("\n")
    # write faces
    for j in range(Nfct_y - 1):
        for i in range(Nfct_x - 1):
            print(f"Writing face {j*(Nfct_x-1) + i + 1:5d}/{(Nfct_x-1)*(Nfct_y-1)}...", end="\r")
            v1 = j * Nfct_x + i + 1
            v2 = j * Nfct_x + (i + 1) + 1
            v3 = (j + 1) * Nfct_x + (i + 1) + 1
            v4 = (j + 1) * Nfct_x + i + 1
            f.write(f"f {v1} {v2} {v3}\n")
            f.write(f"f {v1} {v3} {v4}\n")
