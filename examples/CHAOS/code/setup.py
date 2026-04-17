"""
 * File:        setup.py
 * Author:      Duncan Byrne
 * Institution: University of Colorado Boulder
 * Department:  Aerospace Engineering Sciences
 * Email:       duncan.byrne@colorado.edu
 * Date:        2026-04-16
 *
 * Description:
 *    Generates simulation domain for chaos terrain example. 
 *    Sets up surface facets, and subsurface fissures. 
 * 
 * Notes:
 *    None yet. 
 *
"""

# imports 
import numpy as np
import matplotlib.pyplot as plt
from gstools import SRF, TPLStable
from shapely import Polygon, Point


# === surface parameters === 

# range of block height above matrix material [m]
# SEE: https://doi.org/10.1029/1998GL900144
block_height_min =  40.0
block_height_max = 150.0

# range of block side length [m]
# SEE: https://doi.org/10.1029/1999JE001143
block_size_min   =  1e3
block_size_max   = 20e3/5

# noise variables for chaos
# SEE: https://doi.org/10.1016/j.icarus.2020.113669
chaos_hurst_lrg  = 0.36
chaos_hurst_sml  = 0.69
chaos_break_scl  = 1e3
chaos_variance   = 1.31**2

# matrix mean height
matrix_mean      = 0.0

# random seeds
matrix_seed      = 2002
block_seed       =  702
block_pos_seed   = 2025

# how many blocks?
nblocks          = 20


# == simulation discretization ===

# simulation domain size [m]
domain_size_x    = 50e3
domain_size_y    = 5e3
domain           = (domain_size_x, domain_size_y)

# simulation origin [m]
origin_x         = 0.0
origin_y         = 0.0
origin           = (origin_x, origin_y)

# facet size [m]
facet_size       = 50.0
Nfct_x           = int(domain_size_x / facet_size)
Nfct_y           = int(domain_size_y / facet_size)
Nfct             = Nfct_x * Nfct_y

# arrays
xs               = np.linspace(origin_x, origin_x + domain_size_x, Nfct_x)
ys               = np.linspace(origin_y, origin_y + domain_size_y, Nfct_y)
XX, YY           = np.meshgrid(xs, ys)
ZZ               = np.zeros_like(XX)


# --- GENERATE BLOCKS ---

# block position
np.random.seed(block_pos_seed)
blockXs = np.random.uniform(origin_x, origin_x + domain_size_x, nblocks)
blockYs = np.random.uniform(origin_y, origin_y + domain_size_y, nblocks)

# if any blocks are too close to each other, find a new location
min_dist = 3e3
for i in range(nblocks):
    dist = np.sqrt((blockXs[i] - blockXs)**2 + (blockYs[i] - blockYs)**2)
    dist[i] = np.inf
    while np.any(dist < min_dist):
        new_x = np.random.uniform(origin_x, origin_x + domain_size_x)
        new_y = np.random.uniform(origin_y, origin_y + domain_size_y)
        dist = np.sqrt((new_x - blockXs)**2 + (new_y - blockYs)**2)
        dist[i] = np.inf
        blockXs[i] = new_x
        blockYs[i] = new_y

# block heights
blockHs = np.random.uniform(block_height_min, block_height_max, nblocks)

# block side lengths
blockLs = np.random.uniform(block_size_min, block_size_max, nblocks)
blockWs = np.random.uniform(block_size_min, block_size_max, nblocks)

# block rotations
blockRs = np.random.uniform(0, np.pi, nblocks)

# block corners
blockTLs = np.zeros((nblocks, 2)) # top left corner
blockTRs = np.zeros((nblocks, 2)) # top right corner
blockBLs = np.zeros((nblocks, 2)) # bottom left corner
blockBRs = np.zeros((nblocks, 2)) # bottom right corner

for i in range(nblocks):

    # block center
    cx = blockXs[i]
    cy = blockYs[i]

    # block half dimensions
    hl = blockLs[i] / 2
    hw = blockWs[i] / 2

    # block rotation
    r = blockRs[i]

    # rotation matrix
    R = np.array([[np.cos(r), -np.sin(r)],
                  [np.sin(r),  np.cos(r)]])
    
    # block corners in local coordinates
    local_corners = np.array([[-hl,  hw],  # top left
                              [ hl,  hw],  # top right
                              [-hl, -hw],  # bottom left
                              [ hl, -hw]]) # bottom right
    
    # block corners in global coordinates
    global_corners = (R @ local_corners.T).T + np.array([cx, cy])

    # add to arrays
    blockTLs[i] = global_corners[0]
    blockTRs[i] = global_corners[1]
    blockBLs[i] = global_corners[2]
    blockBRs[i] = global_corners[3]


# --- CREATE BLOCK MASK ---

block_mask = np.ones_like(XX) * -1 # -1 indicates matrix material, block index otherwise

for i in range(nblocks):
    print(f"Masking block {i+1:2d}/{nblocks:2d}...")
    # generate shapely polygon for block
    block_poly = Polygon([blockTLs[i], blockTRs[i], blockBRs[i], blockBLs[i]])
    # make a meshgrid for the polygon
    for j in range(Nfct_x):
        for k in range(Nfct_y):
            point = Point(XX[k, j], YY[k,j])
            if block_poly.contains(point):
                block_mask[k, j] = i

# --- GENERATE FRACTAL NOISE ---

noise_sml = TPLStable(dim=2, hurst=chaos_hurst_sml, len_low=0, len_scale=chaos_break_scl, var=chaos_variance)
noise_lrg = TPLStable(dim=2, hurst=chaos_hurst_lrg, len_low=chaos_break_scl, len_scale=domain_size_x, var=chaos_variance)

def chaos_noise(xs, ys, seed):
    nSml = SRF(noise_sml, mean=0.0, seed=seed)
    nLrg = SRF(noise_lrg, mean=0.0, seed=seed)
    return (nSml((xs, ys), mesh_type="structured") + nLrg((xs, ys), mesh_type="structured")).T

# noise for matrix material
matrix_noise = chaos_noise(xs, ys, matrix_seed)

# noise for block material
block_noise  = chaos_noise(xs, ys, block_seed)


# --- TURN INTO HEIGHT FIELD ---

# first start with matrix
ZZ = matrix_mean + matrix_noise

# add blocks on top
for i in range(nblocks):
    reduced_mask = block_mask == i
    ZZ[reduced_mask] = blockHs[i] + block_noise[reduced_mask]


# --- EXPORT TO OBJ FILE ---

filename = "chaos_terrain.obj"
with open(filename, 'w') as f:
    f.write("# Chaos terrain simulation domain\n")
    f.write(f"# {Nfct_x} x {Nfct_y} facets\n")
    f.write(f"# {nblocks} blocks\n")
    f.write("\n")
    # write vertices
    for j in range(Nfct_y):
        for i in range(Nfct_x):
            print(f"Writing vertex {j*Nfct_x + i + 1:5d}/{Nfct}...", end="\r")
            f.write(f"v {XX[j,i]} {YY[j,i]} {ZZ[j,i]}\n")
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


# plot
lim = 4
extent = np.array([origin_x, origin_x + domain_size_x, origin_y, origin_y + domain_size_y])
fig, ax = plt.subplots(4, figsize=(12, 5), sharex=True)
ax[0].set_title("Block noise")
n1 = ax[0].imshow(block_noise, origin="lower", vmin=-1*lim, vmax=lim, extent=extent/1e3)
fig.colorbar(n1, ax=ax[0], label="Noise [m]")
ax[1].set_title("Matrix noise")
n2 = ax[1].imshow(matrix_noise, origin="lower", vmin=-1*lim, vmax=lim, extent=extent/1e3)
fig.colorbar(n2, ax=ax[1], label="Noise [m]")
ax[2].set_title("Block mask")
n3 = ax[2].imshow(block_mask, origin="lower", extent=extent/1e3, cmap="tab20")
fig.colorbar(n3, ax=ax[2], label="Block index")
ax[3].set_title("Height map")
n4 = ax[3].imshow(ZZ, origin="lower", extent=extent/1e3)
fig.colorbar(n4, ax=ax[3], label="Elevation [m]")
for a in ax: a.set_ylabel("Y [km]")
ax[-1].set_xlabel("X [km]")
# add block locations on top
for a in ax[:-1]:
    a.scatter(blockXs/1e3, blockYs/1e3, color="red", marker="x")
# add block outlines on top
for i in range(nblocks):
    for a in ax[:-1]:
        a.plot([blockTLs[i,0]/1e3, blockTRs[i,0]/1e3], [blockTLs[i,1]/1e3, blockTRs[i,1]/1e3], color="red")
        a.plot([blockTRs[i,0]/1e3, blockBRs[i,0]/1e3], [blockTRs[i,1]/1e3, blockBRs[i,1]/1e3], color="red")
        a.plot([blockBRs[i,0]/1e3, blockBLs[i,0]/1e3], [blockBRs[i,1]/1e3, blockBLs[i,1]/1e3], color="red")
        a.plot([blockBLs[i,0]/1e3, blockTLs[i,0]/1e3], [blockBLs[i,1]/1e3, blockTLs[i,1]/1e3], color="red")
for a in ax:
    a.set_xlim(origin_x/1e3, (origin_x + domain_size_x)/1e3)
    a.set_ylim(origin_y/1e3, (origin_y + domain_size_y)/1e3)
plt.tight_layout()
plt.show()