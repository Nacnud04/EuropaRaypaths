import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../src/PYTHON")
import output_handling as oh

# --- load radargram ---
params = oh.load_params("inputs/params.pkl", "inputs/targets.txt")

rdrgrm = oh.compile_rdrgrm("radargram", params)
print(f"Radargram shape: {rdrgrm.shape}")

rdrgrm_dB = 20 * np.log10(np.abs(rdrgrm))

range_to_surface = 25
dielectric = np.sqrt(3.15)
rx_start = (params['rx_window_offset_m'] / 1e3) - range_to_surface
win_size = params['rx_window_m'] / (1e3 * dielectric)
extent = [
    params['sx0'] / 1e3,
    (params['sx0'] + params['ns'] * params['sdx']) / 1e3,
    rx_start + win_size,
    rx_start,
]

# --- load fracture plane ---
fracture_data = np.loadtxt("inputs/fracture_plane.txt", delimiter=",")
# columns: x, y, z, nx, ny, nz, facet_id
x = fracture_data[:, 0]
y = fracture_data[:, 1]
z = fracture_data[:, 2]
nx = fracture_data[:, 3]
ny = fracture_data[:, 4]
nz = fracture_data[:, 5]

# since y_values = [0], just plot that slice
mask = np.isclose(y, 0.0)

# sort by depth / x for a clean line
sort_idx = np.argsort(z[mask])
x_f = x[mask][sort_idx]
z_f = z[mask][sort_idx]
nx_f = nx[mask][sort_idx]
nz_f = nz[mask][sort_idx]


# --- load surface profile ---
profile = np.load("inputs/profile.npy")
pxs = profile[0, :]
pzs = profile[1, :]


# --- plot ---
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 6), 
    gridspec_kw={"width_ratios": [2, 1]},
    sharey = True
    )

# left: radargram
vmin = -260
vmax = -220
im = ax1.imshow(rdrgrm_dB, aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
fig.colorbar(im, ax=ax1, label="Power [dB]")
ax1.set_xlabel("Corrected Range [km]")
ax1.set_ylabel("Azimuth [km]")

# right: fracture profile
facet_len = 200.0
hair_len = 100.0
zscl = -1e-3
xscl = 1e-3
for i, (xm, zm, nxi, nzi) in enumerate(zip(x_f, z_f, nx_f, nz_f)):

    if i == 0:
        l1 = "Facets"
        l2 = "Normals"
    else:
        l1 = None
        l2 = None

    # tangent to the facet: perpendicular to normal
    tx = -nzi
    tz =  nxi

    # blue facet segment
    x1 = xm - 0.5 * facet_len * tx
    x2 = xm + 0.5 * facet_len * tx
    z1 = zm - 0.5 * facet_len * tz
    z2 = zm + 0.5 * facet_len * tz
    ax2.plot(xscl * np.array([x1, x2]), zscl * np.array([z1, z2]), 
            color="blue", linewidth=1.0, label=l1)

    # red normal "hair"
    ax2.plot(
        xscl * np.array([xm - 0.5 * hair_len * nxi, xm + 0.5 * hair_len * nxi]),
        zscl * np.array([zm - 0.5 * hair_len * nzi, zm + 0.5 * hair_len * nzi]),
        color="red",
        linewidth=0.8, label=l2
    )

ax2.plot(pxs/1e3, pzs/-1e3, linewidth=1, color="black", label="Surface")

ax2.set_xlabel("X [km]")
ax2.set_aspect("equal", adjustable="box")
lim = 2
ax2.set_xlim(-1 * lim, lim)
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/FractureRadargram.png")
plt.show()

np.save("radargram/radargram.npy", rdrgrm)

