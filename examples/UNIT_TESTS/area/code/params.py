import pickle, json, sys
import numpy as np

sys.path.append("../../../archive/src")
from terrain import Terrain

params = {

    # radar parameters
    "power": 100,             # Transmitter power [W]
    "frequency": 60e6,         # Radar frequency [Hz]
    "bandwidth": 10e6,         # Radar bandwidth [Hz]
    "surface_gain": 7.3,       # Antenna gain [dB]
    "subsurface_gain": 7.3,   # Subsurface antenna gain [dB]
    "polarization": "HH",     # polarization (HH, VV, HV, VH)
    "aperture": 120,           # aperture (from nadir->edge) [deg]

    # receive window parameters
    "rx_window_offset_m":  199.75e3,         # receive window length [m]
    "rx_window_m": 0.75e3,  # receive window offset [m]
    "rx_sample_rate": 60e6,       # receive sample rate [Hz]

    # surface parameters
    "rms_height": 0.4,       # surface roughness [m]
    "buff": 1.1,             # buffer for facet estimate

    # atmosphere/subsurface parameters
    "eps_1": 1.0,            # permittivity of medium 1 
    "eps_2": 3.15,           # permittivity of medium 2
    "sig_1": 0.0,            # conductivity of medium 1 [S/m]
    "sig_2": 1e-6,           # conductivity of medium 2 [S/m]
    "mu_1": 1.0,             # permeability of medium 1
    "mu_2": 1.0,             # permeability of medium 2

    # source parameters 
    "sy": 0,                # source y location       [m]
    "sz": 200e3,             # source z location       [m]
    "sdx": 10,              # source x discretization [m]
    "sx0": 0,            # source x origin         [m]
    "ns": 1,             # source count            [.]

    # facet array params
    "oz": 0,
    "fs": 100,

    # target params
    "rerad_funct": 2,  # 1-degree boxcar

    # processing parameters (BOOLEAN)
    "convolution": True,   # use convolution-based processing
    "convolution_linear": True,  # use linear convolution instead of circular
    "specular": False,     # use specular computation methods for specific circumstances only
    "lossless": True,      # simulate without loss (spreading not included)
    "incoherent": True,

    # enable debug to see surface response phasor trace
    "debug_surface": True,

}

origin = np.linspace(-10e3, -0.25e3, 100)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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

for i, o in enumerate(origin):

    params["ox"] = o
    params["oy"] = o
    params["nx"] = int((np.abs(o) * 2) / params['fs'])
    params["ny"] = int((np.abs(o) * 2) / params['fs'])
    params['incoherent'] = True

    with open(f"inputs/incoh_params{i}.json", "w") as f:
        json.dump(params, f, indent=4)

    with open(f"inputs/incoh_params{i}.pkl", 'wb') as hdl:
        pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)

    params['incoherent'] = False
    with open(f"inputs/coh_params{i}.json", "w") as f:
        json.dump(params, f, indent=4)

    with open(f"inputs/coh_params{i}.pkl", 'wb') as hdl:
        pickle.dump(params, hdl, protocol=pickle.HIGHEST_PROTOCOL)
    
    fxs = np.arange(params["nx"])*params["fs"] + params["ox"]
    fys = np.arange(params["ny"])*params["fs"] + params["oy"]
    fxxs, fyys = np.meshgrid(fxs, fys)

    alt = params["sz"]

    fxys = np.sqrt(fxxs**2 + fyys**2)
    R = params["sz"]   # or whatever radius you want
    r2 = fxxs**2 + fyys**2
    fzzs = R - np.sqrt(np.maximum(0, R**2 - r2))

    # generate facet normals
    # first compute surface normals via central differences
    dZdx = np.zeros_like(fzzs)
    dZdy = np.zeros_like(fzzs)
    dZdx[:, 1:-1] = (fzzs[:, 2:] - fzzs[:, :-2]) / (2 * params["fs"])
    dZdy[1:-1, :] = (fzzs[2:, :] - fzzs[:-2, :]) / (2 * params["fs"])

    # fill in edges with copy
    dZdx[:, 0] = dZdx[:, 1]
    dZdx[:, -1] = dZdx[:, -2]
    dZdy[0, :] = dZdy[1, :]
    dZdy[-1, :] = dZdy[-2, :]

    # compute normal vectors
    norms = np.sqrt(dZdx**2 + dZdy**2 + 1)
    n_hat = np.zeros((params["ny"], params["nx"], 3))
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
    fxs = fxxs.flatten()
    fys = fyys.flatten()
    fzs = fzzs.flatten()
    fnxs = n_hat[:,:,0].flatten()
    fnys = n_hat[:,:,1].flatten()
    fnzs = n_hat[:,:,2].flatten()
    fuxs = u_hat[:,:,0].flatten()
    fuys = u_hat[:,:,1].flatten()
    fuzs = u_hat[:,:,2].flatten()
    fvxs = v_hat[:,:,0].flatten()
    fvys = v_hat[:,:,1].flatten()
    fvzs = v_hat[:,:,2].flatten()

    """
    export_obj_points_colored(
        "figures/facets.obj",
        fxs, fys, fzs,
        fzs,
        fnxs, fnys, fnzs,
        cmap_name="magma", nscale=200e3
    )
    
    sys.exit()
    """
    
    # export into facet file
    facet_file = f"inputs/facets{i}.fct"

    with open(facet_file, 'w') as f:
        i = 0
        for x, y, z, nx, ny, nz, ux, uy, uz, vx, vy, vz in zip(fxs, fys, fzs, fnxs, fnys, fnzs, fuxs, fuys, fuzs, fvxs, fvys, fvzs):
            if i != len(fxs) - 1:
                f.write(f"{x:.6f},{y:.6f},{z:.6f}:{ux:.6f},{uy:.6f},{uz:.6f}:{vx:.6f},{vy:.6f},{vz:.6f}\n")
            else:
                f.write(f"{x:.6f},{y:.6f},{z:.6f}:{ux:.6f},{uy:.6f},{uz:.6f}:{vx:.6f},{vy:.6f},{vz:.6f}")
            i += 1
        print(f"Exported facet data to: {facet_file}")

    """
    xmin, xmax = params["ox"], params["ox"]+params["nx"]*params["fs"]
    ymin, ymax = params["oy"], params["oy"]+params["ny"]*params["fs"]

    terrain = Terrain(xmin, xmax, ymin, ymax, params["fs"])
    terrain.gen_flat(0)

    # write output facet data
    terrain.export(f"inputs/facets{i}.fct")
    """