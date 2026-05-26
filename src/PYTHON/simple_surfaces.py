import numpy as np

def gen_normals(params, fzzs):

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

    return n_hat, u_hat, v_hat

def export_obj_points_colored(filename, xs, ys, zs, values, nxs, nys, nzs, cmap_name="magma", vmin=None, vmax=None, nscale=0.5e3):
    import matplotlib
    import matplotlib.colors as colors
    # Normalize values to [0, 1]
    if vmin and vmax:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    cmap = matplotlib.colormaps.get_cmap(cmap_name)

    print(f"Exporting obj to: {filename}")

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

def export_facets(fxs, fys, fzs, fnxs, fnys, fnzs, fuxs, fuys, fuzs, fvxs, fvys, fvzs, filename, obj=None):
    
    # export into facet file
    with open(filename, 'w') as f:
        i = 0
        for x, y, z, nx, ny, nz, ux, uy, uz, vx, vy, vz in zip(fxs, fys, fzs, fnxs, fnys, fnzs, fuxs, fuys, fuzs, fvxs, fvys, fvzs):
            if i != len(fxs) - 1:
                f.write(f"{x:.6f},{y:.6f},{z:.6f}:{nx:.6f},{ny:.6f},{nz:.6f}:{ux:.6f},{uy:.6f},{uz:.6f}:{vx:.6f},{vy:.6f},{vz:.6f}\n")
            else:
                f.write(f"{x:.6f},{y:.6f},{z:.6f}:{nx:.6f},{ny:.6f},{nz:.6f}:{ux:.6f},{uy:.6f},{uz:.6f}:{vx:.6f},{vy:.6f},{vz:.6f}")
            i += 1
        print(f"Exported facet data to: {filename}")

    if obj:
        export_obj_points_colored(
            obj,
            fxs, fys, fzs,
            fzs,
            fnxs, fnys, fnzs,
            cmap_name="magma", nscale=500
        )


def export_targets(fxs, fys, fzs, fnxs, fnys, fnzs, filename, obj=None):
    
    # export into facet file
    with open(filename, 'w') as f:
        i = 0
        for x, y, z, nx, ny, nz in zip(fxs, fys, fzs, fnxs, fnys, fnzs):
            if i != len(fxs) - 1:
                f.write(f"{x:.6f},{y:.6f},{z:.6f},{nx:.6f},{ny:.6f},{nz:.6f},1\n")
            else:
                f.write(f"{x:.6f},{y:.6f},{z:.6f},{nx:.6f},{ny:.6f},{nz:.6f},1")
            i += 1

    if obj:
        export_obj_points_colored(
            obj,
            fxs, fys, fzs,
            fzs,
            fnxs, fnys, fnzs,
            cmap_name="magma", nscale=500
        )


def gen_flat(params, fzzs):
    return np.zeros_like(fzzs)

def gen_concave(params, fxxs, fyys, fzzs):
    R = params["sz"] 
    r2 = fxxs**2 + fyys**2
    return R - np.sqrt(np.maximum(0, R**2 - r2))

def gen_convex(params, fxxs, fyys, fzzs):
    R = params["sz"]
    r2 = fxxs**2 + fyys**2
    fzzs = np.sqrt(np.maximum(0, R**2 - r2)) - R
    return fzzs

def calc_fresnel(params, debug=False):
    Fr = np.sqrt((params["sz"] + params['lam'] / 4)**2 - params["sz"]**2)
    if debug == True: print(f"Calculated fresnel zone radius as {Fr:.2f} meters")
    return Fr

def calc_fresnel_spherical(params, debug=False):
    h = params['sz']
    r = params['sz'] # setting sphere radius equal to altitude (arbitrary)
    Fr = np.sqrt((params['lam'] * h * r) / (2 * (h + r)))
    if debug == True: print(f"Calculated fresnel zone radius as {Fr:.2f} meters")
    return Fr

def gen_fresnel(params, fxxs, fyys, fzzs, stype="flat"):
    fxys = np.sqrt(fxxs**2 + fyys**2)
    if stype == "flat":
        fzzs = gen_flat(params, fzzs)
        Fr   = calc_fresnel(params)
    elif stype == "convex":
        fzzs = gen_convex(params, fxxs, fyys, fzzs)
        Fr   = calc_fresnel_spherical(params)
    mask = fxys <= Fr
    return fzzs, mask

def make_surface(params, stype, filename, obj=None):

    # first make xy grid
    fxs = np.arange(params["nx"])*params["fs"] + params["ox"]
    fys = np.arange(params["ny"])*params["fs"] + params["oy"]
    fxxs, fyys = np.meshgrid(fxs, fys)
    fzzs       = np.zeros_like(fxxs)

    mask = None

    if stype == "flat":
        fzzs = gen_flat(params, fzzs)
    elif stype == "concave":
        fzzs = gen_concave(params, fxxs, fyys, fzzs)
    elif stype == "fresnel":
        fzzs, mask = gen_fresnel(params, fxxs, fyys, fzzs, stype="flat")
    elif stype == "fresnel-convex":
        fzzs, mask = gen_fresnel(params, fxxs, fyys, fzzs, stype="convex")
    else:
        raise ValueError(f"Unknown surface type: {stype}")
    
    n_hat, u_hat, v_hat = gen_normals(params, fzzs)

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

    if mask is not None:

        mask = mask.flatten()

        arrays = [
            fxs, fys, fzs, 
            fnxs, fnys, fnzs, 
            fuxs, fuys, fuzs, 
            fvxs, fvys, fvzs
        ]

        (
            fxs, fys, fzs, 
            fnxs, fnys, fnzs, 
            fuxs, fuys, fuzs, 
            fvxs, fvys, fvzs
        ) = (a[mask] for a in arrays)

    export_facets(fxs, fys, fzs, fnxs, fnys, fnzs, fuxs, fuys, fuzs, fvxs, fvys, fvzs, filename, obj=obj)
    

def make_target_array(params, ttype, filename, obj=None, zoffset=0):

    # first make xy grid
    fxs = np.arange(params["nx"])*params["fs"] + params["ox"]
    fys = np.arange(params["ny"])*params["fs"] + params["oy"]
    fxxs, fyys = np.meshgrid(fxs, fys)
    fzzs       = np.zeros_like(fxxs)

    mask = None

    if ttype == "flat":
        fzzs = gen_flat(params, fzzs)
    fzzs += zoffset

    n_hat, u_hat, v_hat = gen_normals(params, fzzs)

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

    if mask is not None:

        mask = mask.flatten()

        arrays = [
            fxs, fys, fzs, 
            fnxs, fnys, fnzs, 
            fuxs, fuys, fuzs, 
            fvxs, fvys, fvzs
        ]

        (
            fxs, fys, fzs, 
            fnxs, fnys, fnzs, 
            fuxs, fuys, fuzs, 
            fvxs, fvys, fvzs
        ) = (a[mask] for a in arrays)


    print(f"Target count: ntargets = {len(fxs)}")

    export_targets(fxs, fys, fzs, fnxs, fnys, fnzs, filename, obj=obj)