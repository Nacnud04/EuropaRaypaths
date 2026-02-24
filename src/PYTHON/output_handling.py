import numpy as np
import glob, pickle

def compile_rdrgrm(path, par, rx_win_file=None):

    filenames = glob.glob(f"{path}/s*.txt")

    # sort filenames to ensure correct order
    filenames.sort()

    if rx_win_file:
        rx_win_adj = np.load(rx_win_file)
        rx_win_adj_rb = (rx_win_adj - np.min(rx_win_adj)) // ((1/par["rx_sample_rate"]) * 299792458)

    rdrgrm = []
    for i, f in enumerate(filenames):
        if i < par['ns']:
            index = int(f.split("/")[-1][1:-4])
            print(f"Compiling radargram... {index}/{par['ns']}", end="           \r")
            arr = np.loadtxt(f).T
            col = arr[0] + 1j * arr[1]
            if rx_win_file:
                col = np.roll(col, rx_win_adj_rb[index])
            rdrgrm.append(col)

    print("",end='\n')

    rdrgrm = np.array(rdrgrm).T

    return rdrgrm


def get_simulation_range(path):

    filenames = glob.glob(f"{path}/s*.txt")

    # sort filenames to ensure correct order
    filenames.sort()

    indices = []
    for f in filenames:
        index = int(f.split("/")[-1][1:-4])
        indices.append(index)

    return np.min(indices), np.max(indices)


def load_params(param_path, target_path, source_path=None):

    with open(param_path, 'rb') as hdl:
        params = pickle.load(hdl)

    # load targets and add to parameter matrix
    targets = np.genfromtxt(target_path, delimiter=',')
    if len(targets.shape) > 1:
        print(f"Multiple targets detected. Just using the first")
        targets = targets[0]

    params['tx'] = targets[0]
    params['ty'] = targets[1]
    params['tz'] = targets[2]

    if source_path:
        # load source path and count number of sources
        source_coords = np.genfromtxt(source_path, delimiter=',')
        params['ns'] = source_coords.shape[0]

    return params

def load_attenuation_geom(filepath):

    # example file contents:
    # 1e-05, -5000.0, -2500.0, -500.0, -1000.0, 2500.0, 0
    # 1e-05, 1000.0, -2500.0, -500.0, 5000.0, 2500.0, 0
    # cond, xmin, xmax, ymin, ymax, zmin, zmax

    xmins, xmaxs, ymins, ymaxs, zmins, zmaxs = [], [], [], [], [], []
    conductivities = []

    with open(filepath, 'r') as f:
        for line in f:
            vals = line.strip().split(',')
            conductivities.append(float(vals[0]))
            xmins.append(float(vals[1]))
            ymins.append(float(vals[2]))
            zmins.append(float(vals[3]))
            xmaxs.append(float(vals[4]))
            ymaxs.append(float(vals[5]))
            zmaxs.append(float(vals[6]))

    return (conductivities, xmins, xmaxs, ymins, ymaxs, zmins, zmaxs)

def roll_rdrgrm(rdrgrm, shifts):

    corrected = np.zeros_like(rdrgrm)
    for i, shift in enumerate(shifts):
        corrected[:, i] = np.roll(rdrgrm[:, i], -1*shift)

    return corrected

def interpolate_rdrgrm(geometry, rx_win, foc_db,
                       sim_st=55, sim_en=1628, 
                       rea_rng_st=314e3, rea_rng_en=318e3,
                       xst=750, xen=1200, 
                       ADC_SAMP_INT=0.0375e-6,
                       c=299792458):

    # first crop orbit geometry
    geom_crop = geometry.iloc[sim_st:sim_en]
    intrp_st, intrp_en = geom_crop.index[0], geom_crop.index[-1]+1

    # fast time sampling for synthetic data
    foc_rng_st = np.min(rx_win)
    foc_rng_en = np.min(rx_win) + 7.5e3
    foc_rng    = np.linspace(foc_rng_st, foc_rng_en, foc_db.shape[0])

    # fast time sampling window for real data
    # NOTE: This is what we want to interpolate onto
    sltrng = rea_rng_en - rea_rng_st
    samples = int(sltrng // (ADC_SAMP_INT*c/2))
    rea_rng    = np.linspace(rea_rng_st, rea_rng_en, samples)

    # interpolate (iterate over traces)
    foc_intrp = np.zeros((samples, xen - xst))
    for t in range(xen - xst):

        t_full = t + xst
        
        # stack traces
        trace_ids = [i-intrp_st for i, row in geom_crop.iterrows() if int(row['COL']) == t_full]
        trace_stack = np.zeros_like(foc_db[:,0])
        if len(trace_ids) == 0:
            print(f"Warning: No traces found for trace {t_full}", end="        \r")
        for tid in trace_ids:
            clean = foc_db[:, tid]
            clean[np.isnan(clean)] = np.nanmin(clean)
            trace_stack += clean
        trace_stack /= len(trace_ids)

        # interpolate
        foc_intrp[:, t] = np.interp(rea_rng, foc_rng, trace_stack)

    # clean up infs and nans
    foc_intrp[np.isinf(foc_intrp)] = np.nanmin(foc_intrp[np.isfinite(foc_intrp)])
    foc_intrp[np.isnan(foc_intrp)] = np.nanmin(foc_intrp[np.isfinite(foc_intrp)])

    return foc_intrp

