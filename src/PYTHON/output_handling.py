import numpy as np
import glob, pickle

def compile_rdrgrm(path, par, debug=False):

    filenames = glob.glob(f"{path}/s*.txt")

    # sort filenames to ensure correct order
    filenames.sort()

    rdrgrm = []
    lengths = []
    for i, f in enumerate(filenames):
        if i < par['ns']:
            print(f"Compiling radargram... {i}/{par['ns']}", end="           \r")
            arr = np.loadtxt(f).T
            col = arr[0] + 1j * arr[1]
            if debug:
                lengths.append(len(col))
            rdrgrm.append(col)

    print("",end='\n')

    if debug:
        print(f"=== Samples by trace: ===")
        print(lengths)
        print(f"=========================")

    rdrgrm = np.array(rdrgrm).T

    return rdrgrm

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


