import numpy as np
import glob, pickle

def compile_rdrgrm(path, par):

    filenames = glob.glob(f"{path}/s*.txt")

    # sort filenames to ensure correct order
    filenames.sort()

    rdrgrm = []

    for i, f in enumerate(filenames):
        if i < par['ns']:
            arr = np.loadtxt(f).T
            rdrgrm.append(arr[0] + 1j * arr[1])

    rdrgrm = np.array(rdrgrm).T

    return rdrgrm

def load_params(param_path, target_path):

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

    return params