import sys
sys.path.append("../../../.")

from surface import *
from source import *
from simulator import *

import numpy as np
from time import time as Time

# tuple with facet sizes
fss = (20, 10, 5, 3, 2)

# do not overlap facets at all
overlap = 0 # percentage for facets to overlap

# compute dimensions based on facet size
ftprnt_size = 1000 # footprint width [m]
dimss = [(int(ftprnt_size/fs),int(ftprnt_size/fs)) for fs in fss]

# origin
origin = (4500, 4500)

# generate a list of surfaces to call

surfs = [Surface(origin=origin, dims=d, fs=fs, overlap=overlap) for d, fs in zip(dimss, fss)]
for s in surfs:
    s.gen_flat(0)


# generate source locations

f0  = 9e6             # center frequency [Hz]
dt  = 1 / (8 * f0)    # time delta to avoid aliasing [s]
dur = 0.5e-6          # how long to make source chirp [s]
sy  = 5050            # source y [m]
sz  = 25000           # source z [m] - this is like orbital altitude

# source list
ss = []
# how many sources to place in transit?
n = 100 
for x in np.linspace(-10e3, 20e3, n):
    source = Source(dt, dur, (x, sy, sz))
    source.chirp(f0, 1e6)
    ss.append(source)


## Point target location
tx =  5000 # target x [m]
ty =  5050 # target y [m]
tz = -1000 # target z [m]


# do not simulate surface reflection
reflect = False
# output array to house radar images
rdrgrms = []
# for focusing
sltrngs = []
# system start time
st = Time()

# iterate through facet sizes
for i, surf in enumerate(surfs):

    # clock
    st_fs = Time()

    # generate empty array to fill
    rdrgrm = np.zeros((4803, n), np.complex128)

    # array which contains pathtimes to target for focusing
    sltrng = []

    # simulate
    print(f"Simulating at facet size: {fss[i]:03d} m | dims: {dimss[i][0]:03d}x{dimss[i][0]:03d}")

    # iterate through sources in transit
    for j, s in enumerate(ss):
        print(f"Simulating: {j+1}/{len(ss)} ({round(100*((j+1)/len(ss)), 1)}%)", end="     \r")
        model = Model(surf, s, reflect=reflect, vec=True)
        model.set_target((tx, ty, tz))
        model.gen_raypaths()
        model.comp_dopplers()
        model.gen_timeseries_vec(show=False, doppler=False)
        rdrgrm[:,j] = model.signal
        # get highest amplitude return raypath
        max_idx = np.argmax(model.tr)
        y_max, x_max = np.unravel_index(max_idx, model.tr.shape)
        # append travel time to pathtimes
        sltrng.append(model.slant_range[y_max, x_max])
        
    print(f"\nProcessing time for fs of {fss[i]:03d} m : {round((Time() - st_fs)/60)} minutes and {round((Time() - st_fs) % 60,2)} seconds")

    # add to radargram list
    rdrgrms.append(rdrgrm)
    # add pathtimes
    print(np.array(sltrng).shape)
    sltrngs.append(np.array(sltrng))

print(f"\n\nTotal processing time: {round((Time() - st)/60)} minutes and {round((Time() - st) % 60,2)} seconds")


# export the radar data
np.save("variable_fs", np.array(rdrgrms))
np.save("sltrange_fs", np.array(sltrngs))