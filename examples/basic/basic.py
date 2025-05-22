# system imports
import sys, os
sys.path.append("../../../.")

# custom imports
import surface   as sf
import source    as so
import simulator as si

# profiling imports
from time import time as Time


# --- STEP 1: Define surface ---

origin  = (0, 0)          # surface origin (x, y): [m, m]
dims    = (1000, 1000)    # facet count (nx, ny)
fs      = 2               # facet size [m]
verb    = True            # verbose?
surf = sf.Surface(origin=origin, dims=dims, fs=fs, verb=verb)

# define surface structure
surf.gen_flat(0)


# --- STEP 2: Define source ---

f0 = 9e6                  # center frequency of 9 MHz
B  = 1e6                  # bandwidth of 1 MHz
dt = 1 / (8 * f0)         # sampling equal to 1/8 of period
T  = 0.5e-6               # chirp duration of 500 ns
Cs = (1000, 1000, 25000)  # source location (x, y, z) [m]

source = so.Source(dt, T, Cs)
source.chirp(f0, B, correlate=False)
source.plot()


# --- STEP 3: Set up model ---

Ct = (1000, 1250, -10e3)  # target location (x, y, z) [m]

model = si.Model(surf, source, vec=True)        # init model
print(f"Range bins: {model.rb}")
model.set_target(Ct)                  # set target location
signals = []; compare = []
for i in range(2):
    if i == 0: print(f"Running fast, vectorized version...")
    else: print(f"Runnig slow nonvectorized version...")
    st = Time()
    model.gen_raypaths(progress_bar=True) # generate raypaths from source->facet->target
    #model.plot_s2f_angle()                # plot angles of incidence on surface
    #   model.plot_s2f_rad()                  # plot reradiation maintaned due to angle change by facet


    # --- STEP 4: Simulate ---

    model.comp_dopplers(plot=False) # compute the doppler shift at each facet
    print(f"Took {round(Time()-st, 2)} s")

    # generate synthetic signal
    if model.vec: model.gen_timeseries_vec(time=True, doppler=False)
    else: model.gen_timeseries(time=True)

    signals.append(model.signal)
    compare.append(model.comp_val)

    model.vec = False


# plot difference in signals
import matplotlib.pyplot as plt
import numpy as np

plt.imshow(np.reshape(compare[1], dims).T-compare[0])
plt.title("idx offsets")
plt.show()

plt.plot(model.ts, np.real(signals[0]), linewidth=1)
plt.plot(model.ts, np.real(signals[1]), linewidth=1)
plt.show()

plt.plot(model.ts, np.imag(signals[0]), linewidth=1)
plt.plot(model.ts, np.imag(signals[1]), linewidth=1)
plt.show()

plt.plot(model.ts, np.real(signals[0]) - np.real(signals[1]), color="blue", linewidth=1)
plt.plot(model.ts, np.imag(signals[0]) - np.imag(signals[1]), color="red", linewidth=1)
plt.show()