import os

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

# bring in latex
os.environ["PATH"] += os.pathsep + '/usr/share/texlive/texmf-dist/tex/xelatex'
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def dB(val):
    return 10 * np.log10(val)

layer_dir = "../examples/UNIT_TESTS/subsurface_layer"
filepath = f"{layer_dir}/figures/subsurface_layer.csv"

df = pd.read_csv(filepath)
df = df[df['ALT'] < 75e3]

fig, ax = plt.subplots(2, figsize=(5, 3), height_ratios=[4,1], sharex="col")

# plot power
ax[0].plot(df['ALT']/1e3, dB(df['NUM_POW']), color="red", linewidth=1, label="Numerical")
ax[0].plot(df['ALT']/1e3, dB(df['ANA_POW']), color="black", linewidth=1, linestyle="--", label="Analytic")

# plot error
ax[1].plot(df['ALT']/1e3, df['ERR'], color="blue", linewidth=1)

# limits
ax[1].set_xlim(np.min(df['ALT'])/1e3, np.max(df['ALT'])/1e3)

# titles
ax[0].set_title("20 km deep layer", fontweight="bold")

# axis labels
ylabels = ("Power [dB]", "Error [%]")
for row, ylbl in enumerate(ylabels):
    ax[row].set_ylabel(ylbl)
ax[1].set_xlabel("Altitude [km]")

plt.tight_layout()

plt.savefig("Figures/SubsurfaceLayerValidation.pdf")
plt.close()
