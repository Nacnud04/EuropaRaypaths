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

def dB(x):
    return 10 * np.log10(x)

# load data from previous simulation
nadir_dir = "../examples/UNIT_TESTS/subsurface_target/figures"
off_nadir_dir = "../examples/UNIT_TESTS/subsurface_offset_target/figures"

# iterate over depths and add to plot
depths = (250, 5000)
colors = ("blue", "red")
highlights = ("cyan", "fuchsia")

fig, ax = plt.subplots(3, 2, figsize=(6.5, 8.5), sharex='col', sharey='row')

plt.subplots_adjust(wspace=0.1, hspace=0.1)

for col, directory in enumerate((nadir_dir, off_nadir_dir)):

    for d, c, h in zip(depths, colors, highlights):
    
        df = pd.read_csv(f"{directory}/nadir_{d:04d}.csv")

        ax[0,col].plot(df['ALT']/1e3, dB(df['NUM_POW']), color=c, linewidth=1, label=f"Numerical: {d} m")
        ax[0,col].plot(df['ALT']/1e3, dB(df['ANA_POW']), color=h, linewidth=1, label=f"Analytic: {d} m", linestyle="--")

        ax[1,col].plot(df['ALT']/1e3, df['ERR_POW'], color=c, linewidth=1, label=f"{d} m")

        ax[2,col].plot(df['ALT']/1e3, df['ERR_PHS'], color=c, linewidth=1, label=f"{d} m")
        
        ax[2,col].set_xlim(np.min(df['ALT'])/1e3, np.max(df['ALT'])/1e3)

    ax[2,col].axhline(0, color="black", linestyle=":")

# x axis labels
for c in range(2): ax[2,c].set_xlabel("Altitude [km]")

# y axis labels
ylabels = ("Power [dB]", "Power Error [%]", "Phase Error [deg]")
for r, ylabel in enumerate(ylabels):
    ax[r,0].set_ylabel(ylabel)

# legend
for a in ax.flatten():
    a.legend(fontsize="small")

# column headers
headers = ("Nadir Target", "Off-Nadir Target")
for c, head in enumerate(headers):
    ax[0, c].set_title(head, fontweight="bold")

plt.tight_layout()

plt.savefig("TargetValidation.pdf")
plt.close()
