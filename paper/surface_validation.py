import os

import pandas as pd
import numpy  as np

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

def dB(vals):
    return 10 * np.log10(vals)

finite_dir = "../examples/UNIT_TESTS/finite_surface"
fresnel_dir = "../examples/UNIT_TESTS/fresnel_zone"

file_headers = ("finite", "fresnel")

fig, ax = plt.subplots(3, 2, sharex='col', sharey='row', height_ratios=[3,1,3], figsize=(5, 7))

plt.subplots_adjust(wspace=0.1, hspace=0.1)

for col, (directory, header) in enumerate(zip((finite_dir, fresnel_dir), file_headers)):

    # COHERENT
    filename = f"{directory}/figures/{header}_coherent.csv"
    df = pd.read_csv(filename)

    # add to plot
    ax[0,col].plot(df['ALT']/1e3, dB(df['NUM_POW']), color="red", linewidth=1, label="Numerical")
    ax[0,col].plot(df['ALT']/1e3, dB(df['ANA_POW']), color="black", linewidth=1, label="Analytic", linestyle="--")
    ax[0,col].legend(fontsize="small")

    ax[1,col].plot(df['ALT']/1e3, df['ERR'], color="blue", linewidth=1)

    # INCOHERENT
    filename = f"{directory}/figures/{header}_incoherent.csv"
    df = pd.read_csv(filename)

    ax[2,col].plot(df['ALT']/1e3, dB(df['NUM_POW']), color="red", linewidth=1, label="Numerical")
    ax[2,col].plot(df['ALT']/1e3, dB(df['ANA_POW']), color="black", linewidth=1, label="Analytic", linestyle="--")
    ax[2,col].legend(fontsize="small")

    ax[2,col].set_xlim(np.min(df['ALT'])/1e3, np.max(df['ALT'])/1e3)

# x axis labels
for col in range(2): 
    ax[2, col].set_xlabel("Altitude [km]")

# y axis labels
ylabels = ("Power [dB]", "Error [%]", "Power [dB]")
for row in range(3):
    ax[row, 0].set_ylabel(ylabels[row])

# titles
coh_titles = ("Finite Surface\nCoherent","Fresnel Zone\nCoherent")
for col, title in enumerate(coh_titles):
    ax[0, col].set_title(title, fontweight="bold")
    ax[1, col].set_title("Coherent Error", fontweight="bold")
    ax[2, col].set_title("Incoherent", fontweight="bold")

# subplot labels
labels = ("(a)","(b)","(c)","(d)","(e)","(f)")
yoffs  = (0.08, 0.25, 0.08, 0.08, 0.25, 0.08)
for a, label, yoff in zip(ax.T.flatten(), labels, yoffs):
    a.text(0.02, yoff, label, transform=a.transAxes,
            fontweight="bold", va="top", ha="left", color="black",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

plt.tight_layout()

plt.savefig("Figures/SurfaceValidation.pdf")
plt.close()
