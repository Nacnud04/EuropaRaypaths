import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np

import unit_convs as uc

# matplotlib configuration for LaTeX
os.environ["PATH"] += os.pathsep + '/usr/share/texlive/texmf-dist/tex/xelatex'

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def simple_rdrgrm(rdrgrm, par, path, title=None, linspace=True, vmin=None):

    if linspace:
        xmin = par['sx0']
        xmax = par['sx0'] + par['sdx'] * par['ns']
    else:
        raise NotImplementedError("Cannot handle uneven source spacing yet.")

    plt.imshow(uc.lin_to_db(np.abs(rdrgrm)), aspect='auto', vmin=vmin, 
           extent=[xmin/1e3, xmax/1e3, 2*(par["rx_window_offset_m"] + par["rx_window_m"])/299.792458, 2*par["rx_window_offset_m"]/299.792458])
    plt.colorbar(label='Power [dB]')
    plt.xlabel("Azimuth [km]")
    plt.ylabel("Range [us]")
    if title:
        plt.title(title)
    plt.savefig(path)
    plt.close()

def IGARSS2026_rdrgrm_focused(rdrgrm, focused, par, az_s1, az_s2, rng_s1, rng_s2, filename, 
                              vminrdr=None, vminfoc=None, linspace=True, c1=299792458):

    if linspace:
        azmin = par['sx0']
        azmax = par['sx0'] + par['sdx'] * par['ns']
    else:
        raise NotImplementedError("Cannot handle uneven source spacing yet.")
    
    rb = int((par["rx_window_m"] / c1) / (1 / par["rx_sample_rate"]))
    
    extent = (azmin/1e3, azmax/1e3, 2*((par['rx_window_offset_m'] + par['rx_window_m'])/c1)*10**6,
        2*(par['rx_window_offset_m']/c1)*10**6)

    fig, ax = plt.subplots(3, 1, figsize=(4, 6), constrained_layout=True, dpi=300)

    # radargram panel
    im0 = ax[0].imshow(uc.lin_to_db(np.abs(rdrgrm)), cmap="viridis",
                    aspect=par['aspect'], extent=extent,
                    vmin=vminrdr)

    # focused panel
    im1 = ax[1].imshow(uc.lin_to_db(np.abs(focused)), cmap="viridis",
                    aspect=par['aspect'], extent=extent,
                    vmin=vminfoc)
    
    # zoomed panel
    zoomed = np.abs(focused[int(rng_s1*rb):int(rng_s2*rb),
                        int(az_s1*focused.shape[1]):int(az_s2*focused.shape[1])])
    
    zoomed_extent = ((azmin + (azmax-azmin) * az_s1)/1e3, (azmin + (azmax-azmin) * az_s2)/1e3,
                    2*((par['rx_window_offset_m'] + rng_s2 * par['rx_window_m'])/c1)*10**6,
                    2*((par['rx_window_offset_m'] + rng_s1 * par['rx_window_m'])/c1)*10**6)
    im3 = ax[2].imshow(uc.lin_to_db(zoomed), cmap="viridis",
                    aspect=par['aspect']*0.75, extent=zoomed_extent,
                    vmin=0)
    
    # shrink bottom plot to fit
    pos_top = ax[0].get_position()
    pos_bottom = ax[2].get_position()
    new_width = pos_top.width
    new_x = 0.05 + pos_bottom.x0 + (pos_bottom.width - new_width) / 2  # center it
    ax[2].set_position([new_x, pos_bottom.y0, new_width, pos_bottom.height])

    # add rectangle
    rect = Rectangle((zoomed_extent[0], zoomed_extent[2]),
                    zoomed_extent[1]-zoomed_extent[0],
                    zoomed_extent[3]-zoomed_extent[2],
                    linewidth=1, edgecolor="red", facecolor="none")
    ax[1].add_patch(rect)

    # labels and text
    labels = ["(a)", "(b) Focused", "(c) Focused"]
    fontsizes = (11, 11, 9)
    for a, label, fs in zip(ax, labels, fontsizes):
        a.set_ylabel("Range [µs]", fontsize=11)
        a.tick_params(axis="both", which="major", labelsize=9, direction="out")
        a.tick_params(axis="both", which="minor", direction="out")
        a.text(0.02, 0.95, label, transform=a.transAxes, fontsize=fs,
            fontweight="bold", va="top", ha="left", color="black",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

    ax[2].set_xlabel("Azimuth [km]", fontsize=11)

    # colorbars
    ims = [im0, im1, im3]
    for a, im in zip(ax, ims):
        cax = inset_axes(a, width="3%", height="100%",
                        loc='center right', borderpad=-2)  # negative pad pushes it outward
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Power [dB]", fontsize=8, labelpad=2)

    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{filename}.pgf", dpi=300, bbox_inches="tight")
    plt.close()


def IGARSS2026_rdrgrm_focused_profile(rdrgrm, focused, terrain, par, az_s1, az_s2, rng_s1, rng_s2, filename, 
                                      vminrdr=None, vminfoc=None, linspace=True, c1=299792458):

    if linspace:
        azmin = par['sx0']
        azmax = par['sx0'] + par['sdx'] * par['ns']
    else:
        raise NotImplementedError("Cannot handle uneven source spacing yet.")
    
    rb = int((par["rx_window_m"] / c1) / (1 / par["rx_sample_rate"]))
    
    extent = (azmin/1e3, azmax/1e3, 2*((par['rx_window_offset_m'] + par['rx_window_m'])/c1)*10**6,
        2*(par['rx_window_offset_m']/c1)*10**6)

    fig, ax = plt.subplots(4, 1, figsize=(3.555, 8), constrained_layout=True, dpi=300)

    # add profile to axis
    terrain.add_profile_to_axis(ax[0], 'x', 0)

    # shrink top plot
    pos = ax[0].get_position()
    new_height = pos.height * 0.5
    new_y = pos.y0 + pos.height - new_height
    ax[0].set_position([pos.x0+0.052, new_y, pos.width, new_height])

    # radargram panel
    im0 = ax[1].imshow(uc.lin_to_db(np.abs(rdrgrm)), cmap="viridis",
                    aspect=par['aspect'], extent=extent,
                    vmin=vminrdr)

    # focused panel
    im1 = ax[2].imshow(uc.lin_to_db(np.abs(focused)), cmap="viridis",
                    aspect=par['aspect'], extent=extent,
                    vmin=vminfoc)
    
    # zoomed panel
    zoomed = np.abs(focused[int(rng_s1*rb):int(rng_s2*rb),
                        int(az_s1*focused.shape[1]):int(az_s2*focused.shape[1])])
    
    zoomed_extent = ((azmin + (azmax-azmin) * az_s1)/1e3, (azmin + (azmax-azmin) * az_s2)/1e3,
                    2*((par['rx_window_offset_m'] + rng_s2 * par['rx_window_m'])/c1)*10**6,
                    2*((par['rx_window_offset_m'] + rng_s1 * par['rx_window_m'])/c1)*10**6)
    im3 = ax[3].imshow(uc.lin_to_db(zoomed), cmap="viridis",
                    aspect=par['aspect']*0.75, extent=zoomed_extent,
                    vmin=0)
    
    # shrink bottom plot to fit
    pos_top = ax[1].get_position()
    pos_bottom = ax[3].get_position()
    new_width = pos_top.width
    new_x = 0.05 + pos_bottom.x0 + (pos_bottom.width - new_width) / 2  # center it
    ax[3].set_position([new_x, pos_bottom.y0, new_width, pos_bottom.height])

    # add rectangle
    rect = Rectangle((zoomed_extent[0], zoomed_extent[2]),
                    zoomed_extent[1]-zoomed_extent[0],
                    zoomed_extent[3]-zoomed_extent[2],
                    linewidth=1, edgecolor="red", facecolor="none")
    ax[2].add_patch(rect)

    # labels and text
    labels = ["(a)","(b)", "(c) Focused", "(d) Focused"]
    fontsizes = (11, 11, 11, 9)
    for a, label, fs in zip(ax, labels, fontsizes):
        
        a.tick_params(axis="both", which="major", labelsize=9, direction="out")
        a.tick_params(axis="both", which="minor", direction="out")
        a.text(0.02, 0.95, label, transform=a.transAxes, fontsize=fs,
            fontweight="bold", va="top", ha="left", color="black",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

    ax[3].set_xlabel("Azimuth [km]", fontsize=11)

    # colorbars
    ims = [im0, im1, im3]
    for a, im in zip(ax[1:], ims):
        a.set_ylabel("Range [µs]", fontsize=11)
        cax = inset_axes(a, width="3%", height="100%",
                        loc='center right', borderpad=-2)  # negative pad pushes it outward
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Power [dB]", fontsize=8, labelpad=2)

    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{filename}.pgf", dpi=300, bbox_inches="tight")
    plt.close()
