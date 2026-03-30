import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import unit_convs as uc
import output_handling as oh

# matplotlib configuration for LaTeX
os.environ["PATH"] += os.pathsep + '/usr/share/texlive/texmf-dist/tex/xelatex'


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def simple_rdrgrm(rdrgrm, par, path, title=None, linspace=True, vmin=None, vmax=None):

    if linspace:
        xmin = par['sx0']
        xmax = par['sx0'] + par['sdx'] * par['ns']
    else:
        xmin = 0
        xmax = rdrgrm.shape[1]
        #raise NotImplementedError("Cannot handle uneven source spacing yet.")

    plt.imshow(uc.lin_to_db(np.abs(rdrgrm)), aspect='auto', vmin=vmin, vmax=vmax,
           extent=[xmin/1e3, xmax/1e3, (par["rx_window_offset_m"] + par["rx_window_m"]), par["rx_window_offset_m"]])
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


def plot_rdr_attenuation_prof(rdr, focused, attenuation_file, params, savefig):

    range_extent = [
        -5, 5,
        2 * (params["rx_window_offset_m"] + params["rx_window_m"]) / 299.792458,
        2 * params["rx_window_offset_m"] / 299.792458
    ]

    conductivity, ex1_xmins, ex1_xmaxs, ex1_ymins, ex1_ymaxs, ex1_zmins, ex1_zmaxs = oh.load_attenuation_geom(attenuation_file)

    # debug (print attenuation boundaries)
    print("Attenuation boundaries:")
    for c, xmin, xmax, ymin, ymax, zmin, zmax in zip(conductivity, ex1_xmins, ex1_xmaxs, ex1_ymins, ex1_ymaxs, ex1_zmins, ex1_zmaxs):
        print(f"  Conductivity: {c}, X: [{xmin}, {xmax}], Y: [{ymin}, {ymax}], Z: [{zmin}, {zmax}]")

    cmap = plt.get_cmap('inferno')
    norm = plt.Normalize(vmin=params['sig_1'], vmax=np.max(conductivity))

    fig, ax = plt.subplots(3, figsize=((7, 12)), sharex=True)

    # set xlim and ylim in km
    ax[0].set_ylim((params['rx_window_offset_m']+params['rx_window_m'])/1e3, params['rx_window_offset_m']/1e3)
    ax[0].set_xlim(params['sx0']/1e3, (params['sx0']+params['sdx']*params['ns'])/1e3)

    # plot atmosphere conductivity
    rect_x = params['sx0'] / 1e3
    rect_y = params['sz'] / 1e3
    rect_w = (params['sdx'] * params['ns']) / 1e3
    rect_h = -1*params['rx_window_m'] / 1e3
    ax[0].add_patch(Rectangle((rect_x, rect_y), rect_w, rect_h,
                        color=cmap(norm(params['sig_1'])), zorder=0))

    # plot subsurface conductivity
    rect_x = params['sx0'] / 1e3
    rect_y = params['sz'] / 1e3
    rect_w = (params['sdx'] * params['ns']) / 1e3
    rect_h = params['rx_window_m'] / 1e3
    ax[0].add_patch(Rectangle((rect_x, rect_y), rect_w, rect_h,
                        color=cmap(norm(params['sig_2'])), zorder=0))

    # plot conductivity rectangles
    for c, xmin, xmax, ymin, ymax, zmin, zmax in zip(conductivity, ex1_xmins, ex1_xmaxs, ex1_ymins, ex1_ymaxs, ex1_zmins, ex1_zmaxs):
        rect_x = xmin / 1e3
        rect_y = (params['sz'] + zmax) / 1e3
        rect_w = (xmax - xmin) / 1e3
        rect_h = (zmax - zmin) / 1e3
        ax[0].add_patch(Rectangle((rect_x, rect_y), rect_w, rect_h,
                            color=cmap(norm(c)), zorder=1))

    # add conductivity plot labels
    ax[0].set_title("Attenuation Geometry")
    ax[0].set_ylabel("Range [km]")

    # add radargram plot labels
    im1 = ax[1].imshow(uc.lin_to_db(np.abs(rdr)), vmin=-20, extent=range_extent, aspect='auto')
    ax[1].set_title("Radar Image")
    ax[1].set_ylabel("Range [us]")

    im2 = ax[2].imshow(uc.lin_to_db(np.abs(focused)), vmin=0, extent=range_extent, aspect='auto')
    ax[2].set_title("Focused Radar Image")
    ax[2].set_xlabel("Azimuth [km]")
    ax[2].set_ylabel("Range [us]")


    # Colorbar for conductivity (subplot 0)
    cbar0 = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax[0],
        orientation='vertical',
        label='Conductivity [S/m]'
    )

    # Colorbar for radargram (subplot 1)
    cbar1 = fig.colorbar(im1, ax=ax[1], orientation='vertical')
    cbar1.set_label("Power [dB]")

    # Colorbar for focused radargram (subplot 2)
    cbar2 = fig.colorbar(im2, ax=ax[2], orientation='vertical')
    cbar2.set_label("Power [dB]")

    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()


def plot_SHARAD_comparison(real, synth, geometry, aeroid, 
                           mola, plotpar, layers=False):

    # set sharad params
    ADC_SAMP_INT = 0.0375e-6
    c = 299792458

    # extract plotting parameters from dict
    trc_st = plotpar['trc_st']
    trc_en = plotpar['trc_en']
    rea_rng_st = plotpar['rea_rng_st']
    rea_rng_en = plotpar['rea_rng_en']
    ymin = plotpar['ymin']
    ymax = plotpar['ymax']

    # crop real data down
    rb_crop_st = int((rea_rng_st - ymax) // (ADC_SAMP_INT*c/2))
    rb_crop_en = int((rea_rng_en - ymax) // (ADC_SAMP_INT*c/2))
    real = real[rb_crop_st:rb_crop_en-1, trc_st:trc_en]

    # adjust if necessary
    if real.shape[1] > synth.shape[1]:
        real = real[:, :synth.shape[1]]
    if synth.shape[0] > real.shape[0]:
        synth = synth[:real.shape[0], :]

    # scale rdr images
    rea_scl = uc.scale_range(real,  plotpar['rea_min'], plotpar['rea_max'])
    syn_scl = uc.scale_range(synth, plotpar['syn_min'], plotpar['syn_max'])

    # RGB color scale
    rea_clr = np.array([0xFF, 0x00, 0x80])
    syn_clr = np.array([0x00, 0xFF, 0x80])

    # move radargrams into RGB
    rgb = np.zeros((rea_scl.shape[0], rea_scl.shape[1], 3))
    for i in range(3):
        rgb[:, :, i] += rea_scl * rea_clr[i]
        rgb[:, :, i] += syn_scl * syn_clr[i]
    rgb[:, :, 2][rgb[:, :, 2] > 255] = 255 # fix blue channel wrapping

    # export
    if "unformatted_path" in plotpar:
        plt.imsave(plotpar['unformatted_path'], rgb.astype(np.uint8))

    # --- GENERATE FORMATTED PLOT ---

    fig, ax = plt.subplots(figsize=(8,9))

    # radar extents
    extent_syn = [
        np.max(aeroid['LAT'][trc_st:trc_en]),
        np.min(aeroid['LAT'][trc_st:trc_en]),
        rea_rng_en/1e3,
        rea_rng_st/1e3,
    ]

    # plot rgb image
    im = ax.imshow(rgb.astype(np.uint8), extent=extent_syn, aspect='auto')

    # colorbars
    divider = make_axes_locatable(ax)

    # place bars
    cax1 = divider.append_axes("right", size="3%", pad=0.05)
    cax2 = divider.append_axes("right", size="3%", pad=0.4)

    # build colormaps
    rea_clr = np.array([255, 0, 128]) / 255
    syn_clr = np.array([0, 255, 128]) / 255

    cmap_rea = mcolors.LinearSegmentedColormap.from_list("rea_map", [(0,0,0), rea_clr])
    cmap_syn = mcolors.LinearSegmentedColormap.from_list("syn_map", [(0,0,0), syn_clr])

    mappable_rea = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=np.min(rea_scl), vmax=np.max(rea_scl)),
        cmap=cmap_rea
    )
    mappable_rea.set_array([])

    mappable_foc = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=np.min(syn_scl), vmax=np.max(syn_scl)),
        cmap=cmap_syn
    )
    mappable_foc.set_array([])

    cbar1 = fig.colorbar(mappable_foc, cax=cax1)
    cbar1.set_label("Synthetic Power [dB]", rotation=270, labelpad=-6)

    cbar2 = fig.colorbar(mappable_rea, cax=cax2)
    cbar2.set_label("Normalized Real Power [dB]", rotation=270, labelpad=-6)

    ticklocs = [0, 0.25, 0.75, 1.0]
    cbar1.set_ticks(ticklocs)
    cbar1.set_ticklabels([f"{t * (plotpar['syn_max'] - plotpar['syn_min']) + plotpar['syn_min']:.1f}" for t in ticklocs], fontsize=8)

    cbar2.set_ticks(ticklocs)
    cbar2.set_ticklabels([f"{t:.1f}" for t in ticklocs], fontsize=8)

    # if there is a subsurface plot the layers
    if "trc" in plotpar.keys() and "depth" in plotpar.keys() and layers == True:
        for data in zip(plotpar["trc"], plotpar["depth"]):
            lat_subsrf = aeroid['LAT'][data[0]]
            depth_subsrf = data[1] + 315.4
            ax.plot(lat_subsrf, depth_subsrf, color="white", linewidth=1, alpha=0.5)

    # labels
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Range [km]")
    #ax.set_title(f"SHARAD Obs. {plotpar['obs']}", fontsize=18, fontweight="bold")

    plt.savefig(f"{plotpar['savefig']}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plotpar['savefig']}.pgf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_unfoc_foc(rdrgrm, focused, rx_win, OBS, geometry=None,
                   rdrmin=22, rdrmax=35, focmin=35, focmax=45,
                   aspect=250, ymin=314.15, ymax=318, show=False):

    Nr, Na = rdrgrm.shape

    extent = [
        0, Na,
        np.min(rx_win)/1e3 + 7.5, np.min(rx_win)/1e3,
    ]

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharey=True)

    # rdrgrms
    im1 = ax[0].imshow(rdrgrm, vmin=rdrmin, vmax=rdrmax, cmap="viridis", aspect=aspect, extent=extent)
    plt.colorbar(im1, label="dB", shrink=0.9)
    im2 = ax[1].imshow(focused, extent=extent, vmin=focmin, vmax=focmax, aspect=aspect)
    plt.colorbar(im2, label="dB", shrink=0.9)

    # labels
    for a in ax: a.set_ylabel("Range [km]")
    ax[1].set_xlabel("Echo Count")
    ax[0].text(
        -0.16, 0.5, "Unfocused",
        transform=ax[0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )
    ax[1].text(
        -0.16, 0.5, "Focused",
        transform=ax[1].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )
    plt.suptitle(f"SHARAD {OBS}", fontsize=18, fontweight="bold")

    # cropping
    plt.ylim(ymax, ymin)

    if type(geometry) is not type(None):
        # crop geometry to just what successfully simulated
        # upsample the geometry range
        srange = np.interp(np.linspace(0, 1, 2000), np.linspace(0, 1, len(geometry)), geometry["SRANGE"]/1e3)
        for a in ax:
            a.plot(srange[55:1550], color="red", linewidth=1) 

    # export
    plt.tight_layout(rect=(0.08, 0, 1, 1))
    plt.savefig("figures/output.png")

    if show:
        plt.show()

    plt.close()


def TGRS_rdrgrm_focused(rdrgrm, focused, par, filename, 
                        vminrdr=None, vminfoc=None, linspace=True, c1=299792458):

    if linspace:
        azmin = par['sx0']
        azmax = par['sx0'] + par['sdx'] * par['ns']
    else:
        raise NotImplementedError("Cannot handle uneven source spacing yet.")
    
    rb = int((par["rx_window_m"] / c1) / (1 / par["rx_sample_rate"]))
    
    extent = (azmin/1e3, azmax/1e3, 2*((par['rx_window_offset_m'] + par['rx_window_m'])/c1)*10**6,
        2*(par['rx_window_offset_m']/c1)*10**6)

    fig, ax = plt.subplots(2, 1, figsize=(3, 5), constrained_layout=True, dpi=300)

    # radargram panel
    im0 = ax[0].imshow(uc.lin_to_db(np.abs(rdrgrm)), cmap="viridis",
                    aspect=par['aspect'], extent=extent,
                    vmin=vminrdr)

    # focused panel
    im1 = ax[1].imshow(uc.lin_to_db(np.abs(focused)), cmap="viridis",
                    aspect=par['aspect'], extent=extent,
                    vmin=vminfoc)

    # labels and text
    labels = ["(a)", "(b) Focused"]
    fontsizes = (11, 11, 9)
    for a, label, fs in zip(ax, labels, fontsizes):
        a.set_ylabel("Range [µs]", fontsize=11)
        a.tick_params(axis="both", which="major", labelsize=9, direction="out")
        a.tick_params(axis="both", which="minor", direction="out")
        a.text(0.02, 0.95, label, transform=a.transAxes, fontsize=fs,
            fontweight="bold", va="top", ha="left", color="black",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

    ax[1].set_xlabel("Azimuth [km]", fontsize=11)

    # colorbars
    ims = [im0, im1]
    for a, im in zip(ax, ims):
        cax = inset_axes(a, width="3%", height="100%",
                        loc='center right', borderpad=-2)  # negative pad pushes it outward
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Power [dB]", fontsize=8, labelpad=2)

    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{filename}.pgf", dpi=300, bbox_inches="tight")
    plt.close()



def TGRS_rdrgrm_conv(params, directory, focused_files, unfocused_files, names):

    scl = 0.75
    pad = 12
    fig, axes = plt.subplots(3, 2, figsize=(scl*10, scl*12), sharey=True, sharex=True)

    # Column titles
    fig.text(0.31, 0.95, "Unfocused", ha='center', fontsize=16, fontweight="bold")
    fig.text(0.74, 0.95, "Focused", ha='center', fontsize=16, fontweight="bold")

    rdrgrm_unf_orig = np.load(directory + unfocused_files[0])['rdrgrm']
    rdrgrm_foc_orig = np.load(directory + focused_files[0])['focused']

    for i, (unf_file, foc_file, name) in enumerate(zip(unfocused_files, focused_files, names)):

        # Row title
        fig.text(0.035, 0.8 - i*0.30, name, va='center', rotation=90, fontsize=14, fontweight="bold")

        rdrgrm_unf = np.load(directory + unf_file)['rdrgrm']
        rdrgrm_foc = np.load(directory + foc_file)['focused']

        extent = [-5, 5,
                2*(params["rx_window_offset_m"] + params["rx_window_m"])/299.792458,
                2*params["rx_window_offset_m"]/299.792458]

        # Unfocused
        im_unf = axes[i][0].imshow(rdrgrm_unf, aspect='auto', cmap='viridis',
                                interpolation='nearest', vmin=-5, extent=extent)
        cbar = fig.colorbar(im_unf, ax=axes[i][0], fraction=0.046, pad=0.02)
        cbar.set_label("Power [dB]", rotation=270, labelpad=pad)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set_label_position('top')

        # Focused
        im_foc = axes[i][1].imshow(rdrgrm_foc, aspect='auto', cmap='viridis',
                                interpolation='nearest', vmin=10, extent=extent)
        cbar = fig.colorbar(im_foc, ax=axes[i][1], fraction=0.046, pad=0.02)
        cbar.set_label("Power [dB]", rotation=270, labelpad=pad)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set_label_position('top')

        # Add y-axis label only on left column
        axes[i][0].set_ylabel("Range [µs]")

    # Difference row
    j = 0
    conv_idx = 1
    row_idx = len(focused_files) + j

    fig.text(0.035, 0.8 - row_idx*0.30, f"Difference",
            va='center', rotation=90, fontsize=14, fontweight="bold")

    lims = 5
    diff_unf = np.load(directory + unfocused_files[conv_idx])['rdrgrm'] - rdrgrm_unf_orig
    im_diff_unf = axes[row_idx][0].imshow(diff_unf, aspect='auto', cmap='coolwarm',
                                        interpolation='nearest', vmin=-lims, vmax=lims,
                                        extent=extent)

    diff_foc = np.load(directory + focused_files[conv_idx])['focused'] - rdrgrm_foc_orig
    im_diff_foc = axes[row_idx][1].imshow(diff_foc, aspect='auto', cmap='coolwarm',
                                        interpolation='nearest', vmin=-lims, vmax=lims,
                                        extent=extent)

    cbar = fig.colorbar(im_diff_unf, ax=axes[row_idx][0], fraction=0.046, pad=0.02)
    cbar.set_label("Difference [dB]", rotation=270, labelpad=pad)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=8)

    cbar = fig.colorbar(im_diff_foc, ax=axes[row_idx][1], fraction=0.046, pad=0.02)
    cbar.set_label("Difference [dB]", rotation=270, labelpad=pad)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=8)

    # Axis labels for the last row
    axes[row_idx][0].set_xlabel("Azimuth [km]")
    axes[row_idx][1].set_xlabel("Azimuth [km]")
    axes[row_idx][0].set_ylabel("Range [µs]")

    # subplot labels
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    fontsizes = (11, 11, 11, 11, 11, 11)
    for a, label, fs in zip(axes.flatten(), labels, fontsizes):
        a.text(0.04, 0.96, label, transform=a.transAxes, fontsize=fs,
            fontweight="bold", va="top", ha="left", color="black",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

    plt.tight_layout(rect=[0.06, 0, 1, 0.95])
    plt.savefig("TGRS-CONV-COMP.png")
    plt.savefig("TGRS-CONV-COMP.pgf")
    plt.close()


def TGRS_KOR1_SYN(rdrgrm, focused, rx_win, OBS, mola, aeroid, plotpar, geometry=None,
                   trc_st=750, trc_en=1200, c=299.792458,
                   rdrmin=22, rdrmax=35, focmin=35, focmax=45,
                   aspect="auto", ymin=314.15, ymax=318, show=False):

    Nr, Na = rdrgrm.shape

    extent = [
        geometry['LAT'].iloc[0], geometry['LAT'].iloc[-1],
        (np.min(rx_win) + 7.5e3)/c, (np.min(rx_win))/c,
    ]

    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

    ax[0].set_xlim(geometry['LAT'].iloc[0], geometry['LAT'].iloc[-1])
    if "trc" in plotpar.keys() and "depth" in plotpar.keys():
        i = 0
        for data in zip(plotpar["trc"], plotpar["depth"]):
            lat_subsrf = aeroid['LAT'][data[0]]
            depth_subsrf = data[1] + 315.4
            if np.any(depth_subsrf < 316):
                continue
            else:
                if i == 0:
                    ax[0].plot(lat_subsrf, (depth_subsrf*1e3)/c, color="red", linewidth=1, label="Subsurface Layers")
                else:
                    ax[0].plot(lat_subsrf, (depth_subsrf*1e3)/c, color="red", linewidth=1)
                i += 1

    ax[0].plot(mola['LAT'], mola['SRANGE']/c, color="black", linewidth=1, label="Surface Topography")
    ax[0].legend()

    # rdrgrms
    im1 = ax[1].imshow(rdrgrm, vmin=rdrmin, vmax=rdrmax, cmap="viridis", aspect=aspect, extent=extent)
    plt.colorbar(im1, label="Power [dB]", pad=0.01)
    im2 = ax[2].imshow(focused, extent=extent, vmin=focmin, vmax=focmax, aspect=aspect)
    plt.colorbar(im2, label="Power [dB]", pad=0.01)

    # labels
    for a in ax: a.set_ylabel("Range [$\mu$s]")
    ax[2].set_xlabel("Latitude [deg]")

    labels = ["(a) 2D Slice","(b) Unfocused", "(c) Focused"]
    fontsizes = (11, 11, 11)
    for a, label, fs in zip(ax, labels, fontsizes):
        
        a.tick_params(axis="both", which="major", labelsize=9, direction="out")
        a.tick_params(axis="both", which="minor", direction="out")
        a.text(0.011, 0.95, label, transform=a.transAxes, fontsize=fs,
            fontweight="bold", va="top", ha="left", color="black",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

    # cropping
    for a in ax: a.set_ylim((ymax*1e3)/c, (ymin*1e3)/c)

    # export
    plt.savefig("figures/TGRS-KOR1-SYN.png")
    plt.savefig("figures/TGRS-KOR1-SYN.pgf")

    if show:
        plt.show()

    plt.close()
