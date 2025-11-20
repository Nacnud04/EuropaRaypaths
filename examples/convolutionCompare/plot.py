import numpy as np
import matplotlib.pyplot as plt
import pickle

directory = "rdrgrms/"
unfocused_files = ["NoConv_raw.npz", "CircConv_raw.npz", "LinConv_raw.npz"]
focused_files   = ["NoConv_focused.npz", "CircConv_focused.npz", "LinConv_focused.npz"]
names           = ["Original", "Circular Convolution", "Linear Convolution"]

with open("NoConv.pkl", 'rb') as hdl:
    params = pickle.load(hdl)

# 5 rows: 3 methods + 2 diff rows, 2 columns: unfocused and focused
fig, axes = plt.subplots(5, 2, figsize=(15, 18))

# Load original (no convolution) data as baseline for difference computation
rdrgrm_unf_orig = np.load(directory + unfocused_files[0])['rdrgrm']
rdrgrm_foc_orig = np.load(directory + focused_files[0])['focused']

for i, (unf_file, foc_file, name) in enumerate(zip(unfocused_files, focused_files, names)):

    rdrgrm_unf = np.load(directory + unf_file)['rdrgrm']
    rdrgrm_foc = np.load(directory + foc_file)['focused']

    extent = [-5, 5, 2*(params["rx_window_offset_m"] + params["rx_window_m"])/299.792458, 
              2*params["rx_window_offset_m"]/299.792458]

    # Plot unfocused and focused images with colorbars
    im_unf = axes[i][0].imshow(rdrgrm_unf, aspect='auto', cmap='viridis', interpolation='nearest', vmin=-20, extent=extent)
    axes[i][0].set_title(f"{name} - Unfocused")
    fig.colorbar(im_unf, ax=axes[i][0])

    im_foc = axes[i][1].imshow(rdrgrm_foc, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, extent=extent)
    axes[i][1].set_title(f"{name} - Focused")
    fig.colorbar(im_foc, ax=axes[i][1])

# Compute and plot differences for the last two rows: Circular-Original and Linear-Original
for j, conv_idx in enumerate([1, 2]):  # indices for Circular and Linear Convolution
    # Unfocused difference (no absolute value)
    lims = 20
    diff_unf = np.load(directory + unfocused_files[conv_idx])['rdrgrm'] - rdrgrm_unf_orig
    im_diff_unf = axes[3 + j][0].imshow(diff_unf, aspect='auto', cmap='seismic', interpolation='nearest', vmin=-1*lims, vmax=lims, extent=extent)
    axes[3 + j][0].set_title(f"{names[conv_idx]} - Original (Difference, Unfocused)")
    fig.colorbar(im_diff_unf, ax=axes[3 + j][0])

    # Focused difference (no absolute value)
    diff_foc = np.load(directory + focused_files[conv_idx])['focused'] - rdrgrm_foc_orig
    im_diff_foc = axes[3 + j][1].imshow(diff_foc, aspect='auto', cmap='seismic', interpolation='nearest', vmin=-1*lims, vmax=lims, extent=extent)
    axes[3 + j][1].set_title(f"{names[conv_idx]} - Original (Difference, Focused)")
    fig.colorbar(im_diff_foc, ax=axes[3 + j][1])

plt.tight_layout()
plt.savefig("ConvolutionComparison.png")
plt.close()
