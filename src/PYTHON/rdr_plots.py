import matplotlib.pyplot as plt
import numpy as np

import unit_convs as uc

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