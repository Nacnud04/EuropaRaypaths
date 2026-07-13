import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sys.path.append("../../../src/PYTHON")
import output_handling as oh
import unit_convs      as uc

h = np.linspace(100e3, 20e3, 200)

def get_analytic(params, h, d):

    lam = params["lam"]
    G   = 10 ** (params["surface_gain"] / 10)
    P_t = params["power"]
    n   = np.sqrt(params["eps_2"])

    # refraction gain
    g_r = (h + d) / (h + d/n)

    # target gain
    facet_size = params['fs']
    G_T = (4 * np.pi * facet_size**2) / lam**2

    P_r = (P_t * G**2 * lam**4 * G_T**2) / ((4 * np.pi)**4 * (h + d)**4)
    P_r *= g_r**4 / n**2

    return P_r

def get_analytic_layer(params, h, d):

    lam = params["lam"]
    G   = 10 ** (params["surface_gain"] / 10)
    P_t = params["power"]
    n   = np.sqrt(params["eps_2"])

    # refraction gain
    g_r = (h + d) / (h + d/n)

    P_r = (P_t * G**2 * lam**2) / ((4 * np.pi)**2 * (2*(h + d))**2)
    P_r *= g_r**2

    return P_r

def get_target_phase(params, h, d):

    rng = h + np.sqrt(params["eps_2"]) * d
    cmplx = np.exp(((-4j * np.pi) / params['lam']) * rng)
    return np.degrees(np.angle(cmplx))

d = 20000
c = "red"
cont = "black"

params = oh.load_params(f"inputs/params.pkl", "inputs/layer.txt")

P_r = get_analytic(params, h, d)
P_r_layer = get_analytic_layer(params, h, d)

# load radargram
rdrgrm = oh.compile_rdrgrm(f"rdrgrm/", params)

max_row = np.argmax(np.abs(rdrgrm)**2, axis=0)
Emax = rdrgrm[max_row, np.arange(rdrgrm.shape[1])]
    
P_num = np.max(np.abs(rdrgrm)**2, axis=0)

err_mean = np.mean(P_num / P_r_layer)
#err_mean = np.mean(P_num / P_r)

#error = np.abs(P_num - P_r) / P_r * 100
error = np.abs(P_num - P_r_layer) / P_r_layer * 100

# export as csv
df = pd.DataFrame({"ALT":h,"NUM_POW":P_num,"ANA_POW":P_r_layer,"ERR":error})
df.to_csv("figures/subsurface_layer.csv", index=False)

fig, ax = plt.subplots(2, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

ax[0].plot(h/1e3, uc.lin_to_db(P_num), label=f"Numerical: d={d} m", color=c, linewidth=1)
#ax[0].plot(h/1e3, uc.lin_to_db(P_r), color=cont, linestyle="--", label=f"Analytic: d={d} m", linewidth=1)
ax[0].plot(h/1e3, uc.lin_to_db(P_r_layer), color=cont, linestyle="-.", label=f"Layer: d={d} m", linewidth=1)

ax[1].plot(h/1e3, error, color=c, label=f"Error (%): d={d} m", linewidth=1)

ax[0].set_ylabel("Maximum Power [dBW]")
ax[0].legend()
ax[0].set_title("Infinite Subsurface Layer")

ax[1].set_xlabel("Altitude [km]")
ax[1].set_ylabel("Error (%)")
ax[1].set_ylim(0, 10)

plt.xlim(h[-1]/1e3, h[0]/1e3)
plt.savefig("figures/SubsurfaceFacet.png", dpi=300)

plt.show()
