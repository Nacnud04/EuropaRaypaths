import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sys.path.append("../../../src/PYTHON")
import output_handling as oh
import unit_convs      as uc

h = np.linspace(100e3, 19e3, 500)

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

def get_target_phase(params, h, d):

    rng = h + np.sqrt(params["eps_2"]) * d
    cmplx = np.exp(((-4j * np.pi) / params['lam']) * rng)
    return np.degrees(np.angle(cmplx))

plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

depths = (5000, 2500, 1000, 500, 250)
colors = ["red", "blue", "green", "purple", "blue"]
contrast = ["maroon", "orange", "pink", "magenta", "cyan"]

for d, c, cont in zip(depths, colors, contrast):

    if d != 5000 and d != 250:
        continue

    params = oh.load_params(f"inputs/params_{d:04d}.pkl", "inputs/layer.txt")

    P_r = get_analytic(params, h, d)

    # load radargram
    rdrgrm = oh.compile_rdrgrm(f"rdrgrm/{d:04d}", params)

    P_num = np.max(np.abs(rdrgrm)**2, axis=0)

    # find the phase of the maximum signal for each trace
    argmax = np.argmax(np.abs(rdrgrm), axis=0)
    phse   = np.degrees(np.angle(rdrgrm[argmax, range(rdrgrm.shape[1])]))

    error = np.abs(P_num - P_r) / P_r * 100

    ax[0,0].plot(h/1e3, uc.lin_to_db(P_num), label=f"Numerical: d={d} m", color=c, linewidth=1)
    ax[0,0].plot(h/1e3, uc.lin_to_db(P_r), color=cont, linestyle="--", label=f"Analytic: d={d} m", linewidth=1)

    ax[1,0].plot(h/1e3, error, color=c, label=f"Error (%): d={d} m", linewidth=1)

    # find the phase of the maximum signal for each trace
    argmax = np.argmax(np.abs(rdrgrm), axis=0)
    phse   = np.degrees(np.angle(rdrgrm[argmax, range(rdrgrm.shape[1])]))

    ana_phse = get_target_phase(params, h, d)
    phase_error = phse - ana_phse

    # make sure phase error is between -180 and 180
    phase_error = (phase_error + 180) % 360 - 180

    # mean phase error
    mean_error = np.median(np.abs(phase_error))

    print(f"Median errors for d={d} m: {np.median(error):.2f} %, {mean_error:.2f} deg")

    ax[0,1].axhline(mean_error, color=c, alpha=0.5, linestyle="--")
    ax[0,1].axhline(0, color="black", linestyle="--")
    ax[0,1].text(h[0]/1e3 + 1, mean_error, f"{mean_error:.2f}\ndeg", color=c, alpha=0.7, fontsize=12, verticalalignment="center")

    ax[0,1].plot(h/1e3, phase_error, label=f"Target depth: {d} m", color=c)

    ax[0,1].set_ylabel("Phase Error (deg)")
    ax[0,1].set_xlabel("Altitude [km]")
    ax[0,1].legend()
    ax[0,1].set_title("Phase Error of Subsurface Facet")

ax[0,0].set_ylabel("Maximum Power [dBW]")
ax[0,0].legend()
ax[0,0].set_title("Individual Subsurface Target (Facet)")

ax[1,0].set_xlabel("Altitude [km]")
ax[1,0].set_ylabel("Error (%)")
#ax[1].set_ylim(0, 10)

ax[1,1].set_axis_off()

plt.xlim(h[-1]/1e3, h[0]/1e3)
plt.savefig("figures/SubsurfaceFacet.png", dpi=300)

plt.show()
