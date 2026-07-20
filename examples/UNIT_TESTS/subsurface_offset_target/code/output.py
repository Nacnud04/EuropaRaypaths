import sys
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

sys.path.append("../../../src/PYTHON")
import output_handling as oh
import unit_convs      as uc

h = np.linspace(19e3, 300e3, 250)

def sinc(x):

    mask = x != 0
    result = np.zeros_like(x)
    result[mask] = np.sin(x[mask] * np.pi) / (x[mask] * np.pi)
    result[~mask] = 1
    return result

def facet_gain(th, ph, params):

    fs = params["fs"]
    lam = params["lam"]

    piLlam = (np.pi * fs) / lam

    sinc1 = sinc(piLlam * np.sin(th) * np.cos(ph))
    sinc2 = sinc(piLlam * np.sin(th) * np.sin(ph))

    D_norm = (sinc1 * sinc2) ** 2

    D_max = (4 * np.pi * fs**2) / lam**2

    return D_norm * D_max

def facet_gain_wrap(h, d, xoff, params):

    if xoff == 0:
        th = np.zeros_like(h)
    else:
        th = np.arctan(xoff /(h + d))
    ph = 0
    return facet_gain(th, ph, params)

def get_analytic(params, h, d, xoff):

    lam = params["lam"]
    G   = 10 ** (params["surface_gain"] / 10)
    P_t = params["power"]
    n   = np.sqrt(params["eps_2"])

    # refraction gain
    g_r = (h + d) / (h + d/n)

    # target gain
    G_T = facet_gain_wrap(h, d, xoff, params)

    R = np.sqrt((h + d)**2 + xoff**2)

    P_r = (P_t * G**2 * lam**4 * G_T**2) / ((4 * np.pi)**4 * (R)**4)
    P_r *= g_r**4 / n**2

    return P_r

def refracted_angle(th_r, xoff, h, d, eps_2):

    y = np.arcsin(np.sqrt(eps_2) * np.sin(th_r))

    return h * np.tan(y) + d * np.tan(th_r) - xoff

def get_target_phase(params, h, d, xoff):

    if xoff != 0:

        # first find the refracted angle numerically
        th_r = np.zeros_like(h)
        for i, height in enumerate(h):
            sol = root_scalar(refracted_angle,
                            args=(xoff, height, d, params["eps_2"]), bracket=[0, np.radians(20)],
                            method='brentq'
                            )
            
            th_r[i] = sol.root

        th_i = np.arcsin(np.sqrt(params["eps_2"]) * np.sin(th_r))
        #print(th_r)
        #print(th_i)

        r1   = h / np.cos(th_i)
        r2   = d / np.cos(th_r)

        corrected_rng = r1 + r2 * np.sqrt(params["eps_2"])
        #print(corrected_rng)
    else: 

        corrected_rng = h + np.sqrt(params["eps_2"]) * d
    
    cmplx = np.exp(((-4j * np.pi) / params['lam']) * corrected_rng)

    return np.degrees(np.angle(cmplx))


plt.rcParams.update({'font.size': 14})

xoff = 2000

depths = (5000, 1000, 500, 250)
colors = ["red", "green", "purple", "blue"]
contrast = ["maroon", "lime", "fuchsia", "cyan"]

z = np.linspace(0, np.radians(20), 500)

f = refracted_angle(z, xoff, h[0], 250, 3.15)

plt.plot(np.degrees(z), f)
plt.axhline(0, color='k')
plt.show()

for d, c, cont in zip(depths, colors, contrast):

    if d != 5000 and d != 250:
        continue

    params = oh.load_params(f"inputs/params_{d:04d}.pkl", f"inputs/target{d:04d}.txt")

    if d == depths[0]:
        G_T_tmp = facet_gain_wrap(19e3, 250, 500, params)
        print(f"G_T at 19 km altitude and 250 m depth with {xoff} m x-offset: {G_T_tmp:.6f}")
        G_T_max = facet_gain_wrap(19e3, 250, 0, params)
        print(f"Which is {G_T_tmp / G_T_max:.6f} times the max")

    P_r = get_analytic(params, h, d, xoff)

    # load radargram
    rdrgrm = oh.compile_rdrgrm(f"rdrgrm/{d:04d}", params)

    #plt.imshow(np.abs(rdrgrm), aspect="auto")
    #plt.show()

    P_num = np.max(np.abs(rdrgrm)**2, axis=0)

    # find the phase of the maximum signal for each trace
    argmax = np.argmax(np.abs(rdrgrm), axis=0)
    phse   = np.degrees(np.angle(rdrgrm[argmax, range(rdrgrm.shape[1])]))

    error = np.abs(P_num - P_r) / P_r * 100

    ana_phse = get_target_phase(params, h, d, xoff)

    # find the phase of the maximum signal for each trace
    argmax = np.argmax(np.abs(rdrgrm), axis=0)
    phse   = np.degrees(np.angle(rdrgrm[argmax, range(rdrgrm.shape[1])]))

    phase_error = phse - ana_phse

    # make sure phase error is between -180 and 180
    phase_error = (phase_error + 180) % 360 - 180
    
    print(f"Exporting to figures/nadir_{d:04d}.csv")
    df = pd.DataFrame({"ALT": h, "NUM_POW": P_num, "ANA_POW": P_r, "ERR_POW": error, "ERR_PHS": phase_error})
    df.to_csv(f"figures/nadir_{d:04d}.csv", index=False)

    fig, ax = plt.subplots(2, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

    ax[0].plot(h/1e3, uc.lin_to_db(P_num), label=f"Numerical", color=c, linewidth=1)
    ax[0].plot(h/1e3, uc.lin_to_db(P_r), color=cont, linestyle="--", label=f"Analytic", linewidth=1)

    ax[1].plot(h/1e3, error, color=c, label=f"Error (%): d={d} m", linewidth=1)

    ax[0].set_ylabel("Maximum Power [dBW]")
    ax[0].legend()
    ax[0].set_title(f"Off-nadir Subsurface Facet - Depth = {d} m")

    ax[1].set_xlabel("Altitude [km]")
    ax[1].set_ylabel("Error (%)")
    #ax[1].set_ylim(0, 10)

    plt.xlim(h[-1]/1e3, h[0]/1e3)
    plt.savefig(f"figures/SubsurfaceFacet{d:04d}.png", dpi=300)

    plt.close()

#fig, ax = plt.subplots(figsize=(8, 5))
fig, ax = plt.subplots(figsize=(10, 8))

ax.set_xlim(h[-1]/1e3, h[0]/1e3)

for d, c, cont in zip(depths, colors, contrast):

    ana_phse = get_target_phase(params, h, d, xoff)

    # load radargram
    rdrgrm = oh.compile_rdrgrm(f"rdrgrm/{d:04d}", params)

    # find the phase of the maximum signal for each trace
    argmax = np.argmax(np.abs(rdrgrm), axis=0)
    phse   = np.degrees(np.angle(rdrgrm[argmax, range(rdrgrm.shape[1])]))

    phase_error = phse - ana_phse

    # make sure phase error is between -180 and 180
    phase_error = (phase_error + 180) % 360 - 180

    # mean phase error
    mean_error = np.mean(phase_error)

    # line showing mean error with text at end
    ax.axhline(mean_error, color=c, alpha=0.5, linestyle="--")
    ax.text(h[0]/1e3 + 1, mean_error, f"{mean_error:.2f}\ndeg", color=c, alpha=0.7, fontsize=12, verticalalignment="center")

    ax.plot(h/1e3, phase_error, label=f"Target depth: {d} m", color=c)

ax.set_ylabel("Phase Error (deg)")
ax.set_xlabel("Altitude [km]")
ax.legend(fontsize=10)
ax.set_title("Phase Error of Subsurface Facet")
plt.savefig("figures/SubsurfaceFacet_PhaseError.png", dpi=300)
plt.show()
