import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sys.path.append("../../../src/PYTHON")
import output_handling as oh
import unit_convs      as uc

h = np.linspace(30e3, 20e3, 10)

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

a_ss = []
a_os = []

#f = "rdrgrm/Ptarg_s000000_t00.txt"
for i in range(rdrgrm.shape[1]):
    f = f"rdrgrm/s{i:06d}.txt"
    arr = np.loadtxt(f)
    sig = arr[:, 0] + 1j * arr[:, 1]
    idx = np.argmax(np.abs(sig))
    angle = np.degrees(np.angle(sig[idx]))
    #a_o = np.degrees(np.angle(sig[idx]*-1))
    a_o = (angle + 360) % 360 - 180
    print(f"TRC {i}: \n\
          Max norm signal is {sig[idx]/np.abs(sig[idx])}\n\
          With angle {angle:.2f} ({a_o:.2f}) degrees\n\
          At altitude: {h[i]} m\n")
    a_ss.append(angle)
    a_os.append(a_o)
    """
    fig, ax = plt.subplots(2)
    xs = range(len(sig))
    ax[0].plot(xs, np.real(sig), color="red", label="real E")
    ax[0].plot(xs, np.imag(sig), color="blue", label="imag E")
    ax[1].plot(xs, np.degrees(np.angle(sig)), color="purple", label="phase E")
    ax[1].set_ylim(-180, 180)
    plt.show()
    """

fig, ax = plt.subplots(2)
ana_phs = get_target_phase(params, h, d)
ax[0].plot(h/1e3, a_ss, label="True Sim Phase")
ax[0].plot(h/1e3, a_os, label="Corrected Sim Phase")
ax[0].plot(h/1e3, ana_phs, label="Analytic Phase")
ax[0].legend()
ax[1].plot(h/1e3, a_ss - ana_phs, label="True Sim Phase - Analytic Phase")
ax[1].plot(h/1e3, a_os - ana_phs, label="Corrected Sim Phase - Analytic Phase")
ax[1].legend()
plt.show()
    
P_num = np.max(np.abs(rdrgrm)**2, axis=0)
print(P_num)

#err_mean = np.mean(P_num / P_r_layer)
err_mean = np.mean(P_num / P_r)
print(err_mean)

error = np.abs(P_num - P_r) / P_r * 100
print(np.mean(error)/100)

fig, ax = plt.subplots(2, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

ax[0].plot(h/1e3, uc.lin_to_db(P_num), label=f"Numerical: d={d} m", color=c, linewidth=1)
ax[0].plot(h/1e3, uc.lin_to_db(P_r), color=cont, linestyle="--", label=f"Analytic: d={d} m", linewidth=1)
#ax[0].plot(h/1e3, uc.lin_to_db(P_r_layer), color=cont, linestyle="-.", label=f"Layer: d={d} m", linewidth=1)

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
