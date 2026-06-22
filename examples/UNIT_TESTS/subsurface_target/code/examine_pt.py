import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

def load_sig(f):
    sig = np.loadtxt(f)
    out = sig[:,0] + 1j * sig[:,1]
    return out

def sinc(x):
    mask = x != 0
    output = np.ones_like(x)
    output[mask] = np.sin(np.pi * x[mask]) / (np.pi * x[mask])
    return output

def chirp(cen, offset, rng_res):
    return sinc((cen + offset) / rng_res)

def make_half_chirp(chirp_f, L=None):
    if L is None:
        L = len(chirp_f)

    N = len(chirp_f)
    freqs = np.fft.fftfreq(N, d=dt)

    C = np.fft.fft(chirp_f, L)
    Cwin = C[np.abs(freqs) <= B/2]
    Cmean = np.mean(Cwin)

    Cideal = np.zeros(N)
    Cideal[np.abs(freqs) <= B/2] = 1 #np.sqrt(Cmean)

    h = np.fft.ifft(Cideal)

    return h

def gauss(x, cen, width):
    a = 1 / (width * np.sqrt(2 * np.pi))
    expval = -0.5 * ((x - cen)**2 / width**2)
    return a * np.exp(expval)

f_0 = 60e6
B   = 10e6
c   = 299792458
smp = 80e6

dt      = 1 / smp
rng_res = c / B

smpls = np.arange(250) * dt * c
offset = np.mean(smpls) - smpls

chirp_f = chirp(0, offset, rng_res)
#chirp_f = chirp(0, smpls, rng_res)
half_chirp = make_half_chirp(chirp_f)

# load phasor trace @ target
TARID = 0
DEPTH = 5000

def fft_convolve(x, h):
    N = len(x)
    M = len(h)
    L = N + M - 1

    X = np.fft.fft(x, L)
    H = np.fft.fft(h, L)
    #H = np.sqrt(H.astype(np.complex128))

    # remove stuff outside 2x bandwidth
    freqs = np.fft.fftfreq(L, d=1/smp)

    X = np.fft.fft(x, L)

    X[np.abs(freqs) > B] = 0
    
    Y = X * H
    y_full = np.fft.ifft(Y)

    # Crop to same length as x (same behavior as linear convolution)
    start = (M - 1) // 2
    return y_full[start:start + N]

def fft_deconvolve(y, h, eps=1e-8):
    N = len(y)
    M = len(h)
    L = N + M - 1

    Y = np.fft.fft(y, L)
    H = np.fft.fft(h, L)

    X_est = Y / (H + eps)   # stable inverse
    x_full = np.fft.ifft(X_est).real

    start = (M - 1) // 2
    return x_full[start:start + N]

def plot_fft(signal, axis, title, fs, color="black"):
    N = len(signal)
    fft_vals = np.fft.fftshift(np.fft.fft(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))

    intf = np.sum(np.abs(fft_vals))

    axis.plot(freqs/1e6, np.abs(fft_vals), color=color)
    axis.set_title(title)
    axis.set_xlabel("Frequency (MHz)")
    axis.set_ylabel("|FFT|")

    axis.set_xlim(-1*fs/2e6, fs/2e6)

def plot_components(signal, axis, linestyle=None):

    axis.plot(np.real(signal), color="red", linewidth=1, linestyle=linestyle)
    axis.plot(np.imag(signal), color="blue", linewidth=1, linestyle=linestyle)

reconstructed = np.convolve(
    half_chirp,
    half_chirp,
    mode="same"
)
"""
fig, ax = plt.subplots(1,2)
ax[0].plot(chirp_f, label="chirp")
ax[0].plot(reconstructed, '--', label="h*h")
plot_fft(chirp_f, ax[1], "", smp, color="blue")
plot_fft(reconstructed, ax[1], "", smp, color="orange")
plt.legend()
plt.show()
"""

for TRCID in np.arange(0, 500, 100):

    ptTarg = load_sig(f"rdrgrm/{DEPTH:04d}/Ptarg_s{TRCID:06d}_t{TARID:02d}.txt")
    ptSour = load_sig(f"rdrgrm/{DEPTH:04d}/Psour_s{TRCID:06d}_t{TARID:02d}.txt")
    ptTmp  = load_sig(f"rdrgrm/{DEPTH:04d}/PTTmp_s{TRCID:06d}_t{TARID:02d}.txt")

    signal = load_sig(f"rdrgrm/{DEPTH:04d}/s{TRCID:06d}.txt")
    
    fig, ax = plt.subplots(4, 3, figsize=(12, 8))

    #inConv = np.convolve(ptTarg, chirp_f / 2, mode="same")
    #inConv = np.convolve(ptTarg, half_chirp, mode="same")
    #inConv = fft_convolve(ptTarg, chirp_f)
    #inConv = fft_convolve(ptTarg, half_chirp)
    inConv = fft_convolve(ptTarg, chirp_f)
    ax[0,0].plot(np.abs(inConv), color="red")
    ax[0,0].plot(np.abs(ptTarg), color="black")
    ax[0,0].set_title("Phasor trace @ target (time)")
    plot_components(inConv, ax[0,2], linestyle="--")
    plot_components(ptTarg, ax[0,2], linestyle=None)

    #outConv = np.convolve(ptSour, chirp_f / 2, mode="same")
    #outConv = np.convolve(ptSour, half_chirp, mode="same")
    #outConv = fft_convolve(ptSour, chirp_f)
    #outConv = fft_convolve(ptSour, half_chirp)
    outConv = ptSour
    ax[1,0].plot(np.abs(outConv), color="red")
    ax[1,0].plot(np.abs(ptSour), color="black")
    ax[1,0].set_title("Phasor trace @ source (time)")
    plot_components(outConv, ax[1,2], linestyle="--")
    plot_components(ptSour, ax[1,2], linestyle=None)

    fConv = np.convolve(inConv, outConv, mode="same")
    ax[2,0].plot(np.abs(fConv), color="red")
    ax[2,0].plot(np.abs(ptTmp), color="black")
    ax[2,0].set_title("2‑way phasor trace (time)")
    plot_components(fConv, ax[2,2], linestyle="--")
    plot_components(ptTmp, ax[2,2], linestyle=None)

    #final = np.convolve(fConv, chirp_f, mode="same")
    ax[3,0].plot(np.abs(fConv), color="red")
    ax[3,0].plot(np.abs(signal), color="black")
    ax[3,0].set_title("Final signal")
    plot_components(fConv, ax[3,2], linestyle="--")
    plot_components(signal, ax[3,2], linestyle=None)

    plot_fft(ptTarg, ax[0,1], "Phasor trace @ target", smp, color="black")
    plot_fft(ptSour, ax[1,1], "Phasor trace @ source", smp, color="black")
    plot_fft(ptTmp,  ax[2,1], "2-Way phasor trace", smp, color="black")
    plot_fft(signal, ax[3,1], "Final signal", smp, color="black")

    plot_fft(inConv, ax[0,1], "Target phasor (frequency)", smp, color="red")
    plot_fft(outConv, ax[1,1], "Source phasor (frequency)", smp, color="red")
    plot_fft(fConv, ax[2,1], "2‑way phasor (frequency)", smp, color="red")
    #plot_fft(final, ax[3,1], "Final signal (frequency)", smp, color="red")
    plot_fft(fConv, ax[3,1], "Final signal (frequency)", smp, color="red")

    plt.tight_layout()
    plt.show()

    print(f"Maximum amplitude of simulator output is: {np.max(np.abs(signal))}")
    print(f"Maximum ampltiude of new method is: {np.max(np.abs(fConv))}")
    print(f"Maximum power of simulator output is: {np.max(np.abs(signal)**2)}")
    print(f"Maximum power of new method is: {np.max(np.abs(fConv)**2)}")

    import sys
    sys.exit()
