import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

def load_sig(f):
    sig = np.loadtxt(f)
    out = sig[:,0] + 1j * sig[:,1]
    return out

def load_floats(f):
    sig = np.loadtxt(f)
    return sig

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
smp = 60e6

dt      = 1 / smp
rng_res = c / B

smpls = np.arange(250) * dt * c
offset = np.mean(smpls) - smpls

chirp_f = chirp(0, offset, rng_res * 2)
#chirp_f = chirp(0, smpls, rng_res)
half_chirp = make_half_chirp(chirp_f)

# load phasor trace @ target
TARID = 0
DEPTH = 5000

def fft_convolve(x, h, full=False):
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

    if full == False:

        # Crop to same length as x (same behavior as linear convolution)
        start = (M - 1) // 2
        return y_full[start:start + N]
    
    return y_full

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

    axis.set_xlim(-1*fs/2e6, fs/2e6)

def plot_components(signal, axis, linestyle=None):

    axis.plot(np.real(signal), color="red", linewidth=1, linestyle=linestyle)
    axis.plot(np.imag(signal), color="blue", linewidth=1, linestyle=linestyle)

for TRCID in np.arange(0, 500, 100):

    ptTarg = load_sig(f"rdrgrm/{DEPTH:04d}/Ptarg_s{TRCID:06d}_t{TARID:02d}.txt")
    ptTargSAVE = ptTarg#load_sig(f"tmp/Ptarg_s{TRCID:06d}_t{TARID:02d}.txt")
    ptSour = load_sig(f"rdrgrm/{DEPTH:04d}/Psour_s{TRCID:06d}_t{TARID:02d}.txt")
    ptTmp  = load_sig(f"rdrgrm/{DEPTH:04d}/PTTmp_s{TRCID:06d}_t{TARID:02d}.txt")

    sPad = load_sig(f"rdrgrm/{DEPTH:04d}/sPad_s{TRCID:06d}_t{TARID:02d}.txt")
    kPad = load_sig(f"rdrgrm/{DEPTH:04d}/kPad_s{TRCID:06d}_t{TARID:02d}.txt")
    mPad = load_sig(f"rdrgrm/{DEPTH:04d}/mPad_s{TRCID:06d}_t{TARID:02d}.txt")

    signal = load_sig(f"rdrgrm/{DEPTH:04d}/s{TRCID:06d}.txt")
    
    fig, ax = plt.subplots(4, 3, figsize=(12, 8), sharex='col')

    # force everything into scientific notation
    for a in ax.flatten(): 
        a.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

    inConv = fft_convolve(ptTargSAVE, chirp_f, full=False)
    #inConv = ptTarg
    ax[0,0].plot(np.abs(inConv), color="red")
    ax[0,0].plot(np.abs(ptTarg), color="black")
    for r in (0, 2): ax[0,r].set_xlim(0, len(inConv))
    plot_components(inConv, ax[0,2], linestyle="--")
    plot_components(ptTarg, ax[0,2], linestyle=None)

    outConv = ptSour
    ax[1,0].plot(np.abs(outConv), color="red")
    ax[1,0].plot(np.abs(ptSour), color="black")
    plot_components(outConv, ax[1,2], linestyle="--")
    plot_components(ptSour, ax[1,2], linestyle=None)

    fConv = fft_convolve(inConv, outConv, full=True)[::2]
    #fConv = np.convolve(inConv, outConv, mode="full")[::2]
    ax[2,0].plot(np.abs(fConv), color="red")
    ax[2,0].plot(np.abs(ptTmp), color="black")
    plot_components(fConv, ax[2,2], linestyle="--")
    plot_components(ptTmp, ax[2,2], linestyle=None)

    ax[3,0].plot(np.abs(fConv), color="red")
    ax[3,0].plot(np.abs(signal), color="black")

    plot_components(fConv, ax[3,2], linestyle="--")
    plot_components(signal, ax[3,2], linestyle=None)

    plot_fft(ptTarg, ax[0,1], "", smp, color="black")
    plot_fft(ptSour, ax[1,1], "", smp, color="black")
    plot_fft(ptTmp,  ax[2,1], "", smp, color="black")
    plot_fft(signal, ax[3,1], "", smp, color="black")

    plot_fft(inConv, ax[0,1], "", smp, color="red")
    plot_fft(outConv, ax[1,1], "", smp, color="red")
    plot_fft(fConv, ax[2,1], "", smp, color="red")
    plot_fft(fConv, ax[3,1], "", smp, color="red")

    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal)*2-1, d=1/smp))

    #ax[1,1].plot(freqs/1e6, np.fft.fftshift(np.abs(sPad)), color="blue", linewidth=1)
    #ax[0,1].plot(freqs/1e6, np.fft.fftshift(np.abs(kPad)), color="blue", linewidth=1)
    #ax[2,1].plot(freqs/1e6, np.fft.fftshift(np.abs(mPad)), color="blue", linewidth=1)

    ax[0,1].plot(freqs/1e6, np.fft.fftshift(np.abs(sPad[:-1])), color="cyan", linewidth=1)
    ax[0,1].plot(freqs/1e6, np.fft.fftshift(np.abs(kPad[:-1])), color="magenta", linewidth=1)
    ax[0,1].plot(freqs/1e6, np.fft.fftshift(np.abs(mPad[:-1])), color="lime", linewidth=1)

    for r in range(0, 4):
        ax[r,1].axvline(-B/2e6, color="grey", alpha=0.7)
        ax[r,1].axvline(B/2e6, color="grey", alpha=0.7)

    # column headers
    headers = ("Power", "Frequency", "Components")
    for c, h in enumerate(headers):
        ax[0,c].set_title(h, fontsize=14, fontweight="bold")

    # row headers
    headers = ("Inward Trc", "Outward PT", "Joint", "Final Signal")
    ys      = [np.max(np.abs(ptTarg))/2, np.max(np.abs(ptSour))/2, np.max(np.abs(ptTmp))/2, np.max(np.abs(signal))/2]
    xpos    = -0.2 * len(fConv)
    for r, (h, y) in enumerate(zip(headers, ys)):
        ax[r,0].text(xpos, y, s=h, fontsize=14, fontweight="bold", rotation=90, verticalalignment="center")

    # x labels
    labels = ("Sample #", "Freq [MHz]", "Sample #")
    for c, l in enumerate(labels):
        ax[3,c].set_xlabel(l)

    print(f"Maximum amplitude of simulator output is: {np.max(np.abs(signal))}")
    print(f"Maximum ampltiude of new method is: {np.max(np.abs(fConv))}")
    print(f"Maximum power of simulator output is: {np.max(np.abs(signal)**2)}")
    print(f"Maximum power of new method is: {np.max(np.abs(fConv)**2)}")

    max_old = np.max(np.abs(signal))
    max_new = np.max(np.abs(fConv))
    trueV   = 2.5308183001e-9

    plt.suptitle(f"OLD: {max_old:.4E}      NEW: {max_new:.4E}      TRUE: {trueV:.4E}",
                 fontweight="bold", fontsize=14)

    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.savefig("figures/ConvDebug.png")
    plt.show()
    #plt.close()

    import sys
    sys.exit()
