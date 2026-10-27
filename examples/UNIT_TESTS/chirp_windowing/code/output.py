import pickle
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
WIN_TYPES = ("None", "Blackman", "Nuttall")

SIM_COLORS = {
    "None": "red",
    "Blackman": "blue",
    "Nuttall": "green",
}

ANALYTIC_COLORS = {
    "None": "maroon",
    "Blackman": "cyan",
    "Nuttall": "lime",
}

EPS = 1e-12  # Prevent log10(0)

# ------------------------------------------------------------
# Window definitions
# ------------------------------------------------------------
# Each window is specified by its cosine-sum coefficients:
#
# w(r) = sum_{m=0}^{K} a[m] * cos(m * pi * r),  0 <= r <= 1
#
# Here r = abs(f) / (bandwidth / 2), so:
#   r = 0: center of spectrum (DC)
#   r = 1: bandwidth edge
#
# Coefficients are chosen so that w(0) ~= 1 and w(1) ~= 0.
#
# "None" is handled separately: rectangular passband.

WINDOW_COEFFICIENTS = {
    "Blackman": np.array([0.42, 0.50, 0.08]),
    "Nuttall": np.array([
        0.355768,
        0.487396,
        0.144232,
        0.012604,
    ]),
}


def cosine_sum_window(r, coefficients):
    """
    Evaluate a generalized cosine-sum window over normalized
    frequency distance r in [0, 1].

    w(r) = sum_m coefficients[m] * cos(m * pi * r)

    The window is zero outside [0, 1].
    """
    r = np.asarray(r)
    window = np.zeros_like(r, dtype=float)
    inside = (r >= 0.0) & (r <= 1.0)

    r_inside = r[inside]

    for m, coefficient in enumerate(coefficients):
        window[inside] += (
            coefficient * np.cos(m * np.pi * r_inside)
        )

    return window


def make_frequency_window(freqs, bandwidth, win_type):
    """
    Construct a symmetric frequency-domain window centered at DC.
    The passband extends from -bandwidth/2 to +bandwidth/2.
    """
    half_width_hz = bandwidth / 2.0
    abs_freqs = np.abs(freqs)

    # Initialize to zero outside the passband.
    window = np.zeros_like(freqs, dtype=float)

    if win_type == "None":
        window[abs_freqs <= half_width_hz] = 1.0
        return window

    if win_type not in WINDOW_COEFFICIENTS:
        raise ValueError(f"Unknown window type: {win_type}")

    # Normalized distance from DC to the bandwidth edge.
    r = abs_freqs / half_width_hz

    return cosine_sum_window(
        r,
        WINDOW_COEFFICIENTS[win_type],
    )


# ------------------------------------------------------------
# Load parameters
# ------------------------------------------------------------
with open("inputs/none.pkl", "rb") as file:
    params = pickle.load(file)

c = 299792458.0
bandwidth = params["bandwidth"]
sample_rate = params["rx_sample_rate"]
dr = c / sample_rate
range_res = c / bandwidth

# Range axis, centered on the middle of the window.
range_axis = np.arange(0, params["rx_window_m"], dr)[1:]
range_axis -= np.mean(range_axis)
N = len(range_axis)

# Frequency axis in FFT order and centered order.
freqs = np.fft.fftfreq(N, d=1.0 / sample_rate)
freqs_shifted = np.fft.fftshift(freqs)

half_width_hz = bandwidth / 2.0

# ------------------------------------------------------------
# Analytic unwindowed time-domain chirp
# ------------------------------------------------------------
analytic_none = np.sinc(range_axis / range_res)

# FFT of the unwindowed sinc.
analytic_none_fft = np.fft.fft(analytic_none)

# ------------------------------------------------------------
# Construct analytic signals for all window types
# ------------------------------------------------------------
frequency_windows = {}
analytic_signals = {}
analytic_spectra_unshifted = {}
analytic_spectra = {}

for win_type in WIN_TYPES:

    # Construct the frequency-domain window.
    frequency_windows[win_type] = make_frequency_window(
        freqs,
        bandwidth,
        win_type,
    )

    # Apply the window to the unwindowed sinc spectrum.
    analytic_spectra_unshifted[win_type] = (
        analytic_none_fft * frequency_windows[win_type]
    )

    # Inverse FFT to get the corresponding time-domain signal.
    analytic_signals[win_type] = np.fft.ifft(
        analytic_spectra_unshifted[win_type]
    )

    # Center the frequency spectrum for plotting.
    analytic_spectra[win_type] = np.fft.fftshift(
        analytic_spectra_unshifted[win_type]
    )

# ------------------------------------------------------------
# Load simulated chirps
# ------------------------------------------------------------
sim_signals = {}
sim_spectra = {}

for win_type in WIN_TYPES:
    data = np.loadtxt(f"outputs/{win_type}/s000000.txt")
    sig = data[:, 0] + 1j * data[:, 1]

    if len(sig) != N:
        raise ValueError(
            f"Length mismatch for {win_type}: "
            f"simulation has {len(sig)} samples, "
            f"analytic model has {N} samples."
        )

    sim_signals[win_type] = sig
    sim_spectra[win_type] = np.fft.fftshift(np.fft.fft(sig))

# ------------------------------------------------------------
# Normalize analytic signals to simulated peak amplitudes
# ------------------------------------------------------------
# Time-domain normalization.
analytic_time = {}

for win_type in WIN_TYPES:
    analytic = analytic_signals[win_type].copy()

    sim_peak = np.max(np.abs(sim_signals[win_type]))
    analytic_peak = np.max(np.abs(analytic))

    if analytic_peak > 0.0:
        analytic *= sim_peak / analytic_peak

    analytic_time[win_type] = analytic

# Frequency-domain normalization.
analytic_spectra_normalized = {}

for win_type in WIN_TYPES:
    analytic = analytic_spectra[win_type].copy()

    sim_peak = np.max(np.abs(sim_spectra[win_type]))
    analytic_peak = np.max(np.abs(analytic))

    if analytic_peak > 0.0:
        analytic *= sim_peak / analytic_peak

    analytic_spectra_normalized[win_type] = analytic

# ------------------------------------------------------------
# Convert to dB
# ------------------------------------------------------------
def to_db(signal):
    return 20.0 * np.log10(np.maximum(np.abs(signal), EPS))


# ------------------------------------------------------------
# Plot time-domain and frequency-domain results
# ------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Time domain.
for win_type in WIN_TYPES:
    ax[0].plot(
        range_axis,
        to_db(sim_signals[win_type]),
        color=SIM_COLORS[win_type],
        linewidth=1.2,
        label=f"Simulated {win_type}",
    )

    ax[0].plot(
        range_axis,
        to_db(analytic_time[win_type]),
        color=ANALYTIC_COLORS[win_type],
        linestyle="--",
        linewidth=1.2,
        label=f"Analytic {win_type}",
    )

ax[0].set_xlabel("Range (m)")
ax[0].set_ylabel("Amplitude (dB)")
ax[0].set_title("Range-Compressed Chirp")
ax[0].grid(True, alpha=0.3)
ax[0].legend()

# Frequency domain.
for win_type in WIN_TYPES:
    ax[1].plot(
        freqs_shifted / 1e6,
        to_db(sim_spectra[win_type]),
        color=SIM_COLORS[win_type],
        linewidth=1.2,
        label=f"Simulated {win_type}",
    )

    ax[1].plot(
        freqs_shifted / 1e6,
        to_db(analytic_spectra_normalized[win_type]),
        color=ANALYTIC_COLORS[win_type],
        linestyle="--",
        linewidth=1.2,
        label=f"Analytic {win_type}",
    )

ax[1].axvline(
    -half_width_hz / 1e6,
    color="gray",
    linestyle=":",
    label="Window edges",
)
ax[1].axvline(
    half_width_hz / 1e6,
    color="gray",
    linestyle=":",
)

ax[1].set_xlabel("Frequency (MHz)")
ax[1].set_ylabel("Magnitude (dB)")
ax[1].set_title("Frequency Spectrum")
ax[1].grid(True, alpha=0.3)
ax[1].legend()
ax[1].set_ylim(-150, -50)

plt.tight_layout()
plt.savefig("figures/windows.png")
plt.show()
