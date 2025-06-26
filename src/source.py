import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import chirp
import matplotlib.pyplot as plt

def sinc(x):
    return np.sin(np.pi * x)/(np.pi*x)

def source_linspace(axis, axmin, axmax, cval, alt, N, f0, B, dt=None, dur=0.5e-6):
    
    """
    Generate a linspace of sources. Uses a chirp. 

    axis: axis to linspace over (x or y)
    axmin: start of linspace
    axmax: end of linspace
    cval: constant value for coordinate of other axis (x or y)
    alt: spacecraft altitude
    N: how many sources in linspace?
    f0: center frequency of chirp
    B: bandwidth of chirp
    dt: time discretization, defaults to 1/8 of wavelength
    dur: duration of sample chirp (unused for actual radar image construction). defaults to 500 ns
    """

    if dt == None:
        dt = 1 / (8 * f0)

    ss = []
    for c in np.linspace(axmin, axmax, N):
        if axis == 'x':
            coord = (c, cval, alt)
        elif axis == 'y':
            coord = (cval, c, alt)
        else:
            raise TypeError(f"Only axis x and y are supported. Received axis: {axis}")
        source = Source(dt, dur, coord)
        source.chirp(f0, B)
        ss.append(source)

    return ss

class Source():

    # define source
    def __init__(self, dt, dur, coord):

        self.dt = dt
        self.sr = 1/dt # sampling rate
        self.n = int(dur/dt) # num of samples
        self.dur = dur
        
        self.c = 299792458 # speed of light

        self.rr = self.dt * self.c # range resolution

        self.x, self.y, self.z = coord # location of source
        
        self.speed = 3500 # speed of instrument in m/s
        self.u = np.array((1, 0, 0)) * self.speed # velocity vector
        
        self.coord = np.array(coord)

    # generate a gaussian sin function source
    def gauss_sin(self, f0, offset=0):

        t = np.linspace(0, self.dur, int(self.sr * self.dur))
        self.signal = self.power * np.exp(-(t-offset)**2 / (2 * (self.dur / 6)**2)) * np.cos(2 * np.pi * f0 * (t-offset))

        self.t = t
        self.f0 = f0
        self.lam = self.c / f0

        return self.t, self.signal

    # generate a ricker wavelet as the source
    def ricker(self, f0, offset=0):
        
        t = np.linspace(0, self.dur, int(self.sr * self.dur))
    
        a = 2 * (np.pi * f0)**2
        self.signal = (1 - a * (t-offset)**2) * np.exp(-a * (t-offset)**2 / 2)

        self.t = t
        self.f0 = f0
        self.lam = self.c / f0

        return self.t, self.signal
    
    # generate a chirp (which is range compressed)
    def chirp(self, freq, bandwidth, correlate=False):

        self.f0 = freq
        self.wc = freq * 2 * np.pi
        self.lam = self.c / freq
        rho_0 = 0
        
        if correlate:
            
            t = np.arange(0, self.dur, self.dt)
            signal = np.exp(1j * 2 * np.pi * (freq * t + (bandwidth / (2 * self.dur)) * t**2))
                  
            self.t = np.arange(-1.5*self.dur, 1.5*self.dur, self.dt)
            zeros = np.zeros(len(signal))
            self.signal = np.correlate(np.hstack((zeros, signal, zeros)), signal, mode="same")

        else:
            # time axis
            self.t = np.arange(-1.5*self.dur, 1.5*self.dur, self.dt)

            # space axis
            #rax = self.t * self.c
            #self.signal = sinc((self.f0 * (1/self.c)) * (rax-rho_0)/self.rr) * np.exp(2j * self.k * rho_0)

            self.signal = sinc(2 * self.f0 * self.t)
            
        
        return self.t, self.signal
    
    
    def conjugate(self):
        
        chirp_fft = (np.fft.fft(self.signal, 4096)) / np.sqrt(4096)
        chirp_fft = chirp_fft*self.range_weighting_window
        chirp_fft_conj = np.conj(chirp_fft).T
        
        return chirp_fft_conj
        

    # add the source location to a 3d plot of the model
    def add_to_axis(self, fig, surf):

        scatter = go.Scatter3d(x=[self.x], y=[self.y], z=[self.z], mode='markers')
        
        line = go.Scatter3d(x=[self.x, self.x], y=[self.y, self.y], 
                            z=[self.z, np.min(surf.zs)], mode='lines',
                            line=dict(color='red', dash='dash'))
        
        fig.add_trace(scatter)
        fig.update_xaxes(range=[0, 3e-6])
        fig.add_trace(line)
        
        return fig

    # plot the source and its magnitude spectra
    def plot(self, plotly=False):

        N = len(self.signal)
        T = self.t[1] - self.t[0]
        yf = fft(self.signal)[:N//2]
        xf = fftfreq(N, T)[:N//2]

        if plotly:
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Time Domain", "Frequency Spectrum"), horizontal_spacing=0.15)
    
            fig.add_trace(go.Scatter(x=self.t, y=np.real(self.signal), mode='lines', name='Real'), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.t, y=np.imag(self.signal), mode='lines', name='Imaginary'), row=1, col=1)
    
            fig.add_trace(go.Scatter(x=xf, y=2.0/N * np.abs(yf[:N//2]), mode='lines', name='Spectrum'), row=1, col=2)
    
            fig.update_layout(title="Source wavelet", showlegend=False, template="plotly_white")
            
            fig.update_xaxes(title_text='Time (s)', row=1, col=1)
            fig.update_yaxes(title_text='Signal', row=1, col=1)
            fig.update_xaxes(title_text='Frequency (Hz)', type='log', row=1, col=2)
            fig.update_yaxes(title_text='Amplitude', row=1, col=2)
    
            fig.show()

        else:

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].plot(self.t, np.real(self.signal), color="blue", label="Real")
            ax[0].plot(self.t, np.imag(self.signal), color="red", label="Imag")
            ax[1].plot(xf, np.abs(yf))
            ax[1].axvline(self.f0, color="red", linestyle=":")
            plt.suptitle("Source wavelet")
            ax[0].set_xlabel("Time (s)")
            ax[0].set_ylabel("Signal")
            ax[1].set_xlabel("Frequency (Hz)")
            ax[1].set_ylabel("Amplitude")
            ax[1].set_xscale("log")
            ax[1].set_yscale("log")
            plt.show()