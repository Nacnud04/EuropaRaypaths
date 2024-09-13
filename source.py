import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import chirp

class Source():

    # define source
    def __init__(self, dt, dur, coord):

        self.dt = dt
        self.sr = 1/dt # sampling rate
        self.n = int(dur/dt) # num of samples
        self.dur = dur
        
        self.c = 299792458 # speed of light

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
    def chirp(self, freq, bandwidth):

        t = np.linspace(0, self.dur, int(self.sr * self.dur))
        signal = np.exp(1j * 2 * np.pi * (freq * t + (bandwidth / (2 * self.dur)) * t**2))
              
        self.t = np.linspace(-1.5*self.dur, 1.5*self.dur, int(self.sr * 3* self.dur))
        zeros = np.zeros(len(signal))
        self.signal = np.correlate(np.hstack((zeros, signal, zeros)), signal, mode="same")
        self.f0 = freq
        self.wc = freq * 2 * np.pi
        self.lam = self.c / freq
        
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
    def plot(self):

        N = len(self.signal)
        T = self.t[1] - self.t[0]
        yf = fft(self.signal)
        xf = fftfreq(N, T)[:N//2]

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