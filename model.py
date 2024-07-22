import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from math import sqrt

from source import *
from surface import *
from raypath import *

class Model():

    def __init__(self, surface, source):
        
        self.c = 299792458 # speed of light

        # set surface and source objects
        self.surface = surface
        self.source = source

        # define vars for target location
        self.tx, self.ty, self.tz = None, None, None
        
        # material dielectrics
        self.eps1 = 1
        self.eps2 = 10
        
        # turn dielectrics into velocities
        self.c1 = self.c / sqrt(self.eps1)
        self.c2 = self.c / sqrt(self.eps2)

    # set target location
    def set_target(self, coord):
        self.tx, self.ty, self.tz = coord
        
    # find angle between two 3D vectors
    @staticmethod
    def vec_dif_angle(u, v):

        dot_product = np.dot(u, v)

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        cos_theta = dot_product / (norm_u * norm_v)

        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  

        return angle
    
    # find angle of a vector relative to a plane
    @staticmethod
    def vec_dif_plane(vector, normal_vector):
        
        dot_product = np.dot(normal_vector, vector)
    
        norm_n = np.linalg.norm(normal_vector)
        norm_v = np.linalg.norm(vector)

        cos_phi = dot_product / (norm_n * norm_v)

        cos_phi = np.clip(cos_phi, -1.0, 1.0)

        phi = np.arccos(cos_phi)

        theta = np.pi / 2 - phi

        theta_degrees = np.degrees(theta)

        return theta_degrees
    
    # beam pattern for square facets
    @staticmethod
    def beam_pattern(theta, lam, r):
        p = (np.sin((np.pi*r/lam)*np.sin(theta)) / ((np.pi*r/lam)*np.sin(theta)))**2
        return p
    
    # limit angles to being from -90 to 90
    @staticmethod
    def ang_lim(a):
        if a > np.pi / 2: a = a - np.pi
        if a < -np.pi / 2: a = a + np.pi
        return a
    
    # plot beam pattern
    def show_beam_pattern(self):
        
        x = np.linspace(-3, 3, 250)
        ptrn = Model.beam_pattern(x, self.lam, self.surface.fs)
        ptrn = 20 * np.log10(ptrn)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ptrn, mode='lines'))
        fig.update_layout(title="Reradiation Pattern", xaxis_title='Angle (rad)', yaxis_title='dB', template="plotly_white")
        fig.update_yaxes(range=[-60, 0])
        fig.show()

    # create rays from source to target. 
    # calculate frac of transmit power.
    def gen_raypaths(self):

        self.raypaths = []
        
        self.lam = self.c / self.source.f0
        
        for i, x in enumerate(self.surface.x):
            for j, y in enumerate(self.surface.y):
                
                fnorm = self.surface.normals[i, j]
                
                # create rays
                rp = RayPaths(self.source.coord, 
                             [x, y, self.surface.zs[i, j]],
                             [self.tx, self.ty, self.tz])
                
                # compute refracted ray angle'
                rp.set_source(self.source)
                rp.set_facet(fnorm, self.surface.fs)
                th1 = rp.comp_refracted(self.c1, self.c2)
                rp.th1 = th1
                    
                # compute forced ray angle
                th2 = Model.vec_dif_angle(rp.norms[1], fnorm)
                th2 = Model.ang_lim(th2)
                rp.th2 = th2
                   
                # compute trasmitted power based on difference in refracted and forced ray angle
                if rp.th1:
                    rp.tr *= Model.beam_pattern(th2 - th1, self.lam, self.surface.fs)
                else:
                    rp.tr = 0
                        
                self.raypaths.append(rp)
                
    # use raypaths to generate a timeseries
    # show output timeseries as well as frequecy spec
    # received by the radar
    def gen_timeseries(self, n=250):
        
        gain = 49.5 # gain in dB
        gval = 10**(gain / 20)
        
        times = [rp.path_time for rp in self.raypaths]
        
        # compute index offsets
        rel_times = np.array(times) - min(times)
        idx_offsets = np.round(rel_times / self.source.dt).astype(int)
        
        # create an empty output array
        output = np.zeros(np.max(idx_offsets) + len(self.source.signal))
        
        # iterate through raypaths
        i = 0
        for rp, offset in zip(self.raypaths, idx_offsets):
            # add 0 padding to account for different ray arrival time to front
            # of the signal matrix
            tmp = np.concatenate((np.zeros(offset), self.source.signal))
            # add 0 padding to the back, to make it the same length as the output array
            tmp = np.concatenate((tmp, np.zeros(len(output) - len(tmp))))
            # sum the new output signal with the output. Taking into account the amount
            # of energy transferred in that direction, as well as r^2
            output += (rp.tr * tmp * gval) / sum(rp.mags) ** 2
            i += 1
            
        ts = np.linspace(0, len(output)*self.source.dt, num=len(output))
        ts += min(times)
        
        self.ts = ts
        self.data = output
                    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=output, mode='lines'))
        fig.update_layout(title="Final Signal", xaxis_title='Time (s)', yaxis_title='Signal', template="plotly_white")
        fig.show()
        
        N = len(output)
        T = ts[1] - ts[0]
        yf = fft(output)
        xf = fftfreq(N, T)[:N//2]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xf, y=2.0/N * np.abs(yf[:N//2]), mode='lines', name='Spectrum'))
        fig.update_layout(title="Output Spectrum", template="plotly_white")
        fig.update_xaxes(title_text='Frequency (Hz)', type='log')
        fig.update_yaxes(title_text="Amplitude")
        fig.show()
    
    # plot angle difference between refracted ray
    # and forced ray to target
    def plot_s2f_angle(self):
        
        angles = np.zeros(self.surface.zs.shape)
        for i, x in enumerate(self.surface.x):
            for j, y in enumerate(self.surface.y):
                rp = self.raypaths[i*len(self.surface.x)+j]
                if rp.th2 and rp.th1:
                    angles[i, j] = rp.th2 - rp.th1
                else:
                    angles[i, j] = None
                
        fig = go.Figure(data=go.Heatmap(z=angles))

        fig.update_layout(
            title='Angle between refracted ray and ray to target from facet',
            xaxis_title='Facet X #',
            yaxis_title='Facet Y #', template="plotly_white"
        )

        fig.show()
        
    # plot fraction of radiated power which returns to
    # source after reflecting off target by facet
    def plot_s2f_tr(self):
        
        angles = np.zeros(self.surface.zs.shape)
        for i, x in enumerate(self.surface.x):
            for j, y in enumerate(self.surface.y):
                rp = self.raypaths[i*len(self.surface.x)+j]
                angles[i, j] = rp.tr
                
        fig = go.Figure(data=go.Heatmap(z=angles))

        fig.update_layout(
            title='Reradiation',
            xaxis_title='Facet X #',
            yaxis_title='Facet Y #', template="plotly_white"
        )

        fig.show()

    # plot 3d model showing everything
    def plot(self):

        fig = go.Figure()
        
        fig = self.surface.add_to_axis(fig)
        
        fig = self.source.add_to_axis(fig, self.surface)
        
        for rp in self.sraypaths:
            fig = rp.add_to_axis(fig)
        
        if len(self.traypaths) != 0:
            for rp in self.traypaths:
                fig = rp.add_to_axis(fig)
        
        fig.show()
