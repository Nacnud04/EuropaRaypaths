import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from math import sqrt

from source import *
from surface import *
from raypath import *
from util import *

class Model():

    def __init__(self, surface, source, power=11.75):
        
        self.c = 299792458 # speed of light

        # set surface and source objects
        self.surface = surface
        self.source = source

        # define vars for target location
        self.tx, self.ty, self.tz = None, None, None
        
        # material dielectrics
        self.eps1 = 1
        self.eps2 = 5
        
        # turn dielectrics into velocities
        self.c1 = self.c / sqrt(self.eps1)
        self.c2 = self.c / sqrt(self.eps2)
        
        self.power = power
        self.db = 80
        self.gain = db_to_mag(self.db)

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
    
    # 3d beam pattern for square facets
    @staticmethod
    def beam_pattern_3D(th, ph, lam, r, R):
        k = 2 * np.pi / lam
        c = ((1j * r**2)/lam) * (np.exp(-1j*k*R)/R) * k 
        p = c * np.sinc(((r) / lam) * np.sin(ph) * np.cos(th))
        p *= np.sinc(((r) / lam) * np.sin(ph) * np.sin(th))
        return p
    
    # limit angles to being from -90 to 90
    @staticmethod
    def ang_lim(a):
        if a > np.pi / 2: a = a - np.pi
        if a < -np.pi / 2: a = a + np.pi
        return a
    
    # plot beam pattern
    def show_beam_pattern(self, dim=3):
        
        if dim == 2:
            x = np.linspace(-3, 3, 250)
            ptrn = Model.beam_pattern(x, self.lam, self.surface.fs)
            ptrn = 20 * np.log10(ptrn)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=ptrn, mode='lines'))
            fig.update_layout(title="Reradiation Pattern", xaxis_title='Angle (rad)', yaxis_title='dB', template="plotly_white")
            fig.update_yaxes(range=[-60, 0])
            fig.show()
            
        elif dim == 3:
            th = np.linspace(0, 2 * np.pi, 360)
            ph = np.linspace(0, np.pi, 240)
            phGrid, thGrid = np.meshgrid(th, ph)
            r = Model.beam_pattern_3D(thGrid, phGrid, self.lam, self.surface.fs, 1)
            
            R_abs = np.abs(r)
            R_phase = np.angle(r)

            # Convert into xyz coordinates
            x = R_abs * np.cos(thGrid) * np.sin(phGrid)  # x = r*cos(s)*sin(t)
            y = R_abs * np.sin(thGrid) * np.sin(phGrid)  # y = r*sin(s)*sin(t)
            z = R_abs * np.cos(phGrid)                  # z = r*cos(t)
            
            range_min = min(np.min(x), np.min(y), np.min(z))
            range_max = max(np.max(x), np.max(y), np.max(z))

            # Create subplots with two columns, both of type 'scene'
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Magnitude", "Phase"),
                                specs=[[{'type': 'scene'}, {'type': 'scene'}]])

            # Add the surface plot of the absolute value of R
            surface_abs = go.Surface(x=x, y=y, z=z, surfacecolor=R_abs, colorscale='Viridis', coloraxis='coloraxis1')
            fig.add_trace(surface_abs, row=1, col=1)

            # Add the surface plot of the phase of R
            surface_phase = go.Surface(x=x, y=y, z=z, surfacecolor=R_phase, colorscale='Phase', coloraxis='coloraxis2')
            fig.add_trace(surface_phase, row=1, col=2)
            
            fig.update_layout(
                title="Absolute Value and Phase of R", 
                template="plotly_white",
                coloraxis1=dict(colorbar=dict(x=0.45, len=0.75, title="Mag")),
                coloraxis2=dict(colorbar=dict(x=1.05, len=0.75, title="Phase")),
                scene=dict(
                    aspectmode='cube',
                    xaxis=dict(range=[range_min, range_max]),
                    yaxis=dict(range=[range_min, range_max]),
                    zaxis=dict(range=[range_min, range_max])
                ),
                scene2=dict(
                    aspectmode='cube',
                    xaxis=dict(range=[range_min, range_max]),
                    yaxis=dict(range=[range_min, range_max]),
                    zaxis=dict(range=[range_min, range_max])
                )
                        )

            # Update layout and show plot
            fig.update_layout(title="Reradiation Pattern", template="plotly_white")
            fig.show()

    # create rays from source to target. 
    # calculate frac of transmit power.
    def gen_raypaths(self):

        self.raypaths = []
        
        self.lam = self.c / self.source.f0
        
        for i, x in enumerate(self.surface.x):
            for j, y in enumerate(self.surface.y):
                
                fnorm = self.surface.normals[i, j]
                spfnorm = cart_to_sp(fnorm)
                
                # create rays
                rp = RayPaths(self.source.coord, 
                             [x, y, self.surface.zs[i, j]],
                             [self.tx, self.ty, self.tz])
                
                # compute refracted ray angle'
                rp.set_source(self.source)
                rp.set_facet(fnorm, self.surface.fs)
                rp.refracted = rp.comp_refracted(self.c1, self.c2)
                
                if rp.refracted is not None:
                    
                    # compute forced ray angle
                    rp.forced = cart_to_sp(rp.norms[1]) - spfnorm

                    _, dth, dph = rp.forced - rp.refracted

                    # compute loss from propagating to the target
                    # compute trasmitted power based on difference in refracted and forced ray angle
                    rp.tr *= np.abs(Model.beam_pattern_3D(dth, dph, self.lam, self.surface.fs, rp.mags[1]))
                    
                    # compute loss from propagating back to the source
                    refracted_reverse = rp.comp_rev_refracted(self.c1, self.c2)
                    
                    if refracted_reverse is not None:
                        spfnorm_reverse = cart_to_sp(fnorm * -1)
                        forced_reverse = cart_to_sp(rp.norms[0] * -1) - spfnorm_reverse

                        # find difference in angle
                        _, dth, dph = forced_reverse - refracted_reverse
                        rp.tr *= np.abs(Model.beam_pattern_3D(dth, dph, self.lam, self.surface.fs, rp.mags[0]))

                        # radar eq
                        rp.tr *= radar_eq(self.power, self.gain, 1, self.lam, sum(rp.mags))
                        
                    else:
                        rp.tr = 0
                    
                else:
                    rp.tr = 0
                        
                self.raypaths.append(rp)
                
    # use raypaths to generate a timeseries
    # show output timeseries as well as frequecy spec
    # received by the radar
    def gen_timeseries(self, show=True):
        
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
            output += rp.tr * tmp
            i += 1
            
        # normalize to one
        output /= np.max(output)
            
        ts = np.linspace(0, len(output)*self.source.dt, num=len(output))
        ts += min(times)
        
        self.ts = ts
        self.data = output
        
        if show:
                    
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
        
        angles_2nd_index = np.zeros(self.surface.zs.shape)
        angles_1st_index = np.zeros(self.surface.zs.shape)
        for i, x in enumerate(self.surface.x):
            for j, y in enumerate(self.surface.y):
                rp = self.raypaths[i*len(self.surface.x)+j]
                if rp.refracted is not None:
                    angles_2nd_index[i, j] = rp.forced[2] - rp.refracted[2]
                    angles_1st_index[i, j] = rp.forced[1] - rp.refracted[1]  # Difference of the 1st indices
                else:
                    angles_2nd_index[i, j] = None
                    angles_1st_index[i, j] = None

        # Create subplots
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.3, subplot_titles=("$\Delta\phi$", "$\Delta\\theta$"))

        # Add the first heatmap
        heatmap_2nd_index = go.Heatmap(z=angles_2nd_index, coloraxis='coloraxis1', colorscale='Aggrnyl')
        fig.add_trace(heatmap_2nd_index, row=1, col=1)

        # Add the second heatmap
        heatmap_1st_index = go.Heatmap(z=angles_1st_index, coloraxis='coloraxis2', colorscale='Phase')
        fig.add_trace(heatmap_1st_index, row=1, col=2)

        # Update layout
        fig.update_layout(
            title='Angle Differences',
            template="plotly_white",
            coloraxis1=dict(colorscale='Aggrnyl', colorbar=dict(x=0.35, len=0.9, title="rad")),
            coloraxis2=dict(colorscale='Phase', colorbar=dict(x=1.00, len=0.9, title="rad")),
        )

        # Update x and y axis titles for each subplot
        fig.update_xaxes(title_text='Facet X #', row=1, col=1)
        fig.update_yaxes(title_text='Facet Y #', row=1, col=1)
        fig.update_xaxes(title_text='Facet X #', row=1, col=2)
        fig.update_yaxes(title_text='Facet Y #', row=1, col=2)

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
