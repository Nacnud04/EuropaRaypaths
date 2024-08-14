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
        self.nu0 = 376.7 # intrinsic impedance of free space
        self.mu0 = (4 * np.pi) * 1e-7 # magnetic permeability of free space
        self.eps0 = 8.85e-12 # permittivity of free space

        # set surface and source objects
        self.surface = surface
        self.source = source

        # define vars for target location
        self.tx, self.ty, self.tz = None, None, None
        
        # --- MATERIAL PARAMETERS ---
        
        # material relative dielectric
        self.eps1 = 1
        self.eps2 = 3.15
        
        # material magnetic permability
        self.mu1 = self.mu0
        self.mu2 = self.mu1
        
        # material conductivity
        self.sig1 = 0
        self.sig2 = 1e-6
        
        # --- VELOCITIES ---
        
        self.c1 = self.c / sqrt(self.eps1)
        self.c2 = self.c / sqrt(self.eps2)
        
        # --- IMPEDANCES ---
        
        self.nu1 = self.nu0 / sqrt(self.eps1)
        self.nu2 = self.nu0 / sqrt(self.eps2)
        
        # --- INDEX OF REFRACTION ---
        
        self.n1 = sqrt(self.eps1)
        self.n2 = sqrt(self.eps2)
        
        # --- REFLECTION AND TRANSMISSION COEFFS --- 
        
        self.rho = (self.nu2 - self.nu1) / (self.nu2 + self.nu1) # reflection coeff
        self.tau = (2 * self.nu2) / (self.nu2 + self.nu1)        # transmission coeff
        
        # --- ATTENUATION CONSTANTS ---
        
        # calc for above surface
        eps_pp = self.sig1 / (2 * np.pi * source.f0 * self.eps0)
        alpha1 = sqrt(1 + (eps_pp/self.eps1)**2) - 1
        alpha1 = sqrt(0.5 * self.eps1 * alpha1)
        self.alpha1 = (alpha1 * 2 * np.pi) / source.lam
        
        # calc for below surface
        eps_pp = self.sig2 / (2 * np.pi * source.f0 * self.eps0)
        alpha2 = sqrt(1 + (eps_pp/self.eps2)**2) - 1
        alpha2 = sqrt(0.5 * self.eps2 * alpha2)
        self.alpha2 = (alpha2 * 2 * np.pi) / source.lam
        
        # --- ANTENNA POWER & GAIN ---
        
        # antenna power
        self.power = power
        
        # gain for subsurface
        self.db = 80
        self.gain = db_to_mag(self.db)
        
        # gain for surface
        surf_db = 64
        self.surf_gain = db_to_mag(surf_db)
        
        
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
                
                # --- START REFLECTION CALCS ---
                
                rp.re = self.rho * -1
                
                # find reflected raypath
                inbound = (cart_to_sp(rp.norms[0]) - np.pi) - spfnorm
                dph = inbound[2] * 2
                rp.refl_dph = dph
                rp.refl_dth = np.pi
                
                # find energy reflecting back to source
                rp.re *= np.abs(Model.beam_pattern_3D(np.pi, dph, self.lam, self.surface.fs, rp.mags[0]))
                
                # use radar equation
                rp.re *= radar_eq(self.power, self.surf_gain, 1, self.lam, rp.mags[0])
                
                # attenuation
                rp.re *= np.exp(-1 * self.alpha1 * 2 * rp.mags[0])
                
                # --- END REFLECTION CALCS ---
                
                # --- START REFRACTION CALCS ---
                
                rp.tr = self.tau
                
                # compute refracted ray angle
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
                        
                        # attenuation (source -> surf)
                        rp.tr *= np.exp(-1 * self.alpha1 * 2 * rp.mags[0])
                        
                        # attenuation (surf -> target)
                        rp.tr *= np.exp(-1 * self.alpha2 * 2 * rp.mags[1])
                        
                    else:
                        rp.tr = 0
                    
                else:
                    rp.tr = 0
                    
                # --- END REFRACTION CALCS ---
                        
                self.raypaths.append(rp)   


    @staticmethod
    def freq_shift(t, signal, f_shift):

        return signal * np.exp(1j * 2 * np.pi * f_shift * t)
                
                
    @staticmethod
    def comp_dop(u, R, lam):
        
        f_d = -1 * (2 * np.dot(u, R)) / (lam * np.linalg.norm(R))
        return f_d
                
                
    # compute doppler shift due to instrument velocity        
    def comp_dopplers(self, plot=False):
        
        u = self.source.u
        sloc = self.source.coord
        
        Rs = np.array([rp.coord - sloc for rp in self.raypaths])
        f_ds = np.array([Model.comp_dop(u, R, self.source.lam) for R in Rs])

        # source object to modify
        source = Source(1e-9, 1.0e-6, (1050, 5050, 25000))

        for f, rp in zip(f_ds, self.raypaths):
            # frequency shift the source wavelet
            #rp.wavelet = Model.freq_shift(self.source.t, self.source.signal, f)
            # instead of frequency shifting the source we can just generate a new source centered
            # around a different frequency
            source.chirp(9e6+f, 1e6, 0.5e-6)
            rp.wavelet = source.signal

        if plot:
        
            f_ds_shaped = np.reshape(f_ds, self.surface.zs.shape)
            
            fig = go.Figure(data=go.Contour(
                z=f_ds_shaped,
                colorscale='Viridis',
                contours=dict(
                    showlabels=True
                )
            ))
    
            fig.update_layout(
                title='Contour Plot of Doppler Values',
                xaxis_title='Facet Y#',
                yaxis_title='Facet X#',
                xaxis=dict(
                    scaleanchor="y",  # This makes the x-axis scale the same as the y-axis
                ),
                yaxis=dict(
                    scaleratio=1  # Ensures the ratio of the x and y axes are 1:1
                )
            )
    
            fig.show()
    
                
    # use raypaths to generate a timeseries
    # show output timeseries as well as frequecy spec
    # received by the radar
    def gen_timeseries(self, show=True):
        
        times = np.array([rp.path_time for rp in self.raypaths])
        
        # compute index offsets
        mintimes = np.min(list(times) + [rp.refl_time for rp in self.raypaths])
        rel_times = times - mintimes
        idx_offsets = np.round(rel_times / self.source.dt).astype(int)
        
        # create an empty output array
        output = np.zeros(np.max(idx_offsets) + len(self.source.signal))

        # time axis
        ts = np.linspace(0, len(output)*self.source.dt, num=len(output))
        ts += mintimes
        
        # --- ADD REFRACTED RAYPATHS ---
        for rp, offset in zip(self.raypaths, idx_offsets):
            
            # add 0 padding to account for different ray arrival time to front
            # of the signal matrix
            tmp = np.concatenate((np.zeros(offset), rp.wavelet * rp.tr))
            # add 0 padding to the back, to make it the same length as the output array
            tmp = np.concatenate((tmp, np.zeros(len(output) - len(tmp))))
            
            # sum the new output signal with the output.
            output += tmp
            
        # --- ADD REFLECTED RAYPATHS
        rel_times = np.array([rp.refl_time for rp in self.raypaths]) - mintimes
        idx_offsets = np.round(rel_times / self.source.dt).astype(int)
        
        for rp, offset in zip(self.raypaths, idx_offsets):
            
            # add 0 padding to account for different ray arrival time to front
            # of the signal matrix
            tmp = np.concatenate((np.zeros(offset), rp.wavelet * rp.re))
            # add 0 padding to the back, to make it the same length as the output array
            tmp = np.concatenate((tmp, np.zeros(len(output) - len(tmp))))
            
            # sum the new output signal with the output
            output += tmp
        
        # normalize to one
        output /= np.max(output)
        
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
    
    
    def theta_funct(self):
        
        # for the theta function we need the raypath distance for the
        # portion in the air and in the ice. specifically for that of
        # the highest amplitude return (the dominant raypath)
        
        imax = int(len(self.raypaths)/2) #np.array([rp.tr for rp in self.raypaths]).argmax()
        rp = self.raypaths[imax]
        theta = (2 * (rp.mags[0] + self.n2 * rp.mags[1])) / self.c
        return theta
    
    
    def ref_funct(self, x, t):
        
        b = 1 # complex correction factor
        r2m = b * np.exp(-1j * 2 * np.pi * self.source.f0 * self.theta_funct())
        return r2m
    
    
    def ref_funct_conj(self, x, t):
        
        r2m = self.ref_funct(x, t)
        r2m_conj = r2m.real - 1j*r2m.imag
        return r2m_conj
    
    
    # plot angle difference between refracted ray
    # and forced ray to target
    def plot_s2f_angle(self):

        # --- CREATION OF PLOT FOR REFRACTED ---
        
        dphs = np.reshape([rp.forced[2] - rp.refracted[2] if rp.refracted is not None else None for rp in self.raypaths], self.surface.zs.shape)
        dths = np.reshape([rp.forced[1] - rp.refracted[1] if rp.refracted is not None else None for rp in self.raypaths], self.surface.zs.shape)

        # Create subplots
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.3, subplot_titles=("Δφ", "Δθ"))

        # Add the first heatmap
        heatmap_2nd_index = go.Heatmap(z=dphs, coloraxis='coloraxis1')
        fig.add_trace(heatmap_2nd_index, row=1, col=1)

        # Add the second heatmap
        heatmap_1st_index = go.Heatmap(z=dths, coloraxis='coloraxis2', colorscale='Phase')
        fig.add_trace(heatmap_1st_index, row=1, col=2)

        # Update layout
        fig.update_layout(
            title='Angle Differences (Refracted)',
            template="plotly_white",
            coloraxis1=dict(
                colorbar=dict(x=0.35, len=0.9, title="rad"),
                cmin=-np.pi/2, cmax=0
            ),
            coloraxis2=dict(
                colorscale='Phase', 
                colorbar=dict(x=1.00, len=0.9, title="rad"),
                cmin=-np.pi, cmax=np.pi
            )
        )

        # Update x and y axis titles for each subplot
        fig.update_xaxes(title_text='Facet Y #', row=1, col=1)
        fig.update_yaxes(title_text='Facet X #', row=1, col=1)
        fig.update_xaxes(title_text='Facet Y #', row=1, col=2)
        fig.update_yaxes(title_text='Facet X #', row=1, col=2)

        fig.show()
        
        
        # --- NOW CREATE PLOT FOR REFLECTED ---
        
        dphs = np.reshape([rp.refl_dph for rp in self.raypaths], self.surface.zs.shape)
        dths = np.reshape([rp.refl_dth for rp in self.raypaths], self.surface.zs.shape)
        
        # Create subplots
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.3, subplot_titles=("Δφ", "Δθ"))

        # Add the first heatmap
        hm2 = go.Heatmap(z=dphs, coloraxis='coloraxis1')
        fig.add_trace(hm2, row=1, col=1)

        # Add the second heatmap
        hm1 = go.Heatmap(z=dths, coloraxis='coloraxis2', colorscale='Phase')
        fig.add_trace(hm1, row=1, col=2)

        # Update layout
        fig.update_layout(
            title='Angle Differences (Reflected)',
            template="plotly_white",
            coloraxis1=dict(
                colorbar=dict(x=0.35, len=0.9, title="rad"),
                cmin=-np.pi/2, cmax=0
            ),
            coloraxis2=dict(
                colorscale='Phase', 
                colorbar=dict(x=1.00, len=0.9, title="rad"),
                cmin=-np.pi, cmax=np.pi
            )
        )

        # Update x and y axis titles for each subplot
        fig.update_xaxes(title_text='Facet Y #', row=1, col=1)
        fig.update_yaxes(title_text='Facet X #', row=1, col=1)
        fig.update_xaxes(title_text='Facet Y #', row=1, col=2)
        fig.update_yaxes(title_text='Facet X #', row=1, col=2)

        fig.show()

        
    def show_travel_time(self):
        
        # --- SHOW FACET TRAVEL TIME IF DESIRED
        refl = np.reshape([rp.refl_time for rp in self.raypaths], self.surface.zs.shape)
        refr = np.reshape([rp.path_time for rp in self.raypaths], self.surface.zs.shape)
        
        print(f"Minimum time for reflected raypath: {np.min(refl)*1e6}us")
        
        # Create subplots
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.3, subplot_titles=("Reflected", "Refracted"))

        # Add the first heatmap
        hm2 = go.Heatmap(z=refl, coloraxis='coloraxis1')
        fig.add_trace(hm2, row=1, col=1)

        # Add the second heatmap
        hm1 = go.Heatmap(z=refr, coloraxis='coloraxis2')
        fig.add_trace(hm1, row=1, col=2)

        # Update layout
        fig.update_layout(
            title='Two way travel time',
            template="plotly_white",
            coloraxis1=dict(
                colorbar=dict(x=0.35, len=0.9, title="rad"),
            ),
            coloraxis2=dict( 
                colorbar=dict(x=1.00, len=0.9, title="rad"),
            )
        )

        # Update x and y axis titles for each subplot
        fig.update_xaxes(title_text='Facet Y #', row=1, col=1)
        fig.update_yaxes(title_text='Facet X #', row=1, col=1)
        fig.update_xaxes(title_text='Facet Y #', row=1, col=2)
        fig.update_yaxes(title_text='Facet X #', row=1, col=2)

        fig.show()
        
        
    # plot fraction of radiated power which returns to
    # source after reflecting off target by facet
    def plot_s2f_rad(self):
        
        tr = np.reshape([rp.tr for rp in self.raypaths], self.surface.zs.shape)
        re = np.reshape([rp.re for rp in self.raypaths], self.surface.zs.shape)
        
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.3, subplot_titles=("Reflected", "Refracted"))
                
        refl = go.Heatmap(z=re, coloraxis='coloraxis1')
        fig.add_trace(refl, row=1, col=1)
        
        refr = go.Heatmap(z=tr, coloraxis='coloraxis2')
        fig.add_trace(refr, row=1, col=2)

        fig.update_layout(
            title='Reradiation by Facet',
            template="plotly_white",
            coloraxis1=dict(colorbar=dict(x=0.35, len=0.9, title="W")),
            coloraxis2=dict(colorbar=dict(x=1.00, len=0.9, title="W"))
        )

        # Update x and y axis titles for each subplot
        fig.update_xaxes(title_text='Facet Y #', row=1, col=1)
        fig.update_yaxes(title_text='Facet X #', row=1, col=1)
        fig.update_xaxes(title_text='Facet Y #', row=1, col=2)
        fig.update_yaxes(title_text='Facet X #', row=1, col=2)

        fig.show()

        
    def attenuation_plot(self, st=0, en=10000, n=150):
        
        z = np.linspace(st, en, n)
        y = np.exp(-1 * self.alpha2 * z) * 100
        
        fig = go.Figure(data=go.Scatter(x=z, y=y, mode='lines'))

        fig.update_layout(
            title='Attenuation over Distance',
            xaxis_title='Total Distance (m)',
            yaxis_title='Signal retention (%)',
            template='plotly_white'
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
