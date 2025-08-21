import numpy as np
import plotly.graph_objects as go # type: ignore
import matplotlib.pyplot as plt

from math import sqrt
from time import time as Time
import copy, os

from source import *
from surface import *
from raypath import *
from util import *

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

nGPU = int(os.getenv("nGPU"))

if nGPU > 0:
    print(f"GPU's detected. Enabling CUDA compute")
    import cupy as cp

def run_sim_ms(surf, sources, target, reflect=True, progress=True, doppler=False, 
                     phase=False, polarization=None, rough=True, pt_response=None,
                     sltrng=True, plot=True, xmin=-10, xmax=20, refl_center=False):
    """
    Run sim over a bunch of sources and construct radargram
    """

    rdrgrm = np.zeros((4803, len(sources)), np.complex128)

    if nGPU > 0:
        rdrgrm = cp.asarray(rdrgrm)
    
    sltrng = []

    if doppler:
        phase=False

    if phase: 
        phase_hist = []

    start_time = time.time()

    for j, s in enumerate(sources):

        if progress:
            elapsed = time.time() - start_time
            avg_time = elapsed / (j + 1)
            remaining = avg_time * (len(sources) - j - 1)
            mins, secs = divmod(int(remaining), 60)
            eta_str = f"{mins:02d}:{secs:02d}"

            print(f"Simulating: {j+1}/{len(sources)} ({round(100*((j+1)/len(sources)), 1)}%) | ETA: {eta_str}", end="     \r")

        model = Model(surf, s, reflect=reflect, vec=True, polarization=polarization, rough=rough, pt_response=pt_response)
        model.set_target(target)

        model.gen_raypaths(refl_center=refl_center)

        if doppler:
            model.comp_dopplers()

        if nGPU == 0:
            model.gen_timeseries_vec(show=False, doppler=doppler)
        else:
            model.gen_timeseries_gpu(show=False, doppler=doppler)

        rdrgrm[:,j] = model.signal

        # get highest amplitude return raypath
        if nGPU > 0:
            max_idx = np.argmax(cp.asnumpy(model.tr))
        else:
            max_idx = np.argmax(model.tr)
            
        y_max, x_max = np.unravel_index(max_idx, model.tr.shape)

        # append travel time to pathtimes
        if nGPU > 0:
            sltrng.append(cp.asnumpy(model.slant_range)[y_max, x_max])
        else:
            sltrng.append(model.slant_range[y_max, x_max])
        if phase:
            phase_hist.append(model.phase_hist)

    if nGPU > 0:
        rdrgrm = cp.asnumpy(rdrgrm)
        ts = cp.asnumpy(model.ts)
    else:
        ts = model.ts
    
    if plot:
        extent = (xmin, xmax, np.max(ts) / 1e-6, np.min(ts) / 1e-6)
        fig, ax = plt.subplots(1, figsize=(10, 5), constrained_layout=True)
        ax.imshow(np.abs(rdrgrm), cmap="gray", aspect=0.5, extent=extent)
        ax.set_xlabel("Azumith [km]", fontsize=8)
        ax.set_ylabel("Range [us]", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_title("Specular Point Target in Subsurface")
        plt.show()

    if phase == True:
        return rdrgrm, sltrng, phase_hist

    if sltrng == True:
        return rdrgrm, sltrng

    else:
        return rdrgrm, model.ts


class Model():

    def __init__(self, surface, source, power=11.75, reflect=True, eps2=3.15, sig2=1e-6, 
                       vec=False, polarization=None, rough=True, pt_response=None):
        
        self.c = 299792458 # speed of light
        self.nu0 = 376.7 # intrinsic impedance of free space
        self.mu0 = (4 * np.pi) * 1e-7 # magnetic permeability of free space
        self.eps0 = 8.85e-12 # permittivity of free space

        # set surface and source objects
        self.surface = surface
        self.source = source
        self.lam = self.c / self.source.f0

        # define vars for target location
        self.tx, self.ty, self.tz = None, None, None

        # do we compute the surface reflection?
        self.reflect = reflect

        # do we compute everything in a vectorized form?
        self.vec = vec
        
        # --- MATERIAL PARAMETERS ---
        
        # material relative dielectric
        self.eps1 = 1
        self.eps2 = eps2
        
        # material magnetic permability
        self.mu1 = self.mu0
        self.mu2 = self.mu1
        
        # material conductivity
        self.sig1 = 0
        self.sig2 = sig2
        
        # --- VELOCITIES ---
        
        self.c1 = self.c / sqrt(self.eps1)
        self.c2 = self.c / sqrt(self.eps2)
        
        # --- IMPEDANCES ---
        
        self.nu1 = self.nu0 / sqrt(self.eps1)
        self.nu2 = self.nu0 / sqrt(self.eps2)
        
        # --- INDEX OF REFRACTION ---
        
        self.n1 = sqrt(self.eps1)
        self.n2 = sqrt(self.eps2)

        # --- POLARIZATION ---

        self.polarization = polarization
        
        # --- REFLECTION AND TRANSMISSION COEFFS --- 
        
        if polarization is None:
            self.rho = (self.nu2 - self.nu1) / (self.nu2 + self.nu1) # reflection coeff
            self.tau = (2 * self.nu2) / (self.nu2 + self.nu1)        # transmission coeff
            self.tau_rev = (2 * self.nu1) / (self.nu2 + self.nu1)    # coeff for the other direction
        
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

        # --- FACTOR APPLIED TO WAVENUMBER ---
        
        self.k_fac1 = np.sqrt(self.mu0*self.eps1*self.eps0)
        self.k_fac2 = np.sqrt(self.mu0*self.eps2*self.eps0)
        
        # --- ANTENNA POWER & GAIN ---
        
        # antenna power
        self.power = power
        
        # gain for subsurface
        self.db = 80
        self.gain = db_to_mag(self.db)
        
        # gain for surface
        surf_db = 90#64
        self.surf_gain = db_to_mag(surf_db)

        # --- WAVELET CONSTRUCTION ---
        # assuming HF (9 MHz)

        self.range_resolution = 300        # range resolution [m]
        self.lam              = 33.3       # wavelength       [m]
        self.rx_window_m      = 30e3       # rx window        [m]
        self.rx_window_offset = 20e3       # rx window offset [m]
        self.sampling         = 48e6       # rx sampling rate [Hz]

        # angular wavenumber
        self.k                = (2 * np.pi) / self.lam

        # compute range bin count from that
        self.rb = int((self.rx_window_m / self.c) / (1 / self.sampling))

        # sampled slant range of echo
        self.ssl = np.linspace(self.rx_window_offset, self.rx_window_offset + self.rx_window_m, self.rb)

        # --- SURFACE ROUGHNESS ---
        
        # electromagnetic roughness [.]
        self.ks = self.k * self.surface.s

        # simulate roughness?
        self.rough = rough

        # --- POINT TARGET RESPONSE ---
        
        self.pt_response = pt_response
        
        
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



    def compute_raypath(self, i, j, fast=False):
        x = self.surface.x[i]
        y = self.surface.y[j]
        fnorm = self.surface.normals[i, j]
        spfnorm = cart_to_sp(fnorm)
    
        # create rays
        rp = RayPaths(self.source.coord,
                      [x, y, self.surface.zs[i, j]],
                      [self.tx, self.ty, self.tz], i, j)
    
        # --- START REFLECTION CALCS ---
        if self.reflect:
            rp.re = self.rho * -1
            inbound = (cart_to_sp(rp.norms[0]) - np.pi) - spfnorm
            dph = inbound[2] * 2
            rp.refl_dph = dph
            rp.refl_dth = np.pi
            rp.re *= np.abs(Model.beam_pattern_3D(np.pi, dph, self.lam, self.surface.fs, rp.mags[0]))
            rp.re *= radar_eq(self.power, self.surf_gain, 1, self.lam, rp.mags[0])
            rp.re *= np.exp(-1 * self.alpha1 * 2 * rp.mags[0])
        # --- END REFLECTION CALCS ---
    
        # --- START REFRACTION CALCS ---
        rp.tr = self.tau
        rp.set_source(self.source)
        rp.set_facet(fnorm, self.surface.fs)
        rp.refracted = rp.comp_refracted(self.c1, self.c2)
    
        if rp.refracted is not None and not fast:
            rp.forced = cart_to_sp(rp.norms[1]) - spfnorm
            _, dth, dph = rp.forced - rp.refracted
            rp.tr *= np.abs(Model.beam_pattern_3D(dth, dph, self.lam, self.surface.fs, rp.mags[1]))
            refracted_reverse = rp.comp_rev_refracted(self.c1, self.c2)
    
            if refracted_reverse is not None:
                spfnorm_reverse = cart_to_sp(fnorm * -1)
                forced_reverse = cart_to_sp(rp.norms[0] * -1) - spfnorm_reverse
                _, dth, dph = forced_reverse - refracted_reverse
                rp.tr *= np.abs(Model.beam_pattern_3D(dth, dph, self.lam, self.surface.fs, rp.mags[0]))
                rp.tr *= radar_eq(self.power, self.gain, 1, self.lam, sum(rp.mags))
                rp.tr *= np.exp(-1 * self.alpha1 * 2 * rp.mags[0])
                rp.tr *= np.exp(-1 * self.alpha2 * 2 * rp.mags[1])
            else:
                rp.tr = 0
        else:
            rp.tr = 0
        # --- END REFRACTION CALCS ---
    
        return rp

    def gen_s2f_cpu(self, offsetx=0, offsety=0):

        rp_s2f_C = np.stack(((self.source.x - self.surface.X) + offsetx, 
                             (self.source.y - self.surface.Y) + offsety,
                             (self.source.z - self.surface.zs)))
        rp_s2f_S = cart_to_sp(rp_s2f_C, vec=True)
        # reverse of source to facet
        rp_s2f_S = cart_to_sp(-1 * rp_s2f_C, vec=True)

        return rp_s2f_C, rp_s2f_S
    
    def compute_raypaths_vectorized(self, pt_debug=False, refl_center=True):

        # get facet normal vectors in cartesian and spherical
        surf_norms    = np.rollaxis(self.surface.normals, -1, 0)
        surf_norms_S  = cart_to_sp(surf_norms, vec=True)
        # reverse of facet normal for reverse refracted raypath computation
        surf_norms_SR = cart_to_sp(surf_norms * -1, vec=True)

        # first we need the raypath from source to facet
        rp_s2f_C, rp_s2f_S = self.gen_s2f_cpu()

        rp_s2f_S_rF = rp_s2f_S - surf_norms_S
        rp_s2f_S_rF[2, :, :] -= np.pi
        theta_1 = rp_s2f_S_rF[2, :, :]

        # --- START REFRACTION CALCS ---
        # THIS SECTION IS FOR INTO THE SUBSURFACE ONLY
        # make raypaths from facet to target
        rp_f2t_C = np.stack((self.tx - self.surface.X, 
                             self.ty - self.surface.Y,
                            self.tz - self.surface.zs))
        rp_f2t_S = cart_to_sp(rp_f2t_C, vec=True)
        # get the refracted vector from the facet in spherical coordinates
        # stands for raypath_facetToInside_spherical_relativeToFacet
        rp_f2i_S_rF = comp_refracted_vectorized(surf_norms, normalize_vectors(rp_s2f_C),
                                                 self.c1, self.c2)
        # angle delta between the facet to target raypath and the refracted raypath
        # relative to facet normal in spherical coordinates
        rp_f2t_CN = normalize_vectors(rp_f2t_C)
        rp_f2t_SD_rF = cart_to_sp(rp_f2t_CN - surf_norms, vec=True) - rp_f2i_S_rF
        theta_2 = rp_f2t_SD_rF[2, :, :]

        # develop reflection and transmission coefficients
        if self.polarization == "h":
            rho_h = (self.nu2 * np.cos(theta_1) - self.nu1 * np.cos(theta_2)) / (self.nu2 * np.cos(theta_1) + self.nu1 * np.cos(theta_2))
            re = np.abs(rho_h)**2
            tr = 1 - re
        elif self.polarization == "v":
            rho_v = (self.nu2 * np.cos(theta_2) - self.nu1 * np.cos(theta_1)) / (self.nu2 * np.cos(theta_2) + self.nu1 * np.cos(theta_1))
            re = np.abs(rho_v)**2
            tr = 1 - re

        # manipulate coefficients if surface roughness enabled
        if self.rough == True:
            psi_1 = self.ks * np.cos(theta_1)
            re *= np.exp(-4 * psi_1**2)
            psi_2 = self.ks * np.cos(theta_2)
            tr *= np.exp(-4 * psi_2**2)

        # generate array for % energy transmitted
        if self.polarization is None:
            tr = np.ones_like(self.surface.zs) * self.tau * self.tau_rev

        facet_incident = cart_to_sp(rp_f2t_CN, vec=True)

        # implement point target response
        if self.pt_response == "sinusoidal":
            tr *= target_function_sinusoidal(facet_incident[2, :, :], facet_incident[1, :, :], f=100)
            
        elif self.pt_response == "gaussian":
            
            gauss_transmit = target_function_gaussian(facet_incident[2, :, :], 0, phi0=180)
            
            if pt_debug == True:
                data = [phi_deg, tr, gauss_transmit]
                fig, ax = plt.subplots(1, 3)
                for i in range(3):
                    im = ax[i].imshow(data[i])
                    plt.colorbar(im, ax=ax[i], shrink=0.33)
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])
                plt.show()
                
            tr *= gauss_transmit

        del facet_incident

        # account for aperture losses
        tr *= np.abs(Model.beam_pattern_3D(rp_f2t_SD_rF[1, :, :], rp_f2t_SD_rF[2, :, :], self.lam,
                                           self.surface.fs, rp_f2t_S[0, :, :]))
        # THIS SECTION IS FOR EXITING THE SUBSURFACE
        # reverse refracted raypath
        rp_f2o_S_rF = comp_refracted_vectorized(surf_norms, normalize_vectors(rp_f2t_C),
                                                 self.c1, self.c2, rev=True)
        # angle delta between the forced reverse raypath and the reversed refracted raypath
        # relative to reversed facet normal
        forced_rev = normalize_vectors(rp_s2f_C) + surf_norms
        rp_f2s_SD_rF = cart_to_sp(forced_rev, vec=True) - rp_f2o_S_rF
        # --- FIXED TO HERE ---
        # account for Nan 
        tr[(rp_f2i_S_rF[0, :, :] * rp_f2o_S_rF[0, :, :]) == None] = 0
        # compute losses from second refraction through aperture
        tr *= np.abs(Model.beam_pattern_3D(rp_f2s_SD_rF[1, :, :], rp_f2s_SD_rF[2, :, :], self.lam,
                                           self.surface.fs, rp_s2f_S[0, :, :]))
        # implement radar equation
        tr *= radar_eq(self.power, self.gain, 1, self.lam, rp_s2f_S[0, :, :] + rp_f2t_S[0, :, :])
        # phase change
        tr *= np.exp(-1 * self.alpha1 * 2 * rp_s2f_S[0, :, :])
        tr *= np.exp(-1 * self.alpha2 * 2 * rp_f2t_S[0, :, :])
        # vars to save for time series or computation
        self.tr = tr
        self.comp_val = np.copy(tr)
        self.path_time = (rp_s2f_S[0, :, :] / self.c1 + rp_f2t_S[0, :, :] / self.c2) * 2
        self.slant_range = rp_s2f_S[0, :, :] + rp_f2t_S[0, :, :]
        self.refr_dth  = rp_f2t_SD_rF[1, :, :]
        self.refr_dph  = rp_f2t_SD_rF[2, :, :]
        
        if self.reflect:
            # --- START REFLECTION CALCS ---

            # center the facets under the source for full surface sim
            if refl_center == True:
                
                rp_s2f_C, rp_s2f_S = self.gen_s2f_cpu(
                    offsetx=self.surface.xcenter-self.source.x,
                    offsety=self.surface.ycenter-self.source.y
                )

                rp_s2f_S_rF = rp_s2f_S - surf_norms_S
                rp_s2f_S_rF[2, :, :] -= cp.pi
            
            if self.polarization is None:
                # generate array for % energy reflected
                re = np.ones_like(self.surface.zs) * self.rho * -1
                
            # add influence from beam pattern reflection by facets
            re *= np.abs(Model.beam_pattern_3D(np.pi, rp_s2f_S_rF[2, :, :] * 2, self.lam,
                         self.surface.fs, rp_s2f_S[0, :, :]))
            # add distance loss
            re *= radar_eq(self.power, self.surf_gain, 1, self.lam, rp_s2f_S[0, :, :])
            # attenuation
            re *= np.exp(-1 * self.alpha1 * 2 * rp_s2f_S[0, :, :])
            
            # vars to save for time series computation or plotting
            self.re = re
            self.refl_time = (rp_s2f_S[0, :, :] / self.c1) * 2
            self.refl_dth  = np.ones_like(rp_s2f_S_rF[2, :, :])*np.pi
            self.refl_dph  = rp_s2f_S_rF[2, :, :] * 2
            # --- END REFLECTION CALCS ---

        self.refl_slant_range = rp_s2f_S[0, :, :]

    def gen_s2f_gpu(self, offsetx=0, offsety=0):

        rp_s2f_C = cp.stack(((cp.asarray(self.source.x) - cp.asarray(self.surface.X)) + offsetx, 
                             (cp.asarray(self.source.y) - cp.asarray(self.surface.Y)) + offsety,
                             (cp.asarray(self.source.z) - cp.asarray(self.surface.zs))))
        rp_s2f_S = cart_to_sp(rp_s2f_C, vec=True)
        # reverse of source to facet
        rp_s2f_S = cart_to_sp(-1 * rp_s2f_C, vec=True)

        return rp_s2f_C, rp_s2f_S

    def gen_raypaths_gpu(self, refl_center=True):

        # convert surface and source/target arrays from numpy → cupy
        surf_norms    = cp.rollaxis(cp.asarray(self.surface.normals), -1, 0)
        surf_norms_S  = cart_to_sp(surf_norms, vec=True)
        # reverse of facet normal for reverse refracted raypath computation
        surf_norms_SR = cart_to_sp(surf_norms * -1, vec=True)

        # first we need the raypath from source to facet
        rp_s2f_C, rp_s2f_S = self.gen_s2f_gpu()

        rp_s2f_S_rF = rp_s2f_S - surf_norms_S
        rp_s2f_S_rF[2, :, :] -= cp.pi
        theta_1 = rp_s2f_S_rF[2, :, :]

        # --- START REFRACTION CALCS ---
        # THIS SECTION IS FOR INTO THE SUBSURFACE ONLY
        rp_f2t_C = cp.stack((cp.asarray(self.tx) - cp.asarray(self.surface.X), 
                             cp.asarray(self.ty) - cp.asarray(self.surface.Y),
                             cp.asarray(self.tz) - cp.asarray(self.surface.zs)))
        rp_f2t_S = cart_to_sp(rp_f2t_C, vec=True)

        # refracted vector from the facet in spherical coordinates
        rp_f2i_S_rF = comp_refracted_vectorized(surf_norms, normalize_vectors(rp_s2f_C),
                                                 self.c1, self.c2)

        rp_f2t_CN = normalize_vectors(rp_f2t_C)
        rp_f2t_SD_rF = cart_to_sp(rp_f2t_CN - surf_norms, vec=True) - rp_f2i_S_rF
        theta_2 = rp_f2t_SD_rF[2, :, :]

        # develop reflection and transmission coefficients
        if self.polarization == "h":
            rho_h = (self.nu2 * cp.cos(theta_1) - self.nu1 * cp.cos(theta_2)) / (self.nu2 * cp.cos(theta_1) + self.nu1 * cp.cos(theta_2))
            re = cp.abs(rho_h)**2
            tr = 1 - re
        elif self.polarization == "v":
            rho_v = (self.nu2 * cp.cos(theta_2) - self.nu1 * cp.cos(theta_1)) / (self.nu2 * cp.cos(theta_2) + self.nu1 * cp.cos(theta_1))
            re = cp.abs(rho_v)**2
            tr = 1 - re

        # surface roughness
        if self.rough:
            psi_1 = self.ks * cp.cos(theta_1)
            re *= cp.exp(-4 * psi_1**2)
            psi_2 = self.ks * cp.cos(theta_2)
            tr *= cp.exp(-4 * psi_2**2)

        # default transmission coefficient
        if self.polarization is None:
            tr = cp.ones_like(cp.asarray(self.surface.zs)) * self.tau * self.tau_rev

        facet_incident = cart_to_sp(rp_f2t_CN, vec=True)

        if self.pt_response == "sinusoidal":
            tr *= target_function_sinusoidal(facet_incident[2, :, :], facet_incident[1, :, :], f=100)
            
        elif self.pt_response == "gaussian":
            gauss_transmit = target_function_gaussian(facet_incident[2, :, :], 0, phi0=180)          
            tr *= gauss_transmit

        del facet_incident

        # account for aperture losses
        tr *= cp.abs(Model.beam_pattern_3D(rp_f2t_SD_rF[1, :, :], rp_f2t_SD_rF[2, :, :], self.lam,
                                           self.surface.fs, rp_f2t_S[0, :, :]))
        # exiting subsurface
        rp_f2o_S_rF = comp_refracted_vectorized(surf_norms, normalize_vectors(rp_f2t_C),
                                                 self.c1, self.c2, rev=True)

        forced_rev = normalize_vectors(rp_s2f_C) + surf_norms
        rp_f2s_SD_rF = cart_to_sp(forced_rev, vec=True) - rp_f2o_S_rF

        # clean NaNs
        tr[cp.isnan(rp_f2i_S_rF[0, :, :] * rp_f2o_S_rF[0, :, :])] = 0

        # second refraction losses
        tr *= cp.abs(Model.beam_pattern_3D(rp_f2s_SD_rF[1, :, :], rp_f2s_SD_rF[2, :, :], self.lam,
                                           self.surface.fs, rp_s2f_S[0, :, :]))

        # radar equation
        tr *= radar_eq(self.power, self.gain, 1, self.lam, rp_s2f_S[0, :, :] + rp_f2t_S[0, :, :])

        # attenuation
        tr *= cp.exp(-1 * self.alpha1 * 2 * rp_s2f_S[0, :, :])
        tr *= cp.exp(-1 * self.alpha2 * 2 * rp_f2t_S[0, :, :])

        # save vars
        self.tr = tr
        self.comp_val = cp.asnumpy(cp.copy(tr))
        self.path_time = cp.asnumpy((rp_s2f_S[0, :, :] / self.c1 + rp_f2t_S[0, :, :] / self.c2) * 2)
        self.slant_range = rp_s2f_S[0, :, :] + rp_f2t_S[0, :, :]
        self.refr_dth  = cp.asnumpy(rp_f2t_SD_rF[1, :, :])
        self.refr_dph  = cp.asnumpy(rp_f2t_SD_rF[2, :, :])
        
        if self.reflect:

            # center the facets under the source for full surface sim
            if refl_center == True:
                
                rp_s2f_C, rp_s2f_S = self.gen_s2f_gpu(
                    offsetx=self.surface.xcenter-self.source.x,
                    offsety=self.surface.ycenter-self.source.y
                )

                rp_s2f_S_rF = rp_s2f_S - surf_norms_S
                rp_s2f_S_rF[2, :, :] -= cp.pi

            if self.polarization is None:
                re = cp.ones_like(cp.asarray(self.surface.zs)) * self.rho * -1

            re *= cp.abs(Model.beam_pattern_3D(cp.pi, rp_s2f_S_rF[2, :, :] * 2, self.lam,
                         self.surface.fs, rp_s2f_S[0, :, :]))
            re *= radar_eq(self.power, self.surf_gain, 1, self.lam, rp_s2f_S[0, :, :])
            re *= cp.exp(-1 * self.alpha1 * 2 * rp_s2f_S[0, :, :])

            self.re = re
            self.refl_time = cp.asnumpy((rp_s2f_S[0, :, :] / self.c1) * 2)
            self.refl_dth  = cp.asnumpy(cp.ones_like(rp_s2f_S_rF[2, :, :]) * cp.pi)
            self.refl_dph  = cp.asnumpy(rp_s2f_S_rF[2, :, :] * 2)
        
        self.refl_slant_range = rp_s2f_S[0, :, :]
    
    def gen_raypaths_threaded(self, fast=False):
        self.raypaths = []
        self.lam = self.c / self.source.f0
        
        # Use ThreadPoolExecutor to parallelize the computation
        with ThreadPoolExecutor() as executor:
            futures = []
            
            # Submit tasks for each surface point (i, j)
            for i in range(len(self.surface.x)):
                for j in range(len(self.surface.y)):
                    futures.append(executor.submit(self.compute_raypath, j, i, fast))
            
            # Collect the results as they complete
            for future in as_completed(futures):
                self.raypaths.append(future.result())

        

            
    # create rays from source to target. 
    # calculate frac of transmit power.
    def gen_raypaths(self, fast=False, progress_bar=False, refl_center=False):

        if nGPU > 0:

            self.gen_raypaths_gpu(refl_center=refl_center)
        
        elif self.vec:

            self.compute_raypaths_vectorized(refl_center=refl_center)
        
        else:

            self.raypaths = []
            
            self.lam = self.c / self.source.f0

            start_time = time.time()

            self.comp_val = []
            
            for i, x in enumerate(self.surface.x):

                if progress_bar:
                    elapsed_time = time.time() - start_time
                    avg_time_per_iter = elapsed_time / (i + 1)
                    remaining_iters = len(self.surface.x) - (i + 1)
                    eta = avg_time_per_iter * remaining_iters
                    eta_formatted = time.strftime('%H:%M:%S', time.gmtime(eta))
                    print(f"Generating raypaths... {i+1}/{len(self.surface.x)} ({round(100 * ((i+1) / len(self.surface.x)), 2)}%) | ETA: {eta_formatted}", end="    \r")
                
                for j, y in enumerate(self.surface.y):
                    
                    fnorm = self.surface.normals[j, i]
                    spfnorm = cart_to_sp(fnorm)
                    
                    # create rays
                    rp = RayPaths(self.source.coord, 
                                [x, y, self.surface.zs[j, i]],
                                [self.tx, self.ty, self.tz], xid=i, yid=j)
                    
                    # --- START REFLECTION CALCS ---

                    if self.reflect:
                        
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
                    
                    rp.tr = self.tau * self.tau_rev   # energy maintained through double refraction to target and then to surface
                    rp.trt = self.tau                 # energy maintained through single refraction to target
                    
                    # compute refracted ray angle
                    rp.set_source(self.source)
                    rp.set_facet(fnorm, self.surface.fs)
                    
                    rp.refracted = rp.comp_refracted(self.c1, self.c2)
                    
                    if rp.refracted is not None and not fast:
                        
                        # compute forced ray
                        rp.forced = rp.norms[1] - rp.fnorm
                        
                        # find forced ray relative to refracted angle
                        # this computation is done entirely when relative to facet normal
                        spforced = cart_to_sp(rp.forced)
                        
                        # compute angle differences
                        dth, dph = spforced[1] - rp.refracted[1], spforced[2] - rp.refracted[2]
                        rp.dph = dph
                        rp.dth = dth
                        # compute loss from propagating to the target
                        # compute trasmitted power based on difference in refracted and forced ray angle
                        facet_loss = np.abs(Model.beam_pattern_3D(dth, dph, self.lam, self.surface.fs, rp.mags[1]))
                        rp.tr *= facet_loss
                        rp.trt *= facet_loss
                        # compute loss from propagating back to the source
                        refracted_reverse = rp.comp_rev_refracted(self.c1, self.c2)
                        
                        if refracted_reverse is not None:

                            # compute the reverse forced raypath
                            rp.forced_rev = -1 * rp.norms[0] + rp.fnorm

                            # move into spherical
                            spforced_rev = cart_to_sp(rp.forced_rev)
                            
                            # find difference in angle
                            dth = spforced_rev[1] - refracted_reverse[1]
                            dph = spforced_rev[2] - refracted_reverse[2]

                            # compute transmittance via beam pattern
                            rp.tr *= np.abs(Model.beam_pattern_3D(dth, dph, self.lam, self.surface.fs, rp.mags[0]))

                            # radar eq
                            rp.tr *= radar_eq(self.power, self.gain, 1, self.lam, sum(rp.mags))
                            rp.trt *= radar_eq(self.power, self.gain, 1, self.lam, sum(rp.mags)/2)
                            
                            # attenuation (source -> surf)
                            atten_loss = np.exp(-1 * self.alpha1 * 2 * rp.mags[0])
                            rp.tr *= atten_loss
                            rp.trt *= atten_loss
                            
                            # attenuation (surf -> target)
                            rp.tr *= np.exp(-1 * self.alpha2 * 2 * rp.mags[1])

                            self.comp_val.append(rp.tr)
                            
                        else:
                            rp.tr = 0
                        
                    else:
                        rp.tr = 0
                        rp.trt = 0
                        
                    # --- END REFRACTION CALCS ---
                            
                    self.raypaths.append(rp)   


    @staticmethod
    def freq_shift(t, signal, f_shift):

        return signal * np.exp(1j * 2 * np.pi * f_shift * t)
                
                
    @staticmethod
    def comp_dop(u, R, f0):

        R_hat = R / np.linalg.norm(R)
        f_d = 2 * (np.dot(u, R_hat) / 299792458) * f0

        return f_d


    @staticmethod
    def comp_dop_vectorized(u, Rs, f0):
        # Normalize R vectors
        R_hats = Rs / np.linalg.norm(Rs, axis=1)[:, np.newaxis]
        
        # Compute Doppler shifts for all ray paths at once
        f_ds = 2 * (np.dot(R_hats, u) / 299792458) * f0
    
        return f_ds
                
    def comp_doppler_shift_vectorized(self):

        # first we need the raypath from source to facet
        rp_s2f_C = np.stack((self.surface.X - self.source.x, 
                            self.surface.Y - self.source.y,
                            self.surface.zs - self.source.z))
        rp_s2f_CN = normalize_vectors(rp_s2f_C)

        self.dopplers = 2 * (fast_dot_product(rp_s2f_CN, self.source.u) / self.c1) * self.source.f0


    # compute doppler shift due to instrument velocity
    @deprecated("Avoid doppler computation if possible.")
    def comp_dopplers(self, plot=False):
        
        if self.vec:

            self.comp_doppler_shift_vectorized()

            if plot:
                plt.imshow(self.dopplers)
                plt.title("Doppler shift")
                plt.show()

        else:

            u = self.source.u
            sloc = self.source.coord
            
            Rs = np.array([rp.coord for rp in self.raypaths]) - sloc
            #f_ds = np.array([Model.comp_dop(u, R, self.source.f0) for R in Rs])
            f_ds = Model.comp_dop_vectorized(u, Rs, self.source.f0)
            
            #self.relvels = [np.dot(u, R/np.linalg.norm(R)) for R in Rs]
            self.dopplers = f_ds

            for f, rp in zip(f_ds, self.raypaths):
                # instead of frequency shifting the source we can just generate a new source centered
                # around a different frequency
                self.source.chirp(9e6+f, 1e6)
                rp.wavelet =self.source.signal

            if plot:
            
                f_ds_shaped = np.reshape(f_ds, self.surface.zs.shape)

                plt.figure(figsize=(10, 8))
                contour = plt.contourf(f_ds_shaped, cmap='viridis')
                plt.colorbar(contour, label='Frequency Shift [Hz]')
                plt.title('Contour Plot of Doppler Values')
                plt.xlabel('Facet Y#')
                plt.ylabel('Facet X#')
                plt.gca().set_aspect('equal', adjustable='box')  # Ensures equal aspect ratio
                plt.tight_layout()
                plt.show()

    def gen_timeseries_vec(self, refl_mag=3e-6, show=True, tst=None, ten=None, time=False, doppler=True):
        
        st = Time()

        # turn times into index offsets
        if self.reflect:
            mintimes = np.min(np.stack((self.path_time, self.refl_time)))
        else:
            mintimes = np.min(self.path_time)
        # compute relative times
        rel_times = self.path_time - mintimes
        # turn into range bin indicies
        idx_offsets = np.round(rel_times / self.source.dt).astype(int)

        # create empty output array
        #sig_s = np.zeros(np.max(idx_offsets) + len(self.source.signal)).astype(np.complex128)
        sig_s = np.zeros(self.rb).astype(np.complex128)

        # make another for signal at target
        sig_t = np.zeros_like(sig_s)

        # time axis
        ts = np.arange(len(sig_s)) * self.source.dt
        ts += mintimes

        # length of wavelet sample
        wavlen = len(self.source.signal)
        dt = self.source.dt

        # compute imaginary factor
        k = (2 * np.pi) / self.lam

        # --- ADD REFRACTED RAYPATHS ---

        if doppler:

            wavelets = np.sinc(2 * (self.dopplers + self.source.f0)[..., None] * self.source.t[None, None, :])

            # Get phase term and scaling
            exp = np.exp(2j * k * (idx_offsets * dt * self.c))        # shape (N, M)
            scales = exp * self.tr                                     # shape (N, M)

            # Broadcast scales to wavelet shape: (N, M, 1) * (N, M, wavlen)
            scaled_wavelets = scales[..., None] * wavelets             # shape (N, M, wavlen)

            # Flatten everything for vectorized accumulation
            flat_offsets = idx_offsets.flatten()                       # (N*M,)
            flat_wavelets = scaled_wavelets.reshape(-1, wavelets.shape[-1])  # (N*M, wavlen)

            indices = flat_offsets[:, None] + np.arange(self.source.t.size)   # (N*M, wavlen)

            np.add.at(sig_s, indices, flat_wavelets)
            np.add.at(sig_t, indices, flat_wavelets)  # If this is still needed
        
        else: 

            # THIS USES THE PROPER RANGE COMPRESSED EQUATION!!!!
            # get the phase history of the highest amplitude return
            max_idx = np.argmax(self.tr)
            rb_max, trc_max = np.unravel_index(max_idx, self.tr.shape)
            # compute the effective slant range - correcting for velocity change in subsurface
            eff_slant_range = self.refl_slant_range + (self.slant_range - self.refl_slant_range) * (self.c1 / self.c2)
            refrwav, self.phase_hist = compute_wav(self.tr, eff_slant_range, self.ssl, self.range_resolution, self.lam, rb_max, trc_max)
            sig_s += refrwav
            sig_t += sig_s
            self.slant_range = eff_slant_range
        
        # --- ADD REFLECTED RAYPATHS ---
        if self.reflect:

            refl_wav, _ = compute_wav(self.re, self.refl_slant_range, self.ssl, self.range_resolution, self.lam, rb_max, trc_max, scale=refl_mag)
            sig_s += refl_wav
            sig_t += refl_wav

        # received signal at surface
        self.ts = ts
        self.signal = sig_s

        # received signal at target
        self.tar_ts = ts / 2
        self.tar_signal = sig_t

        if time:
            print(f"Signal construction elapsed time: {Time()-st} seconds.")

        if show:

            fig, ax = plt.subplots(2, figsize=(8, 10))
            ax[0].plot(ts, np.real(sig_s), color="blue", label="Real")
            ax[0].plot(ts, np.imag(sig_s), color="red", label="Imag")
            ax[0].set_title("Simulated signal")
            ax[0].set_xlabel("Time (s)"); ax[0].set_ylabel("Signal")
            if tst or ten:
                ax[0].set_xlim(tst, ten)
            ax[0].legend()

            N = len(sig_s)
            T = ts[1] - ts[0]
            yf = np.abs(fft(sig_s)[:N//2])
            xf = fftfreq(N, T)[:N//2]

            ax[1].plot(xf, yf, color="blue", label="spectrum")
            ax[1].set_title("Simulated spectrum")
            ax[1].set_xlabel("Frequency [Hz]"); ax[0].set_ylabel("Power")
            ax[1].set_xscale("log")

            plt.show()

    def gen_timeseries_gpu(self, refl_mag=3e-6, show=True, tst=None, ten=None, time=False, doppler=True):
        
        st = Time()

        # turn times into index offsets
        if self.reflect:
            mintimes = cp.min(cp.stack((cp.asarray(self.path_time), cp.asarray(self.refl_time))))
        else:
            mintimes = cp.min(cp.asarray(self.path_time))

        # compute relative times
        rel_times = cp.asarray(self.path_time) - mintimes
        # turn into range bin indices
        idx_offsets = cp.round(rel_times / self.source.dt).astype(cp.int32)

        # output arrays
        sig_s = cp.zeros(self.rb, dtype=cp.complex128)
        sig_t = cp.zeros_like(sig_s)

        # time axis
        ts = cp.arange(len(sig_s)) * self.source.dt
        ts += mintimes

        # wavelet properties
        wavlen = len(self.source.signal)
        dt = self.source.dt

        # compute wavenumber
        k = (2 * cp.pi) / self.lam

        # --- ADD REFRACTED RAYPATHS ---
        if doppler:

            # wavelets: sinc of doppler shift
            wavelets = cp.sinc(2 * (cp.asarray(self.dopplers) + self.source.f0)[..., None] * cp.asarray(self.source.t)[None, None, :])

            # Get phase term and scaling
            exp = cp.exp(2j * k * (idx_offsets * dt * self.c))     # shape (N, M)
            scales = exp * cp.asarray(self.tr)                     # shape (N, M)

            # Broadcast scales to wavelet shape: (N, M, 1) * (N, M, wavlen)
            scaled_wavelets = scales[..., None] * wavelets         # shape (N, M, wavlen)

            # Flatten everything for accumulation
            flat_offsets = idx_offsets.ravel()                     # (N*M,)
            flat_wavelets = scaled_wavelets.reshape(-1, wavelets.shape[-1])  # (N*M, wavlen)

            indices = flat_offsets[:, None] + cp.arange(self.source.t.size)[None, :]  # (N*M, wavlen)

            # scatter add into GPU arrays
            sig_s = cp.scatter_add(sig_s, indices.ravel(), flat_wavelets.ravel())
            sig_t = cp.scatter_add(sig_t, indices.ravel(), flat_wavelets.ravel())

        else: 
            # proper range compressed equation
            max_idx = cp.argmax(self.tr)
            rb_max, trc_max = cp.unravel_index(max_idx, self.tr.shape)

            eff_slant_range = self.refl_slant_range + (self.slant_range - self.refl_slant_range) * (self.c1 / self.c2)

            refrwav, self.phase_hist = compute_wav_gpu(self.tr, eff_slant_range, cp.asarray(self.ssl),
                                                   self.range_resolution, self.lam, rb_max, trc_max)
            sig_s += refrwav
            sig_t += sig_s
            self.slant_range = eff_slant_range
        
        # --- ADD REFLECTED RAYPATHS ---
        if self.reflect:
            refl_wav, _ = compute_wav_gpu(self.re, self.refl_slant_range, cp.asarray(self.ssl),
                                      self.range_resolution, self.lam, rb_max, trc_max, scale=refl_mag)
            sig_s += refl_wav
            sig_t += refl_wav

        # received signal at surface
        self.ts = ts
        self.signal = sig_s

        # received signal at target
        self.tar_ts = ts / 2
        self.tar_signal = sig_t

    # use raypaths to generate a timeseries
    # show output timeseries as well as frequecy spec
    # received by the radar
    def gen_timeseries(self, show=True, plotly=False, tst=None, ten=None, refl_mag=1e-4, time=False):

        st = Time()
        
        times = np.array([rp.path_time for rp in self.raypaths])
        
        # compute index offsets
        mintimes = np.min(list(times) + [rp.refl_time for rp in self.raypaths])
        rel_times = times - mintimes
        idx_offsets = np.round(rel_times / self.source.dt).astype(int)
        
        # create an empty output array
        output = np.zeros(np.max(idx_offsets) + len(self.source.signal)).astype(np.complex128)

        # make another for signal at target
        output_target = np.zeros_like(output)

        # time axis
        ts = np.linspace(0, len(output)*self.source.dt, num=len(output))
        ts += mintimes

        # length of wavelet sample
        wavlen = len(self.source.signal)
        dt = self.source.dt

        # compute imaginary factor
        k = (2 * np.pi) / self.lam
        
        # --- ADD REFRACTED RAYPATHS ---
        
        exp = np.exp(2j * k * (idx_offsets * dt * self.c))

        # add raypaths
        for rp, offset, e in zip(self.raypaths, idx_offsets, exp):
            output[offset:offset+wavlen] += rp.wavelet * e * rp.tr 
            output_target[offset:offset+wavlen] += rp.wavelet * e * rp.trt
            
        # --- ADD REFLECTED RAYPATHS ---

        if self.reflect:
        
            rel_times = np.array([rp.refl_time for rp in self.raypaths]) - mintimes
            idx_offsets = np.round(rel_times / self.source.dt).astype(int)
    
            exp = np.exp(2j * k * (idx_offsets * dt * self.c))
    
            # add raypaths
            for rp, offset, e in zip(self.raypaths, idx_offsets, exp):
                output[offset:offset+wavlen] += refl_mag * rp.wavelet * e * rp.re

        # received signal at surface
        self.ts = ts
        self.signal = output

        # received signal at target
        self.tar_ts = ts / 2
        self.tar_signal = output_target

        if time:
            print(f"Signal construction elapsed time: {Time()-st} seconds.")
        
        if show and plotly:
                    
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts, y=np.real(output), mode='lines', name='real'))
            fig.add_trace(go.Scatter(x=ts, y=np.imag(output), mode='lines', name='imag'))
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

        if show and not plotly:

            fig, ax = plt.subplots(2, figsize=(8, 10))
            ax[0].plot(ts, np.real(output), color="blue", label="Real")
            ax[0].plot(ts, np.imag(output), color="red", label="Imag")
            ax[0].set_title("Simulated signal")
            ax[0].set_xlabel("Time (s)"); ax[0].set_ylabel("Signal")
            if tst or ten:
                ax[0].set_xlim(tst, ten)
            ax[0].legend()

            N = len(output)
            T = ts[1] - ts[0]
            yf = np.abs(fft(output)[:N//2])
            xf = fftfreq(N, T)[:N//2]

            ax[1].plot(xf, yf, color="blue", label="spectrum")
            ax[1].set_title("Simulated spectrum")
            ax[1].set_xlabel("Frequency [Hz]"); ax[0].set_ylabel("Power")
            ax[1].set_xscale("log")

            plt.show()
            
    
    
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

    def threaded_unscramble(self):

        zeros = np.empty((len(self.surface.x), len(self.surface.y)), dtype=object)
        for rp in self.raypaths:
            zeros[rp.xid, rp.yid] = rp

        return zeros
    
    # plot angle difference between refracted ray
    # and forced ray to target
    def plot_s2f_angle(self, pltly=False):

        if pltly:

            # --- CREATION OF PLOT FOR REFRACTED ---
            
            dphs = np.reshape([(rp.dph) if rp.refracted is not None else None for rp in self.raypaths], self.surface.zs.shape)
            dths = np.reshape([(rp.dth) if rp.refracted is not None else None for rp in self.raypaths], self.surface.zs.shape)

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
            if self.reflect:
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

        else:

            # --- PLOT FOR REFRACTED ---
            if self.vec:
                dphs = self.refr_dph
                dths = self.refr_dth
            else:
                dphs = np.reshape([(rp.dph) if rp.refracted is not None else None for rp in self.raypaths], (len(self.surface.y), len(self.surface.x)))
                dths = np.reshape([(rp.dth) if rp.refracted is not None else None for rp in self.raypaths], (len(self.surface.y), len(self.surface.x)))

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle('Angle Differences (Refracted)', fontsize=16)

            # Δφ Heatmap
            c1 = axs[0].imshow(dphs, cmap='viridis')
            axs[0].set_title('Δφ')
            axs[0].set_xlabel('Facet Y #')
            axs[0].set_ylabel('Facet X #')
            fig.colorbar(c1, ax=axs[0], fraction=0.046, pad=0.04, label='rad')

            # Δθ Heatmap
            c2 = axs[1].imshow(dths, cmap='twilight', vmin=-np.pi, vmax=np.pi)
            axs[1].set_title('Δθ')
            axs[1].set_xlabel('Facet Y #')
            axs[1].set_ylabel('Facet X #')
            fig.colorbar(c2, ax=axs[1], fraction=0.046, pad=0.04, label='rad')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

            # --- PLOT FOR REFLECTED ---
            if self.reflect:
                if self.vec:
                    dphs = self.refl_dph
                    dths = self.refl_dth
                else:
                    dphs = np.reshape([rp.refl_dph for rp in self.raypaths], self.surface.zs.shape)
                    dths = np.reshape([rp.refl_dth for rp in self.raypaths], self.surface.zs.shape)

                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle('Angle Differences (Reflected)', fontsize=16)

                # Δφ Heatmap
                c1 = axs[0].imshow(dphs, cmap='viridis', vmin=-np.pi/2, vmax=0)
                axs[0].set_title('Δφ')
                axs[0].set_xlabel('Facet Y #')
                axs[0].set_ylabel('Facet X #')
                fig.colorbar(c1, ax=axs[0], fraction=0.046, pad=0.04, label='rad')

                # Δθ Heatmap
                c2 = axs[1].imshow(dths, cmap='twilight', vmin=-np.pi, vmax=np.pi)
                axs[1].set_title('Δθ')
                axs[1].set_xlabel('Facet Y #')
                axs[1].set_ylabel('Facet X #')
                fig.colorbar(c2, ax=axs[1], fraction=0.046, pad=0.04, label='rad')

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()


        
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
    def plot_s2f_rad(self, pltly=False):

        if self.vec:
            tr = self.tr
            re = self.re
        else:
            tr = np.reshape([rp.tr for rp in self.raypaths], self.surface.zs.shape)
            re = np.reshape([rp.re for rp in self.raypaths], self.surface.zs.shape)
        
        if pltly:

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

        else:

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle('Reradiation by Facet', fontsize=16)

            # Reflected Heatmap
            c1 = axs[0].imshow(re, cmap='viridis')
            axs[0].set_title('Reflected')
            axs[0].set_xlabel('Facet Y #')
            axs[0].set_ylabel('Facet X #')
            fig.colorbar(c1, ax=axs[0], fraction=0.046, pad=0.04, label='W')

            # Refracted Heatmap
            c2 = axs[1].imshow(tr, cmap='viridis')
            axs[1].set_title('Refracted')
            axs[1].set_xlabel('Facet Y #')
            axs[1].set_ylabel('Facet X #')
            fig.colorbar(c2, ax=axs[1], fraction=0.046, pad=0.04, label='W')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        
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

        return z, y
        
        
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
