import numpy as np
import plotly.graph_objects as go

from util import *

class RayPaths():

    def __init__(self, source, facetloc, target):
        
        self.c = 299792458 # speed of light

        # source & target coordinates
        self.start = source
        self.end   = target
        
        # facet location
        fx, fy, fz = facetloc
        self.fx, self.fy, self.fz = fx, fy, fz

        # comp raypaths
        vec1 = np.array([fx - source[0], fy - source[1], fz - source[2]])
        vec2 = np.array([target[0] - fx, target[1] - fy, target[2] - fz])
        self.vecs = (vec1, vec2)
        self.mags = (np.linalg.norm(vec1), np.linalg.norm(vec2))
        self.norms = (vec1/self.mags[0], vec2/self.mags[1])
        
        # angles, transmission coefficient and whatnot
        self.tr = None # how much energy makes it through the raypath??
        self.re = None # how much energy reflects off the surface?
        self.th1, self.th2 = None, None
        self.dth = None
        
    # find the equivalent angle from -90 to 90 instead of -180 to 180
    @staticmethod
    def ang_lim(a):
        if a > np.pi / 2: a = a - np.pi
        if a < -np.pi / 2: a = a + np.pi
        return a
        
    # find the angle between two vectors in a 3D space
    @staticmethod
    def vec_dif_angle(u, v):

        dot_product = np.dot(u, v)

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        cos_theta = dot_product / (norm_u * norm_v)

        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  

        return RayPaths.ang_lim(angle)
    
    # set source location and capture wavelength
    def set_source(self, source):
        self.ss = source
        self.lam = self.c / source.f0
        
    # set facet
    def set_facet(self, facet_norm, fs):
        
        self.fnorm = facet_norm
        
    # compute the refracted ray angle
    # NOTE: this is NOT the forced ray (ray to target), this is
    # the ray which follows the path of greatest radiation
    def comp_refracted(self, vel1, vel2):
        
        # compute path time for whole raypath
        self.refl_time = (self.mags[0] / vel1) * 2
        self.path_time = (self.mags[0] / vel1 + self.mags[1] / vel2) * 2
        
        # compute the spherical coordinate for the facet normal and inbound ray
        fspher = cart_to_sp(self.fnorm)
        inbound = cart_to_sp(self.norms[0])
        
        # compute a new coordinate for inbound ray, relative to facet normal as origin
        inbound -= fspher
        
        # snells law to find the new phi value (inclincation relative to facet)
        k = (vel2 / vel1) * np.sin(inbound[2])
        
        if abs(k) < 1:
            phi = np.arcsin(k) + np.pi
            
            return np.array((inbound[0], inbound[1], phi))
        
        return None
    
    # compute the reverse refracted ray angle
    # NOTE: this is NOT the forced ray (facet to source), this is
    # the ray which follows the path of greatest radiation
    def comp_rev_refracted(self, vel1, vel2):
        
        # compute the spherical coordinate for the facet normal and inbound ray
        fspher = cart_to_sp(self.fnorm*-1)
        inbound = cart_to_sp(self.norms[1]*-1)
        
        # compute a new coordinate for inbound ray, relative to facet normal as origin
        inbound -= fspher
        
        # snells law to find the new phi value (inclincation relative to facet)
        k = (vel1 / vel2) * np.sin(inbound[2])
        
        if abs(k) < 1:
            phi = np.arcsin(k) + np.pi
            
            return np.array((inbound[0], inbound[1], phi))
        
        return None

