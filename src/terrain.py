import numpy as np
import matplotlib.pyplot as plt

from surface import Surface
from util import simple_gaussian

class Terrain():

    def __init__(self, xmin, xmax, ymin, ymax, fs, overlap=0):

        self.bounds = (xmin, xmax, ymin, ymax)
        self.fs = fs

        self.xs = np.arange(xmin, xmax, fs)
        self.ys = np.arange(ymin, ymax, fs)

        self.zs = np.zeros((len(self.ys), len(self.xs)))

        self.XX, self.YY = np.meshgrid(self.xs, self.ys)

        self.overlap = overlap
        self.offset = 1 - overlap


    def gen_normals(self):
        
        # first find delta z along x axis
        dz_dy = (np.roll(self.zs, 1, axis=0) - self.zs) / (self.fs * self.offset)
        dz_dy[0,:] = dz_dy[1,:]
        
        # find delta z along y axis
        dz_dx = (np.roll(self.zs, 1, axis=1) - self.zs) / (self.fs * self.offset)
        dz_dx[:,0] = dz_dx[:,1]
        
        # compute facet normals
        gradient_x = np.array([np.ones_like(dz_dx), np.zeros_like(dz_dx), dz_dx])
        gradient_y = np.array([np.zeros_like(dz_dy), np.ones_like(dz_dy), dz_dy])
        
        normals = np.cross(gradient_x, gradient_y, axis=0)

        norm = np.linalg.norm(normals, axis=0)
        normals /= norm
        
        self.normals = np.transpose(normals, (1, 2, 0))

        self.uvectors = np.zeros_like(self.normals)
        self.uvectors[:, :, 0] = 1
        self.uvectors[:, :, 2] = dz_dx
        norm = np.linalg.norm(self.uvectors, axis=2)
        self.uvectors /= norm[:, :, np.newaxis]

        self.vvectors = np.zeros_like(self.normals)
        self.vvectors[:, :, 1] = 1
        self.vvectors[:, :, 2] = dz_dy
        norm = np.linalg.norm(self.vvectors, axis=2)
        self.vvectors /= norm[:, :, np.newaxis]

    
    def gen_flat(self, z, normal=(0, 0, 1)):

        self.zs += z
        
        self.gen_normals()


    def sinusoid(self, amp, axis, period, z):

        if axis == 'x' or axis == 1:
            self.zs = amp * np.sin((self.XX * np.pi)/period) + z
        elif axis == 'y' or axis == 0:
            self.zs = amp * np.sin((self.YY * np.pi)/period) + z

        self.gen_normals()

    
    def double_ridge(self, amp1, amp2, peak_dist, ridge_width, center, axis='x'):

        if axis != 'x':

            raise NotImplementedError()

        cen1 = center - peak_dist / 2
        cen2 = center + peak_dist / 2

        sig = ridge_width / 2.5

        self.zs = simple_gaussian(self.XX, amp1, sig, xcen=cen1) + \
                  simple_gaussian(self.XX, amp2, sig, xcen=cen2)
        
        self.gen_normals()

    
    def get_surf(self, center, dims):

        # turn the center coordinate into index
        ix = (center[0] - self.bounds[0]) // self.fs
        iy = (center[1] - self.bounds[2]) // self.fs

        # get the bounds of the dem
        ixmin = int(ix - dims[0] // 2)
        ixmax = int(ixmin + dims[0])

        iymin = int(iy - dims[1] // 2)
        iymax = int(iymin + dims[1])

        # windowed facets
        win_zs = self.zs[iymin:iymax, ixmin:ixmax]

        # windowed normals
        win_ns = self.normals[iymin:iymax, ixmin:ixmax, :]

        # move into surface object
        xorigin = center[0] - self.fs * (dims[0] / 2)
        yorigin = center[1] - self.fs * (dims[1] / 2)
        surf = Surface(fs=self.fs, origin=(xorigin, yorigin), dims=dims, overlap=self.overlap)
        surf.zs_from_arr(win_zs, win_ns)

        return surf


    def show_2d_heatmap(self, ss=None, t=None, savefig=None, show=True):

        fig, ax = plt.subplots()

        c = ax.pcolormesh(self.XX, self.YY, self.zs, cmap='viridis', shading='auto')
        fig.colorbar(c, ax=ax, label='Z', shrink=0.4)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        if t is not None:
            plt.scatter(t[0], t[1], marker="+", color="black", s=5)

        if ss is not None:
            sxs = [s.coord[0] for s in ss]
            sys = [s.coord[1] for s in ss]
            plt.plot(sxs, sys, linestyle=":", color="red")

        plt.gca().set_aspect('equal')
        plt.tight_layout()

        if savefig:
            plt.savefig(savefig, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()


    def show_profile(self, axis, val, savefig=None, show=False, shape=(4, 3)):

        if axis == 'x':

            i = np.argmin(np.abs(self.ys - val))
            prof = self.zs[i, :]
            scale = self.xs

        elif axis == 'y':

            i = np.argmin(np.abs(self.xs - val))
            prof = self.zs[:, i]
            scale = self.ys

        fig, ax = plt.subplots(1, 1, figsize=shape)

        ax.plot(scale/1e3, prof, color="black", linewidth=1)
        ax.set_xlabel("Profile distance [km]")
        ax.set_ylabel("Height [m]")
        plt.grid()

        if savefig:
            plt.savefig(savefig, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()
