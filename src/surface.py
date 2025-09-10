import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Surface():

    def __init__(self, fs=10, origin=(0, 0), dims=(10, 10), overlap=0, verb=False):

        self.fs = fs # facet size in m
        self.origin = origin # origin. defined as minx and miny
        self.dims = dims # (x, y) dimension in terms of facets
        
        self.zs = np.zeros((dims[1], dims[0]))
        self.normals = np.zeros((dims[1], dims[0], 3))

        # how much is each adjacent facet offset?
        # if overlap = 0 then the facets do not overlap.
        # if overlap = 0.5 then the ajacent facet covers half the previous facet
        self.offset = 1 - overlap

        if verb:
            print(f"Facet offset: {self.offset} m")

        # x and y limits
        self.xmin = origin[0]
        self.ymin = origin[1]
        self.xmax = origin[0] + fs * dims[0] * self.offset
        self.ymax = origin[1] + fs * dims[1] * self.offset
        self.xcenter = (self.xmax + self.xmin) / 2
        self.ycenter = (self.ymax + self.ymin) / 2

        if verb:
            print(f"X range: {self.xmin} m, {self.xmax} m \nY range: {self.ymin} m, {self.ymax} m")

        # generate x and y spaces for surface mesh
        self.x = np.linspace(origin[0], self.xmax, dims[0])
        self.y = np.linspace(origin[1], self.ymax, dims[1])

        # turn into mesh
        self.X, self.Y = np.meshgrid(self.x, self.y)    

        # surface roughness
        # derived from Steinbrugge et al:
        # https://www.sciencedirect.com/science/article/pii/S0019103519301526
        self.s = 0.4 # rms height [m]
        
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

        
    # generate a perfectly flat surface
    def gen_flat(self, z, normal=(0, 0, 1)):

        self.zs += z
        
        self.gen_normals()
        
    def gen_sin(self, axis, amp, period, z):
        
        if axis == 'x':
            self.zs = amp * np.sin((self.X * np.pi)/period) + z
        elif axis == 'y':
            self.zs = amp * np.sin((self.Y * np.pi)/period) + z
            
        self.gen_normals()
        
    def arr_along_axis(self, arr, axis=0):

        if axis==0:
            for i in range(self.zs.shape[0]):
                self.zs[:, i] = arr
        elif axis == 1:
            for i in range(self.zs.shape[1]):
                self.zs[i, :] = arr

        self.gen_normals()

    def zs_from_arr(self, zs, normals):

        self.zs = zs
        self.normals = normals

        if zs.shape[0] != self.X.shape[0] or zs.shape[1] != self.X.shape[1]:
            raise IndexError(f"Surface failed to succesfully generate. Provided array \
of dimensions {self.zs.shape} does not match surface \
dimensions of {self.X.shape}")
    
    # make 3D plot of surface
    def show_surf(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(self.X, self.Y, self.zs, cmap='viridis')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        plt.show()


    def show_2d_heatmap(self, ss=None, t=None, savefig=None, show=True):

        fig, ax = plt.subplots(figsize=(12, 4))

        c = ax.pcolormesh(self.X, self.Y, self.zs, cmap='viridis', shading='auto')
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

        if savefig:
            plt.savefig(savefig)

        if show:
            plt.show()
        else:
            plt.close()


    def show_normals(self, ss=None, t=None, show=True):
        if not hasattr(self, "normals"):
            raise AttributeError("Run gen_normals() first to compute self.normals")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        components = ["X-component", "Y-component", "Z-component"]
        for i, ax in enumerate(axes):
            im = ax.imshow(self.normals[:, :, i], cmap="coolwarm", origin="lower")
            ax.set_title(components[i])
            fig.colorbar(im, ax=ax, shrink=0.7)
        
        if t is not None:
            plt.scatter(t[0], t[1], marker="+", color="black", s=5)

        if ss is not None:
            sxs = [s.coord[0] for s in ss]
            sys = [s.coord[1] for s in ss]
            plt.plot(sxs, sys, linestyle=":", color="red")

        plt.suptitle("Normal Vector Components")
        plt.tight_layout()
        plt.gca().set_aspect('equal')

        if show:
            plt.show()
        else:
            plt.close()

    def get_profile(self, axis=0, offset=0, save=None, show=True):

        if axis == 0:

            i = np.argmin(np.abs(self.y - offset))
            profile = self.zs[:, i]
            axis = self.y

        elif axis == 1:

            i = np.argmin(np.abs(self.x - offset))
            profile = self.zs[i, :]
            axis = self.x

        plt.plot(axis, profile, color="black", linewidth=1)
        plt.xlabel("Profile distance [m]")
        plt.ylabel("Height [m]")

        if save:
            plt.savefig(save)

        if show:
            plt.show()
        else:
            plt.close()

    def reduce_by_aperture(self, aperture, alt):

        """
        CRITICAL: THIS ASSUMES THAT THE FACETED SURFACE IS AT NADIR WHICH IS INVALID FOR THE REFRACTED SURFACE
        """

        radius   = np.sin(np.radians(aperture)) * alt
        n_facets = radius // self.fs 

        half_x   = self.dims[0] // 2
        half_y   = self.dims[1] // 2

        self.X = self.X[int(half_x - n_facets):int(half_x + n_facets + 1),
                        int(half_y - n_facets):int(half_y + n_facets + 1)]
        self.Y = self.Y[int(half_x - n_facets):int(half_x + n_facets + 1),
                        int(half_y - n_facets):int(half_y + n_facets + 1)]
        
        self.zs = self.zs[int(half_x - n_facets):int(half_x + n_facets + 1),
                        int(half_y - n_facets):int(half_y + n_facets + 1)]

        self.normals = self.normals[int(half_x - n_facets):int(half_x + n_facets + 1),
                        int(half_y - n_facets):int(half_y + n_facets + 1)]