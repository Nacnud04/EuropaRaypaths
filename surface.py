import numpy as np
import plotly.graph_objects as go

class Surface():

    def __init__(self, fs=10, origin=(0, 0), dims=(10, 10)):

        self.fs = fs # facet size in m
        self.origin = origin # origin. defined as minx and miny
        self.dims = dims # (x, y) dimension in terms of facets
        
        self.zs = np.zeros(self.dims)
        self.normals = np.zeros((dims[0], dims[1], 3))

        # generate x and y spaces for surface mesh
        self.x = np.linspace(origin[0], origin[0] + dims[0] * self.fs, dims[0])
        self.y = np.linspace(origin[1], origin[1] + dims[1] * self.fs, dims[1])
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        
    def gen_normals(self):
        
        # first find delta z along x axis
        dz_dx = (np.roll(self.zs, 1, axis=0) - self.zs) / self.fs
        dz_dx[0,:] = dz_dx[1,:]
        
        # find delta z along y axis
        dz_dy = (np.roll(self.zs, 1, axis=1) - self.zs) / self.fs
        dz_dy[:,0] = dz_dy[:,1]
        
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
        

    # add surface to 3D plot of model axis
    def add_to_axis(self, fig):
        surface = go.Surface(x=self.X, y=self.Y, z=self.zs, colorscale='Viridis')
        fig.add_trace(surface)
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ))
        return fig
    
    # make 3D plot of surface
    def show_surf(self):
        fig = go.Figure(data=[go.Surface(x=self.X, y=self.Y, z=self.zs, colorscale='Viridis')])
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ))
        fig.show()

        