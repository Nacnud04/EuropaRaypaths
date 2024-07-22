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

    # generate a perfectly flat surface
    def gen_flat(self, z, normal=(0, 0, 1)):

        self.zs += z
        self.normals[:, :, 0] = normal[0]
        self.normals[:, :, 1] = normal[1]
        self.normals[:, :, 2] = normal[2]

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

        