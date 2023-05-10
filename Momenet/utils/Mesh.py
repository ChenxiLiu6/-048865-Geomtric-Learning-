import meshio
import numpy as np
import pyvista as pv
import plotly.graph_objects as go


def np2pv(v, f):
    '''
    numpy-to-pyvista convertor for the private case of triangular meshes.
    '''
    return pv.PolyData(v, np.concatenate((np.full((len(f), 1), 3), f), 1)) # add a column for n_nodes=3

class Mesh:
    def __init__(self, fpath):
        mesh = meshio.read(fpath)
        self.v = mesh.points
        self.f = mesh.cells[0][1]
        
    def render_wireframe(self, notebook=False, show_faces=False, **kwargs):
        pp = np2pv(self.v, self.f)
        pp.plot(show_edges=True, culling=False if show_faces else 'front', notebook=notebook, **kwargs)
        return pp
    
    def render_pointcloud(self, vertex=None, col_fun=None, vectoric_fun=True, col_lab='color', notebook=False, **kwargs):
        # get colors
        if col_fun is not None:
            if isinstance(col_fun, np.ndarray):
                col = col_fun
            else:
                if vectoric_fun:
                    col = col_fun(self.v)
                else:
                    col = np.array([col_fun(v) for v in self.v])
        
        # render
        if vertex is not None:
            pp = pv.PolyData(vertex)
        else:
            pp = pv.PolyData(self.v)
        if col_fun is not None:
            pp[col_lab] = col
        pp.plot(render_points_as_spheres=True, notebook=notebook, **kwargs)
        return pp
    
    def render_surface(self, col_fun=None, apply_to_faces=None, vectoric_fun=True, show_edges=True, **kwargs):
        # get colors
        if col_fun is None:
            col = np.random.random(len(self.f))
        else:
            if isinstance(col_fun, (np.ndarray,list)):
                apply_to_faces = len(col_fun)==len(self.f) if apply_to_faces is None else apply_to_faces
                if apply_to_faces:
                    col = np.array(col_fun)
                else:
                    col_v = np.array(col_fun)
                    col = [np.mean(col_v[f]) for f in self.f]
            else:
                if vectoric_fun:
                    col = col_fun(self.f)
                else:
                    col = np.array([col_fun(f) for f in self.f])
        
        # render
        pp = np2pv(self.v, self.f)
        pp.plot(scalars=col, show_edges=show_edges, **kwargs)
        return pp

    def plot_arrows(self, centers, directions, faces_col=None, label='color'):
        pp = pv.Plotter()
        m = np2pv(self.v, self.f)
        if faces_col is not None:
            m[label] = faces_col
        #pp.add_mesh(m, show_edges=True)
        pp.add_arrows(centers, directions)
        pp.show()