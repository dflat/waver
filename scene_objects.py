import numpy as np
from pyrr import Matrix44, Vector3, Quaternion
import moderngl
from utils import Color, rescale, Mat4
from splines import Spline, SplinePatch, grid, wave_mesh

class SceneObject:
    group = []
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])
    data_format = '3f 3f'
    attribute_names = ['in_position', 'in_color']

    def __init__(self, game):
        self.group.append(self)
        self.game = game
        self.ctx = game.ctx
        self.program = game.program
        self.render_mode = moderngl.TRIANGLES
        self.verts = None
        self.colors = None
        #self._model = Matrix44.identity()
        self.rot = Quaternion()
        self.R = Matrix44.identity()
        self.pos = Vector3()
        self.vel = Vector3()
        self.acc = Vector3()
        self.scale = 1
        self._load()

    @property
    def model(self):
        Tr = Mat4.from_translation(self.pos)
        return (Tr @ self.R) #.T.ravel()

    @property
    def model_to_array(self):
        return self.model.T.ravel()

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, v):
        self._pos = v

    @property
    def up_normal(self):
        return np.array((0.,1.,0.))
        #return self.R[:3,2]
        local_normal = Vector3([0, 1, 0])
        M = Matrix44.from_quaternion(self.rot)
        #print(M)
        v = M * local_normal
        #print(v)
        return v# Vector3((0,1,0))

    def match_normals(self, other):
        self.R = Mat4.axis_to_axis(self.up_normal, other) # TODO multiply in, dont set?
        #self.rot = self.align_normals(self.normal, other) * self.rot

    @classmethod
    def align_normals(cls, normal, other):
        """
        Aligns the normal of this to an axis.
        """
        # Normalize the input normals
        normal = Vector3(normal).normalized
        other = Vector3(other).normalized

        # Compute the cross product to get the axis of rotation
        cross = -np.cross(normal, other) # testing negative here TODO
        cross_norm = np.linalg.norm(cross)
        #print('cross', cross)

        if cross_norm < 1e-6:  # Check if the vectors are parallel
            print('small cross norm')
            if np.dot(normal, other) > 0:  # Same direction, no rotation needed
                return Quaternion()
            else:  # Opposite direction, rotate 180 degrees around any perpendicular axis
                arbitrary_axis = Vector3([1, 0, 0]) if abs(normal[0]) < 1 else Vector3([0, 1, 0])
                axis = Vector3(np.cross(normal, arbitrary_axis)).normalized
                return Quaternion.from_axis_rotation(axis, np.pi)

        # Compute the angle using the dot product
        dot = np.dot(normal, other)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))  # Clip to avoid numerical errors

        # Create a quaternion from the axis and angle
        axis = Vector3(cross).normalized
        return Quaternion.from_axis_rotation(axis, angle)

    def translate(self, v):
        #self.trans = Matrix44.from_translation(v) @ self.trans
        self.pos += v
        #self._model = Matrix44.from_translation(v) @ self.model

    def load_mesh(self):
        raise NotImplementedError()

    def update(self, t, dt):
        pass

    def render(self):
        self.vao.render(self.render_mode)

    def _load(self):
        self.verts, self.colors = self.load_mesh()
        self.vbo = self.ctx.buffer(np.hstack((self.verts, self.colors)).astype('f4').tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, self.data_format, *self.attribute_names)]
        )

class Cube(SceneObject):
    maxvel = 5

    def __init__(self, game, size=1):
        self.size = size
        self.forward_offset_angle = 0
        super().__init__(game)
    
    def rotate_about_local_up(self, theta):
        Rup = Mat4.from_y_rotation(theta)
        self.R = (self.R @ Rup)# @ self.R.T) @ self.R

    @property
    def model(self):
        return super().model
        Tr2 = Mat4.from_translation((0,1,0))
        Tr3 = Mat4.from_translation((0,2,0))
        Tr4 = Mat4.from_translation((0,1/np.sin(-np.pi/4),0))
        Tr = Mat4.from_translation((2,1,4))
        self.R = Mat4.from_x_rotation(-np.pi/4)
        #return (self.R @ Tr2 @ Tr3) #.T.ravel()
        return  (self.R @ Tr4 @ self.R.T) @ Tr2 @ self.R

    def integrate(self,t,dt):
        if self.game.controls.left:
            vx = self.maxvel 
        if self.game.controls.right:
            vz = self.maxvel
        self.vel = Vector3((vx,0,vz))
        pass

    def update(self, t, dt):
        hz = 1/4
        w = 2*np.pi*hz
        r = 1.5 + np.cos(w*t/2)


        if self.game.controls.right:
            self.forward_offset_angle -= w*dt*3
        if self.game.controls.left:
            self.forward_offset_angle += w*dt*3

        self.pos = Vector3((r*np.cos(w*t), 0, r*np.sin(w*t)))

        self.game.patch.stick_to_surface(self)
        self.rotate_about_local_up(self.forward_offset_angle)

    def load_mesh(self):
        """Create vertices and colors for a cube."""
        w = self.size/2
        vertices = np.array([
            # Front face
            [-w, -w,  w],
            [ w, -w,  w],
            [ w,  w,  w],
            [-w,  w,  w],

            # Back face
            [-w, -w, -w],
            [ w, -w, -w],
            [ w,  w, -w],
            [-w,  w, -w],

        ], dtype='f4')

        indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front (z +)
            6, 5, 4, 4, 7, 6,  # Back  (z -)
            5, 1, 0, 0, 4, 5,  # Bottom (y -)
            7, 3, 2, 2, 6, 7,  # Top (y +)
            0, 3, 7, 7, 4, 0,  # Left (x -)
            6, 2, 1, 1, 5, 6,  # Right (x +)
        ], dtype='i4')

        tile = (6,1)
        colors = np.concatenate((
            np.tile(Color.RED,tile),
            np.tile(Color.GREEN,tile),
            np.tile(Color.BLUE,tile),
            np.tile(Color.MAGENTA,tile),
            np.tile(Color.CYAN,tile),
            np.tile(Color.YELLOW,tile),
        ), dtype='f4')

        return vertices[indices], colors 


class SplineMesh(SceneObject):
    def __init__(self, game, interval=(0,3), n_samps=11, origin=np.array([0,0,0], dtype='f4')):
        self.origin = origin
        self.interval = interval
        self.n_samps = n_samps
        super().__init__(game)
        #self.render_mode = moderngl.POINTS
        game.ctx.point_size=5

    def load_mesh_OLD(self):
        o = self.origin
        wm = wave_mesh(*self.interval, 4) # 4 x 4 grid over interval
        sp = SplinePatch(wm)
        self.patch = sp
        ts = np.linspace(0,1,self.n_samps)
        P = sp.eval_vec(ts)
        verts = P.reshape(-1,3) # as points
        return verts, np.tile(Color.GREY, (len(verts),1))

    def eval(self, u, v):
        mn, mx = self.interval
        u = rescale(u, mn, mx, 0, 1)
        v = rescale(v, mn, mx, 0, 1)
        return self.patch.eval_one(u, v)
           
    def rescale(self, u, v):
        mn, mx = self.interval
        u = rescale(u, mn, mx, 0, 1)
        v = rescale(v, mn, mx, 0, 1)
        return u, v

    def stick_to_surface(self, obj):
        x,y,z = obj.pos
        #print('cube pos', obj.pos)
        surface_point = self.eval(x,z)
        #print('surf point', surface_point)
        obj.pos = Vector3(surface_point) + Vector3((0,obj.size/2 + 0.005,0))

        # get normal from discrete diff in x and z directions
        #eps = 0.001
        #pdx = self.eval(x + eps,z) - surface_point
        #pdz = self.eval(x,z + eps) - surface_point
        #normal = Vector3(np.cross(pdz,pdx)).normalized

        normal2 = self.patch.eval_tangent(*self.rescale(x,z)).normalized
        #print(normal, normal2)
        #dir = np.dot(normal2, (0,1,0))
        #if dir < 0:
        #    print('negative normal', dir)

        obj.match_normals(normal2) 

    def load_mesh(self):
        """
        Prepare meshgrid data for rendering with OpenGL, including alternating colors for each triangle pair.

        Parameters:
            V: numpy array of shape (3, n, n) representing the meshgrid vertices.

        Returns:
            vertices: Flattened array of vertex positions (numpy array of shape (n*n, 3)).
            indices: Flattened array of triangle indices for rendering (numpy array of shape (m, 3)).
            colors: Array of color values for each triangle (numpy array of shape (m, 3), where each row is [r, g, b]).
        """
        # Prepare indices and colors

        wm = wave_mesh(*self.interval, 4, A=2) # 4 x 4 grid over interval
        sp = SplinePatch(wm)
        self.patch = sp
        ts = np.linspace(0,1,self.n_samps)
        P = sp.eval_vec(ts)
        vertices = P.reshape(-1,3) # as points

        indices = []
        colors = []
        c1 = [Color.WHITE]*3
        c2 = [Color.LIGHTGREY]*3
        n = self.n_samps
        for i in range(n - 1):
            for j in range(n - 1):
                # Calculate vertex indices for the two triangles of the current cell
                top_left = i * n + j
                top_right = top_left + 1
                bottom_left = (i + 1) * n + j
                bottom_right = bottom_left + 1

                # First triangle: top-left, bottom-left, top-right
                indices.append([top_left, bottom_left, top_right])
                colors.extend(c1)  # White color for the first triangle

                # Second triangle: top-right, bottom-left, bottom-right
                indices.append([top_right, bottom_left, bottom_right])
                colors.extend(c2)  # Gray color for the second triangle

        # Convert to numpy arrays
        indices = np.array(indices, dtype=np.uint32).flatten()  # Shape becomes (m,)
        colors = np.array(colors, dtype=np.float32)  # Shape becomes (m, 3)

        return vertices[indices], colors


class Grid(SceneObject):
    data_format = '3f 3f'
    attribute_names = ['in_position', 'in_color']
    def __init__(self, game, origin=np.array([0,0,0], dtype='f4'),
                 x_range=(-5,5), y_range=(-5,5), x_points=11, y_points=11, unit=1):
        self.origin = origin
        self.unit = unit
        self.x_range = x_range
        self.y_range = y_range
        self.x_points = x_points
        self.y_points = y_points
        super().__init__(game)
        self.render_mode = moderngl.LINES
        #self.render_mode = moderngl.POINTS

    def load_mesh(self):
        o = self.origin
        u = self.unit
        scalar = self.unit*(self.x_range[1]-self.x_range[0])/(self.x_points-1)

        X, Y = grid(self.x_range, self.y_range, self.x_points, self.y_points)
        points = np.dstack((X, np.zeros_like(X), Y)).reshape(-1,3)

        vertices = []

        # Create lines along the rows (constant y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1] - 1):
                vertices.append([X[i, j], 0.0, Y[i, j]])
                vertices.append([X[i, j + 1], 0.0, Y[i, j + 1]])

        # Create lines along the columns (constant x)
        for j in range(X.shape[1]):
            for i in range(X.shape[0] - 1):
                vertices.append([X[i, j], 0.0, Y[i, j]])
                vertices.append([X[i + 1, j], 0.0, Y[i + 1, j]])

        verts = np.array(vertices, dtype=np.float32)
        verts *= scalar
        verts += [0,.0001,0]
        return verts, np.tile(Color.GREY, (len(verts),1))


class Axes(SceneObject):
    def __init__(self, game, origin=np.array([0,0,0], dtype='f4'), size=5):
        self.origin = origin
        self.size = size
        super().__init__(game)
        self.render_mode = moderngl.LINES

    def load_mesh(self):
        o = self.origin
        s = self.size

        verts = np.array([
            o, o + s*self.e1,
            o, o + s*self.e2,
            o, o + s*self.e3
        ], dtype='f4')

        colors = np.array([
            Color.RED, Color.RED,
            Color.GREEN, Color.GREEN,
            Color.BLUE, Color.BLUE
        ])

        return verts, colors