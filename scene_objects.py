import numpy as np
import math
from pyrr import Matrix44, Vector3, Quaternion
import moderngl
from utils import Color, rescale, Mat4, clamp
from splines import Spline, SplinePatch, grid, wave_mesh
import sys

class SceneObject:
    group = []
    e1 = np.array([1,0,0], dtype='f4')
    e2 = np.array([0,1,0], dtype='f4')
    e3 = np.array([0,0,1], dtype='f4')
    data_format = '3f 3f'
    attribute_names = ['in_position', 'in_color']

    def __init__(self, game, frame=None):
        self.group.append(self)
        self.game = game
        self.ctx = game.ctx
        self.program = game.program
        self.render_mode = moderngl.TRIANGLES
        self.verts = None
        self.colors = None

        self._o = Mat4.make_rigid_frame_euler() if frame is None else frame

        self.vel = Vector3()
        self.acc = Vector3()

        self.mass = 1
        self.scale = 1
        self._load()

    @property
    def o(self):
        return self._o

    @o.setter
    def o(self, value):
        self._o = value
    
    def get_axis(self, i):
        return self.o[:3, i]

    def world_transform(self, T):
        """
        Transform local object frame with T,
        with respect to the world frame
        (a shortcut for this common aux transform).
        """
        self.o = T @ self.o

    def transform(self, T):
        """
        Transform local object frame with T
        """
        self.o = self.o @ T

    def aux_transform(self, M, A):
        """
        Apply M to object frame with
        respect to frame A
        """
        M = Mat4.get_transform_in_basis(M, A)
        self.o = M @ self.o
        #Ai = np.linalg.inv(A) # todo: use rigid_inverse
        #ßself.o = A @ M @ Ai @ self.o

    def get_object_matrix(self):
        return self.o

    @property
    def object_matrix_as_array(self):
        return self.get_object_matrix().T.ravel()

    @property
    def pos(self):
        return self.o[:3,3]

    @pos.setter
    def pos(self, v):
        self.o[:3,3] = v

    @property
    def R(self):
        return self.o[:3,:3]

    @R.setter
    def R(self, Rot3):
        self.o[:3,:3] = Rot3
    
    @property
    def up_normal(self):
        return self.o[:3,1] # Y axis of local frame
        #return np.array((0.,1.,0.)) #old way

    def match_normals(self, other):
        R = Mat4.axis_to_axis(self.up_normal, other) # TODO multiply in, dont set?
        self.transform(R)

    def translate(self, v):
        self.pos += v

    def load_mesh(self):
        raise NotImplementedError()

    def handle_input(self, controls):
        pass

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
    maxvel = 1.25

    def __init__(self, game, size=1):
        self.size = size
        self.forward_offset_angle = 0
        self.jumping = False
        self.friction_coefficient = 4.75 # ~4 for 'ice', ~8 for quick stop
        self.mass = 1
        self.y_rot = Mat4.identity()
        self.hover_offset = Mat4.identity()
        self.player = None
        super().__init__(game)
    
    def get_object_matrix(self):
        """
        Compose object matrix with independently stored extra transforms
        """
        # rotate against aux frame (camera basis, cube center)
        #A = Mat4.make_aux(R=self.game.cam.view.T, T=self.o)
        #M = Mat4.get_transform_in_basis(self.y_rot, A)
        #return M @ self.o @ self.hover_offset
        return self.o @ self.y_rot @ self.hover_offset

    def rotate_about_local_up(self, theta):
        #self.transform(Mat4.from_y_rotation(theta))
        self.y_rot = Mat4.from_y_rotation(theta)
        #Rup = Mat4.from_y_rotation(theta)
        #self.R = (self.R @ Rup)# @ self.R.T) @ self.R

    def integrate(self,t,dt):
        # Update velocity with acceleration and apply frictional decay
        friction_force = -self.friction_coefficient * self.vel
        net_acc = self.acc + friction_force / self.mass
        self.vel += net_acc * dt

        # Update position based on velocity
        self.pos += self.vel * dt

    def clamp_x_position_to_surface(self, surface: 'SplineMesh', i=0):
        hs = self.size/2
        xmin, xmax = surface.interval
        xmin += hs
        xmax -= hs
        if self.pos[i] < xmin:
            self.pos[i] = xmin
            self.vel[i] = 0
        elif self.pos[i] > xmax:
            self.pos[i] = xmax
            self.vel[i] = 0

    def handle_input(self, controls):
        k = controls.keys 
        dv = 0.25
        oldvel, oldfric = self.maxvel, self.friction_coefficient
        if controls.was_just_pressed(k.I):
            self.maxvel += dv
        if controls.was_just_pressed(k.K):
            self.maxvel -= dv
        if controls.was_just_pressed(k.J):
            self.friction_coefficient += dv
        if controls.was_just_pressed(k.L):
            self.friction_coefficient -= dv
        if (oldvel, oldfric) != (self.maxvel, self.friction_coefficient):
            print('vel:', self.maxvel, 'fric:', self.friction_coefficient)

    def update(self, t, dt):
        # parameters and constants (here for temporary conveinence)
        hz = 1/4
        w = 2*np.pi*hz
        r = 1.5 + np.cos(w*t/2)
        decay_rate = 0.9
        g = 9.8

        if self.player:
            print('got player')
            if self.player.state.dpright:
                self.vel[0] += self.maxvel 
            if self.player.state.dpleft:
                self.vel[0] -= self.maxvel 
            if self.player.state.dpup:
                self.vel[2] -= self.maxvel 
            if self.player.state.dpdown:
                self.vel[2] += self.maxvel 


        # check for controller input
        if self.game.controls.right:
            self.vel[0] += self.maxvel 
        if self.game.controls.left:
            self.vel[0] -= self.maxvel 
        if self.game.controls.up:
            self.vel[2] -= self.maxvel 
        if self.game.controls.down:
            self.vel[2] += self.maxvel 

        # jumping logic (hack for now)
        if not self.jumping and self.game.controls.space:
            self.jumping = True
            self.vel[1] += self.maxvel/2

        if self.jumping:
            self.vel[1] += -g*dt #self.vel[1]*decay_rate*dt

        # physics step
        self.integrate(t, dt)

        # x-boundary enforcing
        self.clamp_x_position_to_surface(self.game.patch, i=0)
        #self.clamp_x_position_to_surface(self.game.patch, i=2)

        # constrain avatar to surface
        self.stick_to_surface(self.game.patch)

        #self.transform(Mat4.from_translation((0.0,0.25*dt,0.0)))

        self.forward_offset_angle = self.game.controls.cursor[0]*(-math.pi/2)#w*dt*3
        self.rotate_about_local_up(self.forward_offset_angle) #TODO fix for new frame system


    def stick_to_surface(self, surface: 'SplineMesh'):
        x,y,z = self.pos

        surface_point = surface.get_point(x,z)
        surface_normal = surface.get_normal(x,z)
        #print(surface_normal)

        self.ground_point = Vector3(surface_point) #+ Vector3((0,self.size/2 + 0.005,0))
        #self.ground_point = Vector3(surface_point) + surface_normal#+ (self.size/2 + 0.005)*surface_normal
        #self.pos = self.ground_point.copy()
        #self.transform(Mat4.from_translation(surface_normal))
        self.hover_offset[1, 3] = self.size/2 + 0.005


        # trying out direct frame building instead of rotating to match normal
        forward = self.e3
        self.o = Mat4.build_frame(up=surface_normal, forward=forward, origin=self.ground_point)
        #print('before', self.pos)
        #self.transform(Mat4.from_translation((0.0,0.5,0.0)))
        #print('afterT', self.pos)
        #sys.exit()

        #self.match_normals(surface_normal) 


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

    def get_point(self, u, v):
        mn, mx = self.interval
        u = rescale(u, mn, mx, 0, 1)
        v = rescale(v, mn, mx, 0, 1)
        return self.patch.eval_one(u, v)
           
    def get_normal(self, x, z):
        return self.patch.get_normal(*self.rescale(x,z))

    def rescale(self, u, v):
        mn, mx = self.interval
        u = rescale(u, mn, mx, 0, 1)
        v = rescale(v, mn, mx, 0, 1)
        return u, v

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

        wm = wave_mesh(*self.interval, 4, A=6) # 4 x 4 grid over interval
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
    def __init__(self, game, parent=None, frame=None, size=5):
        self.size = size
        self.parent = parent
        super().__init__(game, frame)
        self.render_mode = moderngl.LINES

    @property
    def origin(self):
        return self.pos

    def update(self, t, dt):
        if self.parent:
            self.o = self.parent.o

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