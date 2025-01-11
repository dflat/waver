import numpy as np
import math
from glm import vec3
import glm
import moderngl
from utils import Color, rescale, Mat4, clamp, project_onto_axis
from splines import Spline, SplinePatch, grid, wave_mesh
import sys

PI = glm.pi()

class SceneObject:
    group = []
    e1 = vec3(1,0,0)
    e2 = vec3(0,1,0)
    e3 = vec3(0,0,1)
    up = vec3(e2)
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

        self.vel = glm.vec3()
        self.acc = glm.vec3()

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
        return self.o[i]

    def get_forward(self):
        return self.get_axis(2)

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

    def get_object_matrix(self):
        return self.o

    def set_basis(self, b1, b2, b3):
        self.o[0].xyz = b1.xyz
        self.o[1].xyz = b2.xyz
        self.o[2].xyz = b3.xyz

    @property
    def object_matrix_as_bytes(self):
        return self.get_object_matrix().to_bytes()

    @property
    def pos(self) -> glm.vec3:
        return self.o[3]

    @pos.setter
    def pos(self, v):
        self.o[3] = v
        #self.o[3].xyz = v

    @property
    def R(self):
        return glm.mat3(self.o)

    @R.setter
    def R(self, Rot3):
        self.o[0].xyz = Rot3[0] 
        self.o[1].xyz = Rot3[1] 
        self.o[2].xyz = Rot3[2] 
    
    @property
    def up_normal(self):
        return glm.vec3(self.o[1]) # Y axis of local frame

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
    maxspeed = 8
    max_angular_speed = 5

    def __init__(self, game, size=1):
        self.size = size
        self.forward_offset_angle = 0
        self.jumping = False
        self.friction_coefficient = 4.75 # ~4 for 'ice', ~8 for quick stop
        self.mass = 1
        self.y_rot = Mat4.identity()
        self.dir_rot = Mat4.identity()
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

    def integrate(self,t,dt):
        # Update velocity with acceleration and apply frictional decay
        friction_force = -self.friction_coefficient * self.vel
        net_acc = self.acc + friction_force / self.mass
        self.vel += net_acc * dt

        # Update position based on velocity
        self.pos.xyz += self.vel * dt

    def clamp_x_position_to_surface(self, surface: 'SplineMesh', i=0):
        hs = self.size/2
        xmin, xmax = surface.interval
        xmin += hs
        xmax -= hs
        #print(xmin, xmax, self.pos)
        if self.pos[i] < xmin:
            self.pos[i] = xmin
            #self.vel[i] = 0
        elif self.pos[i] > xmax:
            self.pos[i] = xmax
            #self.vel[i] = 0

    def link_to_gamepad(self, pad):
        self.player = pad

    def unlink_gamepad(self):
        self.player = None

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

    def get_cam_relative_velocity(self, vdir, norm):
        # vdir is the direction of linear velocity from 
        # user input. Calculate rotation by projecting
        # into null space of world up vector

        # WANT: orientation to track linear velocity's (in cam space) direction
        # this direction is calculated and called 'cam_relative_vel'

        #if norm < .25:
        #    return 0
        cam_forward = self.game.cam.get_forward() # World(cam_z) (out from monitor)
        cam_forward_xz = cam_forward - project_onto_axis(self.up, cam_forward)
        cam_forward_xz = glm.normalize(cam_forward_xz)
        cam_right_xz = Mat4.cross(self.up, cam_forward_xz) #

        # basis in xz plane based on camera's forward direction
        # used to take combinations by input vector for camera-relative control
        cam_basis_xz = Mat4.concat_basis(cam_right_xz, self.up, cam_forward_xz)
        cam_relative_vel = cam_basis_xz @ vdir

        #print(cam_forward, cam_forward_xz, cam_right_xz, cam_relative_vel)
        #vcam = np.array(self.game.cam.view[:3,:3] @ vdir)
        #vcamxz = vcam - project_onto_axis(self.up, vcam)
        #assert np.allclose(cam_relative_vel, vcamxz)
        #print('cam_rel:', cam_relative_vel.round(3), 'vcamxz:', vcamxz.round(3))
        #vcamxz /= np.linalg.norm(vcamxz)
        #cam_relative_vel = vcamxz
        # note: cam_relative_vel seems more correct than vcamxz
        self.game.camxzAxes.o = Mat4.from_basis(cam_basis_xz, origin=(0,1,0))


        if norm < .25:
            return glm.vec3(0)

        #print('cam forward:', np.round(cam_forward_xz,2), end=' -- ')
        #print('cam right:', np.round(cam_right_xz,2), end=' -- ')
        #print('input vel:', np.round(vdir,2), end=' -- ')
        #print('cam rel vel:', np.round(cam_relative_vel,2))


        # old way
        ##print(cam_forward_xz)
        cos_angle = glm.dot(cam_forward_xz, cam_relative_vel)
        sin_angle = Mat4.cross(cam_forward_xz, cam_relative_vel)[1]
        angle_of_divergence = math.atan2(sin_angle,cos_angle)
        #self.rotate_about_local_up(angle_of_divergence)
        #print(f"angle: {angle_of_divergence:.1f}")

        return cam_relative_vel


    def update(self, t, dt):
        # parameters and constants (here for temporary conveinence)
        decay_rate = 0.9
        g = 9.8*6

        dpadEnabled = True
        if self.player:
            vdir, norm = self.player.leftaxis #TODO should vdir be normalized here

            if dpadEnabled:
                if self.player.state.dpright or self.game.controls.right:
                    vdir[0] += 1
                    norm = glm.length(vdir)
                if self.player.state.dpleft or self.game.controls.left:
                    vdir[0] -= 1
                    norm = glm.length(vdir)
                if self.player.state.dpup or self.game.controls.up:
                    vdir[2] -= 1
                    norm = glm.length(vdir)
                if self.player.state.dpdown or self.game.controls.down:
                    vdir[2] += 1
                    norm = glm.length(vdir)

            # handle player's looking direction
            # match to velocity direction

            # vdir is now cam relative velocity direction
            vdir = self.get_cam_relative_velocity(vdir, norm)

            self.vel += vdir*norm*self.maxvel
            speed = glm.length(self.vel.xz)
            if speed > self.maxspeed:
                v = glm.normalize(self.vel.xz)
                self.vel.xz = v*self.maxspeed

        # get player's current position
        surface = self.game.patch
        x,y,z = self.pos.xyz
        self.surface_point = glm.vec3(surface.get_point(x,z))

        # jumping logic (hack for now)
        jumpvel = 30
        if not self.jumping and (self.player and self.player.just_pressed('a')):
             self.jumping = True
             self.vel[1] += jumpvel
             print('jump start, pos:', self.pos.y)

        if self.jumping:
             self.vel[1] += -g*dt #self.vel[1]*decay_rate*dt
             if self.pos.y + self.vel[1]*dt < self.surface_point.y:# + dt*g:# + self.size/2:
                #self.pos.y = self.surface_point.y #0
                self.jumping = False
                print('jump end')

        # physics step
        self.integrate(t, dt)

        # x-boundary enforcing
        self.clamp_x_position_to_surface(self.game.patch, i=0)
        self.clamp_x_position_to_surface(self.game.patch, i=2)

        # constrain avatar to surface
        #self.stick_to_surface(self.game.patch)
        self.stick_to_surface(self.game.patch)

        # rotate cube's object matrix to match cam_rel_vel
        # after stick to surface forward is 'undefined' and not useful,
        # calculated as an artifact based on the geometry of the surface
        # it is moving upon

        # defunct
        if self.player and False:
            if glm.length(vdir) > 0:
                vdir = glm.normalize(vdir).xyz
                f = self.get_forward().xyz
                f = glm.normalize(vec3(f.x,0,f.z))
                print(vdir, f)
                q = glm.quat(f, vdir)
                R = glm.mat4(q) # rotation 
                #self.transform(R)
                #print(R, glm.length(q))
                #self.y_rot=R
                #self.dir_rot = R
                #self.transform(glm.mat4(2,2,2,1))

        #self.transform(self.dir_rot)
        #self.transform(Mat4.from_translation((0.0,0.25*dt,0.0)))

        #self.forward_offset_angle = self.game.controls.cursor[0]*(-math.pi/2)#w*dt*3
        #self.rotate_about_local_up(self.forward_offset_angle) #TODO fix for new frame system


    def stick_to_surface(self, surface:'SplineMesh'):
        x,y,z = self.pos.xyz
        surface_point = glm.vec3(surface.get_point(x,z))
        surface_normal = glm.vec3(surface.get_normal(x,z))

        self.hover_offset[3, 1] = self.size/2 + 0.005*10 # applied at end of frame

        # set rotation basis to match surface normal and velocity direction
        b2 = surface_normal

        #vnorm = glm.length(self.vel)
        #b3 = self.vel/vnorm if vnorm > 0 else self.get_forward().xyz

        vnorm = glm.length(self.vel.xz)
        vel_xz = glm.vec3(self.vel.x, 0, self.vel.z)
        b3 = vel_xz/vnorm if vnorm > 0 else self.get_forward().xyz

        if glm.length(self.vel.xz) > 0:
            basis = Mat4.build_basis(up=b2, forward=b3)
            self.set_basis(*basis) #basis[0],basis[1],basis[2])

        # lock position to surface when player is on the ground
        if not self.jumping:
            self.pos.xyz = surface_point

        #R = Mat4.axis_to_axis(self.up_normal, surface_normal)
        #self.transform(deltaTrans*R) # TODO offset here (with frame up_normal) (instead of world)

    def stick_to_surface_delta(self, surface:'SplineMesh'):
        x,y,z = self.pos.xyz
        surface_point = glm.vec3(surface.get_point(x,z))
        surface_normal = glm.vec3(surface.get_normal(x,z))

        deltaPos = surface_point - self.pos.xyz
        print('deltaPos:', deltaPos)
        deltaTrans = glm.translate(deltaPos)
        self.hover_offset[3, 1] = self.size/2 + 0.005*10 # applied at end of frame

        epsilon = 1e-6
        R = glm.mat4()
        close = glm.length(glm.cross(self.up_normal, surface_normal))
        print(f'diff normal: {close:.4f}')
        if close > epsilon:
            RmatchNormal = Mat4.axis_to_axis(self.up_normal, surface_normal)
            R = R * RmatchNormal

        # rotate the current forward vector by the match normal matrix
        forward = self.get_forward().xyz
        altered_forward = R * forward

        # get the current velocity direction, but projected onto perp of surface normal
        # in case of 0 velocity, fall back to using the current forward vector

        vnorm = glm.length(self.vel)
        b3_in_xz = self.vel/vnorm if vnorm > 0 else altered_forward
        b3 = b3_in_xz - project_onto_axis(surface_normal, b3_in_xz)
        b3 = glm.normalize(b3)
        close = glm.length(glm.cross(forward, b3))
        print(f'diff forward: {close:.4f}')
        if close > epsilon:
            RmatchForward = Mat4.axis_to_axis(altered_forward, b3)
            R = R @ RmatchForward
        self.transform(deltaTrans)
        self.transform(R) # TODO offset here (with frame up_normal) (instead of world)

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
    def __init__(self, game, interval=(0,3), n_samps=11, origin=glm.vec3()):
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

        wm = wave_mesh(*self.interval, 4, A=1) # 4 x 4 grid over interval
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
    def __init__(self, game, parent=None, frame=None, size=5, aux_origin=None):
        self.size = size
        self.parent = parent
        self.aux_origin = aux_origin
        super().__init__(game, frame)
        self.render_mode = moderngl.LINES

    @property
    def origin(self):
        return glm.vec3(self.pos)

    def update(self, t, dt):
        if self.parent:
            self.o = self.parent.o
        if self.aux_origin is not None:
            self.pos = self.aux_origin

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