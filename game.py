import moderngl
import numpy as np
import math
from pyrr import Matrix44, Vector3
import moderngl_window as mglw
from pathlib import Path

class Color:
    RED = np.array([1,0,0])
    GREEN = np.array([0,1,0])
    BLUE = np.array([0,0,1])
    MAGENTA = np.array([1,0,1])
    CYAN = np.array([0,1,1])
    YELLOW = np.array([1,1,0])
    GREY = np.array([.4,.4,.4])

class Camera:
    def __init__(self):
        self.azimuth = math.pi/4 # looking down center of +xz
        self.altitude = math.pi/2 # Level with xz-plane
        self.dtheta = 0.00025
        self.radius = 6.0
        #self.elevation = 2
        self.target = Vector3([0.0, 0.0, 0.0])
        self.pos = Vector3([self.radius, 0.0, 0.0])
        self.up = Vector3([0.0, 1.0, 0.0])
        self.view = self.get_view_matrix()

    def orbit(self, u, v, r=5):
        x = np.sin(v) * np.cos(u)
        z = np.sin(v) * np.sin(u)
        y = np.cos(v)
        return r*Vector3([x,y,z])

    def update(self):
        # Update camera azimuth
        self.azimuth += self.dtheta
        if self.azimuth > 2 * math.pi:
            self.azimuth -= 2 * math.pi

        #self.pos = Vector3([
        #    self.radius * np.cos(self.azimuth),
        #    self.elevation,  
            #self.radius * np.sin(self.azimuth)
        #])
        self.pos = self.orbit(self.azimuth, self.altitude, self.radius)
        #print('pos',self.pos)

        self.view = self.get_view_matrix()

    def get_view_matrix(self):
        """Calculates the view matrix for a camera orbiting the cube."""
        return Matrix44.look_at(self.pos, self.target, self.up)

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
        self._model = Matrix44.identity()
        self.rot = Matrix44.identity()
        self.trans = Matrix44.identity()
        self.scale = Matrix44.identity()
        self._load()

    @property
    def model(self):
        return self._model
        #return self.scale@self.rot@self.trans

    def situate(self, v):
        self.trans = Matrix44.from_translation(v)

    def translate(self, v):
        #self.trans = Matrix44.from_translation(v) @ self.trans
        self._model = Matrix44.from_translation(v) @ self.model

    def load_mesh(self):
        raise NotImplementedError()

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
    def __init__(self, game, size=1):
        self.size = size
        super().__init__(game)
    
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
        game.ctx.point_size=5

    def load_mesh(self):
        o = self.origin
        u = self.unit
        scalar = self.unit*(self.x_range[1]-self.x_range[0])/(self.x_points-1)

        x = np.linspace(self.x_range[0], self.x_range[1], self.x_points)
        y = np.linspace(self.y_range[0], self.y_range[1], self.y_points)
        X, Y = np.meshgrid(x, y)
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

def setup_moderngl_antialiasing(ctx):
    """
    Sets up ModernGL context for antialiased line rendering.

    Parameters:
        ctx (moderngl.Context): The ModernGL context.
    """
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

#    ctx.enable(moderngl.LINE_SMOOTH)
    ctx.line_width = 1.5

class Game(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Orbiting Cube"
    window_size = 1280,720
    aspect_ratio = 16 / 9
    #resource_dir = (Path(__file__).parent / 'resources').resolve()
    clear_color_val = 0.09#0.15 #0.9

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ctx.enable(moderngl.DEPTH_TEST)# | moderngl.CULL_FACE)
        #setup_moderngl_antialiasing(self.ctx)

        self.program = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec3 in_position;
            in vec3 in_color;
            out vec3 color;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main() {
                gl_Position = projection * view * model * vec4(in_position, 1.0);
                color = in_color;
            }
            """,
            fragment_shader="""
            #version 330
            in vec3 color;
            out vec4 fragColor;

            void main() {
                fragColor = vec4(color, 1.0);
            }
            """
        )

        # instantiate objects
        cube_size = 0.5
        self.cube = Cube(self, size=cube_size)
        self.cube.translate((.25,0.25,.25))
        self.axes = Axes(self, origin=np.array([0,0,0], dtype='f4'), size=5)
        self.grid = Grid(self, unit=cube_size)


        # setup projection matrices (orthographic and perspective)
        width = 2.0
        height = width / self.aspect_ratio
        self.orthographic_projection = Matrix44.orthogonal_projection( -width, width, -height, height, -10, 10)
        self.perspective_projection = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 100.0)

        self.use_perspective = True

        self.cam = Camera()

    def update(self, t, dt):
        # update view matrix
        self.cam.update()
        view = self.cam.view
        self.program['view'].write(view.astype('f4').tobytes())

        # update projection matrix
        projection = self.perspective_projection if self.use_perspective else self.orthographic_projection
        self.program['projection'].write(projection.astype('f4').tobytes())

    def draw(self):
        self.ctx.clear(self.clear_color_val, self.clear_color_val,  self.clear_color_val)

        # draw objects
        for obj in SceneObject.group:
            self.program['model'].write(obj.model.astype('f4').tobytes())
            obj.render()

    def render(self, time, frame_time):
        self.update(time, frame_time)
        self.draw()

    def key_event(self, key, action, modifiers):
        w = self.cube.size
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.SPACE:
                self.use_perspective = not self.use_perspective
            elif key == self.wnd.keys.W: 
                self.cube.translate(-w*SceneObject.e1)
            elif key == self.wnd.keys.S: 
                self.cube.translate(w*SceneObject.e1)
            elif key == self.wnd.keys.A: 
                self.cube.translate(w*SceneObject.e3)
            elif key == self.wnd.keys.D: 
                self.cube.translate(-w*SceneObject.e3)

    def mouse_scroll_event(self, dx,dy):
    #def mouse_position_event(self, x, y, dx, dy): 
        dx = dx*.1
        self.cam.azimuth += dx*.5
        dy = dy*.1
        self.cam.altitude = clamp(self.cam.altitude + dy*.25, .1, np.pi/2)

        #print(x,y,dx,dy)
        #self.clear_color_val = clamp(self.clear_color_val+dy, 0, 1)

        #print("Mouse wheel:", self.clear_color_val)

def clamp(x, a=0, b=1):
    return min(b, max(a, x))

if __name__ == '__main__':
    mglw.run_window_config(Game)

