import moderngl
import numpy as np
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

class SceneObject:
    group = []
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])

    def __init__(self, game):
        self.group.append(self)
        self.game = game
        self.ctx = game.ctx
        self.program = game.program
        self.verts = None
        self.colors = None
        self.model = Matrix44.identity()
        self._load()

    def translate(self, v):
        self.model[3, :3] = v

    def load_mesh(self):
        raise NotImplementedError()

    def render(self):
        self.vao.render(moderngl.TRIANGLES)

    def _load(self):
        self.verts, self.colors = self.load_mesh()
        self.vbo = self.ctx.buffer(np.hstack((self.verts, self.colors)).astype('f4').tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 3f', 'in_position', 'in_color')]
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

class Axes(SceneObject):
    def __init__(self, game, origin=np.array([0,0,0], dtype='f4'), size=5):
        self.origin = origin
        self.size = size
        super().__init__(game)

    def render(self):
        self.vao.render(moderngl.LINES)

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

class Game(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Orbiting Cube"
    window_size = (800, 600)
    aspect_ratio = 16 / 9
    resource_dir = (Path(__file__).parent / 'resources').resolve()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ctx.enable(moderngl.DEPTH_TEST)# | moderngl.CULL_FACE)

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

        cube = Cube(self, size=0.5)
        axes = Axes(self, origin=np.array([0,0,0], dtype='f4'), size=5)

        self.perspective_projection = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 100.0)

        width = 2.0
        height = width / self.aspect_ratio
        self.orthographic_projection = Matrix44.orthogonal_projection( -width, width, -height, height, -10, 10)

        self.use_perspective = True

        #self.program['projection'].write(self.perspective_projection.astype('f4').tobytes())

        self.angle = 0.0
        self.radius = 3.0

    def calculate_view_matrix(self, angle):
        """Calculates the view matrix for a camera orbiting the cube."""
        camera_position = Vector3([
            self.radius * np.cos(angle),
            self.radius * 0.5,  # Slight elevation
            self.radius * np.sin(angle)
        ])
        target = Vector3([0.0, 0.0, 0.0])
        up = Vector3([0.0, 1.0, 0.0])
        return Matrix44.look_at(camera_position, target, up)

    def update(self, t, dt):
        # Update camera angle
        self.angle += 0.01
        if self.angle > 2 * np.pi:
            self.angle -= 2 * np.pi

        # Calculate view matrix and write to the program
        view = self.calculate_view_matrix(self.angle)
        self.program['view'].write(view.astype('f4').tobytes())

        # Toggle projection type
        projection = self.perspective_projection if self.use_perspective else self.orthographic_projection
        self.program['projection'].write(projection.astype('f4').tobytes())

    def draw(self):
        self.ctx.clear(0.1, 0.1, 0.1)

        # draw objects
        for obj in SceneObject.group:
            self.program['model'].write(obj.model.astype('f4').tobytes())
            obj.render()

    def render(self, time, frame_time):
        self.update(time, frame_time)
        self.draw()

    def key_event(self, key, action, modifiers):
        if key == self.wnd.keys.SPACE and action == self.wnd.keys.ACTION_PRESS:
            self.use_perspective = not self.use_perspective

if __name__ == '__main__':
    mglw.run_window_config(Game)

