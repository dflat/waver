import moderngl
import numpy as np
import math
from pyrr import Matrix44, Vector3
import moderngl_window as mglw
from pathlib import Path
from splines import Spline, SplinePatch, grid, wave_mesh
from scene_objects import SceneObject, Cube, Grid, Axes, SplineMesh
from utils import Color, clamp
from camera import Camera

class Game(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Orbiting Cube"
    window_size = 1280,720
    aspect_ratio = 16 / 9
    resource_dir = (Path(__file__).parent / 'resources').resolve()
    clear_color_val = 0.09#0.15 #0.9

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ctx.enable(moderngl.DEPTH_TEST)# | moderngl.CULL_FACE)
        self.dt = 0
        self.t = 0
        self.events = 0

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
        self.cube.translate(Vector3((.25,0.25,.25)))
        self.axes = Axes(self, origin=np.array([0,0,0], dtype='f4'), size=5)
        #self.grid = Grid(self, unit=cube_size)
        self.patch = SplineMesh(self, interval=(-3,3), n_samps=22*2)


        # setup projection matrices (orthographic and perspective)
        width = 2.0
        height = width / self.aspect_ratio
        self.orthographic_projection = Matrix44.orthogonal_projection( -width, width, -height, height, -10, 10)
        self.perspective_projection = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 100.0)

        self.use_perspective = True

        self.cam = Camera(self)
        self.controls = Controls(self)

    def update(self, t, dt):
        self.controls.update(t, dt)

        # update scene objects
        for obj in SceneObject.group:
            obj.update(t, dt)

        # update view matrix
        self.cam.update(t, dt)
        view = self.cam.view
        self.program['view'].write(view.astype('f4'))#.tobytes())

        # update projection matrix
        projection = self.perspective_projection if self.use_perspective else self.orthographic_projection
        self.program['projection'].write(projection.astype('f4'))#.tobytes())


        # reset control input cache
        self.controls.clear_just_pressed()

    def draw(self):
        self.ctx.clear(self.clear_color_val, self.clear_color_val,  self.clear_color_val)

        # draw objects
        for obj in SceneObject.group:
            self.program['model'].write(obj.model_to_array.astype('f4'))#.tobytes())
            obj.render()

    def render(self, time, frame_time):
        self.dt = frame_time
        self.t += frame_time

        if self.t > 1: # debugging mouse poll rate
            self.t = 0
            #print('mouse drag events per second:', self.events)
            self.events = 0

        self.update(time, frame_time)
        self.draw()

    def key_event(self, key, action, modifiers):
        #w = self.cube.size
        if action == self.wnd.keys.ACTION_PRESS:
            self.controls.press(key)
        elif action == self.wnd.keys.ACTION_RELEASE:
            self.controls.release(key)

            # if key == self.wnd.keys.SPACE:
            #     #self.use_perspective = not self.use_perspective
            #     self.cube.match_normals((1,0,0))

            # elif key == self.wnd.keys.W: 
            #     self.cube.translate(-w*SceneObject.e1)
            # elif key == self.wnd.keys.S: 
            #     self.cube.translate(w*SceneObject.e1)
            # elif key == self.wnd.keys.A: 
            #     self.cube.translate(w*SceneObject.e3)
            # elif key == self.wnd.keys.D: 
            #     self.cube.translate(-w*SceneObject.e3)

    def mouse_drag_event(self, x, y, dx, dy):
        #self.events+=1
        radPerSec = 1.5*self.dt
        dx = dx*radPerSec/2
        self.cam.azimuth += dx*.5
        dy = dy*radPerSec
        self.cam.altitude = clamp(self.cam.altitude - dy*.25, .1, np.pi/2)

    def mouse_scroll_event(self, dx,dy):
    #def mouse_position_event(self, x, y, dx, dy): 
        radPerSec = 10*self.dt
        dy = dy*radPerSec

        self.cam.radius = clamp(self.cam.radius + dy, 1, 10)

        #print(x,y,dx,dy)
        #self.clear_color_val = clamp(self.clear_color_val+dy, 0, 1)

        #print("Mouse wheel:", self.clear_color_val)

    # Windows compatablility
    on_render = render
    on_key_event = key_event
    on_mouse_drag_event = mouse_drag_event
    on_mouse_scroll_event = mouse_scroll_event

class Controls:
    def __init__(self, game):
        self.game = game
        self.keys = self.game.wnd.keys
        self.pressed = {}
        self.just_pressed = {}

    @property
    def left(self):
        return self.keys.A in self.pressed
    @property
    def right(self):
        return self.keys.D in self.pressed
    @property
    def up(self):
        return self.keys.W in self.pressed
    @property
    def down(self):
        return self.keys.S in self.pressed
    @property
    def space(self):
        return self.keys.SPACE in self.pressed
    
    def press(self, key):
        self.pressed[key] = 1
        self.just_pressed[key] = 1
        print('pressed', key)

    def release(self, key):
        self.pressed.pop(key)
        print('released', key)

    def was_just_pressed(self, key):
        return key in self.just_pressed

    def clear_just_pressed(self):
        self.just_pressed = {} # todo: find out the order of key events/render calls

    def update(self, t, dt):
        K = self.keys
        if self.was_just_pressed(K.T):
            self.game.cam.track = not self.game.cam.track



if __name__ == '__main__':
    mglw.run_window_config(Game)

