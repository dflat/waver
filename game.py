import moderngl
import threading
import pyglet
import numpy as np
import math
from pyrr import Matrix44, Vector3
import moderngl_window as mglw
from pathlib import Path
from splines import Spline, SplinePatch, grid, wave_mesh
from scene_objects import SceneObject, Cube, Grid, Axes, SplineMesh
from utils import Mat4, Color, clamp, rescale
from camera import Camera
from controllers.gamepad import GamePadManager
from animation import Animation
from pprint import pprint

class Game(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Orbiting Cube"
    window_size = 1280,720
    aspect_ratio = 16 / 9
    resource_dir = (Path(__file__).parent / 'resources').resolve()
    clear_color_val = 0.19#0.09#0.15 #0.9
    samples = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ctx.enable(moderngl.DEPTH_TEST)# | moderngl.CULL_FACE)
        self.dt = 0
        self.t = 0
        self.frame = 0
        self.drags_per_second = 0
        self.uniforms = { }
        print(self.wnd)
        pprint(vars(self.wnd))
        self.wnd.print_context_info()

        self.program = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec3 in_position;
            in vec3 in_color;
            out vec3 color;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            //uniform float time;

            float pi = 3.14159;
            float freq = 1/4.0;
            float A = 0.25;

            void main() {
                //float vx = in_position.x;
                //float x = sin(2*pi*freq*time*vx);
                vec3 offset = vec3(0,0,0);
                gl_Position = projection * view * model * vec4(offset+in_position, 1.0);
                color = in_color;
            }
            """,
            fragment_shader="""
            #version 330
            in vec3 color;
            out vec4 fragColor;

            float pi = 3.14159;
            float freq = 1.0 + 0*1/2.0;
            vec2 c = vec2(1280/2, 720/2);
            float X = 1280;
            float Y = 720;
            float n = 10;
            uniform float time;

            void main() {

                float vx = gl_FragCoord.x/X;
                float vy = gl_FragCoord.y/Y;
                float xn = sin(2*pi*freq*time*vx);
                float yn = sin(2*pi*freq*time*vy);
                float m = xn*yn;

                float r = length(gl_FragCoord.xy - c);
                float y = gl_FragCoord.y;
                float s = sin(2*pi*freq*y);
                s = floor(mod(y/7.2, 2))/2;
                vec3 offset = vec3(s,s,s);
                float L = pow(1 - r/X, 1);
                float mask = floor(n*L);
                fragColor = vec4(mask/n*color, 1.0);
                fragColor = fragColor;//* m;
            }
            """
        )

        # instantiate objects
        cube_size = 0.5
        self.world = Mat4.make_rigid_frame_euler()
        self.cube = Cube(self, size=cube_size)
        self.cube_frame = Axes(self, parent=self.cube, size=1)
        self.axes = Axes(self, frame=self.world, size=5)
        self.camxzAxes = Axes(self, frame=self.world, size=1)
        #self.grid = Grid(self, unit=cube_size)
        self.patch = SplineMesh(self, interval=(-3,3), n_samps=22*2)


        # setup projection matrices (orthographic and perspective)
        width = 2.0
        height = width / self.aspect_ratio
        self.orthographic_projection = Matrix44.orthogonal_projection( -width, width, -height, height, -10, 10)
        self.perspective_projection = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 100.0)

        self.use_perspective = True

        self.cam = Camera(self)
        self.camAxes = Axes(self, parent=self.cam, aux_origin=(0,2,0), size=1)

        self.controls = Controls(self)
        self.pad_manager = GamePadManager(self)

        # setup uniforms
        self.uniforms['time'] = self.program['time']

        #pyglet.clock.schedule_interval(self.pad_manager.update, 1 / 2)
        #threading.Thread(target=pyglet.app.run).start()

    def pump_events(self):
        """
        This "works for now", but may cause issues down the line
        e.g. with latency, because I am not sure how the control
        flow proceeds...TODO: deal with this. It essentially 
        just needs to listen for controller gamepad connection/
        disconnection events, to trigger creating a new controller object,
        which for now it indeed does.
        """
        #loop = pyglet.app.event_loop

        #dt = loop.clock.update_time()
        #loop.clock.call_scheduled_functions(dt)

        # Update timout
        # TODO.. see what these functions do
        #timeout = loop.clock.get_sleep_time(True)
        timeout = 0
        pyglet.app.platform_event_loop.step(timeout)


    def update(self, t, dt):
        self.frame += 1

        self.controls.update(t, dt)
        self.pad_manager.update(dt)

        self.pump_events()

        # Poll Pyglet events
        #pyglet.clock.tick()  # Process pyglet scheduled tasks
        #print(pyglet.app.event_loop)
        #pyglet.app.event_loop.dispatch_posted_events()
        #self._window.dispatch_events()

        # update scene objects
        for obj in SceneObject.group:
            obj.handle_input(self.controls)
            obj.update(t, dt)

        # update miscelleneous uniforms
        self.program['time'].value = t

        # step animations (todo: figure out where best to do this in control flow)
        for anim in list(Animation.playing.values()):
            anim.update(dt)

        # update view matrix
        self.cam.handle_input(self.controls)
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
        # todo: clear is called in main render loop driver (this is calling it again)

        # draw objects
        for obj in SceneObject.group:
            self.program['model'].write(obj.object_matrix_as_array.astype('f4'))#.tobytes())
            obj.render()

    def render(self, time, frame_time):
        self.dt = frame_time
        self.t += frame_time

        #np.linalg.inv(np.eye(4)*2)
        if self.t > 1: # debugging mouse poll rate
            self.t = 0
            #print('mouse drag events per second:', self.drags_per_second)
            self.drags_per_second = 0

        self.update(time, frame_time)
        self.draw()

    def key_event(self, key, action, modifiers):
        #w = self.cube.size
        if action == self.wnd.keys.ACTION_PRESS:
            self.controls.press(key)
        elif action == self.wnd.keys.ACTION_RELEASE:
            self.controls.release(key)

    def mouse_drag_event(self, x, y, dx, dy):
        self.drags_per_second+=1
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

    def mouse_position_event(self, x, y, dx, dy):
        x = rescale(x, 0, self.window_size[0], -1, 1)
        y = rescale(y, 0, self.window_size[1], 1, -1)
        #print("Mouse position:", x, y, dx, dy)
        self.controls.cursor[0] = x
        self.controls.cursor[1] = y
        #print(self.controls.cursor)

    # Windows compatablility
    on_render = render
    on_key_event = key_event
    on_mouse_drag_event = mouse_drag_event
    on_mouse_scroll_event = mouse_scroll_event
    on_mouse_position_event = mouse_position_event

class Controls:
    def __init__(self, game):
        self.game = game
        self.keys = self.game.wnd.keys
        self.pressed = {}
        self.just_pressed = {}
        self.cursor = np.array((0,0), dtype='f4')

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

