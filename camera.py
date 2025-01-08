import numpy as np
from pyrr import Matrix44, Vector3
import math
from animation import Animation, Interpolant
import glm

class Camera:
    def __init__(self, game):
        self.game = game
        self.azimuth = 0*math.pi/4 # looking down center of +xz
        self.altitude = math.pi/2*0.65 
        self.dtheta = 0.00025*10
        self.spin = False
        self.azimuth_lock = False
        self.radius = 6.0*1.5
        #self.elevation = 2
        self.target = glm.vec4(0.0, 0.0, 0.0, 1.0)
        self.pos = glm.vec4(self.radius, 0.0, 0.0, 1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.view = self.get_view_matrix()
        self.track = True
        self.inverted_x_track = False # reverse the azimuth against target's x translation

    def orbit(self, u, v, r=5):
        x = np.sin(v) * np.cos(u)
        z = np.sin(v) * np.sin(u)
        y = np.cos(v)
        return r*glm.vec4(x,y,z,0)

    def handle_input(self, controls):
        k = controls.keys 
        player = self.game.cube.player
        if (player and player.just_pressed('a')) or controls.was_just_pressed(k.SPACE):
            anim = Animation(obj=self, property='azimuth', deltaval=-math.pi/2, dur=0.5,
                interpolant=Interpolant.quintic, channel=1)
            anim.start()
        elif player and player.just_pressed('x'):
            anim = Animation(obj=self, property='azimuth', deltaval=math.pi/2, dur=0.5,
                interpolant=Interpolant.quintic, channel=1)
            anim.start()

    def update(self, t, dt):
        # Update camera azimuth
        if self.spin:
            self.azimuth += self.dtheta

        if self.azimuth > 2 * math.pi:
            self.azimuth -= 2 * math.pi

        # if position tracking cube
        if self.track:
            self.target = self.game.cube.pos
            if self.azimuth_lock:
                sgn = 1 if self.inverted_x_track else -1
                r = max(3, self.radius)
                self.azimuth = sgn*math.asin(self.game.cube.pos[0]/r) + math.pi/2

        #self.pos = Vector3([
        #    self.radius * np.cos(self.azimuth),
        #    self.elevation,  
            #self.radius * np.sin(self.azimuth)
        #])
        self.pos = self.target + self.orbit(self.azimuth, self.altitude, self.radius)
        #print('pos',np.round(self.pos,2))

        self.view = self.get_view_matrix()
        #print(np.round(self.view,2))

    def get_forward(self):
        """
        view is stored as inverse camera transform, in column major order,
        So world-space forward vector appears in the 3rd row.
        """
        return glm.vec3(glm.row(self.view,2))

    @property
    def o(self):
        return glm.affineInverse(self.view) 
    
    def get_view_matrix(self):
        """Calculates the view matrix for a camera orbiting the cube."""
        return glm.lookAtRH(self.pos.xyz, self.target.xyz, self.up)


