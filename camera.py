import numpy as np
from pyrr import Matrix44, Vector3
import math

class Camera:
    def __init__(self, game):
        self.game = game
        self.azimuth = math.pi/4 # looking down center of +xz
        self.altitude = math.pi/2*0.65 # Level with xz-plane
        self.dtheta = 0.00025*0
        self.radius = 6.0*1.5
        #self.elevation = 2
        self.target = Vector3([0.0, 0.0, 0.0])
        self.pos = Vector3([self.radius, 0.0, 0.0])
        self.up = Vector3([0.0, 1.0, 0.0])
        self.view = self.get_view_matrix()
        self.track = True
        self.inverted_x_track = False # reverse the azimuth against target's x translation

    def orbit(self, u, v, r=5):
        x = np.sin(v) * np.cos(u)
        z = np.sin(v) * np.sin(u)
        y = np.cos(v)
        return r*Vector3([x,y,z])

    def update(self, t, dt):
        # Update camera azimuth
        self.azimuth += self.dtheta
        if self.azimuth > 2 * math.pi:
            self.azimuth -= 2 * math.pi

        # if position tracking cube
        if self.track:
	        self.target = self.game.cube.pos
	        sgn = 1 if self.inverted_x_track else -1
	        r = max(3, self.radius)
	        self.azimuth = sgn*math.asin(self.game.cube.pos[0]/r) + math.pi/2

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


