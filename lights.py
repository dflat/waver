import numpy as np
import glm
import struct
import math

class LightArray:
	# Define the Light struct as a numpy dtype
	gl_array = 'LightBuffer'
	gl_count = 'light_count'
	gl_reflectivity = 'reflectivity'
	binding = 0 # uniform block binding index for glsl

	MAX_LIGHTS = 10

	dtype = np.dtype([
	    ('position', np.float32, 4),  # vec3 -> 3 floats
	    #('_padding1', np.float32), # padding

	    ('color', np.float32, 4),     # vec3 -> 3 floats
	    #('_padding2', np.float32), # padding

	    ('intensity', np.float32),     # float
	    ('_padding3', np.float32, 3) # padding
	])

	lights = np.zeros(MAX_LIGHTS, dtype=dtype)
	named_lights = { }

	def __init__(self, game):
		self.game = game
		self.count = 0
		self.reflectivity = 1
		self.dirty = False
 
	def update(self, t, dt):
		intensity1 = 0.5 + 0.5*math.sin(t)
		intensity2 = 0.5 + 0.5*math.sin(t + math.pi)

		self.lights[0]['intensity']	= 0.2 + 0.1*intensity1 
		self.lights[0]['position'] = self.game.cube.pos + glm.vec4(0,1,0,0)
		self.lights[0]['color'] = glm.vec4(1,1,1,1)

		self.lights[1]['intensity']	= intensity2
		self.dirty = True

	def add(self, pos, color, intensity, name=None):
		assert self.count < self.MAX_LIGHTS
		if name is not None:
			self.named_lights[name] = self.count # map light name to array position
		light = self.lights[self.count] #= (pos, color, intensity, 0.0)
		light['position'] = pos
		light['color'] = color
		light['intensity'] = intensity
		self.count += 1
		self.dirty = True

	def send_to_glsl(self):
		if not self.dirty:
			return
		buffer = self.game.ctx.buffer(self.lights.tobytes())  # Create a buffer from the structured array
		prog = self.game.program
		prog[self.gl_count] = self.count         # Pass the number of active lights


		uniform_block = prog[self.gl_array]

		# Bind the buffer to the binding point defined in GLSL (binding = 0)
		buffer.bind_to_uniform_block(binding=uniform_block.binding)	
		#print('size', buffer.size)
		#print('data', buffer.read())
		data = struct.unpack('120f', buffer.read())
		#print('unpacked', data)
		got = prog['LightBuffer']
		#print('as stored on gpu', vars(got))
		prog[self.gl_reflectivity] = self.reflectivity
		self.dirty = False
