import numpy as np
import glm
from glm import vec3

def rescale(x,mn=0,mx=1,a=0,b=1):
	return a + (b-a)*(x - mn)/(mx-mn)

def clamp(x, a=0.0, b=1.0):
	return min(b, max(a, x))

class Color:
	RED = np.array([1,0,0])
	GREEN = np.array([0,1,0])
	BLUE = np.array([0,0,1])
	MAGENTA = np.array([1,0,1])
	CYAN = np.array([0,1,1])
	YELLOW = np.array([1,1,0])
	GREY = np.array([.6,.6,.6])
	LIGHTGREY = np.array([.9,.9,.9])
	WHITE = np.array([1,1,1])
	STEEL = vec3(.3,.3,.4)*1.75

def project_onto_axis(a, v):
	return glm.dot(a,v)*a

class Mat4:
	@classmethod
	def concat_basis(cls, b1:vec3, b2:vec3, b3:vec3):
		return glm.mat3(b1,b2,b3)

	@classmethod
	def from_basis(cls, basis: glm.mat3, origin=(0,0,0)):
		frame = glm.mat4(basis) 
		frame[3] = glm.vec4(origin, 1)
		return frame

	@classmethod
	def build_basis(cls, up, forward):
		"""
		Construct a right-handed 3D orthonormal basis given a unit "up" vector
		and a vector loosely in the "forward" direction.

		Parameters:
			up (np.ndarray): Unit vector pointing "up".
			forward (np.ndarray): Vector loosely in the "forward" direction.
		"""
		forward = glm.normalize(forward)
		
		right = glm.cross(up, forward)
		right = glm.normalize(right)
		
		# Compute the corrected forward vector (cross product of right and up)
		forward = glm.cross(right, up)

		return glm.mat3(right,up,forward)

	@classmethod
	def build_frame(cls, up, forward, origin=(0,0,0)):
		"""
		Construct a right-handed 3D orthonormal basis given a unit "up" vector
		and a vector loosely in the "forward" direction.

		Parameters:
			up (np.ndarray): Unit vector pointing "up".
			forward (np.ndarray): Vector loosely in the "forward" direction.
		"""
		forward = glm.normalize(forward)
		
		right = glm.cross(up, forward)
		right = glm.normalize(right)
		
		# Compute the corrected forward vector (cross product of right and up)
		forward = glm.cross(right, up)

		frame = glm.mat4(glm.mat3(right,up,forward)) 
		frame[3].xyz = origin

		return frame

	@classmethod
	def get_transform_in_basis(cls, M, A):
		Ai = glm.inverse(A) # todo: use rigid_inverse flag?
		return A @ M @ Ai

	@classmethod
	def rigid_inverse(cls, M):
		"""
		Assume M = TR, return Minv = Rinv @ Tinv = R^T @ T(-o),
		taking advantage of this simpler invertiblilty,
		should be more numerically stable.
		"""
		return glm.affineInverse(M)

	@classmethod
	def identity(cls):
		return glm.mat4()

	@classmethod
	def make_rigid_frame_euler(cls, xtheta=0, ytheta=0, ztheta=0, origin=glm.vec3()):
		F = glm.mat4(glm.quat((xtheta, ytheta, ztheta)))
		F[3] = glm.vec4(origin, 1)
		return F

	@classmethod
	def from_translation(cls, v):
		return glm.translate(v)

	@classmethod
	def from_x_rotation(cls, theta):
		return glm.rotate(theta, (1,0,0))

	@classmethod
	def from_y_rotation(cls, theta):
		return glm.rotate(theta, (0,1,0))

	@classmethod
	def from_z_rotation(cls, theta):
		return glm.rotate(theta, (0,0,1))

	@classmethod
	def axis_to_axis(cls, s, t):
		"""
		from Real Time Rendering, chapter 4.3, page 83.
		Original by Tomas, Hughes:
			'Efficiently Building a Matrix to Rotate One Vector to Another'
		"""
		s = glm.normalize(s)
		t = glm.normalize(t)
		v = glm.cross(s,t)
		e = glm.dot(s,t)
		h = 1/(1 + e)
		M = glm.mat4() 
		M[0,0] = e + h*v[0]**2
		M[1,1] = e + h*v[1]**2
		M[2,2] = e + h*v[2]**2
		M[3,3] = 1 

		M[1,0] = h*v[0]*v[1] - v[2]
		M[2,0] = h*v[0]*v[2] + v[1]

		M[0,1] = h*v[0]*v[1] + v[2]
		M[2,1] = h*v[1]*v[2] - v[0]

		M[0,2] = h*v[0]*v[2] - v[1]
		M[1,2] = h*v[1]*v[2] + v[0]

		#return M #glm.mat4(M)
		q = glm.quat(s, t)
		M2 = glm.mat4(q) # faster: try this if surface normal issue TODO
		#assert np.allclose(M, M2) or np.allclose(M, glm.mat4(-q))
		return M2

	@staticmethod
	def cross(a,b):
		return glm.cross(a,b)
