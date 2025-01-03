import numpy as np
import math

def rescale(x,mn=0,mx=1,a=0,b=1):
	return a + (b-a)*(x - mn)/(mx-mn)

def clamp(x, a=0, b=1):
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

class Vec4:
	def __init__(self, v=None):
		self.v = np.array((0,0,0,1), dtype='f4') if v is None else numpy.array(v, dtype='f4')

class Mat4:
	def __init__(self, m:np.ndarray):
		self.m = m

	@classmethod
	def build_frame(cls, up, forward, origin=(0,0,0)):
	    """
	    Construct a right-handed 3D orthonormal basis given a unit "up" vector
	    and a vector loosely in the "forward" direction.

	    Parameters:
	        up (np.ndarray): Unit vector pointing "up".
	        forward (np.ndarray): Vector loosely in the "forward" direction.
	    """
	    forward /= np.linalg.norm(forward)
	    
	    right = np.cross(up, forward)
	    right /= np.linalg.norm(right)
	    
	    # Compute the corrected forward vector (cross product of right and up)
	    forward = np.cross(right, up)

	    frame = Mat4.identity()
	    frame[:3,0] = right
	    frame[:3,1] = up
	    frame[:3,2] = forward
	    frame[:3,3] = origin
	    return frame


	@classmethod
	def make_aux(cls, R, T):
		P = np.eye(4)
		P[:3,:3] = R[:3,:3]
		P[:3, 3] = T[:3, 3]
		return P

	@classmethod
	def get_transform_in_basis(cls, M, A):
		Ai = np.linalg.inv(A) # todo: use rigid_inverse flag?
		return A @ M @ Ai

	@classmethod
	def rigid_inverse(cls, M):
		"""
		Assume M = TR, return Minv = Rinv @ Tinv = R^T @ T(-o),
		taking advantage of this simpler invertiblilty,
		should be more numerically stable.
		"""
		P = np.eye(4)
		RT = M[:3,:3].T
		o = RT @ -M[:3,3]
		P[:3,:3] = RT
		P[:3,3] = o
		return P

	@classmethod
	def identity(cls):
		return np.eye(4, dtype='f4')

	@classmethod
	def make_rigid_frame_euler(cls, xtheta=0, ytheta=0, ztheta=0, origin=(0,0,0)):
		R = Mat4.from_x_rotation(xtheta)
		R = R @ Mat4.from_y_rotation(ytheta)
		R = R @ Mat4.from_z_rotation(ztheta)
		T = Mat4.from_translation(origin)
		return T @ R 

	@classmethod
	def from_translation(cls, v):
		M = np.eye(4, dtype='f4')
		M[:3, 3] = v
		return M

	@classmethod
	def from_x_rotation(cls, theta):
		M = np.eye(4)
		c = math.cos(theta)
		s = math.sin(theta)
		M[1,1] = c
		M[2,1] = s
		M[1,2] = -s
		M[2,2] = c
		return M

	@classmethod
	def from_y_rotation(cls, theta):
		M = np.eye(4)
		c = math.cos(theta)
		s = math.sin(theta)
		M[0,0] = c
		M[2,0] = -s
		M[0,2] = s
		M[2,2] = c
		return M

	@classmethod
	def from_z_rotation(cls, theta):
		M = np.eye(4)
		c = math.cos(theta)
		s = math.sin(theta)
		M[0,0] = c
		M[1,0] = s
		M[0,1] = -s
		M[1,1] = c
		return M

	@classmethod
	def axis_to_axis(cls, s, t):
		"""
		from Real Time Rendering, chapter 4.3, page 83.
		Original by Tomas, Hughes:
			'Efficiently Building a Matrix to Rotate One Vector to Another'
		"""
		s /= np.linalg.norm(s)
		t /= np.linalg.norm(t)
		v = np.cross(s,t)
		e = np.dot(s,t)
		h = 1/(1 + e)
		M = np.eye(4)
		M[0,0] = e + h*v[0]**2
		M[1,1] = e + h*v[1]**2
		M[2,2] = e + h*v[2]**2
		M[3,3] = 1 

		M[0,1] = h*v[0]*v[1] - v[2]
		M[0,2] = h*v[0]*v[2] + v[1]

		M[1,0] = h*v[0]*v[1] + v[2]
		M[1,2] = h*v[1]*v[2] - v[0]

		M[2,0] = h*v[0]*v[2] - v[1]
		M[2,1] = h*v[1]*v[2] + v[0]

		return M
