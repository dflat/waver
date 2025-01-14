import time
from utils import clamp, rescale
from splines import bez
from collections import defaultdict
from scene_objects import SceneObject

class Interpolant:
	@classmethod
	def linear(cls, a, b):
		def f(t):
			return a + t*(b-a) 
		return f

	@classmethod
	def smoothstep(cls, a, b):
		def s(t):
			return 3*t*t - 2*t**3 
		def f(t):
			return a + s(t)*(b-a)
		return f

	@classmethod
	def quintic(cls, a, b):
		def q(t):
			return 35*t**4 - 84*t**5 + 70*t**6 - 20*t**7
			return 6 * t**5 - 15 * t**4 + 10 * t**3
		def f(t):
			return a + q(t)*(b-a)
		return f

	@classmethod
	def overshoot(cls, a, b):
		P = (0,0,-0.4,1)
		P = (0,0.5,-0.3,1)
		p = bez(*P)
		def f(t):
			return a + p(t)*(b-a)
		return f

class Animation:
	_id = 0
	playing = { }
	finished = { }
	channels = defaultdict(list)

	def __init__(self, obj:'SceneObject', 
						property=None,
						startval=None,
						endval=None,
						deltaval=None,
						dur=1,
						interpolant=None,
						update_func=None,
						startup_func=None,
						teardown_func=None,
						on_cancel=None,
						interruptable=True,
						channel=None,
						drop_if_busy=True):
		"""
		args:
			obj: SceneObject, reference to object to be animated

			property: str, name of property on ::obj:: to animate

			startval and endval may be single values or sequences of values,
				to be interpolated together.

			update_func(l, obj, anim) will receive an interpolating value, a
				refrence to the animating object, and the Animation instance itself.
		"""
		self._id = Animation._id
		Animation._id += 1

		self.obj = obj
		self.property = property
		self.startval = startval
		self.endval = endval
		self.deltaval = deltaval
		self.dur = dur

		self.interpolant = interpolant
		self.update_func = update_func
		self.startup_func = startup_func
		self.teardown_func = teardown_func
		self.on_cancel = on_cancel

		self.interruptable = interruptable
		self.channel = channel
		self.drop_if_busy = drop_if_busy

		self._running = False
		self.elapsed = 0
		self._prepare()

	def _prepare(self):
		if self.deltaval and self.property:
			self.startval = getattr(self.obj, self.property)
			self.endval = self.startval + self.deltaval
		if self.interpolant is None:
			assert self.startval is not None and self.endval is not None and self.property
			self.interpolant = Interpolant.linear(self.startval, self.endval)
		else:
			self.interpolant = self.interpolant(self.startval, self.endval)
		if self.update_func is None:
			# default update func for simple single-valued interpolations of floats.
			assert self.startval is not None and self.endval is not None and self.property
			self.update_func = _default_update_func

	def start(self):
		self.t0 = time.time()	
		self.t1 = self.t0 + self.dur # maybe not useful to store this
		self.elapsed = 0
		self.value = self.startval
		Animation.playing[self._id] = self

		if self.channel:
			if self.drop_if_busy:
				if len(Animation.channels[self.channel]) > 0:
					return self.cancel()
			Animation.channels[self.channel].append(self)

		self.startup()
		print(f'animation {self._id} started.')

	def cancel(self, hard=True):
		if not self.interruptable:
			raise RuntimeError(f'Cannot interrupt this animation, id:{self._id}')
		if self.on_cancel:
			self.on_cancel()
		self.remove()
		print(f'animation {self._id} cancelled.')

	def remove(self):
		self._running = False
		Animation.playing.pop(self._id)
		if self.channel:
			try:
				Animation.channels[self.channel].remove(self) # TODO: Fix this...
			except ValueError:
				pass

	def update(self, dt):
		if self._running:
			self.elapsed += dt
			s = clamp(self.elapsed/self.dur) # Normalized progress
			self.value = self.interpolant(s)
			self.update_func(s, self.obj, self)
			if s == 1:
				return self.finish()
		else:
			return self.finish()

	def startup(self):
		if self.startup_func:
			self.startup_func()
		self._running = True

	def finish(self):
		if self.teardown_func:
			self.teardown_func()
		self.remove()
		Animation.finished[self._id] = self
		print(f'animation {self._id} finished.')
		return True

def _default_update_func(s, obj, anim):
	setattr(obj, anim.property, anim.value)




