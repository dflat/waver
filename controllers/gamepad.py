import pyglet
from pyrr import Vector3
import numpy as np
import time
from copy import deepcopy
from collections import deque

# Add PS5 controller to database...
# for some reason it isn't listed in pyglet's
# autogenerated list of gamepads for darwin

pyglet.input.controller_db.mapping_list.append("030000004c050000e60c000000010000,PS5 Controller,a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b10,lefttrigger:a3,leftx:a0,lefty:a1,misc1:b13,rightshoulder:b5,rightstick:b11,righttrigger:a4,rightx:a2,righty:a5,start:b9,x:b0,y:b3,")

# Initialize controller manager


class GamePadState:
    def __init__(self):
        self.a = False
        self.b = False
        self.x = False
        self.y = False
        self.sticks = [Vector3(), Vector3()]
        self.dpleft = False
        self.dpright = False
        self.dpup = False
        self.dpdown = False
    
    def __repr__(self):
        return repr(self.__dict__)

class GamePad:
    dead_zone_thresh = 0.04 # magnitude under which to treat as zero

    def __init__(self, controller, _id):
        self.controller = controller # reference to underlying HID object interface
        self._id = _id # id to manage connection and disconnection by GamePadManager
        controller.open()  # Open the controller for input
        self.state = GamePadState()
        self.state_history = deque(maxlen=60) # store a sequence of past states
        self.player_no = None

    def update(self, dt):
        """
        Cache a 'snapshot' of controller state every update interval,
        defined by self.poll_rate.
        """
        self.state_snapshot()

    @property
    def leftaxis(self):
        v = self.state.sticks[0]
        norm = np.linalg.norm(v)
        #vdir = v/norm if norm > 0 else v
        if norm < self.dead_zone_thresh:
            norm = 0
            vdir = Vector3((0,0,0))
        else:
            vdir = v/norm
        return vdir, norm
    
    def state_snapshot(self):
        c = self.controller
        self.state_history.appendleft(self.state)
        self.state = GamePadState()
        s = self.state
        s.a = c.a
        s.b = c.b
        s.x = c.x
        s.y = c.y
        s.dpleft = c.dpleft
        s.dpright = c.dpright
        s.dpup = c.dpup
        s.dpdown = c.dpdown
        s.sticks[0][0] = c.leftx
        s.sticks[0][2] = c.lefty
        s.sticks[1][0] = c.rightx
        s.sticks[1][2] = c.righty
        #print(c.__dict__)
        #print(s)


class GamePadManager:
    poll_rate = 1/60
    controller_id = 0
    connected_pads = { } # id => GamePad
    max_players = 4
    players = [None]*max_players # four player positions

    def __init__(self, game):
        self.game = game
        self.t0 = time.time()
        self.controller_manager = pyglet.input.ControllerManager()
        self.find_controllers()
        #self.state = GamePadState()

    def find_controllers(self):
        # Detect already-connected controllers
        controllers = self.controller_manager.get_controllers()

        if controllers:
            for controller in controllers:
                #controller = controllers[0]  # Use the first available controller
                self.register_pad(controller)

        # monitor new controller connections and disconnections
        self.controller_manager.on_connect = self.on_connect
        self.controller_manager.on_disconnect = self.on_disconnect

    def connect_pad(self, controller):
        print(f"Detected controller: {controller.name}")
        _id = GamePadManager.controller_id
        GamePadManager.controller_id += 1
        pad = GamePad(controller, _id)
        GamePadManager.connected_pads[_id] = pad
        controller._id = _id # tag controller object itself for later disconnection
        #pyglet.clock.schedule_interval(lambda dt: self.update_controller_state(controller), self.poll_rate)
        return pad

    def unregister_pad(self, controller):
        print(f"Controller disconnected: {controller.name}")
        del GamePadManager.connected_pads[controller._id]

    def register_pad(self, controller):
        pad = self.connect_pad(controller)
        self.assign_player_position(pad)

    def assign_player_position(self, pad):
        player_no = 0
        # find first empty player slot
        for player in self.players:
            if player is None:
                break
            player_no += 1
        pad.player_no = player_no
        self.players[player_no] = pad
        self.game.cube.player = pad # TODO testing this side effect assignment here

    def on_connect(self, controller):
        """Handles controller connection."""
        self.register_pad(controller)

    def on_disconnect(self, controller):
        """Handles controller disconnection."""
        self.unregister_pad(controller)

    def update(self, dt):
        # Poll Pyglet events
        #pyglet.clock.tick()  # Process pyglet scheduled tasks
        #pyglet.app.platform_event_loop.dispatch_posted_events()
        for pad in self.connected_pads.values():
            pad.update(dt)


