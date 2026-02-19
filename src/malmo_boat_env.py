import gym
from gym import spaces
import numpy as np
import MalmoPython
import time
import json
import math
import random

from ice_track_testing import generate_star_race_track, create_mission_xml


class MalmoBoatEnv(gym.Env):
    
    def __init__(self):
        super(MalmoBoatEnv, self).__init__()

        self.agent_host = MalmoPython.AgentHost()

        # Discrete actions
        # 0 = nothing
        # 1 = forward
        # 2 = left
        # 3 = right
        # 4 = brake
        self.action_space = spaces.Discrete(5)

        # Observation: x, z, yaw, distance_to_next_checkpoint
        self.observation_space = spaces.Box(
            low=-1000,
            high=1000,
            shape=(4,),
            dtype=np.float32
        )

        self.checkpoints = []
        self.current_checkpoint = 0
        self.prev_distance = None

    # ---------------------------------------------------

    def reset(self):

        track_xml, cp_pos, _, _, spawn, start_idx = generate_star_race_track()
        mission_xml = create_mission_xml(track_xml, spawn)

        self.checkpoints = cp_pos
        self.current_checkpoint = 0

        my_mission = MalmoPython.MissionSpec(mission_xml, True)
        my_mission_record = MalmoPython.MissionRecordSpec()

        self.agent_host.startMission(my_mission, my_mission_record)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        time.sleep(1)

        obs = self._get_obs()
        self.prev_distance = obs[3]

        return obs

    # ---------------------------------------------------

    def step(self, action):
        world_state = self.agent_host.getWorldState()
        if not world_state.is_mission_running:
            obs = self.reset()  # auto-reset
            reward = 0
            done = False
            return obs, reward, done, {}


        self._take_action(action)
        time.sleep(0.01)

        obs = self._get_obs()

        reward = self.prev_distance - obs[3]
        self.prev_distance = obs[3]

        done = False

        # Check checkpoint reached
        if obs[3] < 5:
            reward += 100
            self.current_checkpoint += 1
            if self.current_checkpoint >= len(self.checkpoints):
                done = True

        return obs, reward, done, {}

    # ---------------------------------------------------

    def _take_action(self, action):

        if action == 1:
            self.agent_host.sendCommand("move 1")
        elif action == 2:
            self.agent_host.sendCommand("turn -1")
        elif action == 3:
            self.agent_host.sendCommand("turn 1")
        elif action == 4:
            self.agent_host.sendCommand("move -1")
        else:
            self.agent_host.sendCommand("move 0")

    # ---------------------------------------------------

    def _get_obs(self):

        world_state = self.agent_host.getWorldState()

        while world_state.number_of_observations_since_last_state == 0:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        msg = world_state.observations[-1].text
        data = json.loads(msg)

        x = data.get("XPos", 0)
        z = data.get("ZPos", 0)
        yaw = data.get("Yaw", 0)

        if self.current_checkpoint < len(self.checkpoints):
            cx, cz = self.checkpoints[self.current_checkpoint]
            dist = math.sqrt((x - cx) ** 2 + (z - cz) ** 2)
        else:
            dist = 0

        return np.array([x, z, yaw, dist], dtype=np.float32)
