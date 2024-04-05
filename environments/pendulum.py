import math
from typing import Any, Dict, Tuple

import gym
import numpy as np
from gym.envs.registration import register


class PendulumEnv(gym.Env):

    def __init__(self):
        super(PendulumEnv, self).__init__()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(
                low=np.array([-np.pi, -1]),
                high=np.array([np.pi, 1]))

        self.init_space = gym.spaces.Box(low=np.array([-0.01, -0.001]),
                                         high=np.array([0.01, 0.001]))

        self.state = np.zeros(2)
        self.unsafe_reward = -10
        self._max_episode_steps = 100
        self.unsafe_episode = 0
        self.use_learned_model = False
        self.env_name = "pendulum"

    def load_model(self, model_path="learned_dynamics.txt", preprocess=False):
        model_path = f"{self.env_name}_learned_dynamics.txt"
        self.use_learned_model = True
        self.preprocess = preprocess
        with open(model_path, "r") as f:
            lines = f.readlines()
            self.learned_dynamic_model = [line[:-1] for line in lines]
            print(f"loading dyancmic model from {model_path}")

    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state

    def reset_to_zero(self):
        self.state = np.zeros(2)
        self.steps = 0
        return self.state

    def step(self, action: np.ndarray) -> \
            Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        dt = 0.02
        if self.use_learned_model == False:
            m = 0.25
            ln = 2.0
            g = 9.81
            theta = self.state[0] + dt * self.state[1]
            omega = self.state[1] + \
                dt * (g / (2 * ln) * np.sin(self.state[0]) +
                    3.0 / (m * ln ** 2) * action[0])
            self.state = np.array([theta, omega])
            old_state = self.state.copy()
            print(action)
            if action[0] < -1 or action[0] > 1:
                print(f"action {action[0]} outside action space")
            self.state = np.clip(
                    np.asarray([theta, omega]),
                    self.observation_space.low,
                    self.observation_space.high)
            if old_state[0] != self.state[0] or old_state[1] != self.state[1]:
                print(f"old state {old_state}, clipped state {self.state}")
        else:
            x1, x2 = self.state
            x3 = action[0]
            cos = math.cos
            sin = math.sin
            next_state = []
            for i in range(0, len(self.learned_dynamic_model)):
                next_state.append(eval(self.learned_dynamic_model[i]))
            self.state = np.array(next_state)
      
        reward = -abs(self.state[0])
        done = self.steps >= self._max_episode_steps
        self.steps += 1
        if self.unsafe() == True:
            if self.use_learned_model == False:
                self.unsafe_episode += 1
                print(f"unsafe episode: {self.unsafe_episode}")
            reward += self.unsafe_reward
        
        if self.steps == self._max_episode_steps:
            print("last state is ", self.state)
            print("num timestep is", self.steps)
        return self.state, reward, done, {}

    def simulate(self, state: np.ndarray, action: np.ndarray, model_path="learned_dynamics.txt") -> np.ndarray:
        model_path = f"{self.env_name}_learned_dynamics.txt"
        with open(model_path, "r") as f:
            lines = f.readlines()
            self.learned_dynamic_model = [line[:-1] for line in lines]
            print(f"loading dyancmic model from {model_path}")
        
        x1, x2 = state
        x3 = action[0]
        cos = math.cos
        sin = math.sin

        next_state = []
        for i in range(0, len(self.learned_dynamic_model)):
            next_state.append(eval(self.learned_dynamic_model[i]))
        # next_state = [eval(expr) for expr in learned_model]
        next_state = np.array(next_state) 
        return next_state

    def predict_done(self, state: np.ndarray) -> bool:
        return abs(state[0]) >= 0.4

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self) -> bool:
        return abs(self.state[0]) >= 0.4


class Pendulum:

    environment_name = "pendulum"
    entry_point = "environments.pendulum:PendulumEnv"
    max_episode_steps = 100
    reward_threshold = 21

    version = 1

    def __init__(self, **kwargs):
        config = {
            # 'image': kwargs.pop('image', False),
            # 'sliding_window': kwargs.pop('sliding_window', 0),
            # 'image_dim': kwargs.pop('image_dim', 32),
        }

        env_name = "Marvel%s-v%u" % (self.environment_name, self.version)
        self.env_name = env_name
        print(f"config1 : {config}")
        register(
            id=env_name,
            entry_point=self.entry_point,
            max_episode_steps=self.max_episode_steps,
            reward_threshold=self.reward_threshold,
            kwargs=config,
        )
        Pendulum.version += 1
        self._config = config
        self.__dict__.update(config)
        print(f"env_name : {env_name}")
        print(f"entry_point : {self.entry_point}")
        print(f"config2 : {config}")
        self.gym_env = gym.make(env_name)
        self.state = None
