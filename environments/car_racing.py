from typing import Any, Dict, Tuple

import gym
import numpy as np
from gym.envs.registration import register
from scipy.integrate import odeint


class CarRacingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(4,))

        self.init_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(4,))

        self._max_episode_steps = 400

    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.steps = 0
        self.corner = False
        return self.state

    def step(self, action: np.ndarray) -> \
            Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        dt = 0.02
        x = self.state[0] + dt * self.state[2]
        y = self.state[1] + dt * self.state[3]
        vx = self.state[2] + dt * action[0]
        vy = self.state[3] + dt * action[1]

        self.state = np.clip(np.array([x, y, vx, vy]),
                             self.observation_space.low,
                             self.observation_space.high)

        if x >= 3.0 and y >= 3.0:
            self.corner = True

        if self.corner:
            reward = -(abs(x) + abs(y))
        else:
            reward = -(6.0 + abs(x - 3.0) + abs(y - 3.0))

        done = self.corner and x <= 0.0 and y <= 0.0
        done = self.steps >= self._max_episode_steps - 1
            # self.unsafe(self.state)
        self.steps += 1

        return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        return False

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self) -> bool:
        return self.state[0] >= 1.0 and self.state[0] <= 2.0 and \
            self.state[1] >= 1.0 and self.state[1] <= 2.0


class CarRacing:

    environment_name = "car_racing"
    entry_point = "environments.car_racing:CarRacingEnv"
    max_episode_steps = 400
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
        CarRacing.version += 1
        self._config = config
        self.__dict__.update(config)
        print(f"env_name : {env_name}")
        print(f"entry_point : {self.entry_point}")
        print(f"config2 : {config}")
        self.gym_env = gym.make(env_name)
        self.state = None
