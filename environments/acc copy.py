from typing import Any, Dict, Tuple

import gym
import numpy as np
from gym.envs.registration import register
from scipy.integrate import odeint


class AccEnv(gym.Env):
    def __init__(self):
        super(AccEnv, self).__init__()

        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(1,))
        self.observation_space = gym.spaces.Box(
            low=np.array([-10, -10]), high=np.array([10, 10])
        )

        self.init_space = gym.spaces.Box(
            low=np.array([-1.1, -0.1]), high=np.array([-0.9, 0.1])
        )
        self.state = np.zeros(2)

        self.rng = np.random.default_rng()

        # self.concrete_safety = [
        #     lambda x: x[0]
        # ]

        self._max_episode_steps = 300

    def load_model(self):
        pass

    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state

    def f(self, state, t, u, noise):
        x, v = state
        ff = np.array([v, u + noise])
        return ff

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        dt = 0.02
        x = self.state[0] + dt * self.state[1]
        v = self.state[1] + dt * (action[0] + self.rng.normal(loc=0, scale=0.5))
        self.state = np.array([x, v])
        # self.state = np.clip(
        #         np.asarray([x, v]),
        #         self.observation_space.low,
        #         self.observation_space.high)
        reward =  2.0 + x if x < 0 else -10 - x
        # done = bool(x >= 0) or self.steps > self._max_episode_steps
        done = self.steps >= self._max_episode_steps - 1
        self.steps += 1
        # if self.unsafe():
        #     print("unsafe")
        if done:
            print(f"last state is {self.state}")
        return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        return state[0] >= 0

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)
        self.rng = np.random.default_rng(np.random.PCG64(seed))

    def unsafe(self) -> bool:
        return self.state[0] >= 0

    def render(self, mode="human"):
        print(self.state)

    def simulate(self, action):
        dt = 0.02
        t = np.linspace(0, dt, 2)
        u = action[0]
        # noise = self.rng.normal(loc=0, scale=0.5)
        noise = 0
        return odeint(self.f, self.state, t, args=(u, noise))[-1, :]


class ACC:

    environment_name = "acc"
    entry_point = "environments.acc:AccEnv"
    max_episode_steps = 300
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
        ACC.version += 1
        self._config = config
        self.__dict__.update(config)
        print(f"env_name : {env_name}")
        print(f"entry_point : {self.entry_point}")
        print(f"config2 : {config}")
        self.gym_env = gym.make(env_name)
        self.state = None
