import math
from typing import Any, Dict, Tuple

import gym
import numpy as np
from gym.envs.registration import register
from scipy.integrate import odeint

from models.dso_model import evaluate_dynamics, make_context_with_action


class NoisyRoad2dEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-20, high=20, shape=(4,))
        self.init_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(4,))
        self.rng = np.random.default_rng()

        self.max_speed = 10.0
        self._max_episode_steps = 300
        self.unsafe_episode = 0
        self.use_learned_model = False
        self.env_name = "noisy_road_2d"

    def load_model(self, model_path="learned_dynamics.txt", preprocess=False):
        model_path = f"{self.env_name}_learned_dynamics.txt"
        self.use_learned_model = True
        self.preprocess = preprocess
        with open(model_path, "r") as f:
            lines = f.readlines()
            self.learned_dynamic_model = [line[:-1] for line in lines]
            print(f"loading dyancmic model from {model_path}")
        
        std_path = f"{self.env_name}_learned_stds.txt"
        with open(std_path, "r") as f:
            lines = f.readlines()
            self.stds = [float(line[:-1]) for line in lines]
            print(f"loading stds from {std_path}")

    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state

    def reset_to_zero(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.steps = 0
        return self.state

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        dt = 0.02
        if self.use_learned_model == False or self.use_learned_model == True:
            x = self.state[0] + dt * self.state[2]
            y = self.state[1] + dt * self.state[3]
            vx = self.state[2] + dt * action[0] + self.rng.normal(loc=0, scale=0.05)
            vy = self.state[3] + dt * action[1] + self.rng.normal(loc=0, scale=0.05)

            self.state = np.array([x, y, vx, vy])
            # self.state += self.rng.normal(loc=0, scale=0.05, size=(4,))
            # self.state = np.clip(
            # self.state, self.observation_space.low, self.observation_space.high
            # )
        # else:
        #     ctx = make_context_with_action(self.state, action)
        #     self.state = evaluate_dynamics(self.learned_dynamic_model, ctx)
        #     stds = []
        #     dim = len(self.stds)
        #     for i in range(0, dim):
        #         noise = np.random.normal(loc=0.0, scale=self.stds[i])
        #         stds.append(noise)
        #     import pdb
        #     pdb.set_trace()
        #     self.state = self.state + np.array(stds)

        x, y = self.state[0], self.state[1]
        reward = -(abs(x - 3.0) + abs(y - 3.0))
        self.steps += 1
        done = self.steps >= self._max_episode_steps

        if self.unsafe():
            if not self.use_learned_model:
                self.unsafe_episode += 1
                print(f"unsafe episodes: {self.unsafe_episode}")
            reward -= 20
        if self.steps == self._max_episode_steps:
            print("last state is ", self.state)
            print("num timestep is", self.steps)
        return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        return state[0] >= 3.0 and state[1] >= 3.0

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)
        self.rng = np.random.default_rng(np.random.PCG64(seed))

    def unsafe(self) -> bool:
        return self.state[0] ** 2 + self.state[1] ** 2 >= self.max_speed ** 2


class NoisyRoad2D:

    environment_name = "noisy_road_2d"
    entry_point = "environments.noisy_road_2d:NoisyRoad2dEnv"
    max_episode_steps = 200
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
        NoisyRoad2D.version += 1
        self._config = config
        self.__dict__.update(config)
        print(f"env_name : {env_name}")
        print(f"entry_point : {self.entry_point}")
        print(f"config2 : {config}")
        self.gym_env = gym.make(env_name)
        self.state = None
