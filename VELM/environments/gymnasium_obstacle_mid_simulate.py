import copy
import math
import pdb
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from . import simulated_env_util


class Gymnasium_ObstacleMidSimulateEnv(gym.Env):
    def __init__(self, learned_model: List[str]=[]):
        super().__init__()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(4,))

        self.init_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(4,))

        self.model = copy.deepcopy(learned_model)

        self._max_episode_steps = 200
        self.env_name = "obstacle_mid"

    def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
        # pdb.set_trace()
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        self.state = simulated_env_util.eval_model(self.model, self.state, action)
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        x = self.state[0]
        y = self.state[1]

        # reward = -(abs(x - 3.0) + abs(y - 3.0))
        reward = -1 * ((x-3)**2 + (y-3)**2)
        if self.unsafe():
            reward -= 20

        return self.state, reward, False, False, {}

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self) -> bool:
        return (
            self.state[0] >= 1.0
            and self.state[0] <= 2.0
            and self.state[1] >= 1.0
            and self.state[1] <= 2.0
        )

    def simulate(self, state, action):
        self.state = copy.deepcopy(state)
        next_state, rwd, _, _, _ = self.step(action)
        self.state = next_state
        return copy.deepcopy(next_state)


class GymnasiumObstacleMidSimulate:

    environment_name = "gymnasium_obstacle_mid_simulate"
    entry_point = (
        "environments.gymnasium_obstacle_mid_simulate:Gymnasium_ObstacleMidSimulateEnv"
    )
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
        gym.register(
            id=env_name,
            entry_point=self.entry_point,
            max_episode_steps=self.max_episode_steps,
            reward_threshold=self.reward_threshold,
            kwargs=config,
        )
        GymnasiumObstacleMidSimulate.version += 1
        self._config = config
        self.__dict__.update(config)
        print(f"env_name : {env_name}")
        print(f"entry_point : {self.entry_point}")
        print(f"config2 : {config}")
        self.gym_env = gym.make(env_name)
        self.state = None

        # lagrange method parameters
        self.lagrange_config = {
            "dso_dataset_size": 2000,
            "num_traj": 100,
            "horizon": 200,
            "alpha": 0.001,
            "N_of_directions": 3,
            "b": 2,
            "noise": 0.001,
            "initial_lambda": 0.5,
            "iters_run": 1,
        }

        def plot_other_components():
            # plot unsafe set
            plt.plot([1, 2, 2, 1, 1], [2, 2, 1, 1, 2])

            # plot goal position
            plt.plot([3.0], [3.0], "p-r")

        def plot_state_to_xy(state):
            return state[0], state[1]

        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
