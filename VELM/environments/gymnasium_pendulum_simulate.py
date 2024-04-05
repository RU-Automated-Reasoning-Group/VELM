import copy
import math
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from . import simulated_env_util


class Gymnasium_PendulumSimulateEnv(gym.Env):
    def __init__(self, learned_model: List[str] = []):
        super().__init__()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.pi, -1]), high=np.array([np.pi, 1])
        )

        self.init_space = gym.spaces.Box(
            low=np.array([-0.01, -0.001]), high=np.array([0.01, 0.001])
        )

        self.model = copy.deepcopy(learned_model)

        self.unsafe_reward = -10
        self._max_episode_steps = 100
        self.env_name = "pendulum_simulatw"

    def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        self.state = simulated_env_util.eval_model(self.model, self.state, action)

        reward = -abs(self.state[0])
        if self.unsafe():
            reward += self.unsafe_reward

        return self.state, reward, False, False, {}

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self) -> bool:
        return abs(self.state[0]) >= 0.4

    def simulate(self, state, action):
        self.state = copy.deepcopy(state)
        next_state, rwd, _, _, _ = self.step(action)
        self.state = next_state
        return copy.deepcopy(next_state)


class GymnasiumPendulumSimulate:

    environment_name = "gymnasium_pendulum_simulate"
    entry_point = (
        "environments.gymnasium_pendulum_simulate:Gymnasium_PendulumSimulateEnv"
    )
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
        gym.register(
            id=env_name,
            entry_point=self.entry_point,
            max_episode_steps=self.max_episode_steps,
            reward_threshold=self.reward_threshold,
            kwargs=config,
        )
        GymnasiumPendulumSimulate.version += 1
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
            "horizon": 100,
            "alpha": 0.001,
            "N_of_directions": 3,
            "b": 2,
            "noise": 0.001,
            "initial_lambda": 0.5,
            "iters_run": 1,
        }

        def plot_other_components():
            # plot unsafe set
            plt.plot([-0.4, -0.4], [-1, 1], "b")
            plt.plot([0.4, 0.4], [-1, 1], "b")

            # plot goal position
            plt.plot([0, 0], [-1, 1], "p--r")

        def plot_state_to_xy(state):
            return state[0], state[1]

        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
