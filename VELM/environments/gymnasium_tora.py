from typing import Any, Dict, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class Gymnasium_ToraEnv(gym.Env):
    def __init__(self):
        super(Gymnasium_ToraEnv, self).__init__()

        self.threshold = 2
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(1,))
        self.observation_space = gym.spaces.Box(
            low=-2 * self.threshold, high=2 * self.threshold, shape=(4,)
        )

        self.init_space = gym.spaces.Box(
            low=np.array([0.6, -0.7, -0.4, 0.5]), high=np.array([0.7, -0.6, -0.3, 0.6])
        )

        self.rng = np.random.default_rng()

        self._max_episode_steps = 500

    def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        dt = 0.1
        x1, x2, x3, x4 = self.state

        self.state = np.array(
            [
                x1 + dt * x2,
                x2 + dt * (-1 * x1 + 0.1 * np.sin(x3)),
                x3 + dt * x4,
                x4 + dt * 10*action[0],
            ]
        )

        print(f"action {action}")
        reward = -1 * np.linalg.norm(self.state).item()
        self.steps += 1
        # if self.unsafe():
        #     print("unsafe")
        # if self.steps == self._max_episode_steps:
        # print("last state is ", self.state)
        # print("num timestep is", self.steps)
        return self.state, reward, False, False, {}

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)
        self.rng = np.random.default_rng(np.random.PCG64(seed))

    def unsafe(self) -> bool:
        return (
            abs(self.state[0]) > 2
            or abs(self.state[1]) > 2
            or abs(self.state[2]) > 2
            or abs(self.state[3]) > 2
        )


class Gymnasium_Tora:

    environment_name = "gymnasium_tora"
    entry_point = "environments.gymnasium_tora:Gymnasium_ToraEnv"
    max_episode_steps = 500
    reward_threshold = 1000

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
        Gymnasium_Tora.version += 1
        self._config = config
        self.__dict__.update(config)
        print(f"env_name : {env_name}")
        print(f"entry_point : {self.entry_point}")
        print(f"config2 : {config}")
        self.gym_env = gym.make(env_name)
        self.state = None

        self.lagrange_config = {
            "dso_dataset_size": 2000,
            "num_traj": 100,
            "horizon": 300,
            "alpha": 0.001,
            "N_of_directions": 3,
            "b": 2,
            "noise": 0.001,
            "initial_lambda": 0.5,
            "iters_run": 1,
        }

        def plot_other_components():
            plt.plot([-2, 2, 2, -2, -2], [2, 2, -2, -2, 2])
            plt.plot([0], [0], "p-r")

        def plot_state_to_xy(state):
            return state[0], state[1]

        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
