from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class Gymnasium_LaloEnv(gym.Env):
    def __init__(self):
        super(Gymnasium_LaloEnv, self).__init__()

        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(1,))
        self.observation_space = gym.spaces.Box(
            low=np.array([-3.8, -3.95, -3.50, -2.60, -4.00, -4.90, -4.55]), high=np.array([6.20, 6.05, 6.50, 7.40, 6.00, 5.10, 5.45])
        )

        self.init_space = gym.spaces.Box(
            low=np.array([1.15, 1.00, 1.45, 2.35, 0.95, 0.05, 0.40]), high=np.array([1.25, 1.10, 1.55, 2.45, 1.05, 0.15, 0.50])
        )

        self.rng = np.random.default_rng()

        self._max_episode_steps = 100


    def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state, {}

    def f(self, state, u):
        x1, x2, x3, x4, x5, x6, x7 = state
        
        x1_dot = 1.4 * x3 - 0.9 * x1
        x2_dot = 2.5 * x5 - 1.5 * x2 + u
        x3_dot = 0.6 * x7 - 0.8 * x2 * x3
        x4_dot = 2 - 1.3 * x3 * x4
        x5_dot = 0.7 * x1 - x4 * x5
        x6_dot = 0.3 * x1 - 3.1 * x6
        x7_dot = 1.8 * x6 - 1.5 * x2 * x7
        return np.array([x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot])


    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        dt = 0.1
        fxu = self.f(self.state, action[0])

        self.state = self.state + dt * fxu

        # self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        reward = -1 * np.linalg.norm(fxu).item()
        # if np.any(self.state < self.observation_space.low).item() or np.any(self.state > self.observation_space.high).item():
            # reward -= 100
        if self.unsafe():
            reward -= 10

        return self.state, reward, False, False, {}

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)
        self.rng = np.random.default_rng(np.random.PCG64(seed))

    def unsafe(self) -> bool:
        lower_bound = np.array([-3.30, -3.45, -3.00, -2.10, -3.50, -4.40, -4.05])
        upper_bound = np.array([1.25, 1.10, 1.55, 2.45, 1.05, 1.05, 0.50])
        return np.all(lower_bound <= self.state).item() and np.all(self.state <= upper_bound).item()


class Gymnasium_Lalo:

    environment_name = "gymnasium_lalo"
    entry_point = "environments.gymnasium_lalo:Gymnasium_LaloEnv"
    max_episode_steps = 100
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
        Gymnasium_Lalo.version += 1
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
            "horizon": 100,
            "alpha": 0.001,
            "N_of_directions": 3,
            "b": 2,
            "noise": 0.001,
            "initial_lambda": 0.5,
            "iters_run": 1,
        }

        def plot_other_components():
            plt.plot([0, 0], [-2, 2], linestyle="--")

        def plot_state_to_xy(state):
            return state[0], state[1]
        
        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
