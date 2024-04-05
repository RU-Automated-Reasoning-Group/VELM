import copy
import math
from typing import List, Optional, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.classic_control import utils
from scipy.integrate import odeint

from . import simulated_env_util


class Gymnasium_CartPoleSimulateEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, learned_model: List[str] = []):
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(-10, 10, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.init_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(4,))

        self.model = copy.deepcopy(learned_model)

        self.state = None

        self.steps_beyond_terminated = None

    def seed(self, seed: int):
        return

    def step(self, action):
        self.state = simulated_env_util.eval_model(self.model, self.state, action)
        # self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        state_vec = self.state.copy()
        reward = -1 * np.linalg.norm(state_vec).item()
        # if not self.unsafe():
        #     reward = 1.0
        # else:
        #     reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, False, False, {}

    def unsafe(self):
        x, x_dot, theta, theta_dot = self.state
        return bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def simulate(self, state, action):
        self.state = copy.deepcopy(state)
        next_state, rwd, _, _, _ = self.step(action)
        self.state = next_state
        return copy.deepcopy(next_state)


class GymnasiumCartPoleSimulate:
    environment_name = "gymnasium_cartpole_simulate"
    entry_point = (
        "environments.gymnasium_cartpole_simulate:Gymnasium_CartPoleSimulateEnv"
    )
    max_episode_steps = 200
    reward_threshold = 10000

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
        GymnasiumCartPoleSimulate.version += 1
        self._config = config
        self.__dict__.update(config)
        print(f"env_name : {env_name}")
        print(f"entry_point : {self.entry_point}")
        print(f"config2 : {config}")
        self.gym_env = gym.make(env_name)
        self.state = None

        self.lagrange_config = {
            "dso_dataset_size": 2000,
            "num_traj": 500,
            "horizon": 200,
            "alpha": 0.001,
            "N_of_directions": 10,
            "b": 3,
            "noise": 0.001,
            "initial_lambda": 0.5,
            "iters_run": 50,
        }

        def plot_other_components():
            # plot unsafe set
            theta_limit = 0.20943951023931953
            x_limit = 2.4
            plt.plot(
                [-x_limit, x_limit, x_limit, -x_limit, -x_limit],
                [theta_limit, theta_limit, -theta_limit, -theta_limit, theta_limit],
            )

        def plot_state_to_xy(state):
            return state[0], state[2]

        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
