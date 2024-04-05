import math
from typing import Optional, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.classic_control import utils
from scipy.integrate import odeint


class Gymnasium_CartPoleSwingEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 1.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

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

        self.init_space = gym.spaces.Box(low=0.0, high=0.0, shape=(4,))

        self.state = None

        self.steps_beyond_terminated = None

    def seed(self, seed: int):
        return

    def step(self, action):
        # assert self.action_space.contains(
        # action
        # ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag * action[0]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        # if not self.unsafe():
        #     reward = 1.0
        # else:
        #     reward = 0.0
        # state_vec = np.array([x, x_dot, theta, theta_dot])
        # reward = -1 * np.linalg.norm(state_vec).item()
        # terminate = False
        reward = (self.state[2])**2
        if self.unsafe():
            reward = -30
            # terminate = True

        # import pdb
        # pdb.set_trace()
        return np.array(self.state, dtype=np.float32), reward, False, False, {}

    def unsafe(self):
        x, x_dot, theta, theta_dot = self.state
        return bool(
            x < -0.9
            or x > 0.9
            or theta < -1.5
            or theta > 1.5
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
            options, 0.0, 0.0  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}


class GymnasiumCartPoleSwing:
    environment_name = "gymnasium_cartpole_swing"
    entry_point = "environments.gymnasium_cartpole_swing:Gymnasium_CartPoleSwingEnv"
    max_episode_steps = 100
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
        GymnasiumCartPoleSwing.version += 1
        self._config = config
        self.__dict__.update(config)
        print(f"env_name : {env_name}")
        print(f"entry_point : {self.entry_point}")
        print(f"config2 : {config}")
        self.gym_env = gym.make(env_name)
        self.state = None

        self.lagrange_config = {
            "dso_dataset_size": 2000,
            "num_traj": 10,
            "horizon": 100,
            "alpha": 0.0003,
            "N_of_directions": 6,
            "b": 3,
            "noise": 0.001,
            "initial_lambda": 0.5,
            "iters_run": 1,
        }

        def plot_other_components():
            # plot unsafe set
            theta_limit = 1.5
            x_limit = 0.9
            plt.plot(
                [-x_limit, x_limit, x_limit, -x_limit, -x_limit],
                [theta_limit, theta_limit, -theta_limit, -theta_limit, theta_limit],
            )

        def plot_state_to_xy(state):
            return state[0], state[2]

        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
