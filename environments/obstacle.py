import math
from typing import Any, Dict, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.registration import register
from scipy.integrate import odeint


class ObstacleEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(4,))

        self.init_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(4,))

        self._max_episode_steps = 200
        self.unsafe_episode = 0
        self.integrate = "not_ode"
        self.use_learned_model = False
        self.preprocess = False
        self.env_name = "obstacle"

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
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.steps = 0
        return self.state

    def dynamics(self, state, t, action):
        return np.array([state[2], state[3], action[0], action[1]])

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        dt = 0.02
        if self.use_learned_model == False:
            if self.integrate == "ode":
                N = 10
                t = np.linspace(0, dt, N)
                self.state = odeint(self.dynamics, self.state, t, args=(action,))[-1, :]
            else:
                # self.state = self.state + dt * self.dynamics(self.state, None, action)
                x = self.state[0] + dt * self.state[2]
                y = self.state[1] + dt * self.state[3]
                vx = self.state[2] + dt * action[0]
                vy = self.state[3] + dt * action[1]
                self.state = np.array([x, y, vx, vy])
        else:
            x1, x2, x3, x4 = self.state
            x5 = action[0]
            x6 = action[1]
            cos = math.cos
            sin = math.sin
            if self.preprocess:
                # use dt to calculate next state
                dt = 0.02
                change_in_state = np.array(
                    [eval(expr) for expr in self.learned_dynamic_model]
                )
                self.state = self.state + dt * change_in_state
            else:
                # directly give the next step
                x1, x2, x3, x4 = self.state
                x5 = action[0]
                x6 = action[1]
                cos = math.cos
                sin = math.sin
                next_state = []
                for i in range(0, len(self.learned_dynamic_model)):
                    next_state.append(eval(self.learned_dynamic_model[i]))
                # next_state = [eval(expr) for expr in learned_model]
                self.state = np.array(next_state)

        # TODO: decide whether to use clip
        # self.state = np.clip(np.array([x, y, vx, vy]),
        #                      self.observation_space.low,
        #                      self.observation_space.high)\
        x = self.state[0]
        y = self.state[1]
        reward = -(abs(x - 3.0) + abs(y - 3.0))
        # reward = -1 * np.linalg.norm(np.array([x-3, y-3]))
        # done = x >= 3.0 and y >= 3.0
        self.steps += 1
        # print(self.steps)
        if self.unsafe() == True and self.use_learned_model == False:
            self.unsafe_episode += 1
            print(f"unsafe episodes: {self.unsafe_episode}")
            reward -= 20
            # reward -= 5000
        if self.steps == self._max_episode_steps:
            print("last state is ", self.state)
            print("num timestep is", self.steps)

        return self.state, reward, False, {}

    def simulate(
        self, state: np.ndarray, action: np.ndarray, model_path="learned_dynamics.txt"
    ) -> np.ndarray:
        # with open(model_path, "r") as f:
        #     lines = f.readlines()
        #     self.learned_dynamic_model = [line[:-1] for line in lines]
        #     print(f"loading dyancmic model from {model_path}")
        # dt = 0.02

        x1, x2, x3, x4 = state
        x5 = action[0]
        x6 = action[1]
        cos = math.cos
        sin = math.sin
        return np.array(
            [x1 + 0.02 * x3, x2 + 0.02 * x4, x3 + 0.02 * x5, x4 + 0.02 * x6]
        )
        # if self.preprocess:
        #     # use dt to calculate next state
        #     dt = 0.02
        #     change_in_state = np.array([eval(expr) for expr in self.learned_dynamic_model])
        #     next_state = state + dt * change_in_state
        # else:
        #     # directly give the next step
        #     x1, x2, x3, x4 = state
        #     x5 = action[0]
        #     x6 = action[1]
        #     cos = math.cos
        #     sin = math.sin
        #     next_state = []
        #     for i in range(0, len(self.learned_dynamic_model)):
        #         next_state.append(eval(self.learned_dynamic_model[i]))
        #     # next_state = [eval(expr) for expr in learned_model]
        #     next_state = np.array(next_state)
        # return next_state

    def predict_done(self, state: np.ndarray) -> bool:
        return state[0] >= 3.0 and state[1] >= 3.0

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self) -> bool:
        return (
            self.state[0] >= 0.0
            and self.state[0] <= 1.0
            and self.state[1] >= 2.0
            and self.state[1] <= 3.0
        )


class Obstacle:

    environment_name = "obstacle"
    entry_point = "environments.obstacle:ObstacleEnv"
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
        Obstacle.version += 1
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
            plt.plot([0, 1, 1, 0, 0], [3, 3, 2, 2, 3])

            # plot goal position
            plt.plot([3.0], [3.0], "p-r")

        def plot_state_to_xy(state):
            return state[0], state[1]

        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
