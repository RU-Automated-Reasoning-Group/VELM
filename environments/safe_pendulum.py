import gym
import numpy as np
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding
from scipy.integrate import odeint


class GymSafePendulum(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, *args, **kwargs):
        self.threshold = 2

        high = np.array([self.threshold, self.threshold], dtype=np.float32)
        low = -high
        self.action_space = spaces.Box(
            low=-self.threshold, high=self.threshold, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.dt = 0.05
        self.c = 0
        self.goal_state = -0.3
        self.ilb = np.array([0.25, -0.95])
        self.iub = np.array([0.35, -0.85])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def f(self, state, t, u):
        x1, x2 = state
        ff = np.array([x2, 15 * np.sin(x1) + 3 * u])
        return ff

    def step(self, action):
        self.c += 1
        u = action[0]
        # print(self.c, u)
        N = 6
        t = np.linspace(0, self.dt, N)
        # self.state = self.state + self.dt * self.f(u)
        self.state = odeint(self.f, self.state, t, args=(u,))[-1, :]
        costs = (angle_normalize(self.state[0]) - self.goal_state) ** 2
        # if self.c >= 9 and (self.state[0] < 0 or self.state[0] > 1.0):
        done = False
        if abs(angle_normalize(self.state[0])) > 1.5:
            done = True
            costs = 5

        #     # unsafe!!
        #     costs += 5
        # print(self.state)
        return np.array(self.state), -costs, done, {}

    def simulate(self, action):
        u = action[0]
        N = 2
        t = np.linspace(0, self.dt, N)
        simulated_state = odeint(self.f, self.state, t, args=(u,))[-1, :]
        return np.array(simulated_state)

    def reset(self):
        self.state = self.np_random.uniform(low=self.ilb, high=self.iub)
        # self.state = np.array([0.3, -0.9])
        # print("initial state", self.state)
        self.c = 0
        return np.array(self.state)

    def render(self, mode="human"):
        print(f"state {self.c}: {self.state}")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class SafePendulum:
    environment_name = "safe_pendulum"
    entry_point = "environments.safe_pendulum:GymSafePendulum"
    max_episode_steps = 200
    reward_threshold = 21

    version = 1

    def __init__(self, **kwargs):
        config = {
            "image": kwargs.pop("image", False),
            "sliding_window": kwargs.pop("sliding_window", 0),
            "image_dim": kwargs.pop("image_dim", 32),
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
        SafePendulum.version += 1
        self._config = config
        self.__dict__.update(config)
        print(f"env_name : {env_name}")
        print(f"entry_point : {self.entry_point}")
        print(f"config2 : {config}")
        self.gym_env = gym.make(env_name)
        self.state = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
