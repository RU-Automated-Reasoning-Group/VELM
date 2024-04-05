import copy
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from . import simulated_env_util


class Gymnasium_AccSimulateEnv(gym.Env):
    def __init__(self, learned_model: List[str] = [], stds: List[float] = []):
        super(Gymnasium_AccSimulateEnv, self).__init__()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(
            low=np.array([-10, -10]), high=np.array([10, 10])
        )

        self.init_space = gym.spaces.Box(
            low=np.array([-1.1, -0.1]), high=np.array([-0.9, 0.1])
        )

        self.rng = np.random.default_rng()

        self.model = copy.deepcopy(learned_model)
        self.stds = copy.deepcopy(stds)

        self._max_episode_steps = 300

    def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        # import pdb
        # pdb.set_trace()
        self.state = simulated_env_util.eval_model(self.model, self.state, action, stds=self.stds)

        if len(self.state) == 0:
            import pdb
            pdb.set_trace()
        x, v = self.state
        # self.state = np.clip(
        #         np.asarray([x, v]),
        #         self.observation_space.low,
        #         self.observation_space.high)
        reward =  2.0 + x if x < 0 else -10
        # done = bool(x >= 0) or self.steps > self._max_episode_steps
        # self.steps += 1
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
        return self.state[0] >= 0
    
    def simulate(self, state, action):
        self.state = copy.deepcopy(state)
        next_state, rwd, _, _, _ = self.step(action)
        self.state = next_state
        return copy.deepcopy(next_state)


class Gymnasium_ACCSimulate:

    environment_name = "gymnasium_acc_simulate"
    entry_point = "environments.gymnasium_acc_simulate:Gymnasium_AccSimulateEnv"
    max_episode_steps = 300
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
        Gymnasium_ACCSimulate.version += 1
        self._config = config
        self.__dict__.update(config)
        print(f"env_name : {env_name}")
        print(f"entry_point : {self.entry_point}")
        print(f"config2 : {config}")
        self.gym_env = gym.make(env_name)
        self.state = None

        def plot_other_components():
            plt.plot([0, 0], [-2, 2], linestyle="--")

        def plot_state_to_xy(state):
            return state[0], state[1]

        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
