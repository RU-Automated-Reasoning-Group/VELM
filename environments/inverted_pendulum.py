import gym
import numpy as np
from gym import register
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.utils.ezpickle import EzPickle


class SafeInvertedPendulumEnv(InvertedPendulumEnv):
    episode_unsafe = False

    def __init__(
        self, threshold=0.2, task="move", random_reset=False, violation_penalty=10
    ):
        self.threshold = threshold
        self.task = task
        self.random_reset = random_reset
        self.violation_penalty = violation_penalty
        self.steps = 0
        self._max_steps = 200
        super().__init__()
        EzPickle.__init__(
            self, threshold=threshold, task=task, random_reset=random_reset
        )  # deepcopy calls `get_state`

    def reset_model(self):
        if self.random_reset:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-0.01, high=0.01
            )
            qvel = self.init_qvel + self.np_random.uniform(
                size=self.model.nv, low=-0.01, high=0.01
            )
            self.set_state(qpos, qvel)
        else:
            self.set_state(self.init_qpos, self.init_qvel)
        self.episode_unsafe = False
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return super()._get_obs().astype(np.float32)

    def step(self, a):
        a = np.clip(a, -1, 1)
        next_state, _, done, info = super().step(a)
        # reward = (next_state[0]**2 + next_state[1]**2)  # + a[0]**2 * 0.01
        # reward = next_state[1]**2  # + a[0]**2 * 0.01

        if self.task == "upright":
            reward = -next_state[1] ** 2
        elif self.task == "swing":
            reward = next_state[1] ** 2
        elif self.task == "move":
            reward = next_state[0] ** 2
        else:
            assert 0

        done = False
        if self.steps >= self._max_steps:
            done = True

        if abs(next_state[..., 1]) > self.threshold or abs(next_state[..., 0]) > 0.9:
            # breakpoint()
            self.episode_unsafe = True
            done = True
            reward -= self.violation_penalty
        self.steps += 1
        # done = False
        return next_state, reward, done, info


class SafeInvertedPendulum:
    environment_name = "inverted_pendulum"
    entry_point = "environments.inverted_pendulum:SafeInvertedPendulumEnv"
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
        SafeInvertedPendulum.version += 1
        self._config = config
        self.__dict__.update(config)
        print(f"env_name : {env_name}")
        print(f"entry_point : {self.entry_point}")
        print(f"config2 : {config}")
        self.gym_env = gym.make(env_name)
        self.state = None
