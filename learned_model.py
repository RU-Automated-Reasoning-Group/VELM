"""Wrapper for using the learned dynamic instead of the original dynamics"""
import copy
from typing import List

import gym


class LearnedModel(gym.Wrapper):
    """This wrapper will issue a `truncated` signal if a maximum number of timesteps is exceeded.

    If a truncation is not defined inside the environment itself, this is the only place that the truncation signal is issued.
    Critically, this is different from the `terminated` signal that originates from the underlying environment as part of the MDP.

    Example:
       >>> from gym.envs.classic_control import CartPoleEnv
       >>> from gym.wrappers import TimeLimit
       >>> env = CartPoleEnv()
       >>> env = TimeLimit(env, max_episode_steps=1000)
    """

    def __init__(
        self,
        env: gym.Env,
        learned_model: List[str],
        preprocess: bool
    ):
        """Initializes the :class:`TimeLimit` wrapper with an environment and the number of steps after which truncation will occur.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps: An optional max episode steps (if ``Ǹone``, ``env.spec.max_episode_steps`` is used)
        """
        super().__init__(env)
        self.learned_model = copy.deepcopy(learned_model)
        self.preprocess = preprocess
        self._max_episode_steps = env._max_episode_steps

    def step(self, action):
        """Steps through the environment using the learned model
        """
        observation, reward, done, info = self.env.learned_model_step(action, self.learned_model, self.preprocess)

        return observation, reward, done, info