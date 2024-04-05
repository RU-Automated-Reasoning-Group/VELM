import copy
import os
import pathlib
import pdb

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from environments.simulated_env_util import make_simulated_env


def visualize_trajectory(
    version: int,
    env_info,
    policy,
    num_traj: int=15,
    simulated=False,
    learned_model=[],
    learned_stds=None,
    random=False,
):
    # make test env
    if simulated:
        test_env = make_simulated_env(
            random, env_info.env_name, learned_model=learned_model, stds=learned_stds
        )
    else:
        test_env = gym.make(env_info.env_name)

    plt.cla()
    all_episode_rwd = []
    for _ in range(0, num_traj):
        state_list = []
        # pdb.set_trace()
        state, _ = test_env.reset()
        state_2d = env_info.plot_state_to_xy(state)
        state_list.append(state_2d)

        done = False
        episode_rwd = 0.0
        while not done:
            action, _ = policy.predict(np.array([state]), deterministic=True)

            action = action[0]
            state, rwd, terminated, trucated, _ = test_env.step(action)
            done = terminated or trucated
            episode_rwd = episode_rwd + rwd

            state_2d = env_info.plot_state_to_xy(state)
            state_list.append(state_2d)

        plt.plot(*zip(*state_list))
        all_episode_rwd.append(episode_rwd)

    # plot unsafe set and save
    env_info.plot_other_components()

    directory = "simulated" if simulated else "real"
    path = pathlib.Path(f"./results/trajectories/{env_info.env_name}/sac_{version}/{directory}")
    path.mkdir(parents=True, exist_ok=True)

    fig_name = os.path.join(path, f"{policy.num_timesteps}.png")
    print(f"saving to {fig_name}")
    plt.savefig(fig_name)

    print(
        f"avg reward: {np.mean(all_episode_rwd)}, std rward {np.std(all_episode_rwd)}"
    )


class PlotCallback(BaseCallback):
    def __init__(
        self,
        version: int,
        env_info,
        simulated_env_info,
        learned_model,
        learned_stds=None,
        random=False,
        verbose=0,
    ):
        super().__init__(verbose)
        self.version = version
        self.env_info = env_info
        self.simulated_env_info = simulated_env_info
        self.learned_model = copy.deepcopy(learned_model)
        self.learned_stds = copy.deepcopy(learned_stds)
        self.random = copy.copy(random)

    def _on_step(self) -> bool:
        visualize_trajectory(self.version, self.env_info, self.model, num_traj=15)
        visualize_trajectory(
            self.version,
            self.simulated_env_info,
            self.model,
            num_traj=15,
            simulated=True,
            learned_model=self.learned_model,
            learned_stds=self.learned_stds,
            random=self.random,
        )
        return True
