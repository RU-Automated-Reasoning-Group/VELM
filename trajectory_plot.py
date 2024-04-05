import pathlib
import sys

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt

from environments.simulated_env_util import load_dynamic_from_file, make_simulated_env
from fullplot import gym_env_info
from utils.get_env import get_env

matplotlib.rcParams.update({"font.size": 16})


def compare_trajectories(env_name, xlabel, ylabel):
    gym_env_name, episode_length = gym_env_info(env_name)
    env_info, simulated_env_info = get_env(env_name)

    real_env = gym.make(env_info.env_name)
    # generate a real trajectory
    xs, ys, actions = [], [], []
    done = False
    current_state, _ = real_env.reset()
    initial_state = current_state.copy()
    while not done:
        x, y = env_info.plot_state_to_xy(current_state)
        xs.append(x)
        ys.append(y)

        action = real_env.action_space.sample()
        actions.append(action)

        next_state, _, terminate, truncated, _ = real_env.step(action)
        done = terminate or truncated

        current_state = next_state

    x, y = env_info.plot_state_to_xy(current_state)
    xs.append(x)
    ys.append(y)

    # load dynamics for this benchmark
    path = pathlib.Path(f"results/learned_dynamics/{gym_env_name}")
    models, stds = load_dynamic_from_file(path)
    random = stds is not None

    simulated_env = make_simulated_env(
        random, simulated_env_info.env_name, learned_model=models, stds=stds
    )
    # plot the using the same action but learned trajectory
    learned_xs, learned_ys = [], []
    done = False
    current_state, _ = simulated_env.reset()
    simulated_env.env.env.env.state = initial_state.copy()
    current_state = initial_state.copy()
    import pdb

    # pdb.set_trace()
    idx = 0
    while not done:
        x, y = simulated_env_info.plot_state_to_xy(current_state)
        learned_xs.append(x)
        learned_ys.append(y)

        action = actions[idx]
        next_state, _, terminate, truncated, _ = simulated_env.step(action)
        done = terminate or truncated

        current_state = next_state
        idx += 1

    x, y = simulated_env_info.plot_state_to_xy(current_state)
    learned_xs.append(x)
    learned_ys.append(y)
    # generate plot
    plt.plot(xs, ys, label="real rollout")
    plt.plot(learned_xs, learned_ys, label="simulated rollout")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(f"compare_{env_name}.png")


if __name__ == "__main__":
    env_name, xlabel, ylabel = sys.argv[1:]
    compare_trajectories(env_name, xlabel, ylabel)
