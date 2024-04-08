import argparse
import csv
import os
import pathlib
import pdb
import sys

import matplotlib.pyplot as plt
import numpy as np
from tbparse import SummaryReader
from itertools import accumulate


def propcess_plot_data(
    episode_length,
    tensorabord_reward,
    tensorboard_violation,
    violation_file_name,
    reward_file_name,
    random_start
):

    tb_episode = []
    tb_reward = []
    tb_violations = []

    reader = SummaryReader(tensorabord_reward, pivot=True)
    reward_df = reader.scalars
    for idx, row in reward_df.iterrows():
        tb_episode.append(row["step"] / episode_length)
        tb_reward.append(row["rollout/ep_rew_mean"])

    # pdb.set_trace()

    if not random_start:
        reader = SummaryReader(tensorboard_violation, pivot=True)
        violation_df = reader.scalars
        for idx, row in violation_df.iterrows():
            if row["step"] % episode_length == 0:
                # finish one episode, log the number of violations
                tb_violations.append(row["safe_violation"])

    # pdb.set_trace()
    print(tb_violations)
    # assert len(tb_episode) == len(tb_reward) and len(tb_reward) == len(tb_violations)

    with open(violation_file_name, "r") as violation_file:
        violation_lines = violation_file.readlines()
        violation_lines = [int(x) for x in violation_lines]
        i = 0
        real_episode = []
        real_violations = []
        while i < len(violation_lines):
            real_episode.append(violation_lines[i] / episode_length)
            num_real_episodes = violation_lines[i + 1]

            i = i + 2
            for j in range(0, num_real_episodes):
                real_violations.append(violation_lines[i])
                i = i + 1
    # pdb.set_trace()

    with open(reward_file_name, "r") as reward_file:
        reward_lines = reward_file.readlines()
        reward_lines = [float(x) for x in reward_lines]
        i = 0
        real_episode = []
        real_rewards = []
        while i < len(reward_lines):
            real_episode.append(int(reward_lines[i]) / episode_length)
            num_real_episodes = int(reward_lines[i + 1])

            i = i + 2
            for j in range(0, num_real_episodes):
                real_rewards.append(reward_lines[i])
                i = i + 1

    # pdb.set_trace()
    assert len(real_episode) * num_real_episodes == len(real_violations) and len(
        real_violations
    ) == len(real_rewards)

    tb_episode_idx = 0
    real_episode_idx = 0
    result_episode_idx = 0
    result_idx = []
    result_violations = []
    result_rewards = []
    # pdb.set_trace()
    for i in range(0, len(real_episode)):
        tb_upper_bound = real_episode[i]

        while tb_episode[tb_episode_idx] <= tb_upper_bound:
            result_idx.append(result_episode_idx)
            result_rewards.append(tb_reward[tb_episode_idx])

            if tb_episode_idx < len(tb_violations):
                result_violations.append(tb_violations[tb_episode_idx])
            else:
                if len(result_violations) == 0:
                    result_violations.append(0)
                else:
                    result_violations.append(result_violations[-1])

            result_episode_idx += 1
            tb_episode_idx += 1
            if tb_episode_idx >= len(tb_episode):
                # pdb.set_trace()
                break
        # pdb.set_trace()
        for j in range(0, num_real_episodes):
            try:
                result_idx.append(result_episode_idx)
                result_violations.append(
                    real_violations[real_episode_idx] + result_violations[-1]
                )  # calculate cumulative violations
            except:
                pass
                # pdb.set_trace()
            result_rewards.append(real_rewards[real_episode_idx])

            result_episode_idx += 1
            real_episode_idx += 1
        # pdb.set_trace()
    
    while tb_episode_idx < len(tb_reward):
        result_idx.append(result_episode_idx)
        result_rewards.append(tb_reward[tb_episode_idx])
        if result_episode_idx < len(tb_violations):
            result_violations.append(tb_violations[tb_episode_idx])
        else:
            result_violations.append(result_violations[-1])

        tb_episode_idx += 1
        result_episode_idx += 1

    if random_start:
        with open(os.path.join(tensorboard_violation, "start.txt")) as f:
            lines = f.readlines()
        start_reward = [float(x) for x in lines[0].split()]
        start_violations = [float(x) for x in lines[1].split()]
        start_violations_episode = [float(x > 0) for x in start_violations]
        start_violations_episode = list(accumulate(start_violations_episode))

        start_violations_step = [x for x in start_violations]
        start_violations_step = list(accumulate(start_violations_step))
        for i in range(0, len(start_reward)):
            result_idx.append(result_idx[-1] + 1)

        for i in range(0, len(result_violations)):
            result_violations[i] = result_violations[i] + start_violations_step[-1]
        
        result_violations = start_violations_step + result_violations
        result_rewards = start_reward + result_rewards
        # pdb.set_trace()
    return result_idx, result_violations, result_rewards


def plot_data(all_xs, all_violations, all_rewards, env_name):
    path = pathlib.Path(f"results/figures/{env_name}")
    path.mkdir(parents=True, exist_ok=True)

    min_len = min([len(xs) for xs in all_xs])
    final_xs = all_xs[0][:min_len]
    final_violations = [violations[:min_len] for violations in all_violations]
    final_rewards = [rewards[:min_len] for rewards in all_rewards]

    final_xs = np.array(final_xs)
    final_violations = np.array(final_violations)
    final_rewards = np.array(final_rewards)

    # takes the processed test as result and directly put them into a figure
    plt.plot(final_xs, np.mean(final_violations, axis=0))
    plt.fill_between(
        final_xs,
        np.min(final_violations, axis=0),
        np.max(final_violations, axis=0),
        alpha=0.3,
    )

    plt.ylabel("cumulative safe violations")
    plt.xlabel("episodes")
    plt.savefig(os.path.join(path, "violations.png"))

    plt.cla()
    plt.plot(final_xs, np.mean(final_rewards, axis=0))
    plt.fill_between(
        final_xs,
        np.min(final_rewards, axis=0),
        np.max(final_rewards, axis=0),
        alpha=0.3,
    )

    plt.ylabel("rewards")
    plt.xlabel("episodes")
    plt.savefig(os.path.join(path, "rewards.png"))


if __name__ == "__main__":
    env_name = sys.argv[1]

    gym_env_name_map = {
        "acc": "Marvelgymnasium_acc-v1",
        "obstacle": "Marvelgymnasium_obstacle-v1",
        "pendulum": "Marvelgymnasium_pendulum-v1",
        "road": "Marvelgymnasium_road-v1",
    }

    env_episode_length = {"acc": 300, "obstacle": 200, "pendulum": 100, "road": 300}

    gym_env_name = gym_env_name_map[env_name]

    log_path = pathlib.Path(f"results/logs/{gym_env_name}/")
    num_runs = len(os.listdir(log_path))

    all_xs = []
    all_violations = []
    all_rewards = []
    for run_id in range(1, num_runs + 1):
        xs, violations, rewards = propcess_plot_data(
            env_episode_length[env_name],
            pathlib.Path(f"results/logs/{gym_env_name}/sac_{run_id}"),
            pathlib.Path(f"results/safe_violation/{gym_env_name}/sac_{run_id}"),
            pathlib.Path(
                f"results/plot_log/{gym_env_name}/sac_{run_id}/safe_violations.txt"
            ),
            pathlib.Path(f"results/plot_log/{gym_env_name}/sac_{run_id}/rewards.txt"),
        )
        all_xs.append(xs)
        all_violations.append(violations)
        all_rewards.append(rewards)

    plot_data(all_xs, all_violations, all_rewards, env_name)
