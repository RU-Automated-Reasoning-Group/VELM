import os
import pathlib
import pdb
import sys
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import parse
from tbparse import SummaryReader

from process_plot_data import propcess_plot_data

matplotlib.rcParams.update({"font.size": 16})


def full_plot(envs: List[List[Tuple[str, bool]]]):
    nrows = 8
    ncols = 4
    height_ratios=[1, 1, 0.01, 1, 1, 0.01, 1, 1]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, height_ratios=height_ratios)
    fig.set_figheight(3.5 * nrows)
    fig.set_figwidth(16.9)

    for i in range(len(ax[2])):
        ax[2, i].set_visible(False)

    for i in range(len(ax[5])):
        ax[5, i].set_visible(False)

    ax[6, 0].set_visible(False)
    ax[6, 3].set_visible(False)
    ax[7, 0].set_visible(False)
    ax[7, 3].set_visible(False)

    for row, row_envs in enumerate(envs):
        for idx, env_name_tuple in enumerate(row_envs):
            if env_name_tuple[0] == "empty":
                continue
            print(f"ploting data for {env_name_tuple}")
            lines = plot_data_one_env(ax, row, idx, env_name_tuple)

    fig.legend(
        (lines[3], lines[0], lines[1], lines[2]),
        ( "VELM", "SAC", "SPICE", "MBPPO-Lagrangian", "CRABS"),
        ncols=5,
        frameon=False,
        bbox_to_anchor=(0.25, 0.93),
        loc='upper left'
    )

    ax[0, 0].set_ylim([-10, 0])
    ax[0, 0].set_xlim([0, 200])
    ax[1, 0].set_xlim([0, 200])
    ax[1, 0].set_ylim([-100, 1000])

    ax[0, 1].set_ylim([200, 600])
    ax[0, 1].set_xlim([0, 200])
    ax[1, 1].set_xlim([0, 200])
    ax[1, 1].set_ylim([-100, 700])

    ax[0, 2].set_ylim([-4000, -300])
    ax[0, 2].set_xlim([0, 400])
    ax[1, 2].set_xlim([0, 400])
    ax[1, 2].set_ylim([-100, 700])

    ax[0, 3].set_ylim([-2000, 100])
    ax[0, 3].set_xlim([0, 200])
    ax[1, 3].set_xlim([0, 200])
    # ax[1, 3].set_ylim([-100, 10000])

    ax[3, 0].set_ylim([-3000, 0])
    ax[3, 0].set_xlim([0, 200])
    ax[4, 0].set_xlim([0, 200])
    # ax[4, 0].set_ylim([-30, 400])

    ax[3, 1].set_ylim([-6000, 0])
    ax[3, 1].set_xlim([0, 100])
    ax[4, 1].set_xlim([0, 100])
    ax[4, 1].set_ylim([-100, 2500])

    ax[3, 2].set_ylim([-6000, -1500])
    ax[3, 2].set_xlim([0, 600])
    ax[4, 2].set_xlim([0, 600])
    ax[4, 2].set_ylim([-100, 1500])


    ax[3, 3].set_ylim([-2000, 100])
    ax[3, 3].set_xlim([0, 300])
    ax[4, 3].set_xlim([0, 300])
    ax[4, 3].set_ylim([-100, 7000])

    ax[6, 1].set_ylim([-4000, 1000])
    ax[6, 1].set_xlim([0, 300])
    ax[7, 1].set_xlim([0, 300])
    ax[7, 1].set_ylim([-100, 10000])

    ax[6, 2].set_ylim([-1000, 0])
    ax[6, 2].set_xlim([0, 300])
    ax[7, 2].set_xlim([0, 300])
    ax[7, 2].set_ylim([-100, 2000])
    fig.align_ylabels()

    plt.subplots_adjust(wspace=0.5)
    # fig.tight_layout()

    # legend(",
    #             bbox_transform=fig.transFigure, ncol=3)
    plt.savefig("performance.png", bbox_inches='tight')


def plot_data_one_env(
    ax, row, idx, env_name_tuple, colors=["blue", "green", "orange", "red", "pink"]
):
    extract_data_fns = [
        extract_velm_data,
        extract_sac_data,
        extract_spice_data,
        extract_mbppo_data,
        extract_crabs_data,
    ]
    lines = []
    env_name, random_start = env_name_tuple
    for i in [1, 2, 3, 0]:
        print(i)
        all_xs, all_violations, all_rewards = extract_data_fns[i](
            env_name, random_start
        )
        # if env_name == "lalo":
        # if i == 1:
            # pdb.set_trace()
        name2title = {
            "pendulum": "Pendulum",
            "acc": "ACC",
            "obstacle_mid": "Obstacle2",
            "cartpole": "CartPole",
            "obstacle": "Obstacle",
            "road_2d": "Road2D",
            "car_racing": "CarRacing",
            "cartpole_move": "CartPoleMove",
            "cartpole_swing": "CartPoleSwing",
            "lalo": "LALO20",
        }

        line = plot_data(
            ax, row, idx, name2title[env_name], colors[i], all_xs, all_violations, all_rewards
        )
        lines.append(line)
    return tuple(lines)

def extract_crabs_data():
    pass

def plot_data(ax, row, idx, env_name, color, all_xs, all_violations, all_rewards):
    if row == 0:
        episodes_row = 1
        rewards_row = 0
    elif row == 1:
        episodes_row = 4
        rewards_row = 3
    elif row == 2:
        episodes_row = 7
        rewards_row = 6
    else:
        assert False

    min_len = min([len(xs) for xs in all_xs]) - 1
    final_xs = all_xs[0][:min_len]
    final_violations = [violations[:min_len] for violations in all_violations]
    final_rewards = [rewards[:min_len] for rewards in all_rewards]

    try:
        ax[episodes_row, idx].plot(final_xs, np.mean(final_violations, axis=0), color=color)
    except:
        pdb.set_trace()
    # print(f"{env_name} {np.mean(final_violations, axis=0)[-1]} {color}")
    # pdb.set_trace()

    ax[episodes_row, idx].fill_between(
        final_xs,
        np.min(final_violations, axis=0),
        np.max(final_violations, axis=0),
        alpha=0.3,
        color=color,
    )
    ax[episodes_row, idx].set_xlabel("# of episodes in total")
    if (idx == 0) or (idx == 1 and episodes_row == 7):
        ax[episodes_row, idx].set_ylabel("# unsafe steps")

    (line,) = ax[rewards_row, idx].plot(
        final_xs, np.mean(final_rewards, axis=0), color=color
    )
    ax[rewards_row, idx].fill_between(
        final_xs,
        np.min(final_rewards, axis=0),
        np.max(final_rewards, axis=0),
        alpha=0.3,
        color=color,
    )
    if (idx == 0) or (idx == 1 and rewards_row == 6):
        ax[rewards_row, idx].set_ylabel("total rewards")

    # title
    ax[rewards_row, idx].set_title(env_name)

    return line


def extract_velm_data(env_name, random_start):
    gym_env_name, eposide_len = gym_env_info(env_name)

    log_path = pathlib.Path(f"results/logs/{gym_env_name}/")
    num_runs = len(os.listdir(log_path))

    all_xs = []
    all_violations = []
    all_rewards = []

    for run_id in range(1, num_runs + 1):
        xs, violations, rewards = propcess_plot_data(
            eposide_len,
            pathlib.Path(f"results/logs/{gym_env_name}/sac_{run_id}"),
            pathlib.Path(f"results/safe_violation/{gym_env_name}/sac_{run_id}"),
            pathlib.Path(
                f"results/plot_log/{gym_env_name}/sac_{run_id}/safe_violations.txt"
            ),
            pathlib.Path(f"results/plot_log/{gym_env_name}/sac_{run_id}/rewards.txt"),
            random_start,
        )
        # violations = unsafe_steps_to_episodes(violations)
        # if env_name == "cartpole":
        #     pdb.set_trace()
        all_xs.append(xs)
        all_violations.append(violations)
        all_rewards.append(rewards)

    return make_all_seq_equal(all_xs, all_violations, all_rewards)

    # return all_xs, all_violations, all_rewards

def make_all_seq_equal(all_xs, all_violations, all_rewards):
    import numpy as np
    # pdb.set_trace()
    all_lens = [len(xs) for xs in all_xs]
    idx = np.argmax(all_lens)
    max_len = all_lens[idx]

    for i in range(len(all_xs)):
        all_xs[i] = all_xs[i] + all_xs[idx][all_lens[i]:]
        all_violations[i] = all_violations[i] + all_violations[idx][all_lens[i]:]
        all_rewards[i] = all_rewards[i] + all_rewards[idx][all_lens[i]:]
    
    return all_xs, all_violations, all_rewards


def extract_sac_data(env_name, random_start):
    gym_env_name, eposide_len = gym_env_info(env_name)

    log_path = pathlib.Path(f"results_sac/logs/{gym_env_name}/")
    num_runs = len(os.listdir(log_path))

    all_xs = []
    all_violations = []
    all_rewards = []

    for run_id in range(1, num_runs + 1):
        print("run_id", run_id)
        xs, violations, rewards = process_sac_plot_data(
            eposide_len,
            pathlib.Path(f"results_sac/logs/{gym_env_name}/sac_{run_id}"),
            pathlib.Path(f"results_sac/safe_violation/{gym_env_name}/sac_{run_id}"),
        )
        all_xs.append(xs)
        all_violations.append(violations)
        all_rewards.append(rewards)

    return all_xs, all_violations, all_rewards


def process_sac_plot_data(episode_length, tensorboard_reward, tensorboard_violation):
    tb_episode, tb_reward, tb_violation = [], [], []
    reward_reader = SummaryReader(tensorboard_reward, pivot=True)
    reward_df = reward_reader.scalars
    for idx, row in reward_df.iterrows():
        tb_episode.append(int(row["step"]) / episode_length)
        tb_reward.append(row["rollout/ep_rew_mean"])

    violation_reader = SummaryReader(tensorboard_violation, pivot=True)
    violation_df = violation_reader.scalars
    for idx, row in violation_df.iterrows():
        if row["step"] % episode_length == 0:
            # finish one episode, log the number of violations
            tb_violation.append(row["safe_violation"])

    # unsafe_episodes = unsafe_steps_to_episodes(tb_violation)

    # pdb.set_trace()
    return tb_episode, tb_violation, tb_reward


def unsafe_steps_to_episodes(unsafe_steps):
    unsafe_episodes = []
    previous_unsafe_steps = 0
    unsafe_count = 0
    for idx, unsafe_steps in enumerate(unsafe_steps):
        if unsafe_steps > previous_unsafe_steps:
            unsafe_count += 1

        previous_unsafe_steps = unsafe_steps
        unsafe_episodes.append(unsafe_count)

    return unsafe_episodes


def process_spice_plot_data(episode_length, logfile):
    with open(logfile) as f:
        lines = f.readlines()

    tb_episode, tb_reward, tb_violation = [], [], []

    new_episode = True
    current_count = 0

    for line in lines:
        if "UNSAFE (outside testing)" in line:
            # if new_episode:
            current_count += 1
            new_episode = False
        elif "total numsteps" in line:
            fmd_str = "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}"
            rst = parse.parse(fmd_str, line)
            # pdb.set_trace()
            tb_episode.append(int(rst[0]))
            tb_reward.append(float(rst[3]))

            new_episode = True
            tb_violation.append(current_count)

    return tb_episode, tb_violation, tb_reward


def extract_spice_data(env_name, random_start):
    gym_env_name, episode_len = gym_env_info(env_name)

    log_path = pathlib.Path(f"/home/yuning/spice_code/spice/results/{env_name}")
    num_runs = len(os.listdir(log_path))

    all_xs = []
    all_violations = []
    all_rewards = []

    for run_id in range(1, num_runs + 1):
        xs, violations, rewards = process_spice_plot_data(
            episode_len, os.path.join(log_path, f"sac_{run_id}")
        )
        all_xs.append(xs)
        all_violations.append(violations)
        all_rewards.append(rewards)

    return all_xs, all_violations, all_rewards


def process_mbppo_plot_data(episode_length, logfile):
    with open(logfile) as f:
        lines = f.readlines()

    tb_episode, tb_reward, tb_violation = [], [], []

    episode = 0
    current_count = 0

    for line in lines:
        if "episode" in line:
            fmt_str = "{} episode: ep_ret: {}, ep_cost: {}, ep_len: {}"
            rst = parse.parse(fmt_str, line)
            
            tb_episode.append(episode)
            episode += 1

            tb_reward.append(float(rst[1]))

            if rst[0] == "Real":
                current_count += int(rst[2])
            tb_violation.append(current_count)

    return tb_episode, tb_violation, tb_reward

def extract_mbppo_data(env_name, random_start):
    gym_env_name, episode_len = gym_env_info(env_name)
    log_path = pathlib.Path(f"/home/yuning/mbppol/final_results/{env_name}")
    num_runs = len(os.listdir(log_path))

    all_xs = []
    all_violations = []
    all_rewards = []

    for run_id in range(0, num_runs):
        xs, violations, rewards = process_mbppo_plot_data(
            episode_len, os.path.join(log_path, f"seed_{run_id}.txt")
        )
        all_xs.append(xs)
        all_violations.append(violations)
        all_rewards.append(rewards)
        
    return all_xs, all_violations, all_rewards

def gym_env_info(env_name):
    gym_env_name_map = {
        "acc": "Marvelgymnasium_acc-v1",
        "obstacle": "Marvelgymnasium_obstacle-v1",
        "pendulum": "Marvelgymnasium_pendulum-v1",
        "road": "Marvelgymnasium_road-v1",
        "cartpole": "Marvelgymnasium_cartpole-v1",
        "cartpole_move": "Marvelgymnasium_cartpole_move-v1",
        "cartpole_swing": "Marvelgymnasium_cartpole_swing-v1",
        "obstacle_mid": "Marvelgymnasium_obstacle_mid-v1",
        "road_2d": "Marvelgymnasium_road_2d-v1",
        "noisy_road_2d": "Marvelgymnasium_noisy_road_2d-v1",
        "car_racing": "Marvelgymnasium_car_racing-v1",
        "lalo": "Marvelgymnasium_lalo-v1",
    }

    env_episode_length = {
        "acc": 300,
        "obstacle": 200,
        "pendulum": 100,
        "road": 300,
        "cartpole": 200,
        "cartpole_move": 100,
        "cartpole_swing": 100,
        "obstacle_mid": 200,
        "road_2d": 300,
        "noisy_road_2d": 300,
        "car_racing": 200,
        "lalo": 100,
    }

    gym_env_name = gym_env_name_map[env_name]
    return gym_env_name, env_episode_length[env_name]


if __name__ == "__main__":
    full_plot(
        [
            [
                ("pendulum", False),
                ("acc", False),
                ("obstacle_mid", False),
                ("cartpole", True),
            ],
            [("obstacle", False), ("road_2d", False), ("car_racing", False), ("cartpole_move", True)],
            [ ("empty", True), ("cartpole_swing", True), ("lalo", True)],
        ]
    )

    # full_plot([[("car_racing", False)]])
    # full_plot([])
