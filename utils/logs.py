import logging
import os

import matplotlib.pyplot as plt
import numpy as np


def init_logging(save_path, save_file="log.txt"):
    logfile = os.path.join(save_path, save_file)

    # clear log file
    with open(logfile, "w"):
        pass
    # remove previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, level=logging.INFO, format="%(message)s")


def log_and_print(line):
    print(line)
    logging.info(line)


class Logger:
    def __init__(self, store_path):
        self.trajectories = {
            "rewards": [],
            "states": [],
            "actions": [],
            "Vs": [],
            "enforce_rewards": [],
            "enforce_states": [],
            "enforce_actions": [],
            "enforce_Vs": [],
        }
        self.loss_dict = {}
        self.store_path = store_path
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)

    def store_trajectory(
        self, rewards, states, actions, Vs, enf_rewards, enf_states, enf_actions, enf_Vs
    ):
        self.trajectories["rewards"].append(rewards)
        self.trajectories["states"].append(states)
        self.trajectories["actions"].append(actions)
        self.trajectories["Vs"].append(Vs)
        self.trajectories["enforce_rewards"].append(enf_rewards)
        self.trajectories["enforce_states"].append(enf_states)
        self.trajectories["enforce_actions"].append(enf_actions)
        self.trajectories["enforce_Vs"].append(enf_Vs)

    def store_loss(self, loss_dict):
        for k in loss_dict:
            if k not in self.loss_dict:
                self.loss_dict[k] = []
            self.loss_dict[k].append(loss_dict[k])

    def store_log(self):
        np.save(os.path.join(self.store_path, "loss.pth"), self.loss_dict)
        np.save(os.path.join(self.store_path, "traj.pth"), self.trajectories)

    def draw_traj(self, rewards, states, actions, epoch, fig_path="figs"):
        # figure root path
        fig_root = os.path.join(self.store_path, fig_path)
        if not os.path.exists(fig_root):
            os.makedirs(fig_root)

        # print rewards
        plt.figure()
        for episode_rewards in rewards:
            plt.plot(np.arange(len(episode_rewards)), episode_rewards)
        plt.savefig(os.path.join(fig_root, "reward_{}.png".format(epoch)))
        plt.close()

        # print states
        for s_id in range(states.shape[-1]):
            plt.figure()
            for episode_state in states:
                plt.plot(np.arange(len(episode_state[:, s_id])), episode_state[:, s_id])
            plt.savefig(os.path.join(fig_root, "state_{}_{}.png".format(s_id, epoch)))
            plt.close()

        # print actions
        for a_id in range(actions.shape[-1]):
            plt.figure()
            for episode_action in actions:
                plt.plot(
                    np.arange(len(episode_action[:, a_id])), episode_action[:, a_id]
                )
            plt.savefig(os.path.join(fig_root, "act_{}_{}.png".format(a_id, epoch)))
            plt.close()
