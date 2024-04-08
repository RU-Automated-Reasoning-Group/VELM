import os
import pathlib
import pdb

import gymnasium as gym
import matplotlib.patches
import numpy as np
from matplotlib import pyplot as plt

from environments.simulated_env_util import make_simulated_env


def check_state_in_box(state, safe_set):
    dim = len(safe_set) // 2
    safe = True
    for i in range(0, dim):
        if state[i] < safe_set[2 * i] or state[i] > safe_set[2 * i + 1]:
            safe = False
            break

    return safe


class safe_agent:
    def __init__(
        self,
        neural_agent,
        linear_models: np.ndarray,
        simulated_env,
        safe_sets,
        horizon: int,
        interval: int = 1,
        current_version: int = 1,
    ):
        # pdb.set_trace()
        self.neural_agent = neural_agent
        self.num_of_controller = horizon // interval
        self.linear_models = linear_models.reshape((self.num_of_controller, -1))
        self.simulated_env = simulated_env
        self.safe_sets = safe_sets
        self.interval = interval
        self.total_query_count = 0
        self.neural_query_count = 0
        self.current_version = current_version

    def sample(
        self,
        args,
        logger,
        replay_buffer,
        env_info,
        simulated_env_info,
        learned_dynamic_model,
        learned_stds,
        episodes: int,
        plot_unsafe_set,
        plot_state_to_xy,
    ):
        real_env = gym.make(env_info.env_name)
        plt.cla()
        copy_buffer = []
        episodes_rwd = []
        episodes_violations = []
        unsafe_count = 0
        for i in range(0, episodes):
            xs = []
            ys = []
            self.current_step = 0
            state, _ = real_env.reset()
            x, y = plot_state_to_xy(state)
            xs.append(x)
            ys.append(y)
            done = False
            episode_rwd = 0.0
            episode_violation = 0
            while not done:
                action, intervented = self(state)
                next_state, rwd, terminated, truncated, info = real_env.step(action)
                done = terminated or truncated
                if real_env.unsafe():
                    unsafe_count += 1
                    episode_violation += 1
                # tracker.add_step(not env.unsafe())
                # print(f"next state is {next_state}")
                # safe = check_state_in_box(
                # next_state, self.safe_sets[self.current_step // self.interval - 1]
                # )
                # print(
                # f"next safe set is {self.safe_sets[self.current_step // self.interval - 1]}"
                # )
                # print(f"next state safety is {safe}")
                # replay_buffer.add(
                #     np.array([state]),
                #     np.array([next_state]),
                #     np.array([action]),
                #     rwd,
                #     done,
                #     [info],
                # )

                # plot
                color = "orange" if intervented else "blue"
                x, y = plot_state_to_xy(state)
                next_x, next_y = plot_state_to_xy(next_state)
                plt.plot([x, next_x], [y, next_y], color=color)

                state = next_state
                if np.any(state <= real_env.observation_space.low) or np.any(state >= real_env.observation_space.high) or \
                   np.any(next_state <= real_env.observation_space.low) or np.any(state >= real_env.observation_space.high):
                    # only add sas tuple to buffer if states are not clipped.
                    copy_buffer.append((state.copy(), action.copy(), next_state.copy()))
                x, y = plot_state_to_xy(state)
                xs.append(x)
                ys.append(y)
                episode_rwd += rwd
                self.current_step += 1
            # plt.plot(xs, ys)
            episodes_rwd.append(episode_rwd)
            episodes_violations.append(episode_violation)
        # plot safe and unsafe sets
        plot_unsafe_set()
        for i in range(len(self.safe_sets)):
            lower_left_point, width, height = self.safe_set_to_rectangle(
                self.safe_sets[i], plot_state_to_xy
            )
            ax = plt.gca()
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    lower_left_point,
                    width,
                    height,
                    facecolor="green",
                    alpha=0.5,
                    ec="green",
                    lw=2,
                )
            )

        path = pathlib.Path(
            f"./results/sampled_strajectories/{env_info.env_name}/sac_{self.current_version}"
        )
        path.mkdir(parents=True, exist_ok=True)
        # pdb.set_trace()

        fig_name = os.path.join(path, f"{self.neural_agent.num_timesteps}.png")
        print(f"saving safe agent trajectories to {fig_name}")
        plt.savefig(fig_name)
        print(f"mean episode reward for safe agent is {np.mean(episodes_rwd)}")
        print(f"unsafe count is {unsafe_count}")

        # add stats to logger
        for (one_rwd, one_violation) in zip(episodes_rwd, episodes_violations):
            logger.manual_add(one_rwd, one_violation)

        # logging safe violation stats for plot
        plot_log = f"./results/plot_log/{env_info.env_name}/sac_{self.current_version}"
        path = pathlib.Path(plot_log)
        path.mkdir(parents=True, exist_ok=True)
        violation_log_name = os.path.join(path, "safe_violations.txt")
        with open(violation_log_name, "a") as file:
            lines = [str(self.neural_agent.num_timesteps), str(episodes)]
            lines += [str(x) for x in episodes_violations]
            file.write("\n".join(lines))
            file.write("\n")

        reward_log_name = os.path.join(path, "rewards.txt")
        with open(reward_log_name, "a") as file:
            lines = [str(self.neural_agent.num_timesteps), str(episodes)]
            lines += [str(x) for x in episodes_rwd]
            file.write("\n".join(lines))
            file.write("\n")

        intervention_log_name = os.path.join(path, "intervention.txt")
        with open(intervention_log_name, "a") as file:
            lines = [
                f"total query count {self.total_query_count}\n",
                f"total nerual count {self.neural_query_count}\n",
            ]
            file.writelines(lines)

        linear_plot_name = os.path.join(
            path, f"linear_plot_{self.neural_agent.num_timesteps}.png"
        )
        self.creat_linear_controller_figure(
            args,
            linear_plot_name,
            env_info,
            simulated_env_info,
            learned_dynamic_model,
            learned_stds,
            plot_unsafe_set,
            plot_state_to_xy,
        )

        return copy_buffer

    def creat_linear_controller_figure(
        self,
        args,
        linear_plot_name,
        env_info,
        simulated_env_info,
        learned_dynamic_model,
        learned_stds,
        plot_unsafe_set,
        plot_state_to_xy,
    ):
        test_env = gym.make(env_info.env_name)
        simulated_test_env = make_simulated_env(
            args.random,
            simulated_env_info.env_name,
            learned_dynamic_model,
            learned_stds,
        )
        plt.cla()

        # deploy linear controller in the real environment
        state, _ = test_env.reset()
        state_copy = state.copy()
        xs, ys = [], []
        x, y = plot_state_to_xy(state)
        xs.append(x)
        ys.append(y)
        terminate, truncated = False, False
        current_step = 0
        while not (terminate or truncated):
            linear_action = self.get_linear_action(current_step, state)
            next_state, _, terminate, truncated, _ = test_env.step(linear_action)
            state = next_state
            x, y = plot_state_to_xy(state)
            xs.append(x)
            ys.append(y)
            current_step += 1
        plt.plot(xs, ys, color="blue", label="real trajectory (linear)")

        # deploy the neural agent in the real environment
        test_env.reset()
        test_env.env.env.env.state = state_copy.copy()
        state = state_copy.copy()

        xs, ys = [], []
        x, y = plot_state_to_xy(state)
        xs.append(x)
        ys.append(y)
        terminate, truncated = False, False
        while not (terminate or truncated):
            neural_action, _ = self.neural_agent.predict(
                np.array([state]), deterministic=True
            )
            neural_action = neural_action[0]
            next_state, _, terminate, truncated, _ = test_env.step(neural_action)
            state = next_state
            x, y = plot_state_to_xy(state)
            xs.append(x)
            ys.append(y)
        plt.plot(xs, ys, color="red", label="real trajectory (neural)")

        simulated_test_env.reset()
        simulated_test_env.env.env.env.state = state_copy.copy()
        state = state_copy.copy()

        xs, ys = [], []
        x, y = plot_state_to_xy(state)
        xs.append(x)
        ys.append(y)
        terminate, truncated = False, False
        current_step = 0
        while not (terminate or truncated):
            linear_action = self.get_linear_action(current_step, state)
            next_state, _, terminate, truncated, _ = simulated_test_env.step(
                linear_action
            )
            state = next_state
            x, y = plot_state_to_xy(state)
            xs.append(x)
            ys.append(y)
            current_step += 1
        plt.plot(xs, ys, color="black", label="simulated trajectory (linear)")

        # plot the safe sets
        for i in range(len(self.safe_sets)):
            lower_left_point, width, height = self.safe_set_to_rectangle(
                self.safe_sets[i], plot_state_to_xy
            )
            ax = plt.gca()
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    lower_left_point,
                    width,
                    height,
                    facecolor="green",
                    alpha=0.5,
                    ec="green",
                    lw=2,
                )
            )

        plot_unsafe_set()
        plt.legend()
        plt.savefig(linear_plot_name)

    def safe_set_to_rectangle(self, safe_set, plot_state_to_xy):
        dim = len(safe_set) // 2
        lower_left_state = [safe_set[2 * x] for x in range(dim)]
        lower_left_point = plot_state_to_xy(lower_left_state)

        upper_right_state = [safe_set[2 * x + 1] for x in range(dim)]
        upper_right_point = plot_state_to_xy(upper_right_state)
        return (
            lower_left_point,
            (upper_right_point[0] - lower_left_point[0]),
            (upper_right_point[1] - lower_left_point[1]),
        )

    def __call__(self, state, debug=False, strict=False):
        # strict means state can only progress box by box. Cannot skip any box

        neural_action, _ = self.neural_agent.predict(
            np.array([state]), deterministic=True
        )
        neural_action = neural_action[0]

        proposed_state = self.simulated_env.simulate(state, neural_action)

        # check if proposed state is in *any* safe sets
        proposed_state_safe = False
        safe_set_idx = -1
        if strict:
            safe_set_idx = self.current_step
            proposed_state_safe = check_state_in_box(
                proposed_state, self.safe_sets[safe_set_idx]
            )
        else:
            safe_set_idx = self.current_step

            for idx, safe_set in enumerate(self.safe_sets[:safe_set_idx + 1]):
                if check_state_in_box(proposed_state, safe_set):
                    # found one safe set that contains the proposed state
                    proposed_state_safe = True
                    safe_set_idx = idx
                    break

        if debug:
            print(f"neural action is {neural_action}")
            print(f"proposed state is {proposed_state}")
            print(
                f"current safe set is {None if safe_set_idx == -1 else self.safe_sets[safe_set_idx]}"
            )
            print(f"proposed_state_safe is {proposed_state_safe}")

        if proposed_state_safe:
            self.neural_query_count += 1
            final_action = neural_action
            intervented = False
        else:
            if strict:
                safe_set_idx = self.current_step - 1
            else:
                # find one safe set that contains the current state
                if self.current_step == 0:
                    safe_set_idx = self.current_step - 1
                else:
                    safe_set_idx = -1
                    for idx, safe_set in enumerate(self.safe_sets[:self.current_step]):
                        if check_state_in_box(state, safe_set):
                            safe_set_idx = idx
                            break

                    if safe_set_idx < 0:
                        # pdb.set_trace()
                        safe_set_idx = self.current_step - 1

            # use the corresponding linear controller to give an action
            linear_model = self.linear_models[safe_set_idx + 1]

            # use linear model to give action
            # print(f"using step {self.current_step // self.interval} while the linear model has shape {self.linear_models.shape}")
            # linear_model = self.linear_models[self.current_step // self.interval]
            state_dim = len(state)
            action_dim = len(linear_model) // (state_dim + 1)
            linear_model = linear_model.reshape((-1, state_dim + 1))
            # print(f"linear model is reshaped to {linear_model.shape}")
            modified_state = np.ones(state_dim + 1)
            modified_state[:state_dim] = state.copy()
            linear_action = np.inner(linear_model, modified_state)
            final_action = linear_action

            intervented = True

            if debug:
                print(f"safe set id for current state is {safe_set_idx}")
                print(f"linear model is {linear_model}")
                print(f"linear action is {linear_action}")

        self.total_query_count += 1

        return final_action, intervented

    def get_linear_action(self, current_step, state):
        linear_model = self.linear_models[current_step]

        state_dim = len(state)
        action_dim = len(linear_model) // (state_dim + 1)
        linear_model = linear_model.reshape((-1, state_dim + 1))
        # print(f"linear model is reshaped to {linear_model.shape}")
        modified_state = np.ones(state_dim + 1)
        modified_state[:state_dim] = state.copy()
        linear_action = np.inner(linear_model, modified_state)
        return linear_action

    def report(self):
        print("[safe agent report]")
        print(
            f"intervention rate is {1 - self.neural_query_count / self.total_query_count}"
        )
        print(f"total query count is {self.total_query_count}")
