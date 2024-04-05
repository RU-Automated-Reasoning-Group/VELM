import csv
import random

import numpy as np
import torch


class ReplayMemory:
    def __init__(self, args, get_torch=True):
        self.capacity = args.capacity
        self.buffer = []
        self.position = 0
        self.get_torch = get_torch

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, mask = map(np.stack, zip(*batch))

        if self.get_torch:
            state = torch.tensor(state).type(torch.float32)
            action = torch.tensor(action).type(torch.float32)
            reward = torch.tensor(reward).type(torch.float32).unsqueeze(1)
            next_state = torch.tensor(next_state).type(torch.float32)
            mask = torch.tensor(mask).type(torch.float32).unsqueeze(1)

        return state, action, reward, next_state, mask

    def sample_for_dso(self, data_num=None, preprocess=False, dt=None):
        if data_num is None:
            data_num = len(self.buffer)

        dataset = random.sample(self.buffer, data_num)
        X_d, y_list = [], [[] for _ in range(len(dataset[0][-2]))]
        # pdb.set_trace()
        for d in dataset:
            X_d.append(np.concatenate([d[0], d[1]]))
            for s_d in range(len(d[-2])):
                if preprocess is True:
                    # append the different between next state and current state divided by dt
                    y_list[s_d].append((d[-2][s_d] - d[0][s_d]) / dt)
                else:
                    # only append the next state
                    y_list[s_d].append(d[-2][s_d])

        return np.array(X_d), np.array(y_list)

    def write_CSV_for_dso(self, benchmark_name, X, y_list):
        n_dim = y_list.shape[0]
        for i in range(0, n_dim):
            ys = y_list[i]
            with open(f"{benchmark_name}_{i}.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                n_data = len(ys)
                for j in range(0, n_data):
                    row = X[j].tolist() + [ys[j]]
                    writer.writerow(row)

    def write_pickle_for_EQL(self, benchmark_name, X, y_list):
        dim = 1
        y_list = y_list.T
        y_list = y_list[:, dim]
        y_list = y_list[:, np.newaxis]
        # pdb.set_trace()
        datfull = self.split_dataset(X, y_list)
        import gzip

        import _pickle as cPickle

        cPickle.dump(
            datfull,
            gzip.open(f"data/{benchmark_name}_dim{dim}.dat.gz", "wb"),
            protocol=2,
        )

    def split_dataset(self, inputs, outputs):
        assert len(inputs) == len(outputs)
        size = len(inputs)
        ts = int(size * 90 / 100)
        # vs = size * 10 / 100
        train_set = (inputs[:ts], outputs[:ts])
        valid_set = (inputs[ts:], outputs[ts:])
        return train_set, valid_set


def push_trajectory(
    env, buffer, policy, traj_len, evaluate=False, enforce_optimal=False
):
    # init state
    init_state = env.reset()
    state = init_state

    # sample and store
    for traj_id in range(traj_len):
        # pdb.set_trace()
        u = policy(
            torch.tensor(np.array([state])).type(torch.float32).to(policy.device),
            evaluate,
        )

        next_state, r, done, _ = env.step(u)
        mask = 1 if traj_id == traj_len - 1 else float(not done)
        # pdb.set_trace()
        buffer.push(state, u, r, next_state, mask)
        if done:
            state = env.reset()
        else:
            state = next_state


def push_trajectory_with_safe_sets(
    env, buffer, policy, traj_len, linear_models, safe_sets, evaluate=False
):
    # init state
    init_state = env.reset()
    state = init_state

    # sample and store
    index = 0
    print(init_state)
    original = 0
    total = 0
    for traj_id in range(traj_len):
        # pdb.set_trace()
        u = policy(
            torch.tensor(np.array([state])).type(torch.float64).to(policy.device),
            evaluate,
        )

        next_state = env.simulate(u)
        if (
            next_state[0] > safe_sets[index][0]
            and next_state[0] < safe_sets[index][1]
            and next_state[1] > safe_sets[index][2]
            and next_state[1] < safe_sets[index][3]
        ):
            # take this step
            next_state, r, done, _ = env.step(u)
            action_taken = u
            # print("original step")
            original += 1
        else:
            # use the linear model
            model_idx = index // 10
            # print(model_idx)
            linear_model = linear_models[model_idx]
            next_state, r, done, _ = env.step(
                np.array(
                    [
                        linear_model[0] * state[0]
                        + linear_model[1] * state[1]
                        + linear_model[2]
                    ]
                )
            )
            action_taken = np.array(
                [
                    linear_model[0] * state[0]
                    + linear_model[1] * state[1]
                    + linear_model[2]
                ]
            )
        mask = 1 if traj_id == traj_len - 1 else float(not done)

        buffer.push(state, action_taken, r, next_state, mask)
        if done:
            state = env.reset()
            index = 0
        else:
            state = next_state
            index += 1
        total += 1
    print("intervention rate", 1 - original / total)
    print("total", total)
    print("==== end of push trajectories === ")
