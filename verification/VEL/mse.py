import gymnasium as gym
import numpy as np
import onnx
import onnx2torch
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import SAC

from environments.simulated_env_util import make_simulated_env


class seq_linear:
    def __init__(self, num_controller, iters_run, controller_sz, params, state_dim, action_dim):
        self.num_controller = num_controller
        self.iters_run = iters_run
        self.controller_sz = controller_sz
        self.horizon = num_controller * iters_run
        self.linear_policy = []
        # self.dim = controller_sz - 1
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.params = params.copy()
        for i in range(0, num_controller):
            for j in range(0, iters_run):
                self.linear_policy.append(
                    np.array(params[i * controller_sz : (i + 1) * controller_sz]).reshape(self.action_dim, -1)
                )

    def get_action(self, o, t):
        # import pdb
        # pdb.set_trace()
        modified_o = np.ones(self.state_dim + 1)
        modified_o[0 : self.state_dim] = o
        return np.inner(self.linear_policy[t], modified_o)

    def sample_from_initial_states(self, initial_states, env_str, learned_model, stds, random, horizon, plot=False):
        gym_env = make_simulated_env(random, env_str, learned_model=learned_model, stds=stds)
        # gym_env.load_model()
        linear_paths = []
        complete_paths = []
        sample_interval = 10
        for idx, initial_state in enumerate(initial_states):
            gym_env.reset()
            gym_env.env.state = initial_state
            obs = initial_state
            path = [obs]
            complete_path = [obs]
            done = False
            t = 0

            while t < horizon and done is not True:
                action = self.get_action(obs, t)
                obs, r, terminated, truncated, _ = gym_env.step(action)
                done = terminated or truncated
                if (t + 1) % sample_interval == 0:
                    path.append(obs)
                complete_path.append(obs)
                # print("t is ", t)
                t += 1
            complete_paths.append(complete_path)
            linear_paths.append(path)

        if plot:
            complete_paths = np.array(complete_paths)
            dims = len(complete_paths[0][0])
            for x in range(0, dims):
                plt.cla()
                for complete_path in complete_paths:
                    plt.plot(complete_path[:, x].reshape(-1))
                plt.savefig(f"linear_path_{x}.png")
        return np.array(linear_paths)


class neural_model_from_onnx:
    def __init__(self, onnx_file):
        self.onnx_model = onnx.load(onnx_file)
        self.torch_model = onnx2torch.convert(self.onnx_model)

    def get_action(self, o, t):
        o = np.float32(o.reshape(1, -1))
        obs = torch.from_numpy(o)
        return self.torch_model(obs).data.numpy().ravel()

    def sample_from_initial_states(self, initial_states, env_str, horizon, plot=False):
        gym_env = gym.make(env_str)
        neural_paths = []
        complete_paths = []
        sample_interval = 10
        for idx, initial_state in enumerate(initial_states):
            print(idx)
            gym_env.reset()
            gym_env.env.state = initial_state
            obs = initial_state
            path = [obs]
            complete_path = [obs]
            done = False
            t = 0

            while t < horizon and done is not True:
                action = self.get_action(obs, t)
                obs, r, done, _ = gym_env.step(action)
                if (t + 1) % sample_interval == 0:
                    path.append(obs)
                complete_path.append(obs)
                t += 1
            complete_paths.append(complete_path)
            neural_paths.append(path)

        # import pdb
        # pdb.set_trace()
        if plot:
            complete_paths = np.array(complete_paths)
            dims = len(complete_paths[0][0])
            for x in range(0, dims):
                plt.cla()
                for complete_path in complete_paths:
                    plt.plot(complete_path[:, x].reshape(-1))
                plt.savefig(f"neural_path_{x}.png")
        return np.array(neural_paths)


class neural_model:
    def __init__(self, model):
        self.model = model

    def get_action(self, o, t):
        o = np.float32(o.reshape(1, -1))
        obs = torch.from_numpy(o)
        if type(self.model) == SAC:
            return self.model.predict(obs, deterministic=True)
        return self.model.model(obs).data.numpy(), None

    def sample_from_initial_states(self, initial_states, env_str, learned_model, stds, random, horizon, plot=True):
        print("==== sampling neural paths ====")
        # gym_env = gym.make(env_str, learned_model=learned_model)
        gym_env = make_simulated_env(random, env_str, learned_model=learned_model, stds=stds)

        # gym_env.load_model()
        neural_paths = []
        complete_paths = []
        sample_interval = 10
        for idx, initial_state in enumerate(initial_states):
            # print(idx)
            gym_env.reset()
            gym_env.env.state = initial_state
            obs = initial_state
            path = [obs]
            complete_path = [obs]
            done = False
            t = 0

            while t < horizon and done is not True:
                action, _ = self.get_action(obs, t)
                action = action[0]
                # import pdb
                # pdb.set_trace()
                obs, r, terminated, truncated, _ = gym_env.step(action)
                done = terminated or truncated
                if (t + 1) % sample_interval == 0:
                    path.append(obs)
                complete_path.append(obs)
                t += 1
            complete_paths.append(complete_path)
            neural_paths.append(path)

        # import pdb
        # pdb.set_trace()
        if plot:
            complete_paths = np.array(complete_paths)
            dims = len(complete_paths[0][0])
            for x in range(0, dims):
                plt.cla()
                for complete_path in complete_paths:
                    plt.plot(complete_path[:, x].reshape(-1))
                plt.savefig(f"neural_path_{x}.png")
        print("==== done ====")
        return np.array(neural_paths)


# class MSE():
#     # this class is used to sample paths and calculate MSE loss
#     def __init__(self, policy_model, env, seed=None):
#         self.policy_model = policy_model
#         self.seed = 0 if seed is None else seed
#         self.dyn_env = env

#     def sample(self, N, horizon=1e6, num_cpu='max', gamma=0.995, env_kwargs=None):
#         ts = timer.time()

#         input_dict = dict(num_traj=N, env=self.dyn_env, policy=self.policy_model, horizon=horizon,
#                           base_seed=self.seed, num_cpu=num_cpu, ignore_done=True, env_kwargs=env_kwargs)
#         paths = trajectory_sampler.sample_paths(**input_dict)

#         if self.save_logs:
#             self.logger.log_kv('time_sampling', timer.time() - ts)

#         self.seed = self.seed + N if self.seed is not None else self.seed

#         # compute returns
#         # process_samples.compute_returns(paths, gamma)

#         return paths

#     def get_loss(self, linear_paths, neural_paths):
#         assert (linear_paths.shape == neural_paths.shape)
#         return np.square(linear_paths - neural_paths).mean()


def get_mse_loss(linear_paths, neural_paths, print_detail):
    # import pdb
    # pdb.set_trace()
    # print(linear_paths.shape, neural_paths.shape)
    # assert linear_paths.shape == neural_paths.shape
    # return np.square(linear_paths - neural_paths).mean()
    # return np.square(linear_paths[:,:,0] - neural_paths[:,:,0]).mean()\
    #     + np.square(linear_paths[:,:,1] - neural_paths[:,:,1]).mean() \
    #     + np.square(linear_paths[:,:,2] - neural_paths[:,:,2]).mean() \
    #     + np.square(linear_paths[:,:,3] - neural_paths[:,:,3]).mean()

    # weighted mse loss
    part1 = np.square(linear_paths - neural_paths).mean()
    # part2 = np.square(linear_paths[:,-1,:] - neural_paths[:,-1,:]).mean()
    # if print_detail:
        # print(f"part 1: {part1}, part 2: {part2}")
        # print("mse loss", part1)
    return part1
