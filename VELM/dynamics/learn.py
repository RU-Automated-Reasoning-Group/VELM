import matplotlib

# import tensorflow as tf
import numpy as np
import torch
import tqdm
from scipy.stats import beta

matplotlib.use("Agg")
import logging
import os
import pathlib
import time as timer
from os import path
from os.path import isfile

import matplotlib.pyplot as plt

logging.disable(logging.CRITICAL)

import sampler.core as trajectory_sampler
import utils.process_samples as process_samples
from absinterp.abs import Linear
from absinterp.deeppoly import Ele
from control.rl.policies.gaussian_prog import ProgPolicy, ToraPolicy
from utils.gym_env import GymEnv
from utils.logger import DataLog
from utils.prior_gmm import ControllerPriorGMM, DynamicsPriorGMM

# from onnx2torch import convert
# import onnx
# import onnxruntime

# gfile = tf.io.gfile


def util_chunk(*data, **kwargs):
    chunk_size = kwargs.pop("chunk_size", 100)
    shuffle = kwargs.pop("shuffle", False)
    show_progress = kwargs.pop("show_progress", None)
    N = len(data[0])
    if shuffle:
        permutation = np.random.permutation(N)
    else:
        permutation = np.arange(N)
    num_chunks = N // chunk_size
    if N % chunk_size > 0:
        num_chunks += 1
    rng = (
        tqdm.trange(num_chunks, desc=show_progress)
        if show_progress is not None
        else range(num_chunks)
    )
    for c in rng:
        chunk_slice = slice(c * chunk_size, (c + 1) * chunk_size)
        idx = permutation[chunk_slice]
        yield idx, tuple(d[idx] for d in data)


def util_linear_fit(Xy, idx_x, idx_y, reg=1e-6, prior=None):
    N = Xy.shape[0]
    mu = Xy.mean(axis=0)
    empsig = np.einsum("ia,ib->ab", Xy - mu, Xy - mu)
    sigma = 0.5 * (empsig + empsig.T) / N

    if prior:
        mu0, Phi, m, n0 = prior
        sigma = (N * sigma + Phi + (N * m) / (N + m) * np.outer(mu - mu0, mu - mu0)) / (
            N + n0
        )

    sigma[idx_x, idx_x] += np.eye(idx_x.stop) * reg
    mat = np.linalg.solve(sigma[idx_x, idx_x], sigma[idx_x, idx_y]).T
    lin = mu[idx_y] - mat.dot(mu[idx_x])
    cov = sigma[idx_y, idx_y] - mat.dot(sigma[idx_x, idx_x]).dot(mat.T)
    return mat, lin, cov


class DynamicsLearner:
    def __init__(self, env, policy_model, prior_type="gmm", save_logs=False, seed=None):
        self.dyn_env = env
        self.policy_model = policy_model
        self.horizon = env.horizon
        self.ds, self.da = env.observation_dim, env.action_dim
        self.prior_type = prior_type
        self.dynamics_stats = None
        self.max_steps = self.horizon - 1

        self.running_score = None
        self.save_logs = save_logs
        if save_logs:
            self.logger = DataLog()

        self.seed = 0 if seed is None else seed

    def sample(self, N, horizon=1e6, num_cpu="max", gamma=0.995, env_kwargs=None):
        print("==== sample paths ====")
        ts = timer.time()

        input_dict = dict(
            num_traj=N,
            env=self.dyn_env,
            policy=self.policy_model,
            horizon=horizon,
            base_seed=self.seed,
            num_cpu=num_cpu,
            ignore_done=True,
            env_kwargs=env_kwargs,
        )
        paths = trajectory_sampler.sample_paths(**input_dict)

        if self.save_logs:
            self.logger.log_kv("time_sampling", timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        print("==== done ====")
        return paths

    def process_paths(self, paths):
        # Concatenate from all the trajectories
        # observations = np.concatenate([path["observations"] for path in paths])
        # actions = np.concatenate([path["actions"] for path in paths])
        observations = np.empty([len(paths), self.horizon] + [self.ds])
        actions = np.empty([len(paths), self.horizon] + [self.da])
        # import pdb

        # pdb.set_trace()
        for i in range(len(paths)):
            observations[i][:, :] = paths[i]["observations"]
            actions[i][:, :] = paths[i]["actions"][:, :]

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        running_score = (
            mean_return
            if self.running_score is None
            else 0.9 * self.running_score + 0.1 * mean_return
        )

        return observations, actions, base_stats, running_score

    def upd_deviation(self):
        self.dv = []
        self.deviation = []
        T = len(self.S_D)
        sigma = np.zeros((T, self.ds + self.da, self.ds + self.da))
        idx_s = slice(self.ds)
        sigma[0, idx_s, idx_s] = self.S_D[0]
        for t in range(T):
            sigma[t] = np.vstack(
                [
                    np.hstack(
                        [
                            sigma[t, idx_s, idx_s],
                            np.matmul(sigma[t, idx_s, idx_s], np.transpose(self.K[t])),
                        ]
                    ),
                    np.hstack(
                        [
                            np.matmul(self.K[t], sigma[t, idx_s, idx_s]),
                            np.matmul(
                                np.matmul(self.K[t], sigma[t, idx_s, idx_s]),
                                np.transpose(self.K[t]),
                            )
                            + self.S_K[t],
                        ]
                    ),
                ]
            )
            if t < T - 1:
                sigma[t + 1, idx_s, idx_s] = self.S_D[t] + np.matmul(
                    np.matmul(self.D[t], sigma[t]), np.transpose(self.D[t])
                )
                # D[t].dot(sigma[t]).dot(D[t].T) + S_D[t]
            self.dv.append(5)
            self.deviation.append(
                torch.sqrt(
                    torch.diagonal(torch.from_numpy(sigma[t, idx_s, idx_s]).float(), 0)
                )
            )

    def model2Torch(self):
        self.dyn_linear = []
        self.pol_linear = []
        T = self.horizon
        for t in range(T):
            wA = torch.from_numpy(self.D[t]).float()
            A = Linear(wA.shape[1], wA.shape[0], bias=True)
            A.weight.data = wA

            bA = torch.from_numpy(self.d[t]).float()
            A.bias.data = bA

            for param in A.parameters():
                param.requires_grad = False

            self.dyn_linear.append(A)

            wK = torch.from_numpy(self.K[t])  # .float()
            K = Linear(wK.shape[1], wK.shape[0], bias=True)
            assert self.ds == wK.shape[1] and self.da == wK.shape[0]
            K.weight.data = wK

            bK = torch.from_numpy(self.k[t])  # .float()
            K.bias.data = bK

            for param in K.parameters():
                param.requires_grad = False

            self.pol_linear.append(K)

    def fit(self, paths, iters_run, horizon=None):
        import time

        a = time.time()
        observations, controls, base_stats, self.running_score = self.process_paths(
            paths
        )
        if self.save_logs:
            self.log_rollout_statistics(paths)
        print("process time is ", time.time() - a)

        N = observations.shape[0]
        T, ds, da = self.horizon if horizon is None else horizon, self.ds, self.da
        ds + da

        states, actions = np.zeros((N, T, ds)), np.zeros((N, T, da))
        chunk_size = 10
        for idx, chunk in tqdm.tqdm(
            util_chunk(observations[:, :], controls[:, :], chunk_size=chunk_size),
            desc="Encoding",
            total=N / chunk_size,
        ):
            states[idx], actions[idx] = chunk[0], chunk[1]
        self.mu_s0 = np.mean(states[:, 0], axis=0)
        self.S_s0 = np.diag(np.maximum(np.var(states[:, 0], axis=0), 1e-6))

        # Fit dynamics #
        gmm = DynamicsPriorGMM(
            {
                "max_samples": N,
                "max_clusters": 20,
                "min_samples_per_cluster": 40,
            }
        )
        a = time.time()
        gmm.update(states, actions)
        print("update time is ", time.time() - a)

        self.D, self.d = np.zeros((T, ds, ds + da)), np.zeros((T, ds))
        self.S_D = np.zeros((T, ds, ds))
        for t in tqdm.trange(T, desc="Fitting dynamics"):
            if t < T - 1:
                SAS_ = np.concatenate(
                    [states[:, t], actions[:, t], states[:, t + 1]],
                    axis=-1,
                )
                if self.prior_type == "gmm":
                    prior = gmm.eval(ds, da, SAS_)
                else:
                    prior = None
                self.D[t], self.d[t], self.S_D[t] = util_linear_fit(
                    SAS_,
                    slice(ds + da),
                    slice(ds + da, ds + da + ds),
                    prior=prior,
                )
                self.S_D[t] = 0.5 * (self.S_D[t] + self.S_D[t].T)

        # Fit controller #
        gmm = ControllerPriorGMM(
            {
                "max_samples": N,
                "max_clusters": 1,
                "min_samples_per_cluster": 10,
            }
        )
        gmm.update(states, actions)

        self.K, self.k = np.zeros((T, da, ds)), np.zeros((T, da))
        self.S_K = np.zeros((T, da, da))
        # for t in tqdm.trange(T, desc='Fitting controllers'):
        #     if t < T - 10:
        #         gg = [np.concatenate([states[:, t+i], actions[:, t+i]], axis=-1) for i in range(0, 10)]
        #         SAS_ = np.concatenate(gg, axis=0)
        #         if self.prior_type == 'gmm':
        #             prior = gmm.eval(ds, da, SAS_)
        #         else:
        #             prior = None
        #         self.K[t], self.k[t], self.S_K[t] = util_linear_fit(
        #                 SAS_, slice(ds), slice(ds, ds+da), prior=prior,
        #         )
        #         self.S_K[t] = 0.5 * (self.S_K[t] + self.S_K[t].T)

        num_controllers = self.horizon // iters_run
        if self.horizon % iters_run != 0:
            assert False, f"horizon {self.horizon}, iters_run {iters_run}"
        # import pdb

        # pdb.set_trace()
        for t in tqdm.trange(num_controllers, desc="Fitting controllers"):
            gg = [
                np.concatenate(
                    [states[:, iters_run * t + i], actions[:, iters_run * t + i]],
                    axis=-1,
                )
                for i in range(0, iters_run)
            ]
            SAS_ = np.concatenate(gg, axis=0)
            if self.prior_type == "gmm":
                prior = gmm.eval(ds, da, SAS_)
            else:
                prior = None
            (
                self.K[iters_run * t],
                self.k[iters_run * t],
                self.S_K[iters_run * t],
            ) = util_linear_fit(
                SAS_,
                slice(ds),
                slice(ds, ds + da),
                prior=prior,
            )
            self.S_K[iters_run * t] = 0.5 * (
                self.S_K[iters_run * t] + self.S_K[iters_run * t].T
            )
            for i in range(1, iters_run):
                (
                    self.K[iters_run * t + i],
                    self.k[iters_run * t + i],
                    self.S_K[iters_run * t + i],
                ) = (
                    self.K[iters_run * t],
                    self.k[iters_run * t],
                    self.S_K[iters_run * t],
                )

        # Take care of noises
        self.upd_deviation()

        self.model2Torch()
        # pdb.set_trace()
        return base_stats

    def train(self, N, num_cpu="max", iters_run=1, out_dir=None):
        paths = self.sample(N, self.horizon, num_cpu=num_cpu)
        eval_statistics = self.fit(paths, iters_run, self.horizon)
        eval_statistics.append(N)
        # log number of samples
        if self.save_logs:
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            self.logger.log_kv("num_samples", num_samples)

        # with gfile.GFile(path.join(out_dir, self.dyn_env.env_id+'.pkl'), 'wb') as fp:
        # policy_params = (self.K, self.k, self.S_K)
        # pickle.dump(policy_params, fp)
        # with gfile.GFile(path.join(out_dir, 'Dy'+self.dyn_env.env_id+'.pkl'), 'wb') as fp:
        # dynamics = (self.D, self.d, self.S_D)
        # pickle.dump(dynamics, fp)
        # with gfile.GFile(path.join(out_dir, 'Dv'+self.dyn_env.env_id+'.pkl'), 'wb') as fp:
        # pickle.dump(self.dv, fp)

        return eval_statistics

    def step(self, x, t):
        u = self.pol_linear[t](x)
        self.action = u
        if isinstance(x, Ele):
            assert isinstance(u, Ele)
            mu = Ele.cat(x, u)
            # next = self.dyn_linear[t](mu)
            # return Ele.add_noise(next, self.dyn_deviation[t]) # incorporate the uncertainty from dynmaics estimation.
        else:
            mu = torch.cat((x, u), 1)
        return self.dyn_linear[t](mu)

    def abstract_interp(self, curr_abs, max_steps, trajectory=None, deviation=False):
        for k in range(max_steps):
            if trajectory is not None:
                if deviation and k > 0:
                    trajectory.append(
                        Ele.add_noise(curr_abs, self.dv[k - 1] * self.deviation[k - 1])
                    )
                else:
                    trajectory.append(curr_abs)

            curr_abs = self.step(curr_abs, k)
            if deviation:
                Ele.add_noise(curr_abs, self.dv[k] * self.deviation[k])
                # print (f"state lb: {full_abs.lb()}")
                # print (f"state ub: {full_abs.ub()}")
            else:
                # print (f"state lb: {curr_abs.lb()}")
                # print (f"state ub: {curr_abs.ub()}")
                pass
            # print (f"action lb: {self.action.lb()}")
            # print (f"action ub: {self.action.ub()}")
        # add the last state into the trajectory
        if trajectory is not None:
            if deviation and k > 0:
                trajectory.append(
                    Ele.add_noise(curr_abs, self.dv[k - 1] * self.deviation[k - 1])
                )
            else:
                trajectory.append(curr_abs)

    def abstract_simulate(self, curr_abs, max_steps, deviation=True):
        trajectory = []
        with torch.no_grad():
            self.abstract_interp(
                curr_abs, max_steps, trajectory=trajectory, deviation=deviation
            )
        return trajectory

    # Statistical verification of the accuracy of our models.
    def model_verify(self, b_u, dSingnificanceLevel, lbs, ubs, num_cpu, out_dir):
        # Inputs:
        #       b_u:                    Probability Thresshold,
        #       dSingnificanceLevel:    Desired sig. level,
        # Outputs:
        #       A:                      Assertation,
        #       N:                      Sampling cost

        a = [0, b_u]
        b = [b_u, 1]
        sigLevel = 1
        Nk = T = 0
        cnt = 0
        N = 200

        print(
            "////////////////// Statistically Verifying the Model ... //////////////////"
        )

        while sigLevel > dSingnificanceLevel:
            cnt += 1
            print(f"////////////////// Verification Round {cnt}: //////////////////")
            # Sampling a set of rollouts.
            paths = self.sample(N, self.horizon, num_cpu=num_cpu)
            observations, controls, _, _ = self.process_paths(paths)

            N = observations.shape[0]
            print(f"====== Number of trajectories {N} ======")
            TH, ds, da = self.horizon, self.ds, self.da
            ds + da

            states, actions = np.zeros((N, TH, ds)), np.zeros((N, TH, da))
            chunk_size = 10
            for idx, chunk in tqdm.tqdm(
                util_chunk(observations[:, :], controls[:, :], chunk_size=chunk_size),
                desc="Encoding",
                total=N / chunk_size,
            ):
                states[idx], actions[idx] = chunk[0], chunk[1]

            f = 0
            for i in range(N):
                print(f"------ validating trajectory {i} -----")
                for j in range(TH):
                    if (
                        np.less_equal(lbs[j], states[i, j]).all()
                        and np.less_equal(states[i, j], ubs[j]).all()
                    ):
                        pass
                    else:
                        print(f"At time {j}, concrete state {states[i, j]}")
                        print(f"           , abstract state lb {lbs[j]}")
                        print(f"           , abstract state ub {ubs[j]}")
                        print(f"diff lb    , {lbs[j] - states[i, j]}")
                        print(f"diff ub    , {states[i, j] - ubs[j]}")
                        max_diff_lb = np.max(lbs[j] - states[i, j])
                        max_diff_ub = np.max(states[i, j] - ubs[j])
                        if max_diff_lb > 0:
                            print(f"max diff lb, {max_diff_lb}")
                        if max_diff_ub > 0:
                            print(f"max diff ub, {max_diff_ub}")
                        f += 1
                        break
            print(f"Total of {f} invalid rollouts in this round")

            Nk += N
            T += f
            if T / Nk < b_u:
                z = 0  # interval [0 b]
            else:
                z = 1  # interval [b 1]
            if T == 0:
                Alpha = (1 - a[z]) ** Nk - (1 - b[z]) ** Nk
            elif T == Nk:
                Alpha = b[z] ** Nk - a[z] ** Nk
            else:
                alpha_a = T
                beta_a = Nk - T + 1
                alpha_b = T + 1
                beta_b = Nk - T
                pd_a = beta(alpha_a, beta_a)  # makedist('Beta','a',alpha_a,'b',beta_a)
                pd_b = beta(alpha_b, beta_b)  # makedist('Beta','a',alpha_b,'b',beta_b)
                Alpha = pd_b.cdf(b[z]) - pd_a.cdf(
                    a[z]
                )  # cdf(pd_b,b[z]) - cdf(pd_a,a[z])
            sigLevel = 1 - Alpha  # the calculated significance level
            print(f"The total number of samples is {Nk}")
            print(f"The total number of errors is {T}")
            print(f"Current sigLevel is {sigLevel}")
        if (T / Nk) < b_u:
            A = 1  # assertation is true
        else:
            A = 0  # assertation is false
        return A, Nk

    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger.log_kv("stoc_pol_mean", mean_return)
        self.logger.log_kv("stoc_pol_std", std_return)
        self.logger.log_kv("stoc_pol_max", max_return)
        self.logger.log_kv("stoc_pol_min", min_return)

    def fit_box_inv(self, rollouts):
        TH, ds, da = self.horizon, self.ds, self.da

        ups = np.zeros((TH, ds))
        los = np.zeros((TH, ds))

        for t in range(TH):
            # print (f'{t} : {rollouts[:, t, :]}')
            # print (f'{t} : {np.min(rollouts[:, t, :], axis=0)}')
            # print (f'{t} : {np.max(rollouts[:, t, :], axis=0)}')
            ups[t] = np.max(rollouts[:, t, :], axis=0)
            los[t] = np.min(rollouts[:, t, :], axis=0)
        boxes = (los, ups)
        return boxes

    def make_rollout_plots(
        self, dims, lbs, ubs, los=None, ups=None, rollouts=None, save_loc=None
    ):
        for ind in range(dims):
            plt.figure(figsize=(10, 6))

            if isinstance(lbs, list):
                for i in range(len(lbs)):
                    if i == 0:
                        c = "red"
                    elif i == 1:
                        c = "blue"
                    else:
                        c = np.random.rand(
                            3,
                        )
                    plt.plot(
                        ubs[i][:, ind],
                        marker="",
                        color=c,
                        linewidth=2,
                        linestyle="dashed",
                        label="$\it{x}$ abstraction upper-bound",
                    )
                    plt.plot(
                        lbs[i][:, ind],
                        marker="",
                        color=c,
                        linewidth=2,
                        linestyle="dashdot",
                        label="$\it{x}$ abstraction lower-bound",
                    )
            else:
                plt.plot(
                    ubs[:, ind],
                    marker="",
                    color="olive",
                    linewidth=2,
                    linestyle="dashed",
                    label="$\it{x}$ abstraction upper-bound",
                )
                plt.plot(
                    lbs[:, ind],
                    marker="",
                    color="olive",
                    linewidth=2,
                    linestyle="dashdot",
                    label="$\it{x}$ abstraction lower-bound",
                )

            if rollouts is not None:
                for rollout in rollouts:
                    plt.plot(
                        rollout[:, ind],
                        marker="",
                        color=np.random.rand(
                            3,
                        ),
                        linewidth=1,
                    )

                # los, ups = self.fit_box_inv(horizon, rollouts)
            if los is not None and ups is not None:
                plt.plot(
                    los[:, ind],
                    marker="",
                    color="magenta",
                    linewidth=2,
                    linestyle="solid",
                    label="$\it{x}$ rollout lower-bound",
                )
                plt.plot(
                    ups[:, ind],
                    marker="",
                    color="magenta",
                    linewidth=2,
                    linestyle="solid",
                    label="$\it{x}$ rollout upper-bound",
                )

            plt.xlabel("timesteps")
            plt.title("Symbolic Rollout")
            plt.legend()
            plt.savefig(save_loc + "/abs" + str(ind) + ".png", dpi=100)
            plt.close()

    def visualize(self, num_episodes):
        dyn_path = path.join(out_dir, "Dy" + str(self.dyn_env.env_id) + ".pkl")
        self.D, self.d, self.S_D = np.load(dyn_path, allow_pickle=True)
        ctrl_path = path.join(out_dir, str(self.dyn_env.env_id) + ".pkl")
        self.K, self.k, self.S_K = np.load(ctrl_path, allow_pickle=True)
        self.upd_deviation()
        # check if a dv file exists
        dv_path = path.join(out_dir, "Dv" + str(self.dyn_env.env_id) + ".pkl")
        if isfile(dv_path):
            dv = np.load(dv_path, allow_pickle=True)
            self.dv = dv
        self.model2Torch()
        # print(self.pol_linear[0])
        # import pdb

        # pdb.set_trace()
        for ep in range(num_episodes):
            o = self.dyn_env.reset()
            d = False
            t = 0
            score = 0.0
            while t < self.horizon and d is False:
                self.dyn_env.render()
                obs = np.float32(o.reshape(1, -1))
                obs = torch.from_numpy(obs)
                a = self.pol_linear[t](obs).data.numpy().ravel()
                # a = self.policy_model.get_action(o)[1]['evaluation']
                o, r, d, _ = self.dyn_env.step(a)
                t = t + 1
                score = score + r
            # print(f'Episode steps = {t}')
            print(f"Episode score = {score}")

    def run(self, in_lb, in_ub, N, num_cpu, out_dir, iters_run=1, use_loaded_dynamics=False):
        # if (use_loaded_dynamics):
        #     dyn_path = path.join(out_dir, 'Dy'+str(self.dyn_env.env_id)+'.pkl')
        #     self.D, self.d, self.S_D = np.load(dyn_path, allow_pickle=True)
        #     ctrl_path = path.join(out_dir, str(self.dyn_env.env_id)+'.pkl')
        #     self.K, self.k, self.S_K = np.load(ctrl_path, allow_pickle=True)
        #     self.upd_deviation()
        #     # check if a dv file exists
        #     dv_path = path.join(out_dir, 'Dv'+str(self.dyn_env.env_id)+'.pkl')
        #     if isfile(dv_path):
        #         dv = np.load(dv_path, allow_pickle=True)
        #         self.dv = dv
        #     self.model2Torch()
        # else:
        #     self.train(N, num_cpu=num_cpu, out_dir=out_dir)

        # self.visualize(2)
        self.train(N, num_cpu=num_cpu, iters_run=iters_run, out_dir=out_dir)

        in_lb = torch.from_numpy(in_lb).float()
        in_ub = torch.from_numpy(in_ub).float()
        in_lb = in_lb.unsqueeze(dim=0)
        in_ub = in_ub.unsqueeze(dim=0)
        # if torch.cuda.is_available():
        # in_lb, in_ub = in_lb.cuda(), in_ub.cuda()
        print(f"in_lb {in_lb}")
        print(f"in_ub {in_ub}")
        Ele.by_intvl(in_lb, in_ub)

        statistical_model_verify = False

        # while model_updated: # Iteratively make the learned model more accurate
        #     model_updated_cnt += 1
        #     print (f'############# Model accuracy refinement at iter {model_updated_cnt} #############')

        #     #abstract_simulate to collect abstract states
        #     #abstract_traj = self.abstract_simulate(curr_abs, self.horizon, deviation=True)

        #     #rollout to collect concrete states
        #     paths = self.sample(N, self.horizon, num_cpu=num_cpu)
        #     observations, controls, _, _ = self.process_paths(paths)
        #     initial_states = observations[:, 0, :]

        #     N = observations.shape[0]
        #     print (f'Number of trajectories {N}')
        #     TH, ds, da = self.horizon, self.ds, self.da
        #     dsa = ds + da

        #     states, actions = np.zeros((N, TH, ds)), np.zeros((N, TH, da))
        #     chunk_size = 10
        #     for idx, chunk in tqdm.tqdm(util_chunk(observations[:, :], controls[:, :],
        #                                     chunk_size=chunk_size), desc='Encoding', total=N / chunk_size):
        #         states[idx], actions[idx] = chunk[0], chunk[1]

        #     #Draw the lower bounds and upper bounds
        #     # lbs = [abstract_traj[j].lb().numpy()[0] for j in range(0, self.horizon)]
        #     # ubs = [abstract_traj[j].ub().numpy()[0] for j in range(0, self.horizon)]
        #     # lbs = np.array(lbs)
        #     # ubs = np.array(ubs)

        #     #Get box invariant
        #     import time
        #     x = time.time()
        #     curr_los, curr_ups = self.fit_box_inv(states)
        #     print("fix box inx time is", time.time() - x)

        #     if model_updated_cnt > 1:
        #         model_updated = False

        #         for j in range(1, TH):
        #             if np.less_equal(lbs[j], curr_los[j]).all() and np.less_equal(curr_ups[j], ubs[j]).all():
        #                 pass
        #             else:
        #                 print ('========================================')
        #                 print (f'At time {j}, box invariant  lb {curr_los[j]}')
        #                 print (f'At time {j}, box invariant  ub {curr_ups[j]}')
        #                 print (f'           , abstract state lb {lbs[j]}')
        #                 print (f'           , abstract state ub {ubs[j]}')
        #                 print (f'diff lb    , {lbs[j] - curr_los[j]}')
        #                 print (f'diff ub    , {curr_ups[j] - ubs[j]}')

        #                 # add_n_lb = (lbs[j] - los[j]) / self.deviation[j-1].numpy()
        #                 # add_n_ub = (ups[j] - ubs[j]) / self.deviation[j-1].numpy()
        #                 # add_n_lb = np.where(add_n_lb<0, 0, add_n_lb)
        #                 # add_n_ub = np.where(add_n_ub<0, 0, add_n_ub)
        #                 #
        #                 # if isinstance(self.dv[j-1], int):
        #                 #     self.dv[j-1] = \
        #                 #         torch.from_numpy(self.dv[j-1] + np.array([add_n_lb, add_n_ub])).float()#np.maximum
        #                 # else:
        #                 #     self.dv[j-1] = \
        #                 #         torch.from_numpy(self.dv[j-1].numpy() + np.array([add_n_lb, add_n_ub])).float()
        #                 #
        #                 # print(f'self.dv = {self.dv[j-1]}')

        #                 model_updated = True
        #                 model_adjusted = True
        #                 #break
        #         lbs = np.minimum(lbs, curr_los) - np.where(lbs - curr_los < 0, 0, 0.005)
        #         ubs = np.maximum(ubs, curr_ups) + np.where(curr_ups - ubs < 0, 0, 0.005)
        #         model_updated = False
        #         print (f"model_updated = {model_updated}")
        #     else:
        #         lbs = curr_los
        #         ubs = curr_ups
        # print (f'############# Model accuracy converged at iter {model_updated_cnt} #############')

        print("Saving linear models")
        path = pathlib.Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path, "model.txt"), "w") as f:
            for linear_controller in self.pol_linear:
                sz = linear_controller.weight.data.numpy().shape[0]
                line = ""
                for i in range(0, sz):
                    weight = linear_controller.weight.data.numpy()[i]
                    bias = linear_controller.bias.data.numpy()[i]
                    weight_str = " ".join([str(x) for x in weight])
                    if i == 0:
                        line += f"{weight_str} {bias}"
                    else:
                        line += f" {weight_str} {bias}"
                    # weight1 = linear_controller.weight.data.numpy()[0]
                    # bias1 = linear_controller.bias.data.numpy()[0]
                    # weight2 = linear_controller.weight.data.numpy()[1]
                    # bias2 = linear_controller.bias.data.numpy()[1]
                    # weight_str1 = " ".join([str(x) for x in weight1])
                    # weight_str2 = " ".join([str(x) for x in weight2])
                    # line = f'{weight_str1} {bias1} {weight_str2} {bias2}\n'
                line += "\n"
                f.write(line)

        # print('Saving lbs')
        # with open(out_dir + 'lbs.txt', "w") as f:
        #     for box in lbs:
        #         box_str = [str(x) for x in box]
        #         f.write(" ".join(box_str))
        #         f.write("\n")

        # print('Saving ubs')
        # with open(out_dir + 'ubs.txt', "w") as f:
        #     for box in ubs:
        #         box_str = [str(x) for x in box]
        #         f.write(" ".join(box_str))
        #         f.write("\n")

        # self.make_rollout_plots(self.ds, lbs, ubs, los=curr_los, ups=curr_ups, rollouts= states, save_loc= out_dir)

        # self.simulate_with_linear_models(1, 1000, out_dir, initial_states=np.array([[0.0, 0.0, 0.0, 0.0]]))
        # Statistically verify model's accuracy. b_u, dSingnificanceLevel, abstract_traj, horizon, out_dir
        if statistical_model_verify:
            A, Nk = self.model_verify(0.01, 0.01, lbs, ubs, num_cpu, out_dir)
            print(f"Verification result {A == 1} estimated with {Nk} samples")

        # Save the synthesized accurate model
        # if model_adjusted:
        #     with gfile.GFile(path.join(out_dir, 'Dv'+self.dyn_env.env_id+'.pkl'), 'wb') as fp:
        #         pickle.dump(self.dv, fp)

    def simulate_with_linear_models(
        self, num_traj, horizon, out_dir, ignore_done=False, initial_states=None
    ):
        import pdb

        # pdb.set_trace()
        print("==== simulate linear and neural models ====")
        env = self.dyn_env
        paths = []
        for i in range(num_traj):
            # print(f"{i}-th traj")
            obs = env.env.reset_to_zero()
            # print(obs)

            if initial_states is not None:
                obs = initial_states[i]
                env.state = initial_states[i]

            path = []
            done = False
            t = 0

            while t < horizon and done is not True:
                obs = np.float64(obs.reshape(1, -1))
                self.pol_linear[t](torch.from_numpy(obs))
                action = self.pol_linear[t](torch.from_numpy(obs)).data.numpy().ravel()
                gt_action = self.policy_model.get_action(obs)[1]['evaluation']
                print(f"time {t} action {action}, gt_action {gt_action}")
                # print(action)
                # action = np.clip(action, -1, 1)
                # print(action)
                next_obs, r, done, _ = env.step(action)
                path.append(obs)
                # print(next_obs)
                obs = next_obs
                t += 1

            paths.append(path)
            # print(path)

        paths = np.array(paths)
        linear_paths = paths
        dims = len(paths[0][0][0])
        for x in range(0, dims):
            plt.cla()
            for path in paths:
                plt.plot(path[:, :, x].reshape(-1))
            plt.savefig(out_dir + f"/simulated_linear_{x}.png")

        print("neural paths")
        env = self.dyn_env
        paths = []
        for i in range(num_traj):
            # print(f"{i}-th traj")
            obs = env.env.reset_to_zero()
            # print(obs)
            if initial_states is not None:
                obs = initial_states[i]
                env.state = initial_states[i]

            path = []
            done = False
            t = 0

            while t < horizon and done is not True:
                obs = np.float64(obs.reshape(1, -1))
                action = self.policy_model.get_action(torch.from_numpy(obs))[1][
                    "evaluation"
                ]
                # linear_action = self.pol_linear[t](torch.from_numpy(obs)).data.numpy().ravel()
                # gt_action = self.policy_model.get_action(obs)[1]['evaluation']
                # print(f"time {t} action {action}, gt_action {gt_action}")
                # print(action)
                # action = np.clip(action, -1, 1)
                # print(action)
                next_obs, r, done, _ = env.step(action)
                path.append(obs)
                obs = next_obs
                # print(next_obs)
                t += 1

            paths.append(path)
            # print(path)

        paths = np.array(paths)
        dims = len(paths[0][0][0])
        for x in range(0, dims):
            plt.cla()
            for path in paths:
                plt.plot(path[:, :, x].reshape(-1))
            plt.savefig(out_dir + f"/simulated_neural_{x}.png")
        neural_paths = paths
        neural_path = neural_paths[0]
        linear_path = linear_paths[0]
        for i in range(len(neural_path)):
            if np.allclose(neural_path[i], linear_path[i]):
                print(i, "close")
                print(neural_path[i], linear_path[i])
            else:
                print(i, "not close")
                print(neural_path[i], linear_path[i])
        # pdb.set_trace()
        for x in self.pol_linear:
            print(x.weight, x.bias)
        print("==== done ====")

    def predict(self, t, state):
        ds, da = self.ds, self.da
        idx_s = slice(ds)

        sigma, mu = np.zeros((ds + da, ds + da)), np.zeros(ds + da)
        mu[idx_s] = state

        D, d, S_D = self.D, self.d, self.S_D
        K, k, S_K = self.K, self.k.self.S_K

        sigma = np.vstack(
            [
                np.hstack(
                    [
                        sigma[idx_s, idx_s],
                        sigma[idx_s, idx_s].dot(K[t].T),
                    ]
                ),
                np.hstack(
                    [
                        K[t].dot(sigma[idx_s, idx_s]),
                        K[t].dot(sigma[idx_s, idx_s]).dot(K[t].T) + S_K[t],
                    ]
                ),
            ]
        )
        mu = np.hstack(
            [
                mu[idx_s],
                K[t].dot(mu[idx_s]) + k[t],
            ]
        )
        sigma[idx_s, idx_s] = D[t].dot(sigma).dot(D[t].T) + S_D[t]
        mu[idx_s] = D[t].dot(mu) + d[t]
        return mu, sigma

    def forward(self, output=False):
        T, ds, da = self.horizon, self.ds, self.da
        idx_s = slice(self.ds)
        D, d, S_D = self.D, self.d, self.S_D
        K, k, S_K = self.K, self.k, self.S_K

        sigma, mu = np.zeros((T, ds + da, ds + da)), np.zeros((T, ds + da))
        sigma[0, idx_s, idx_s] = self.S_s0
        mu[0, idx_s] = self.mu_s0

        for t in range(T):
            if output:
                print(f"time_step {t} at {mu[t, idx_s]}")
            sigma[t] = np.vstack(
                [
                    np.hstack(
                        [
                            sigma[t, idx_s, idx_s],
                            sigma[t, idx_s, idx_s].dot(K[t].T),
                        ]
                    ),
                    np.hstack(
                        [
                            K[t].dot(sigma[t, idx_s, idx_s]),
                            K[t].dot(sigma[t, idx_s, idx_s]).dot(K[t].T) + S_K[t],
                        ]
                    ),
                ]
            )
            mu[t] = np.hstack(
                [
                    mu[t, idx_s],
                    K[t].dot(mu[t, idx_s]) + k[t],
                ]
            )
            if t < T - 1:
                sigma[t + 1, idx_s, idx_s] = D[t].dot(sigma[t]).dot(D[t].T) + S_D[t]
                mu[t + 1, idx_s] = D[t].dot(mu[t]) + d[t]
        return mu, sigma


def evaluate_model(linear_model, neural_model, env, rollout, safe_sets):
    # import pdb
    # pdb.set_trace()
    fail_count = 0
    total_reward = 0.0
    linear_count = 0
    for _ in tqdm.tqdm(range(0, rollout)):
        initial_state = env.gym_env.reset()
        state = initial_state
        eposide_reward = 0.0
        for t in range(0, 70):
            state = torch.from_numpy(np.float32(state))
            if linear_model is None:
                state, rwd, done, info = env.gym_env.step(neural_model(state))
            else:
                simulated_state = env.gym_env.simulate(neural_model(state))
                # if simulated state is in safe state, take the step, else use linear model to take step
                safe = True
                if t >= 0 and t <= 34:
                    if (
                        simulated_state[0] < safe_sets[t][0]
                        or simulated_state[0] > safe_sets[t][1]
                        or simulated_state[1] < safe_sets[t][2]
                        or simulated_state[1] > safe_sets[t][3]
                    ):
                        safe = False
                safe = False
                if safe:
                    state, rwd, done, info = env.gym_env.step(neural_model(state))
                else:
                    # print("linear model is used at iter ", t)
                    linear_count += 1
                    state, rwd, done, info = env.gym_env.step(
                        np.array(
                            [
                                linear_model[0] * state[0]
                                + linear_model[1] * state[1]
                                + linear_model[2]
                            ]
                        )
                    )

            eposide_reward += rwd
            if env.gym_env.c >= 10 and env.gym_env.c <= 20:
                if state[0] > 1.0 or state[0] < 0.0:
                    fail_count += 1
                    print(f"fail with starting at {initial_state}")
                    break
            if t == 69:
                total_reward += eposide_reward

    print(f"linear model used {linear_count} times")
    print(f"fail count {fail_count}")
    print(f"fail rate {fail_count / rollout}")
    print(f"success rate {1 - fail_count / rollout}")
    print(f"average reward {total_reward / (rollout - fail_count)}")


if __name__ == "__main__":
    # out_dir = 'experiments/mjrl/benches/lunarlandercontinuous-v2/'
    # env = GymEnv('LunarLanderContinuous-v2')

    # from experiments.mjrl.benches.pendulum.pendulum import Pendulum
    # pen = Pendulum()
    # env = GymEnv(pen.gym_env)
    from experiments.mjrl.benches.airplane.airplane import Airplane
    from experiments.mjrl.benches.attitude_control.attitude_control import (
        AttitudeControl,
    )
    from experiments.mjrl.benches.cartpole.cartpole import CartPole
    from experiments.mjrl.benches.docking.docking import Docking
    from experiments.mjrl.benches.double_pendulum.double_pendulum import DoublePendulum
    from experiments.mjrl.benches.single_pendulum.single_pendulum import SinglePendulum
    from experiments.mjrl.benches.single_pendulum_inf.single_pendulum_inf import (
        SinglePendulumInf,
    )
    from experiments.mjrl.benches.tora.tora import Tora
    from experiments.mjrl.benches.tora_inf.tora_inf import ToraInf

    bench_name = "double_pendulum"
    out_dir = "experiments/mjrl/benches/" + bench_name + "/"
    if bench_name == "cartpole":
        pole = CartPole()
        env = GymEnv(pole.gym_env)
        high = np.array([0.05, 0.05, 0.05, 0.05])
        low = -high
    elif bench_name == "tora":
        pole = Tora()
        env = GymEnv(pole.gym_env)
        high = np.array([0.7, -0.6, -0.3, 0.6])
        low = np.array([0.6, -0.7, -0.4, 0.5])
        keras_model = tf.keras.models.load_model(out_dir + "controllerTora_nnv.h5")
        torch_model = ToraPolicy(keras_model)
        import pdb

        pdb.set_trace()
        # perform a simple test
        test = np.array([[1.0, 2.0, 3.0, 4.0]])
        print(keras_model.predict(test))
        print(torch_model(torch.from_numpy(np.float32(test))))
        test = np.array([[7.9, 2.1, 6.5, 4.2]])
        print(keras_model.predict(test))
        print(torch_model(torch.from_numpy(np.float32(test))))
    elif bench_name == "tora_inf":
        pole = ToraInf()
        env = GymEnv(pole.gym_env)
        high = np.array([0.1, 0.1, 0.1, 0.1])
        low = np.array([-0.1, -0.1, -0.1, -0.1])
        keras_model = tf.keras.models.load_model(out_dir + "controllerTora_nnv.h5")
        torch_model = ToraPolicy(keras_model)
    elif bench_name == "docking":
        pole = Docking()
        env = GymEnv(pole.gym_env)
        high = np.array([106.0, 106.0, 0.28, 0.28])
        low = np.array([70.0, 70.0, -0.28, -0.28])
    elif bench_name == "single_pendulum":
        pole = SinglePendulum()
        env = GymEnv(pole.gym_env)
        high = np.array([1.2, 0.2])
        low = np.array([1.0, 0.0])
        # high = np.array([0.05, 0.05, 0.05, 0.05])
        # low = np.array([-0.007983,1.3982,-0.80856,-0.5645,-0.00926, -0.18315, 0.0, 0.0])
        # high = np.array([ 0.007983,1.4225, 0.80856, 0.5111, 0.00926 , 0.18315, 0.0, 0.0])
        # pi = out_dir + 'best_policy.pickle'
        # policy_model = pickle.load(open(pi, 'rb'))
        onnx_model = onnx.load(out_dir + "controller_single_pendulum.onnx")
        torch_model = convert(onnx_model)

        # ====================================== single pendulum
        # safe_sets = []
        # with open(out_dir + "safe_sets.txt") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.split()
        #         safe_sets.append([float(x) for x in line])

        # linear_model = np.array([-0.52603303, -0.49399166, -0.10628574])
        # evaluate_model(linear_model, torch_model, SinglePendulum(), 1, safe_sets)
    elif bench_name == "single_pendulum_inf":
        pole = SinglePendulumInf()
        env = GymEnv(pole.gym_env)
        high = np.array([0.05, 0.05])
        low = np.array([-0.05, -0.05])
        onnx_model = onnx.load(out_dir + "controller_single_pendulum.onnx")
        torch_model = convert(onnx_model)
    elif bench_name == "airplane":
        pole = Airplane()
        env = GymEnv(pole.gym_env)
        high = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        low = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        onnx_model = onnx.load(out_dir + "controller_airplane.onnx")
        torch_model = convert(onnx_model)
    elif bench_name == "double_pendulum":
        pole = DoublePendulum()
        env = GymEnv(pole.gym_env)
        high = np.array([1.3, 1.3, 1.3, 1.3])
        # high = np.array([1.0, 1.0, 1.0, 1.0])
        low = np.array([1.0, 1.0, 1.0, 1.0])

        onnx_model = onnx.load(out_dir + "controller_double_pendulum_less_robust.onnx")
        torch_model = convert(onnx_model)
    elif bench_name == "attitude_control":
        pole = AttitudeControl()
        env = GymEnv(pole.gym_env)
        high = np.array([-0.44, -0.54, 0.66, -0.74, 0.86, -0.64])
        low = np.array([-0.45, -0.55, 0.65, -0.75, 0.85, -0.65])

        onnx_model = onnx.load(out_dir + "attitude_control_3_64_torch.onnx")
        torch_model = convert(onnx_model)
        # import pdb
        # pdb.set_trace()
    # ====================================
    # import pdb
    # pdb.set_trace()
    policy_model = ProgPolicy(env.spec, torch_model)

    # for tora
    # policy_model = ProgPolicy(env.spec, onnx_model)

    learner = DynamicsLearner(env, policy_model)
    learner.run(low, high, 500, 12, out_dir)

    # learner.visualize(5)
