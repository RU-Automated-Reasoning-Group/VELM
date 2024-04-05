import time as timer

import numpy as np

import sampler.core as trajectory_sampler


class seq_linear:
    def __init__(self, num_controller, iters_run, controller_sz, params):
        self.num_controller = num_controller
        self.iters_run = iters_run
        self.controller_sz = controller_sz
        self.horizon = num_controller * iters_run
        self.linear_policy = []
        self.dim = controller_sz - 1
        self.params = params.copy()
        for i in range(0, num_controller):
            self.linear_policy.append(
                np.array(params[i * controller_sz : (i + 1) * controller_sz])
            )

    def get_action(self, o, t):
        modified_o = np.ones(self.dim + 1)
        modified_o[0 : self.dim] = o
        return np.inner(self.linear_policy[t], modified_o)


class MSE:
    # this class is used to sample paths and calculate MSE loss
    def __init__(self, policy_model, env, seed=None):
        self.policy_model = policy_model
        self.seed = 0 if seed is None else seed
        self.dyn_env = env

    def sample(self, N, horizon=1e6, num_cpu="max", gamma=0.995, env_kwargs=None):
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
        # process_samples.compute_returns(paths, gamma)

        return paths

    def get_loss(self, N, neural_paths, num_cpu="max"):
        paths = self.sample(N, self.policy_model.horizon, num_cpu=num_cpu)
        return np.square(paths - neural_paths).mean()
