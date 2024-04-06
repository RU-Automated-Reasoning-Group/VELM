import copy
import math
import os
import pathlib
import pdb
from typing import List

# import multiprocessing as mp
import multiprocess as mp
import numpy as np
import scipy
import sympy
import torch.nn as nn
from dso import DeepSymbolicRegressor

# sys.path.append("deep-symbolic-optimization/dso")


class DSO(nn.Module):
    def __init__(self, args):
        self.dso_config = {
            "task": {
                "task_type": "regression",
                "function_set": ["add", "sub", "mul", "div", "sin", "const"],
                "dataset": "dso/custom_benchmarks/pendulum/train_init_nogt_theta.csv",
            },
            "training": {"n_samples": 150000, "n_cores_batch" : 4},
        }
        if args.env == "acc":
            self.dso_config["task"]["metric"] = "inv_nmse_noise"
            self.dso_config["task"]["metric_params"] = [1.0, 5.0]
        self.dso_models = [
            DeepSymbolicRegressor(config=self.dso_config) for _ in range(args.state_dim)
        ]

    def learn_dynamic_model(self, samples, preprocess=False, random=False):
        X, y_list = self.process_for_dso(samples, preprocess=preprocess)
        # pdb.set_trace()
        self.train_dso(X, y_list)
        sympy_model = self.remove_small_numbers()
        str_model = [str(x) for x in sympy_model]
        if random:
            stds = self.compute_standard_deviation(X, y_list)
        else:
            stds = None
        return str_model, stds

    @staticmethod
    def process_for_dso(samples, preprocess=False):
        num_data = samples.observations.shape[0]
        num_dim = samples.observations.shape[1]
        X = []
        y_list = [[] for _ in range(0, num_dim)]
        for i in range(0, num_data):
            obs_part = samples.observations[i].cpu().numpy()
            act_part = samples.actions[i].cpu().numpy()
            X.append(np.concatenate([obs_part, act_part]))
            for j in range(0, num_dim):
                y_list[j].append(samples.next_observations[i][j].cpu().numpy())

        return np.array(X), np.array(y_list)

    def train_dso(self, X, y_list):
        # pdb.set_trace()
        pool = mp.Pool(processes=len(self.dso_models))
        parallel_runs = [
            pool.apply_async(fit, kwds={"dso_model": dso_model, "X": X, "y": y}) for dso_model, y in zip(self.dso_models, y_list)
        ]
        results = [p.get() for p in parallel_runs]
        programs, rwds = zip(*results)
        self.learned_model = copy.deepcopy(programs)
        self.reward = copy.deepcopy(rwds)
        return results

    def get_learned_model(self):
        # each dimension is represented as string
        learned_model = [str(dso_model.program_.sympy_expr) for dso_model in self.dso_models]
        return learned_model
    
    def remove_small_numbers(self, threshold=1e-2):
        new_model = []
        for expr in self.learned_model:
            small_numbers = set([e for e in expr.atoms(sympy.Number) if abs(e) < threshold])
            d = {s: 0 for s in small_numbers}
            new_model.append(expr.subs(d))
        self.learned_model = copy.deepcopy(new_model)
        return new_model

    def compute_standard_deviation(self, X, y_list):
        stds = []
        dim = len(self.learned_model)
        for i in range(0, dim):
            predicted_values = []
            str_model = str(self.learned_model[i])
            print(f"str_model is {str_model}")
            for state in X:
                ctx = make_context(state)
                predicted = eval(str_model, ctx)
                predicted_values.append(predicted)
            # pdb.set_trace()     
            diffs = y_list[i] - np.array(predicted_values)
            _, std = scipy.stats.norm.fit(diffs, floc=0.0)
            stds.append(std)
        self.stds = copy.deepcopy(stds)
        return stds

    
    # def fit_noise(self, X, y_list):
    #     self.noise = []
    #     self.add_noise = False
    #     for idx, rwd in enumerate(self.reward):
    #         if rwd < 0.99:
    #             # fit the noise
    #             predicted_y = []
    #             self.add_noise = True
    #             for one_datum in X:
    #                 ctx = make_context(one_datum)
    #                 y_hat = eval(self.learned_model[idx], ctx)
    #                 predicted_y.append(y_hat)
                
    #             predicted_y = np.array(predicted_y)

    #         else:
    #             self.noise.append(None)
    #     return self.add_noise
    
    def pred_dso(self, X):
        return np.array([dso_model.predict(X) for dso_model in self.dso_models])


def fit(dso_model, X, y):
    dso_model.fit(X, y)
    return dso_model.program_.sympy_expr, dso_model.r_

def make_context(X):
    ctx = {"cos": math.cos, "sin": math.sin}
    idx = 1
    for i in range(0, len(X)):
        ctx[f"x{idx}"] = X[i]
        idx += 1
    return ctx

def make_context_with_action(state, action):
    ctx = {"cos": math.cos, "sin": math.sin}
    idx = 1
    for i in range(0, len(state)):
        ctx[f"x{idx}"] = state[i]
        idx += 1
    
    for i in range(0, len(action)):
        ctx[f"x{idx}"] = action[i]
        idx += 1
    return ctx

def evaluate_dynamics(learned_model: List[str], ctx: dict):
    next_state = []
    for i in range(0, len(learned_model)):
        val = eval(learned_model[i], ctx)
        next_state.append(val)
    return np.array(next_state)

def save_learned_dynamics(env_name, learned_dynamics, stds=None):
    path = pathlib.Path(f"results/learned_dynamics/{env_name}/")
    path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(path, "model.txt"), "w+") as f:
        for equation in learned_dynamics:
            eq_str = str(equation)
            eq_str = eq_str.replace(" ", "")
            f.write(f"{eq_str}\n")

    if stds is not None:
        with open(os.path.join(path, "std.txt"), "w+") as f:
            for std in stds:
                eq_str = str(std)
                eq_str = eq_str.replace(" ", "")
                f.write(f"{eq_str}\n")