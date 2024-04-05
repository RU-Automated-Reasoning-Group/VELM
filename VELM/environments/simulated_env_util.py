import math
import os
import pdb
from typing import List, Optional

import gymnasium as gym
import numpy as np


def construct_local_dict(state: np.ndarray, action: np.ndarray):
    eval_vars = {"cos": math.cos, "sin": math.sin}
    state_dim = len(state)
    action_dim = len(action)

    for i in range(0, state_dim):
        eval_vars[f"x{i+1}"] = state[i]
    for i in range(0, action_dim):
        eval_vars[f"x{i+1+state_dim}"] = action[i]

    return eval_vars


def eval_model_froms_vars(learned_model: List[str], eval_vars: dict):
    sim_state = []
    for line in learned_model:
        # pdb.set_trace()
        dim_val = eval(line, eval_vars)
        sim_state.append(dim_val)
    return np.array(sim_state)


def eval_model(
    learned_model: List[str],
    state: np.ndarray,
    action: np.ndarray,
    stds: Optional[List[float]] = None,
):
    eval_vars = construct_local_dict(state, action)
    result = eval_model_froms_vars(learned_model, eval_vars)

    if stds is None:
        return result
    else:
        # apply std
        for i in range(0, len(stds)):
            noise = np.random.normal(loc=0.0, scale=stds[i])
            result[i] = result[i] + noise

        return result


def make_simulated_env(
    random: bool,
    env_name: str,
    learned_model: List[str] = [],
    stds: Optional[List[float]] = [],
):
    if random:
        simulated_env = gym.make(env_name, learned_model=learned_model, stds=stds)
    else:
        simulated_env = gym.make(env_name, learned_model=learned_model)
    return simulated_env


def load_dynamic_from_file(f_dir, stds=False):
    with open(os.path.join(f_dir, "model.txt")) as f:
        models = f.readlines()

    stds = None
    if os.path.isfile(os.path.join(f_dir, "std.txt")):
        with open(os.path.join(f_dir, "std.txt")) as f:
            stds = f.readlines()
            stds = [float(x) for x in stds]
    return models, stds


# def make_simulated_env(random: bool, env_name: str, learned_model:List[str]=[], stds:Optional[List[float]]=[]):
#     simulated_env = gym.make(env_name)
#     return simulated_env
