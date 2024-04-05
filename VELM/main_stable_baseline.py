import math
import os
import pathlib
import pdb
import sys
import torch
import gymnasium as gym
from collections import namedtuple
import numpy as np
from pyoperon.sklearn import SymbolicRegressor
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.sac.policies import SACPolicy
from sympy import parse_expr

import utils.get_env
import utils.plot
from control.rl.policies.gaussian_prog import ProgPolicy
from dynamics.learn import DynamicsLearner
from environments.simulated_env_util import make_simulated_env
from models.dso_model import DSO, save_learned_dynamics
from models.safe_controller import safe_agent
from utils.arguments import env_to_train_args
from utils.gym_env import GymEnv
from verification.VEL.lagrange_lib import run_lagrange
from verification.VEL.safe_violation_callback import (
    SafeViolationCallback,
    safety_violation_tracker,
)
import time
import tabulate

# from sbx import SAC
# from sbx.sac.policies import SACPolicy


def load_model(path, iters_run):
    with open(path + "model.txt") as file:
        lines = file.readlines()
        controller = []
        for line in lines:
            line = line.split()
            line = [float(x) for x in line]
            controller += line
    controller = np.array(controller)
    return controller


def compute_safe_agent(
    args,
    simulated_env_info,
    learned_model,
    stds,
    neural_agent,
    benchmark_config_dict: dict,
    current_version: int
):
    simulated_env = make_simulated_env(
        args.random, simulated_env_info.env_name, learned_model=learned_model, stds=stds
    )
    # env.load_model()
    # pdb.set_trace()
    learner_env = GymEnv(simulated_env)
    learner = DynamicsLearner(learner_env, ProgPolicy(learner_env.spec, neural_agent))

    learner.run(
        simulated_env.init_space.low,
        simulated_env.init_space.high,
        benchmark_config_dict["num_traj"],
        1,
        f"results/linear_models/{simulated_env_info.env_name}/",
        iters_run=benchmark_config_dict["iters_run"],
    )

    linear_model = load_model(
        f"results/linear_models/{simulated_env_info.env_name}/",
        benchmark_config_dict["iters_run"],
    )

    _, safe_linear_model, safe_sets = run_lagrange(
        action_dim=simulated_env.action_space.shape[0],
        state_dim=simulated_env.observation_space.shape[0],
        params=linear_model,
        env_str=simulated_env_info.env_name,
        learned_model=learned_model,
        stds=stds,
        random=args.random,
        num_traj=benchmark_config_dict["num_traj"],
        horizon=benchmark_config_dict["horizon"],
        neural_agent=neural_agent,
        alpha=benchmark_config_dict["alpha"],
        N_of_directions=benchmark_config_dict["N_of_directions"],
        b=benchmark_config_dict["b"],
        noise=benchmark_config_dict["noise"],
        initial_lambda=benchmark_config_dict["initial_lambda"],
        iters_run=benchmark_config_dict["iters_run"],
        eval_program=f"results/programs/{args.env}",
    )

    simulated_env = make_simulated_env(
        args.random, simulated_env_info.env_name, learned_model=learned_model, stds=stds
    )
    return safe_agent(
        neural_agent,
        safe_linear_model,
        simulated_env,
        safe_sets,
        horizon=benchmark_config_dict["horizon"],
        interval=1,
        current_version=current_version
    )


def check_model_accurate(
    learned_dynamic_model: list,
    state_dim: int,
    action_dim: int,
    new_data: list,
    threshold: float = 0.95,
):
    true_next_state = []
    learned_next_state = []
    for state, action, next_state in new_data:
        true_next_state.append(next_state)

        eval_vars = {"cos": math.cos, "sin": math.sin}
        for i in range(0, state_dim):
            eval_vars[f"x{i+1}"] = state[i]
        for i in range(0, action_dim):
            eval_vars[f"x{i+1+state_dim}"] = action[i]

        sim_state = []
        for line in learned_dynamic_model:
            dim_val = eval(line, eval_vars)
            sim_state.append(dim_val)

        learned_next_state.append(sim_state)

    true_next_state = np.array(true_next_state)
    learned_next_state = np.array(learned_next_state)
    n = true_next_state.shape[0]
    stds = np.std(true_next_state, axis=0)
    nrmse = (
        np.sqrt(np.sum(np.square(true_next_state - learned_next_state), axis=0) / n)
        / stds
    )

    model_R = 1 / (1 + nrmse)
    return np.any(model_R < threshold)


def train(args):
    all_t0 = time.time()
    all_safe_time = 0
    all_dy_time = 0
    # create env
    env_info, simulated_env_info = utils.get_env.get_env(args.env)
    env = gym.make(env_info.env_name)

    # define call backs
    safe_violation_logdir = f"results/safe_violation/{env_info.env_name}"
    safe_violation_path = pathlib.Path(safe_violation_logdir)
    safe_violation_path.mkdir(parents=True, exist_ok=True)
    current_version = len(os.listdir(safe_violation_path)) + 1

    final_safe_violation_logdir = os.path.join(
        safe_violation_logdir, f"sac_{current_version}"
    )

    tracker = safety_violation_tracker(final_safe_violation_logdir)
    safeviolation_callback = SafeViolationCallback(tracker)

    # eval_callback = EvalCallback(test_env, eval_freq=1000, deterministic=True, best_model_save_path=f"results/saved_agents/{env_info.env_name}/sac_warm_checkpoint")

    # stop_train_callback = StopTrainingOnNoModelImprovement(
    #     max_no_improvement_evals=25, verbose=1
    # )

    # eval_callback_stop = EvalCallback(
    #     test_env,
    #     eval_freq=1000,
    #     deterministic=True,
    #     callback_after_eval=stop_train_callback,
    # )
    # callback_list = CallbackList([safeviolation_callback, eval_callback, eval_callback_stop])
    callback_list = CallbackList([safeviolation_callback])

    # start warm up training
    if args.load_neural_model:
        f_name = f"sac_checkpoint_{env_info.env_name}"
        neural_agent = SAC.load(f_name)
        print(f"loading neural agent from {f_name}")
    else:
        # pdb.set_trace()
        # in warm up episodes, no linear controllers are involved and only trained on real data
        policy_kwargs = {"net_arch": [args.arch, args.arch]}
        neural_agent = SAC(
            SACPolicy,
            env,
            verbose=1,
            tensorboard_log=f"./results/logs/{env_info.env_name}",
            policy_kwargs=policy_kwargs,
            buffer_size=1000000,
            batch_size=args.batch_size,
            use_sde=True,
            # device="cpu",
            learning_rate=args.lr,
            stats_window_size=1,
        )
        # neural_agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./results/logs/{env_info.env_name}", policy_kwargs=policy_kwargs)
        # pdb.set_trace()
        if args.env not in ["cartpole", "tora", "lalo", "cartpole_move", "cartpole_swing"]:
            neural_agent.learn(
                total_timesteps=args.warm_up_steps,
                progress_bar=True,
                tb_log_name="sac",
                callback=callback_list,
                log_interval=1,
            )
        else:
            buffer = []
            observations = []
            next_observations = []
            actions = []
            episodes_rwd = []
            episodes_unsafe = []
            # env = gymnasium.make("CartPole-v1")
            for i in range(0, 4):
                state, _ = env.reset()
                done = False
                episode_rwd = 0
                unsafe = 0
                while not done:
                    action = env.action_space.sample()
                    next_state, rwd, terminated, truncated, _ = env.step(action)

                    # pdb.set_trace()
                    if env.unsafe():
                        unsafe += 1
                    episode_rwd += rwd
                    buffer.append((state, action, next_state))
                    observations.append(state)
                    next_observations.append(next_state)
                    actions.append(action)

                    done = terminated or truncated
                    state = next_state
                episodes_rwd.append(episode_rwd)
                episodes_unsafe.append(unsafe)
            
            with open(os.path.join(final_safe_violation_logdir, "start.txt"), "w") as f:
                f.write(" ".join([str(x) for x in episodes_rwd]))
                f.write("\n")
                f.write(" ".join([str(x) for x in episodes_unsafe]))
                f.write("\n")

            print("episodes_rwd", episodes_rwd)
            print("episodes unsafe", episodes_unsafe)
            # exit()
        # neural_agent.save(
        #     f"results/saved_agents/{env_info.env_name}/sac_warm_checkpoint"
        # )

    # Use dso to learn the dynamic model
    dy_t0 = time.time()
    if args.load_dynamic_model:
        model_path = f"results/learned_dynamics/{env_info.env_name}/"
        with open(os.path.join(model_path, "model.txt")) as f:
            lines = f.readlines()
            learned_dynamic_model = [line[:-1] for line in lines]
            print(f"loading dyancmic model from {model_path}")

        if args.random:
            with open(os.path.join(model_path, "std.txt")) as f:
                lines = f.readlines()
                learned_stds = [line[:-1] for line in lines]
                learned_stds = [float(x) for x in learned_stds]
        else:
            learned_stds = None
    else:
        if args.sr_method == "DSO":
            # learn a new model
            dso = DSO(args)
            # if args.env == "acc":
            #     observations = torch.tensor(observations)
            #     next_observations = torch.tensor(next_observations)
            #     actions = torch.tensor(actions)
            #     Custom_buffers = namedtuple("Custom_buffers", ["observations", "next_observations", "actions"])
            #     samples = Custom_buffers(observations=observations, next_observations=next_observations, actions=actions)
            #     # pdb.set_trace()
            # else:
            replay_buffer = neural_agent.replay_buffer
            samples = replay_buffer.sample(env_info.lagrange_config["dso_dataset_size"])
                # pdb.set_trace()
            # pdb.set_trace()
            learned_dynamic_model, learned_stds = dso.learn_dynamic_model(
                samples, random=args.random
            )
        elif args.sr_method == "operon":
            reg = SymbolicRegressor(
                    allowed_symbols='add,sub,mul,div,constant,variable,sin,cos',
                    offspring_generator='basic',
                    local_iterations=5,
                    max_length=50,
                    initialization_method='btc',
                    n_threads=10,
                    objectives = ['mse'],
                    symbolic_mode=False,
                    model_selection_criterion='mean_squared_error',
                    random_state=4,
                    generations=10000,
                    population_size=5000
                    )
            if args.env not in ["cartpole", "tora", "lalo", "cartpole_move", "cartpole_swing"]:
                replay_buffer = neural_agent.replay_buffer
                samples = replay_buffer.sample(env_info.lagrange_config["dso_dataset_size"])
                X, y_list = DSO.process_for_dso(samples)
            else:
                x_list, a_list, y_list = zip(*buffer)

                x_list = np.array(x_list)
                a_list = np.array(a_list)
                y_list = np.array(y_list)

                features = np.concatenate((x_list, a_list), axis=1)

                X = features
                y_list = y_list.T
                # pdb.set_trace()

            learned_dynamic_model = []
            for i in range(0, len(y_list)):
                Y = y_list[i]
                reg.fit(X, Y)
                # pdb.set_trace()
                learned_dynamic_model.append(parse_expr(reg.get_model_string(reg.model_, 5)))
            learned_stds = None

            pdb.set_trace()
            for i in range(0, len(learned_dynamic_model)):
                learned_dynamic_model[i] = str(learned_dynamic_model[i]).lower()
        else:
            assert False, "Unknown Symbolic Regression method"

        # save learned dynamics to be used by the verifier
        save_learned_dynamics(
            env_info.env_name, learned_dynamic_model, stds=learned_stds
        )
        # pdb.set_trace()
    print(f"========= Learning Dynamic Time Is {time.time() - dy_t0} seconds ==============")
    all_dy_time = all_dy_time + (time.time() - dy_t0)

    # create the simulated env
    simulated_env = make_simulated_env(
        args.random,
        simulated_env_info.env_name,
        learned_model=learned_dynamic_model,
        stds=learned_stds,
    )

    test_simulated_env = make_simulated_env(
        args.random,
        simulated_env_info.env_name,
        learned_model=learned_dynamic_model,
        stds=learned_stds,
    )

    # Entering the second phase
    train_on_real_data = False
    first = True
    while neural_agent.num_timesteps < args.total_steps:
        if train_on_real_data is True:
            train_on_real_data = False

            safe_t0 = time.time()
            safe_agent = compute_safe_agent(
                args,
                simulated_env_info,
                learned_dynamic_model,
                learned_stds,
                neural_agent,
                env_info.lagrange_config,
                current_version,
            )
            print("training on real data")
            new_data = safe_agent.sample(
                args,
                tracker,
                neural_agent.replay_buffer,
                env_info,
                simulated_env_info,
                learned_dynamic_model,
                learned_stds,
                episodes=25,
                plot_unsafe_set=env_info.plot_other_components,
                plot_state_to_xy=env_info.plot_state_to_xy,
            )
            # pdb.set_trace()
            safe_agent.report()
            print(f"========= Learning Dynamic Time Is {time.time() - safe_t0} seconds ==============")
            all_safe_time = all_safe_time + (time.time() - safe_t0)
            time_table = [["total time", time.time() - all_t0], ["safe time", all_safe_time], ["model time", all_dy_time], ["VELM time", all_safe_time + all_dy_time]]
            print(tabulate.tabulate(time_table))

            # reg = SymbolicRegressor(
            #         allowed_symbols='add,sub,mul,div,constant,variable,sin,cos',
            #         offspring_generator='basic',
            #         local_iterations=5,
            #         max_length=50,
            #         initialization_method='btc',
            #         n_threads=10,
            #         objectives = ['mse'],
            #         symbolic_mode=False,
            #         model_selection_criterion='mean_squared_error',
            #         random_state=4,
            #         generations=10000,
            #         population_size=5000
            #         )
            # # learn a new model
            # buffer += new_data
            # x_list, a_list, y_list = zip(*buffer)
            # x_list = np.array(x_list)
            # a_list = np.array(a_list)
            # y_list = np.array(y_list)

            # features = np.concatenate((x_list, a_list), axis=1)

            # X = features
            # y_list = y_list.T
            #     # pdb.set_trace()

            # learned_dynamic_model = []
            # for i in range(0, len(y_list)):
            #     Y = y_list[i]
            #     reg.fit(X, Y)
            #     # pdb.set_trace()
            #     learned_dynamic_model.append(parse_expr(reg.get_model_string(reg.model_, 5)))
            # save_learned_dynamics(
            #     env_info.env_name, learned_dynamic_model, stds=learned_stds
            # )
            # if (
            #     check_model_accurate(
            #         learned_dynamic_model,
            #         env.observation_space.shape[0],
            #         env.action_space.shape[0],
            #         new_data,
            #     )
            #     == False
            # ):l
            #     # todo: improve the dso model
            #     pass

        else:
            train_on_real_data = True

            # define the new eval call back and stop when converged
            stop_train_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=args.patience, verbose=1
            )

            eval_callback = EvalCallback(
                test_simulated_env,
                eval_freq=args.eval_freq,
                deterministic=True,
                callback_after_eval=stop_train_callback,
            )

            plot_callback = utils.plot.PlotCallback(
                current_version,
                env_info,
                simulated_env_info,
                learned_dynamic_model,
                learned_stds=learned_stds,
                random=args.random,
            )
            eval_callback_2 = EvalCallback(
                test_simulated_env,
                eval_freq=args.eval_freq,
                deterministic=True,
                callback_after_eval=plot_callback,
                n_eval_episodes=1
            )

            callback_list = CallbackList([eval_callback_2, eval_callback])

            neural_agent.set_env(simulated_env)

            new_run = (args.env in ["cartpole", "tora", "lalo", "cartpole_move", "cartpole_swing"]) and first
            first = False
            neural_agent.learn(
                args.individual_learn_steps,
                reset_num_timesteps=new_run,
                tb_log_name="sac",
                progress_bar=True,
                callback=callback_list,
                log_interval=1,
            )
            time_table = [["total time", time.time() - all_t0], ["safe time", all_safe_time], ["model time", all_dy_time], ["VELM time", all_safe_time + all_dy_time]]
            print(tabulate.tabulate(time_table))


if __name__ == "__main__":
    # train
    env = sys.argv[1]
    args = env_to_train_args(env)
    train(args)
