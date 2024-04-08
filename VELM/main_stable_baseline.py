import copy
import math
import os
import pathlib
import pdb
import sys
import time

import gymnasium as gym
import numpy as np
import tabulate
from pyoperon.sklearn import SymbolicRegressor
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.logger import Logger, make_output_format
from stable_baselines3.sac.policies import SACPolicy
from sympy import parse_expr

import verification.VEL.improve_lib
import dataset.dataset
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
from verification.VEL.safe_violation_callback import SafeViolationLogCallback

operon_list = [
    "cartpole",
    "tora",
    "lalo",
    "cartpole_move",
    "cartpole_swing",
    "pendulum",
    "road_2d",
    "obstacle",
    "obstacle_mid",
    "road_2d",
]


class FinishException(Exception):
    pass


class OurLogger(Logger):
    def __init__(self, folder, output_formats, max_episodes, env_name):
        super().__init__(folder, output_formats)
        self.reward_logs = []
        self.violation_logs = []
        self.max_episodes = max_episodes
        self.protected_episodes = 0
        self.env_name = env_name

    def record(self, key, value, exclude=None):
        super().record(key, value, exclude)
        if key == "rollout/ep_rew_mean":
            self.reward_logs.append(value)
            if len(self.violation_logs) == 0:
                self.violation_logs.append(0)
            else:
                self.violation_logs.append(self.violation_logs[-1])

        self.check_terminate()

    def dump(self, step=0):
        super().dump(step)

    def check_terminate(self):
        if len(self.reward_logs) >= self.max_episodes and self.protected_episodes >= 0:
            self.final_write()
            raise FinishException()

    def final_write(self):
        assert len(self.violation_logs) == len(self.reward_logs)
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/{self.env_name}.txt", "w") as f:
            for rwd, violation in zip(self.reward_logs, self.violation_logs):
                f.write(f"{rwd} {violation}\n")

    def manual_add(self, rwd, violations):
        self.reward_logs.append(rwd)
        self.violation_logs.append(violations + self.violation_logs[-1])
        self.protected_episodes += 1
        self.check_terminate()


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
    current_version: int,
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
        current_version=current_version,
    )


def check_model_accurate(
    learned_dynamic_model: list,
    new_data: list,
    threshold: float = 0.95,
):
    true_next_state = []
    learned_next_state = []
    for state, action, next_state in new_data:
        true_next_state.append(next_state)
        state_dim = len(state)
        action_dim = len(action)
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
    result = np.any(model_R < threshold)
    print(f"dynamic model accuracy checking is {result}")
    return result


def learn_environment_model(args, env_info, buffer=None, neural_agent=None):
    # learn a new environment model
    if args.sr_method == "DSO":
        dso = DSO(args)
        replay_buffer = neural_agent.replay_buffer
        samples = replay_buffer.sample(env_info.lagrange_config["dso_dataset_size"])
        learned_dynamic_model, learned_stds = dso.learn_dynamic_model(
            samples, random=args.random
        )
    elif args.sr_method == "operon":
        reg = SymbolicRegressor(
            allowed_symbols="add,sub,mul,div,constant,variable,sin,cos",
            offspring_generator="basic",
            local_iterations=5,
            max_length=50,
            initialization_method="btc",
            n_threads=10,
            objectives=["mse"],
            symbolic_mode=False,
            model_selection_criterion="mean_squared_error",
            random_state=4,
            generations=10000,
            population_size=5000,
        )
        assert args.env in operon_list
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
            learned_dynamic_model.append(
                parse_expr(reg.get_model_string(reg.model_, 5))
            )
        learned_stds = None

        for i in range(0, len(learned_dynamic_model)):
            learned_dynamic_model[i] = str(learned_dynamic_model[i]).lower()
            learned_dynamic_model[i] = learned_dynamic_model[i].replace("**", "^")
    else:
        assert False, "Unknown Symbolic Regression method"

    # save learned dynamics to be used by the verifier
    save_learned_dynamics(env_info.env_name, learned_dynamic_model, stds=learned_stds)
    return learned_dynamic_model, learned_stds


def train(args):
    all_t0 = time.time()
    all_safe_time = 0
    all_dy_time = 0
    # create env
    env_info, simulated_env_info = utils.get_env.get_env(args.env)
    env = gym.make(env_info.env_name)

    # set the run id to 1
    current_version = 1

    # define call backs
    safeviolation_callback = SafeViolationLogCallback(args.horizon)
    callback_list = CallbackList([safeviolation_callback])

    # initialize real dataset
    real_data = dataset.dataset.dataset()

    # start warm up training
    if args.load_neural_model:
        f_name = f"sac_checkpoint_{env_info.env_name}"
        neural_agent = SAC.load(f_name)
        print(f"loading neural agent from {f_name}")
    else:
        # in warm up episodes, no linear controllers are involved and only trained on real data
        policy_kwargs = {"net_arch": [args.arch, args.arch]}
        neural_agent = SAC(
            SACPolicy,
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            buffer_size=1000000,
            batch_size=args.batch_size,
            use_sde=True,
            device="auto",
            learning_rate=args.lr,
            stats_window_size=1,
        )
        logger = OurLogger(
            folder=None,
            output_formats=[make_output_format("stdout", ".")],
            max_episodes=args.max_episodes,
            env_name=args.env,
        )
        neural_agent.set_logger(logger)

        # gather data to train the first environment model
        if args.env not in operon_list:
            neural_agent.learn(
                total_timesteps=args.warm_up_steps,
                progress_bar=True,
                callback=callback_list,
                log_interval=1,
            )
            neural_agent.logger.violation_logs = copy.deepcopy(
                safeviolation_callback.safe_violations
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

            print("episodes_rwd", episodes_rwd)
            print("episodes unsafe", episodes_unsafe)
            # exit()

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
            real_data.add_new_data_DSO(neural_agent.replay_buffer)
            learned_dynamic_model, learned_stds = learn_environment_model(
                args, env_info, neural_agent=neural_agent
            )
        elif args.sr_method == "operon":
            real_data.add_new_data_operon(buffer)
            learned_dynamic_model, learned_stds = learn_environment_model(
                args, env_info, buffer=buffer
            )
        else:
            assert False
        # pdb.set_trace()
    print(
        f"========= Learning Dynamic Time Is {time.time() - dy_t0} seconds =============="
    )
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
            must_learn_new_model = False
            try:
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
                    logger,
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
                print(
                    f"========= Lagragian Time Is {time.time() - safe_t0} seconds =============="
                )
                all_safe_time = all_safe_time + (time.time() - safe_t0)
                time_table = [
                    ["total time", time.time() - all_t0],
                    ["safe time", all_safe_time],
                    ["model time", all_dy_time],
                    ["VELM time", all_safe_time + all_dy_time],
                ]
                print(tabulate.tabulate(time_table))

                # add new data and learn a new env model if necessary
                real_data.add_new_safe_data(new_data)
            except verification.VEL.improve_lib.EvalControllerFailure as e:
                must_learn_new_model = True
            if not args.random:
                # currently doesn't check with std
                if (
                    not check_model_accurate(learned_dynamic_model, new_data)
                    or must_learn_new_model
                ):
                    if args.sr_method == "operon":
                        # learn a new model
                        print("===== Learning a new environment model ========")
                        learned_dynamic_model, learned_stds = learn_environment_model(
                            args, env_info, buffer=real_data.get_data_for_operon()
                        )
                        print("===== Constructing new simulated env ========")
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
                    else:
                        assert False, "DSO model is not accurate"
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
                n_eval_episodes=1,
            )
            callback_list = CallbackList([eval_callback_2, eval_callback])

            neural_agent.set_env(simulated_env)

            new_run = (args.env in operon_list) and first
            first = False
            neural_agent.learn(
                total_timesteps=args.individual_learn_steps,
                reset_num_timesteps=new_run,
                progress_bar=True,
                callback=callback_list,
                log_interval=1,
            )
            time_table = [
                ["total time", time.time() - all_t0],
                ["safe time", all_safe_time],
                ["model time", all_dy_time],
                ["VELM time", all_safe_time + all_dy_time],
            ]
            print(tabulate.tabulate(time_table))
            pdb.set_trace()


if __name__ == "__main__":
    # train
    env = sys.argv[1]
    args = env_to_train_args(env)
    train(args)
