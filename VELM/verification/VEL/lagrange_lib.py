import subprocess
import time
from typing import List, Optional

import gymnasium as gym
import numpy as np

import verification.VEL.improve_lib as improve_lib
import verification.VEL.mse as mse

from environments.simulated_env_util import make_simulated_env


def convert_to_polar_controller_format(
    state_dim: int, action_dim: int, controller: np.ndarray
):
    polar_controller = [
        f"{state_dim}\n",
        f"{action_dim}\n",
        "0\n",
        "Affine\n",
    ]
    reshaped_controller = np.reshape(
        controller, (-1, action_dim)
    )  # (state_dim + 1, action_dim)
    W_part = reshaped_controller[:-1, :].ravel()
    b_part = reshaped_controller[-1, :].ravel()
    to_line = lambda x: f"{x}\n"
    W_part_str = list(map(to_line, W_part))
    b_part_str = list(map(to_line, b_part))
    polar_controller.extend(W_part_str)
    polar_controller.extend(b_part_str)
    polar_controller.extend(["0\n", "1\n"])
    return polar_controller


def eval_controller_lagrange(
    config: dict, controller: np.ndarray, index: int, args: list = []
):
    state_dim = config["state_dim"]
    action_dim = config["action_dim"]
    # infer the size of each controller from the config
    controller_sz = (state_dim + 1) * action_dim
    num_of_controller = config["horizon"] // config["iters_run"]
    if index == 0:
        print("config[horizon]", config["horizon"])
        print("config[iters_run]", config["iters_run"])
        print("controller_sz", controller_sz)
        print("num of controller", num_of_controller)
        print("controller length", len(controller))
        # print("controller", controller)
        # import pdb
        # pdb.set_trace()
    # assert num_of_controller == len(controller) // controller_sz

    reshaped_controller = np.reshape(controller, (-1, controller_sz))
    for seq, one_controller in enumerate(reshaped_controller):
        fname = f"controller_{index}_{seq}"
        line = convert_to_polar_controller_format(state_dim, action_dim, one_controller)
        f = open(fname, "w")
        f.writelines(line)
        f.close()
    cmd = [f"./{config['eval_program']}", f"controller_{index}"]
    cmd.extend(args)
    # print(f"cmd is {cmd}")
    polar_output = subprocess.run(
        cmd, shell=False, stdout=subprocess.PIPE
    ).stdout.decode("utf-8")

    # print(f"polar output is {polar_output}")
    if args == []:
        safe_loss = float(polar_output)

        # calculate mse
        linear_model = mse.seq_linear(
            num_of_controller,
            config["iters_run"],
            controller_sz,
            controller,
            config["state_dim"],
            config["action_dim"],
        )
        # print("before collecting linear paths")
        linear_paths = linear_model.sample_from_initial_states(
            config["initial_states"],
            config["env_str"],
            config["learned_model"],
            config["stds"],
            config["random"],
            config["horizon"],
        )
        # print("finish samping linear paths")
        mse_loss = mse.get_mse_loss(linear_paths, config["neural_paths"], True)
        # print("mse loss", mse_loss)
        rst = mse_loss + config["lambda"] * safe_loss

        return (rst, safe_loss)

    return polar_output


def run_lagrange(
    action_dim: int,
    state_dim: int,
    params: np.ndarray,
    env_str: str,
    learned_model: List[str],
    stds: Optional[List[float]],
    random: bool,
    num_traj: int,
    horizon: int,
    neural_agent,
    alpha: float,
    N_of_directions: int,
    b: int,
    noise: float,
    initial_lambda: float,
    iters_run: int,
    eval_program: str,
    patience: int = 2,
) -> tuple[float, np.ndarray, float]:
    gym_env = make_simulated_env(random, env_str, learned_model=learned_model, stds=stds)

    initial_states = []
    for _ in range(0, num_traj):
        initial_states.append(gym_env.reset()[0])

    # import pdb
    # pdb.set_trace()

    wrapped_neural_agent = mse.neural_model(neural_agent)
    neural_paths = wrapped_neural_agent.sample_from_initial_states(
        initial_states, env_str, learned_model, stds, random, horizon
    )

    loss_list = []
    safe_loss_list = []
    theta = params.copy()

    config = {}
    # lagrange setting
    config["state_dim"] = state_dim
    config["action_dim"] = action_dim
    config["horizon"] = horizon
    config["lambda"] = initial_lambda
    config["lambda_lr"] = 0.05
    config["env_str"] = env_str
    config["learned_model"] = learned_model
    config["stds"] = stds
    config["random"] = random
    config["iters_run"] = iters_run
    config["eval_program"] = eval_program

    # neural agent
    config["initial_states"] = initial_states
    config["neural_paths"] = neural_paths

    best_theta = theta.copy()
    best_loss = 100
    t = 0
    safe_iter = 0
    wait_count: int = 0
    best_mse_loss = None
    while True:
        current = time.time()
        # eval_controller_lagrange(cmd_first_part, theta.copy(), 0)
        # exit()
        (
            new_theta,
            new_lab,
            loss,
            safe_loss,
        ) = improve_lib.true_lagrange_combined_direction(
            theta.copy(),
            alpha_in=alpha,
            N_of_directions_in=N_of_directions,
            b_in=b,
            noise_in=noise,
            eval_controller=eval_controller_lagrange,
            cmd_first_part=config,
        )

        # current = time.time()
        print(f"----- iteration {t} -------------")
        t += 1
        print("old theta", theta)
        print("updated theta", new_theta)
        print(f"old lambda {initial_lambda}, new lambda {new_lab}")
        loss_list.append(loss)
        safe_loss_list.append(safe_loss)
        print("loss", loss, "safe_loss", safe_loss)
        print("loss list", loss_list)

        should_stop = False
        if safe_loss == 0:
            safe_iter += 1
            # safe right now, start counting patience
            if loss < best_loss:
                best_theta = theta.copy()
                best_loss = loss
                wait_count = 0
            else:
                wait_count += 1
                if wait_count >= patience:
                    should_stop = True
        else:
            # not safe again
            # restart wait_count
            wait_count = 0

        if safe_iter > 100:
            should_stop = True

        if should_stop:
            print("exit the lagrange method")
            print("theta", theta)
            print("time", time.time() - current)
            print("saving safe loss list to safe_loss.txt")
            with open("safe_loss.txt", "a") as file:
                loss_str = [str(x) for x in safe_loss_list]
                loss_str = " ".join(loss_str)
                file.write(loss_str)
                file.write("\n")
            with open("loss.txt", "a") as file:
                loss_str = [str(x) for x in loss_list]
                loss_str = " ".join(loss_str)
                file.write(loss_str)
                file.write("\n")

            # get safe sets for theta
            safe_sets = []
            out = eval_controller_lagrange(config, best_theta, 0, ["--safe_sets"])
            print("safe set is", out)
            lines = out.split("\n")
            for line in lines:
                line = line.split()
                if len(line) > 3:
                    line = [float(x) for x in line]
                    safe_sets.append(line)
            # exit(0)
            break

        theta = new_theta
        initial_lambda = new_lab
        config["lambda"] = new_lab
        print("time", time.time() - current)
        print("----------------------------")
    return best_loss, best_theta, safe_sets
