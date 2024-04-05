import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.append("./")
import ACC.improve_lib as improve_lib
import ACC.mse as mse
import gym

# from single_pendulum import SinglePendulum


def eval_controller_lagrange(cmd_first_part, controller, index, args=None):
    # print(controller)
    controller_copy = controller.copy()
    controller_sz = 3
    num_of_controller = len(controller) // controller_sz
    assert num_of_controller == 30
    controllers = []
    for i in range(0, num_of_controller):
        controllers.append(controller[i * controller_sz : (i + 1) * controller_sz])
    for seq, controller in enumerate(controllers):
        fname = f"controller_{index}_{seq}"
        # print("controller", controller)
        lines = [
            "2\n",
            "1\n",
            "0\n",
            "Affine\n",
            f"{controller[0]}\n",
            f"{controller[1]}\n",
            f"{controller[2]}\n",
            "0\n",
            "1",
        ]
        f = open(fname, "w")
        f.writelines(lines)
        f.close()
    cmd = ["./ACC/acc", f"controller_{index}"]
    # print(cmd)
    if args is not None:
        cmd += args
    out = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )
    if args is None:
        safe_loss = float(out)

        # calculate mse
        linear_model = mse.seq_linear(
            num_of_controller, 10, controller_sz, controller_copy
        )
        linear_paths = linear_model.sample_from_initial_states(
            cmd_first_part["initial_states"],
            cmd_first_part["env_str"],
            cmd_first_part["horizon"],
        )
        mse_loss = mse.get_mse_loss(linear_paths, cmd_first_part["neural_paths"], True)
        print("mse loss", mse_loss)
        rst = mse_loss + cmd_first_part["lambda"] * safe_loss

        return (rst, safe_loss)
    else:
        return out


def eval_controller_lagrange_one_iter(cmd_first_part, controller, index, args=None):
    # print(controller)
    controller_copy = controller.copy()
    controller_sz = 3
    num_of_controller = len(controller) // controller_sz
    assert num_of_controller == 30
    controllers = []
    for i in range(0, num_of_controller):
        controllers.append(controller[i * controller_sz : (i + 1) * controller_sz])
    for seq, controller in enumerate(controllers):
        fname = f"controller_{index}_{seq}"
        # print("controller", controller)
        lines = [
            "2\n",
            "1\n",
            "0\n",
            "Affine\n",
            f"{controller[0]}\n",
            f"{controller[1]}\n",
            f"{controller[2]}\n",
            "0\n",
            "1",
        ]
        f = open(fname, "w")
        f.writelines(lines)
        f.close()
    x1_low = -1.1
    x1_high = -0.9
    x2_low = -0.1
    x2_high = 0.1
    safe_loss = 0.0
    for i in range(0, 300, 10):
        print(x1_low, x1_high, x2_low, x2_high)
        cmd = [
            "./ACC/acc_one_iter",
            f"controller_{index}_{i // 10}",
            str(x1_low),
            str(x1_high),
            str(x2_low),
            str(x2_high),
        ]

        out = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE).stdout.decode(
            "utf-8"
        )
        out = out.split()
        print(out)
        x1_low, x1_high, x2_low, x2_high, current_safe_loss = map(
            lambda x: float(x), out
        )
        safe_loss += current_safe_loss

    if args is None:

        # calculate mse
        linear_model = mse.seq_linear(
            num_of_controller, 10, controller_sz, controller_copy
        )
        linear_paths = linear_model.sample_from_initial_states(
            cmd_first_part["initial_states"],
            cmd_first_part["env_str"],
            cmd_first_part["horizon"],
        )
        mse_loss = mse.get_mse_loss(linear_paths, cmd_first_part["neural_paths"], True)
        print("mse loss", mse_loss)
        rst = mse_loss + cmd_first_part["lambda"] * safe_loss

        return (rst, safe_loss)
    else:
        return out


def eval_controller_lagrange_safe_and_reach(cmd_first_part, controller, index):
    # print(controller)
    controller_copy = controller.copy()
    controller_sz = 3
    num_of_controller = len(controller) // controller_sz
    assert num_of_controller == 8
    controllers = []
    for i in range(0, num_of_controller):
        controllers.append(controller[i * controller_sz : (i + 1) * controller_sz])
    for seq, controller in enumerate(controllers):
        fname = f"controller_{index}_{seq}"
        # print("controller", controller)
        lines = [
            "2\n",
            "1\n",
            "0\n",
            "Affine\n",
            f"{controller[0]}\n",
            f"{controller[1]}\n",
            f"{controller[2]}\n",
            "0\n",
            "1",
        ]
        f = open(fname, "w")
        f.writelines(lines)
        f.close()
    cmd = ["./single_pendulum_whole", f"controller_{index}"]
    # print(cmd)
    out = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )
    out = out.split()
    safe_loss = float(out[0])
    reach_loss = float(out[1])

    # calculate mse
    linear_model = mse.seq_linear(num_of_controller, 10, controller_sz, controller_copy)
    linear_paths = linear_model.sample_from_initial_states(
        cmd_first_part["initial_states"],
        cmd_first_part["env_str"],
        cmd_first_part["horizon"],
    )
    mse_loss = mse.get_mse_loss(linear_paths, cmd_first_part["neural_paths"])
    print("mse loss", mse_loss)
    rst = (
        mse_loss
        + cmd_first_part["lambda_safe"] * safe_loss
        + cmd_first_part["lambda_reach"] * reach_loss
    )

    return (rst, safe_loss, reach_loss)


def eval_controller(cmd_first_part, controller, index, args=None):
    # print(controller)
    controller.copy()
    controller_sz = 3
    num_of_controller = len(controller) // controller_sz
    assert num_of_controller == 30
    controllers = []
    for i in range(0, num_of_controller):
        controllers.append(controller[i * controller_sz : (i + 1) * controller_sz])
    for seq, controller in enumerate(controllers):
        fname = f"controller_{index}_{seq}"
        # print("controller", controller)
        lines = [
            "2\n",
            "1\n",
            "0\n",
            "Affine\n",
            f"{controller[0]}\n",
            f"{controller[1]}\n",
            f"{controller[2]}\n",
            "0\n",
            "1",
        ]
        f = open(fname, "w")
        f.writelines(lines)
        f.close()
    cmd = ["./ACC/acc", f"controller_{index}"]
    if args is not None:
        cmd += args
    # print(cmd)
    out = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )
    if args is None:
        safe_loss = float(out)

        return safe_loss
    else:
        return out


def eval_controller_with_partition(cmd_first_part, controller):
    # first parse the initial lb and ub
    initial_lb = cmd_first_part[1:5]
    initial_lb = np.array([float(x) for x in initial_lb])
    # print("initial lb", initial_lb)
    initial_ub = cmd_first_part[9:13]
    initial_ub = np.array([float(x) for x in initial_ub])
    # print("initial ub", initial_ub)

    # final lb and ub
    final_lb = cmd_first_part[5:9]
    final_ub = cmd_first_part[13:17]

    # get the partition
    num_of_partitions = 2
    partitions = []
    for i in range(0, initial_ub.shape[0]):
        partitions.append(
            np.linspace(initial_lb[i], initial_ub[i], num_of_partitions + 1)
        )

    # enumerate the partitions
    # for i in range(0, 4): # here 4 is the dimension of the observation space
    # print('here1')
    loss_list = []
    for d1 in range(0, num_of_partitions):
        for d2 in range(0, num_of_partitions):
            for d3 in range(0, num_of_partitions):
                for d4 in range(0, num_of_partitions):
                    partition_initial_lb = [
                        str(partitions[0][d1]),
                        str(partitions[1][d2]),
                        str(partitions[2][d3]),
                        str(partitions[3][d4]),
                    ]
                    partition_initial_ub = [
                        str(partitions[0][d1 + 1]),
                        str(partitions[1][d2 + 1]),
                        str(partitions[2][d3 + 1]),
                        str(partitions[3][d4 + 1]),
                    ]
                    partition_cmd = (
                        cmd_first_part[0:1]
                        + partition_initial_lb
                        + final_lb
                        + partition_initial_ub
                        + final_ub
                        + [str(x) for x in controller]
                    )
                    loss = subprocess.run(
                        partition_cmd, shell=False, stdout=subprocess.PIPE
                    ).stdout.decode("utf-8")
                    # print(partition_initial_lb, partition_initial_ub, loss)
                    # print(loss)
                    loss_list.append(float(loss))
    # print('here2')
    total_loss = sum(loss_list)
    print("total_loss", total_loss)
    print("max loss", max(loss_list))
    return total_loss


def run(cmd_first_part, params):
    loss_list = []
    theta = params.copy()
    # theta = np.array([-10.780834,   -8.6560955, -2.725309])
    # if str(arg) == "--eval":
    #     eval_controller(theta)
    #     exit()
    best_theta = theta.copy()
    best_loss = 100
    for t in range(0, 100000):
        current = time.time()
        new_theta, loss = improve_lib.true_ars_combined_direction(
            theta.copy(),
            alpha_in=0.005,
            N_of_directions_in=16,
            b_in=2,
            noise_in=0.001,
            eval_controller=eval_controller,
            cmd_first_part=cmd_first_part,
        )

        # current = time.time()
        print(f"----- iteration {t} -------------")
        print("old theta", theta)
        print("updated theta", new_theta)
        loss_list.append(loss)
        print("loss", loss)
        print("loss list", loss_list)
        if loss < best_loss:
            best_theta = theta.copy()
            best_loss = loss
        if loss == 0:
            print("loss reaches 0")
            print("theta", theta)
            print("time", time.time() - current)
            print("saving loss list to loss.txt")

            # get safe sets for theta
            safe_sets = []
            out = eval_controller("", theta, 0, ["--safe_sets"])
            print("safe set is", out)
            lines = out.split("\n")
            for line in lines:
                line = line.split()
                if len(line) > 3:
                    line = [float(x) for x in line]
                    safe_sets.append(line[0:4])

            with open("loss.txt", "a") as file:
                loss_str = [str(x) for x in loss_list]
                loss_str = " ".join(loss_str)
                file.write(loss_str)
                file.write("\n")
            # exit(0)
            break

        theta = new_theta
        print("time", time.time() - current)
        print("----------------------------")
    return best_loss, best_theta, safe_sets


def run_lagrange(cmd_first_part, params, env_str, num_traj, onnx_file):
    horizon = 300
    gym_env = gym.make(env_str)
    initial_states = []
    for i in range(0, num_traj):
        initial_states.append(gym_env.reset())

    # import pdb
    # pdb.set_trace()

    nn = mse.neural_model_without_read(onnx_file)
    neural_paths = nn.sample_from_initial_states(initial_states, env_str, horizon)

    loss_list = []
    safe_loss_list = []
    theta = params.copy()
    lab = 0.5

    cmd_first_part = {}
    cmd_first_part["horizon"] = horizon
    cmd_first_part["env_str"] = env_str
    cmd_first_part["lambda"] = lab
    cmd_first_part["initial_states"] = initial_states
    cmd_first_part["neural_paths"] = neural_paths

    best_theta = theta.copy()
    best_loss = 100
    t = 0
    safe_iter = 0
    while True:
        current = time.time()
        # eval_controller_lagrange_one_iter(cmd_first_part, theta.copy(), 0)
        # exit()
        (
            new_theta,
            new_lab,
            loss,
            safe_loss,
        ) = improve_lib.true_lagrange_combined_direction(
            theta.copy(),
            alpha_in=0.001,
            N_of_directions_in=11,
            b_in=3,
            noise_in=0.001,
            eval_controller=eval_controller_lagrange,
            cmd_first_part=cmd_first_part,
        )

        # current = time.time()
        print(f"----- iteration {t} -------------")
        print("old theta", theta)
        print("updated theta", new_theta)
        print(f"old lambda {lab}, new lambda {new_lab}")
        loss_list.append(loss)
        safe_loss_list.append(safe_loss)
        print("loss", loss, "safe_loss", safe_loss)
        print("loss list", loss_list)
        if loss < best_loss:
            best_theta = theta.copy()
            best_loss = loss
        if safe_loss == 0:
            safe_iter += 1
        if safe_iter >= 5 and safe_loss == 0:
            print("mse loss reaches less than 1e-4")
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
            out = eval_controller("", theta, 0, ["--safe_sets"])
            print("safe set is", out)
            lines = out.split("\n")
            for line in lines:
                line = line.split()
                if len(line) > 3:
                    line = [float(x) for x in line]
                    safe_sets.append(line[0:4])
            # exit(0)
            break

        theta = new_theta
        lab = new_lab
        cmd_first_part["lambda"] = new_lab
        print("time", time.time() - current)
        print("----------------------------")
    return best_loss, best_theta, safe_sets


def run_lagrange_safe_and_reach(cmd_first_part, params, env_str, num_traj, onnx_file):
    horizon = 80
    gym_env = gym.make(env_str)
    initial_states = []
    for i in range(0, num_traj):
        initial_states.append(gym_env.reset())

    # import pdb
    # pdb.set_trace()

    nn = mse.neural_model(onnx_file)
    neural_paths = nn.sample_from_initial_states(initial_states, env_str, horizon)

    loss_list = []
    safe_loss_list = []
    reach_loss_list = []
    theta = params.copy()
    safe_lab = 0.5
    reach_lab = 0.5

    cmd_first_part = {}
    cmd_first_part["horizon"] = horizon
    cmd_first_part["env_str"] = env_str
    cmd_first_part["lambda_safe"] = safe_lab
    cmd_first_part["lambda_reach"] = reach_lab
    cmd_first_part["initial_states"] = initial_states
    cmd_first_part["neural_paths"] = neural_paths

    best_theta = theta.copy()
    best_loss = 100
    for t in range(0, 100000):
        current = time.time()
        # eval_controller_lagrange(cmd_first_part, theta.copy(), 0)
        # exit()
        (
            new_theta,
            new_safe_lab,
            new_reach_lab,
            loss,
            safe_loss,
            reach_loss,
        ) = improve_lib.true_lagrange_combined_direction_safe_and_reach(
            theta.copy(),
            alpha_in=0.005,
            N_of_directions_in=32,
            b_in=2,
            noise_in=0.0005,
            eval_controller=eval_controller_lagrange_safe_and_reach,
            cmd_first_part=cmd_first_part,
        )

        # current = time.time()
        print(f"----- iteration {t} -------------")
        print("old theta", theta)
        print("updated theta", new_theta)
        print(f"old lambda {safe_lab}, new lambda {new_safe_lab}")
        print(f"old lambda {reach_lab}, new lambda {new_reach_lab}")
        loss_list.append(loss)
        safe_loss_list.append(safe_loss)
        reach_loss_list.append(reach_loss)
        print("loss", loss, "safe_loss", safe_loss, "reach_loss", reach_loss)
        print("loss list", loss_list)
        if loss < best_loss:
            best_theta = theta.copy()
            best_loss = loss
        if loss < 1e-4:
            print("mse loss reaches less than 1e-4")
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
            # exit(0)
            break

        theta = new_theta
        safe_lab = new_safe_lab
        reach_lab = new_reach_lab
        cmd_first_part["lambda_safe"] = new_safe_lab
        cmd_first_part["lambda_reach"] = new_reach_lab
        print("time", time.time() - current)
        print("----------------------------")
    return best_loss, best_theta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", action="store_true")
    parser.add_argument("--clean_dir", action="store_true")
    parser.add_argument("--VEL", action="store_true")
    parser.add_argument("--LAG", action="store_true")
    args = parser.parse_args()

    env = SinglePendulum()
    num_of_controller = 8
    num_iters = 10
    count = 0

    file = open("model.txt", "r")
    lines = file.readlines()
    controller = []
    for idx, line in enumerate(lines):
        if idx % num_iters == 0:
            count += 1
            line = line.split()
            line = [float(x) for x in line]
            controller += line
    assert count == num_of_controller
    controller = np.array(controller)
    file.close()

    # the controller below is a safe controller trained from using abstract interpretation
    #     controller = np.array([-0.46834455, -0.5476798 , -0.10055765, -0.43183517, -0.37431092, -0.02341144
    # , -0.46949769, -0.43075975, -0.02294413, -0.26382837, -0.14718253, -0.0072925
    # , -0.05055127 , 0.22323454 , 0.02394191, -0.18186125, -0.043336   , 0.00560376
    # , -0.06450085 , 0.06752627 , 0.00638927, -0.06709761 , 0.09621851 , 0.00463967])

    # the controller below is a controller trained using lagrange using controller above
    #     controller = np.array([-0.4692185 , -0.54507779, -0.0976085 , -0.42574866, -0.38224013, -0.02947752
    # , -0.47512394, -0.43554899, -0.03027757, -0.26902401, -0.1508592 , -0.007966
    # , -0.04834636 , 0.21779247 , 0.0230056 , -0.17902535, -0.04275337 , 0.00909988
    # , -0.05834771 , 0.06788477 , 0.00425909, -0.0643732  , 0.09642416 , 0.00076406])

    #     controller = """-0.46870268 -0.54475147 -0.10174622 -0.43098645 -0.37815941 -0.01962589
    #  -0.47299457 -0.42528616 -0.01976031 -0.26241122 -0.14740672 -0.00486187
    #  -0.04837329  0.22679932  0.02009649 -0.18625247 -0.04496481  0.00895684
    #  -0.0687733   0.0620177   0.00434253 -0.0701186   0.09482014  0.00132622"""

    # a controller with mse < 1e-4
    #     controller = """-0.46816764 -0.54400407 -0.10072984 -0.43008138 -0.37945108 -0.0195251
    #  -0.47544786 -0.42538596 -0.02311547 -0.26153986 -0.14575746 -0.00806508
    #  -0.04881492  0.22706021  0.01898692 -0.18562969 -0.04726302  0.01047524
    #  -0.06692338  0.06279314  0.0052031  -0.06856866  0.09439275  0.00193335"""

    #     controller = """-4.71982490e-01 -5.45443907e-01 -1.03285169e-01 -4.28587488e-01
    #  -3.77484935e-01 -1.88703156e-02 -4.74798571e-01 -4.27710320e-01
    #  -2.51229063e-02 -2.62656815e-01 -1.46013807e-01 -7.14956513e-03
    #  -4.92920182e-02  2.24196573e-01  1.83514332e-02 -1.86639634e-01
    #  -4.63675731e-02  1.01344237e-02 -6.67653219e-02  6.04615383e-02
    #   6.36081558e-03 -6.94778065e-02  9.37098114e-02  2.91037369e-04"""

    # trained with [0.01, 1] for safety
    #     controller = """-0.48161459 -0.54546431 -0.09075082 -0.43459615 -0.38584539 -0.02243796
    #  -0.45822636 -0.41786628 -0.01909365 -0.27247287 -0.14929318 -0.02148486
    #  -0.03379702  0.2181854   0.02625314 -0.18425352 -0.0578834   0.02462131
    #  -0.07632731  0.05831839  0.00267472 -0.08047719  0.09236808  0.01132441"""

    # trained with [0, 1] for safety
    controller = """-0.46706316 -0.5491939  -0.10118215 -0.43508066 -0.3781324  -0.02270409
 -0.47141366 -0.43059993 -0.02627735 -0.26575408 -0.14986231 -0.009468
 -0.04711865  0.2116693   0.0194962  -0.18867718 -0.04952431  0.00668999
 -0.06141152  0.057293    0.00591904 -0.07316538  0.09542352  0.01024159"""
    if type(controller) == str:
        controller = controller.split()
        controller = [float(x) for x in controller]
        controller = np.array(controller)

    if args.clean_dir:
        os.system("rm controller_?_* controller_??_* controller_test_*")

    if args.controller:
        print("writing controller to controller_test files")
        print(controller.shape)
        controller_sz = 3

        num_of_controller = len(controller) // controller_sz
        assert num_of_controller == 8
        controllers = []
        for i in range(0, num_of_controller):
            controllers.append(controller[i * controller_sz : (i + 1) * controller_sz])
        for seq, controller in enumerate(controllers):
            fname = f"controller_test_{seq}"
            # print("controller", controller)
            lines = [
                "2\n",
                "1\n",
                "0\n",
                "Affine\n",
                f"{controller[0]}\n",
                f"{controller[1]}\n",
                f"{controller[2]}\n",
                "0\n",
                "1",
            ]
            f = open(fname, "w")
            f.writelines(lines)
            f.close()
        exit()

    if args.VEL:
        print("running VEL")
        run("", controller)
        exit()

    if args.LAG:
        print("running LAG")
        # run_lagrange("", controller, env.env_name, 300, "controller_single_pendulum.onnx")
        run_lagrange_safe_and_reach(
            "", controller, env.env_name, 300, "controller_single_pendulum.onnx"
        )
        exit()
