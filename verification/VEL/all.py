import subprocess

import numpy as np
import vel_one


def find_length(dir):
    # read all lower bounds:
    lbs = []
    with open(dir + "lbs.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            lb = line.split()
            lbs.append(lb)
    # lbs[0] = ['-0.05', '-0.05', '-0.05', '-0.05']

    # read all upper bounds:
    ubs = []
    with open(dir + "ubs.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            ub = line.split()
            ubs.append(ub)
    # ubs[0] = ['0.05', '0.05', '0.05', '0.05']

    for idx, _ in enumerate(lbs):
        lb, ub = lbs[idx], ubs[idx]
        flag = True
        for x in lb:
            if float(x) < -0.25:
                flag = False
        for x in ub:
            if float(x) > 0.25:
                flag = False

        if flag:
            print("line", idx + 1, "is correct")
            # break


def verify_all(print_flag, dir, verify_only=False):
    if verify_only:
        print_flag = True
    # verifies all controllers

    # read all controllers:
    controllers = []
    with open(dir + "model.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            c = line.split()
            controllers.append(c)

    # read all lower bounds:
    lbs = []
    with open(dir + "lbs.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            lb = line.split()
            lbs.append(lb)
    # lbs[0] = ['-0.05', '-0.05', '-0.05', '-0.05']

    # read all upper bounds:
    ubs = []
    with open(dir + "ubs.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            ub = line.split()
            ubs.append(ub)

    # controllers = [controllers[0] for x in range(0, len(ubs))]
    # ubs[0] = ['0.05', '0.05', '0.05', '0.05']

    # pdb.set_trace()
    # for index, controller in enumerate(controllers):
    for index in range(0, 150, 25):
        controller = controllers[index]
        # controller = [str(x) for x in [4.894885,  -3.5926018, -6.021182,  -7.5303345, 0.03799889]]
        if index > -1:
            print(f"Verifying controller {index}")
            print("stats")
            print("lbs[index]", lbs[index])
            print("lbs[index + 25]", lbs[index + 25])
            print("ubs[index]", ubs[index])
            print("ubs[index + 25]", ubs[index + 25])
            cmd_first_part = (
                ["./" + dir + "one"]
                + lbs[index]
                + lbs[index + 25]
                + ubs[index]
                + ubs[index + 25]
            )
            print_lst = [] if print_flag is False else ["--print"]
            loss = subprocess.run(
                cmd_first_part + controller + print_lst,
                shell=False,
                stdout=subprocess.PIPE,
            ).stdout.decode("utf-8")
            print(f"Controller {index} has loss {loss}")
            if verify_only:
                out = loss.split()
                with open("verified_lbs.txt", "a") as f:
                    f.write(f"{out[0]} {out[1]} {out[2]} {out[3]}\n")

                with open("verified_ubs.txt", "a") as f:
                    f.write(f"{out[4]} {out[5]} {out[6]} {out[7]}\n")
                continue
            if float(loss) > 0:
                print(f"Starting VEL training for controller {index}")
                controller = np.array([float(x) for x in controller])
                best_loss, best_controller = vel_one.run(cmd_first_part, controller)
                print(f"controller {index} has loss {best_loss} at {best_controller}")
                print("now verifying again to get final box")
                best_controller = [str(x) for x in best_controller]
                out = subprocess.run(
                    cmd_first_part + best_controller + ["--print"],
                    shell=False,
                    stdout=subprocess.PIPE,
                ).stdout.decode("utf-8")
                out = out.split()
                lbs[index + 25] = out[0:4]
                ubs[index + 25] = out[4:8]
                new_loss = out[8]
                # import pdb
                # pdb.set_trace()
                print(
                    f"new loss as single partition is {new_loss} and originall loss is {loss}"
                )
                print(f"target lbs is {out[0:4]} and ubs is {out[4:8]}")
                # vel_one.eval_controller_with_partition(cmd_first_part, best_controller)
            # break


if __name__ == "__main__":
    count = 0
    with open("model.txt", "r") as f:
        lines = f.readlines()
        controllers = []
        for idx, line in enumerate(lines):
            if idx % 5 == 0 and idx != 99:
                line = line.split()
                line = [float(x) for x in line]
                controllers = controllers + line
                count += 1
    print(f"{count} controllers added", len(controllers))
    # print(controllers)
    # controller = np.array([
    #     -0.5266466736793518, -0.6544913649559021, -0.017249464988708496,
    #     -0.4756084382534027, -0.44064220786094666, -0.01947667822241783,
    #     -0.41119319200515747, -0.3381892740726471, -0.02262021042406559,
    #     -0.41201838850975037, -0.35144636034965515, -0.0271429605782032
    #     ])
    # vel_one.run("", np.array(controllers))
    vel_one.eval_controller("", controllers, 1)
    # import sys
    # print_flag = False
    # if (len(sys.argv) == 2):
    #         print_flag = (str(sys.argv[1]) == "--print")
    # bench_name = 'tora'
    # dir = 'experiments/mjrl/benches/' + bench_name + '/'
    # # find_length(dir)
    # verify_all(print_flag, dir, False)
