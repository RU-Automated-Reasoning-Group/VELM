import argparse
import os
import pathlib
import gc

import matplotlib.pyplot as plt

import main_stable_baseline
import utils.arguments


def run_single_benchmark(benchmark: str):
    try:
        train_args = utils.arguments.env_to_train_args(benchmark)
        main_stable_baseline.train(train_args)
    except main_stable_baseline.FinishException:
        print(f"{benchmark} finish successfully")
        return 1
    except Exception as e:
        # import pdb
        # pdb.set_trace()
        print(e)
        print(f"{benchmark} exited unexpectedly")
        return 0


def run_one_row(row: int):
    if row == 0:
        return
    row2bench = {
        1: ["pendulum", "acc", "obstacle_mid", "cartpole"],
        2: ["obstacle", "road_2d", "car_racing", "cartpole_move"],
        3: ["cartpole_swing", "lalo"],
    }
    bench = row2bench[row]
    success_count = 0
    for b in bench:
        rst = run_single_benchmark(b)
        success_count += rst
        gc.collect()

    print(f"{success_count} out of {len(row2bench[row])} benchmarks finish successful")


def run_main(start: int, end: int):
    for row in range(start, end + 1):
        run_one_row(row)

    eval_plot()


def plot_one_env(ax, row, column, env_name, color="blue"):
    path = pathlib.Path(f"logs/{env_name}.txt")
    if path.exists():
        print(f"{env_name}: plotting")
        with open(path) as f:
            lines = f.readlines()

        xs = [i for i in range(len(lines))]
        rwds = [float(log.split()[0]) for log in lines]
        violations = [float(log.split()[1]) for log in lines]

        ax[row][column].plot(xs, rwds, color=color)
        ax[row + 1][column].plot(xs, violations, color=color)

        ax[row][column].set_title(env_name)
        ax[row][column].set_ylabel("total rewards")
        ax[row + 1][column].set_ylabel("# unsafe steps")
        ax[row + 1][column].set_xlabel("# of episodes in total")
    else:
        print(f"{env_name}: logs not found")


def eval_plot():
    nrows = 8
    ncols = 4
    height_ratios = [1, 1, 0.01, 1, 1, 0.01, 1, 1]
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, squeeze=False, height_ratios=height_ratios
    )
    fig.set_figheight(3.5 * nrows)
    fig.set_figwidth(16.9)

    for i in range(len(ax[2])):
        ax[2, i].set_visible(False)

    for i in range(len(ax[5])):
        ax[5, i].set_visible(False)

    ax[6, 0].set_visible(False)
    ax[6, 3].set_visible(False)
    ax[7, 0].set_visible(False)
    ax[7, 3].set_visible(False)

    plot_one_env(ax, 0, 0, "pendulum")
    plot_one_env(ax, 0, 1, "acc")
    plot_one_env(ax, 0, 2, "obstacle_mid")
    plot_one_env(ax, 0, 3, "cartpole")

    plot_one_env(ax, 3, 0, "obstacle")
    plot_one_env(ax, 3, 1, "road_2d")
    plot_one_env(ax, 3, 2, "car_racing")
    plot_one_env(ax, 3, 3, "cartpole_move")

    plot_one_env(ax, 6, 1, "cartpole_swing")
    plot_one_env(ax, 6, 2, "lalo")

    ax[0, 0].set_ylim([-10, 0])
    ax[0, 0].set_xlim(left=0)
    ax[1, 0].set_xlim(left=0)
    ax[1, 0].set_ylim([-100, 1000])

    ax[0, 1].set_ylim([200, 600])
    ax[0, 1].set_xlim(left=0)
    ax[1, 1].set_xlim(left=0)
    ax[1, 1].set_ylim([-100, 700])

    ax[0, 2].set_ylim([-4000, -300])
    ax[0, 2].set_xlim(left=0)
    ax[1, 2].set_xlim(left=0)
    ax[1, 2].set_ylim([-100, 700])

    ax[0, 3].set_ylim([-2000, 100])
    ax[0, 3].set_xlim(left=0)
    ax[1, 3].set_xlim(left=0)
    ax[1, 3].set_ylim([-1000, 24000])

    ax[3, 0].set_ylim([-3000, 0])
    ax[3, 0].set_xlim(left=0)
    ax[4, 0].set_xlim(left=0)
    ax[4, 0].set_ylim([-30, 850])

    ax[3, 1].set_ylim([-6000, 0])
    ax[3, 1].set_xlim(left=0)
    ax[4, 1].set_xlim(left=0)
    ax[4, 1].set_ylim([-100, 2500])

    ax[3, 2].set_ylim([-6000, -1500])
    ax[3, 2].set_xlim(left=0)
    ax[4, 2].set_xlim(left=0)
    ax[4, 2].set_ylim([-100, 1500])

    ax[3, 3].set_ylim([-2000, 100])
    ax[3, 3].set_xlim(left=0)
    ax[4, 3].set_xlim(left=0)
    ax[4, 3].set_ylim([-100, 7000])

    ax[6, 1].set_ylim([-4000, 1000])
    ax[6, 1].set_xlim(left=0)
    ax[7, 1].set_xlim(left=0)
    ax[7, 1].set_ylim([-100, 10000])

    ax[6, 2].set_ylim([-1000, 0])
    ax[6, 2].set_xlim(left=0)
    ax[7, 2].set_xlim(left=0)
    ax[7, 2].set_ylim([-100, 2000])

    fig.align_ylabels()
    plt.subplots_adjust(wspace=0.5)
    plt.savefig("eval.png", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int, default=-1)
    parser.add_argument("--single", type=str, default="")

    args = parser.parse_args()
    if args.row != -1:
        # handle row
        run_main(args.row, args.row)
    elif args.single != "":
        # handle individual benchmark
        assert args.single in [
            "pendulum",
            "acc",
            "obstacle_mid",
            "cartpole",
            "obstacle",
            "road_2d",
            "car_racing",
            "cartpole_move",
            "cartpole_swing",
            "lalo",
        ], f"unknown benchmark {args.single}"
        run_single_benchmark(args.single)
        eval_plot()
    else:
        # no useful command, just update the figure
        eval_plot()
