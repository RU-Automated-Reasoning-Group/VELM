import argparse
from typing import List


def env_to_train_args(env):
    parser = get_argparser()
    env_args = get_args(env)
    return parser.parse_args(env_args)


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="arguments for training.")

    # environment
    parser.add_argument("--env", type=str, help="environment choice")
    parser.add_argument("--state_dim", type=int, help="number of dimensions in state")

    parser.add_argument("--horizon", type=int, help="length of each episode", required=True)
    parser.add_argument("--max_episodes", type=int, help="maximum episodes for plot purpose", required=True)
    # safe learning algorithm
    parser.add_argument(
        "--warm_up_steps",
        type=int,
        default=4000,
        help="number of training eposides using only the neural controller",
    )

    parser.add_argument(
        "--total_steps",
        type=int,
        default=40000,
        help="total number of episodes in training",
    )

    parser.add_argument(
        "--load_neural_model",
        default=False,
        action="store_true",
        help="load stored neural agent",
    )

    parser.add_argument(
        "--load_dynamic_model",
        default=False,
        action="store_true",
        help="load stored dynamic model",
    )

    parser.add_argument(
        "--individual_learn_steps",
        default=10000,
        type=int,
        action="store",
        help="number of timesteps for each of call to the neural agent's learn function",
    )

    parser.add_argument(
        "--random",
        default=False,
        action="store_true",
        help="True if the dynamic model contains noise",
    )

    parser.add_argument(
        "--patience",
        default=10,
        type=int,
        action="store",
        help="patience for training in simulated env",
    )

    parser.add_argument(
        "--eval_freq",
        default=1000,
        type=int,
        action="store",
        help="evaluation and plot frequency in simulated env",
    )

    parser.add_argument(
        "--batch_size", default=256, type=int, action="store", help="SAC batch size"
    )

    parser.add_argument(
        "--arch", default=256, type=int, action="store", help="SAC architecture size"
    )

    parser.add_argument(
        "--lr", default=3e-4, type=float, action="store", help="SAC learning rate"
    )

    parser.add_argument(
        "--sr_method", default="DSO", type=str, action="store", help="symbolic learning method [DSO or operon]"
    )

    return parser


def get_args(env: str) -> List:
    args_dict = {
        "acc": "--env acc --state_dim 2 --warm_up_steps 600 --total_steps 1500000 --individual_learn_steps 150000 --random --patience 5 --eval_freq 1500 --horizon 300 --max_episodes 200".split(),
        "obstacle": "--env obstacle --state_dim 4 --warm_up_steps 400 --total_steps 150000 --individual_learn_steps 100000 --eval_freq 1200 --horizon 200 --max_episodes 200".split(),
        "obstacle_mid": "--env obstacle_mid --state_dim 4 --warm_up_steps 400 --total_steps 150000 --individual_learn_steps 100000 --patience 10 --eval_freq 1200 --horizon 200 --max_episodes 400".split(),
        "pendulum": "--env pendulum --state_dim 2 --warm_up_steps 200 --total_steps 150000 --individual_learn_steps 150000 --patience 3 --eval_freq 1000 --horizon 100 --max_episodes 200".split(),
        "road": "--env road --state_dim 2 --warm_up_steps 300 --total_steps 150000 --individual_learn_steps 150000 --patience 10 --lr 1e-4".split(),
        "cartpole": "--env cartpole --state_dim 4 --warm_up_steps 800 --total_steps 100000 --individual_learn_steps 100000 --lr 1e-4 --sr_method operon --patience 10 --horizon 200 --max_episodes 200".split(), 
        "car_racing": "--env car_racing --state_dim 4 --warm_up_steps 400 --total_steps 150000 --individual_learn_steps 100000 --patience 10 --eval_freq 1200 --horizon 200 --max_episodes 300".split(), 
        "road_2d": "--env road_2d --state_dim 4 --warm_up_steps 600 --total_steps 150000 --individual_learn_steps 100000 --patience 10 --eval_freq 1500 --horizon 300 --max_episodes 100".split(), 
        "noisy_road_2d": "--env noisy_road_2d --state_dim 4 --warm_up_steps 600 --total_steps 1500000 --individual_learn_steps 600000 --patience 10 --eval_freq 1500 --random".split(), 
        "cartpole_move": "--env cartpole_move --state_dim 4 --warm_up_steps 800 --total_steps 150000 --individual_learn_steps 100000 --lr 1e-4 --sr_method operon --patience 10 --horizon 100 --max_episodes 300".split(), 
        "cartpole_swing": "--env cartpole_swing --state_dim 4 --warm_up_steps 800 --total_steps 150000 --individual_learn_steps 100000 --lr 1e-4 --sr_method operon --patience 10 --horizon 100 --max_episodes 300".split(), 
        "tora": "--env tora --state_dim 4 --warm_up_steps 600 --total_steps 150000 --individual_learn_steps 150000 --lr 1e-4 --sr_method operon --patience 1000 --load_dynamic_model".split(), 
        "lalo": "--env lalo --state_dim 7 --warm_up_steps 600 --total_steps 150000 --individual_learn_steps 150000 --lr 1e-6 --sr_method operon --patience 10 --horizon 100 --max_episodes 300".split(), 
    }
    return args_dict[env]
