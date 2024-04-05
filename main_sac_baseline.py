import os
import pathlib
import sys

import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.sac.policies import SACPolicy

from utils.arguments import env_to_train_args
from utils.get_env import get_env
from verification.VEL.safe_violation_callback import (
    SafeViolationCallback,
    safety_violation_tracker,
)


def train(args):
    # define env
    env_info, _ = get_env(args.env)
    env = gym.make(env_info.env_name)
    test_env = gym.make(env_info.env_name)

    # callbacks
    safe_violation_logdir = f"results_sac/safe_violation/{env_info.env_name}"
    safe_violation_path = pathlib.Path(safe_violation_logdir)
    safe_violation_path.mkdir(parents=True, exist_ok=True)
    current_version = len(os.listdir(safe_violation_path)) + 1

    final_safe_violation_logdir = os.path.join(
        safe_violation_logdir, f"sac_{current_version}"
    )

    tracker = safety_violation_tracker(final_safe_violation_logdir)
    safeviolation_callback = SafeViolationCallback(tracker)

    eval_callback = EvalCallback(
        test_env,
        eval_freq=1000,
        deterministic=True,
        best_model_save_path=f"results/saved_agents/{env_info.env_name}/sac_warm_checkpoint",
    )

    callback_list = CallbackList([safeviolation_callback])

    # define SAC agnet
    policy_kwargs = {"net_arch": [args.arch, args.arch]}

    # neural_agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./results_sac/logs/{env_info.env_name}", policy_kwargs=policy_kwargs)
    neural_agent = SAC(
        SACPolicy,
        env,
        verbose=1,
        tensorboard_log=f"./results_sac/logs/{env_info.env_name}",
        policy_kwargs=policy_kwargs,
        buffer_size=10000000,
        batch_size=args.batch_size,
        use_sde=True,
        learning_rate=1e-4,
        stats_window_size=1,
    )

    # start learning
    neural_agent.learn(
        total_timesteps=100_0000,
        progress_bar=True,
        tb_log_name="sac",
        callback=callback_list,
        log_interval=1,
    )


if __name__ == "__main__":
    # train
    env = sys.argv[1]
    # import pdb
    # pdb.set_trace()
    args = env_to_train_args(env)
    train(args)
