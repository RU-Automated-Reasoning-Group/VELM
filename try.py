# import gymnasium
# import stable_baselines3
# from stable_baselines3.common.callbacks import (
#     BaseCallback,
#     CallbackList,
#     EvalCallback,
#     StopTrainingOnNoModelImprovement,
# )

# env = gymnasium.make("MountainCarContinuous-v0")
# test_env = gymnasium.make("MountainCarContinuous-v0")
# model = stable_baselines3.PPO("MlpPolicy", env, batch_size=256, clip_range=0.1, ent_coef=0.00429, gae_lambda=0.9, gamma=0.9999, learning_rate=7.77e-5, max_grad_norm=5, n_epochs=10, n_steps=8, policy_kwargs={"log_std_init": -3.29, "ortho_init": False}, use_sde=True, vf_coef=0.19)
# eval_callback = EvalCallback(test_env, eval_freq=1000, deterministic=True)

# model.learn(100000, callback=eval_callback, progress_bar=True)

import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(0.01, 2, 300)
ys = -xs**4 + xs**3 + xs**3 * np.log(xs) - xs + xs * np.log(xs) + 1 - (xs - 1)**3

plt.plot([-0.3, 2], [0, 0])
plt.plot(xs, ys)
plt.savefig("gg.png")