import datetime
import math
import os
import pdb
import time

import gym

# import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import learned_model

# from Single_Pendulum.vel_one import run, run_lagrange
# from ACC.vel_one import run_lagrange
# from cartpole.vel_one import run_lagrange
from control.rl.policies.gaussian_prog import ProgPolicy
from dynamics.learn import DynamicsLearner
from environments.env import GymENV
from environments.replay_memory import (
    ReplayMemory,
    push_trajectory,
    push_trajectory_with_safe_sets,
)
from models.dso_model import DSO
from models.sac_controller import SAC, SAC_agent
from models.safe_controller import safe_agent
from trainer.evaluate import do_eval, do_eval_with_linear
from trainer.sac_trainer import train_sac
from utils.arguments import train_args
from utils.gym_env import GymEnv
from utils.logs import Logger, init_logging, log_and_print

# from safe_pendulum.vel_one import run_lagrange

# from road.vel_one import run_lagrange


def save_learned_dynamics(learned_dynamics):
    with open("learned_dynamics.txt", "w") as f:
        for equation in learned_dynamics:
            f.write(f"{equation}\n")


def load_model(path):
    num_of_controller = 400
    num_iters = 1
    count = 0

    file = open(path + "model.txt", "r")
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
    return controller


def train_one_episode(
    args,
    writer,
    env,
    agent,
    replay_buffer,
    updates,
    total_numsteps,
    total_unsafe_episodes,
    device,
    real_data=True,
):
    # adapted from spice code
    if real_data:
        print("training on real data")
    else:
        print("training on data from learned model")

    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        import pdb

        # pdb.set_trace()
        state_tensor = torch.tensor(np.array([state])).type(torch.float32).to(device)
        action = agent(state_tensor, evaluate=False)

        if len(replay_buffer) > args.sac_batch_size:
            # Number of updates per step in environment
            for _ in range(args.sac_updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_l, alph = agent.train(
                    replay_buffer
                )

                writer.add_scalar("loss/critic_1", critic_1_loss, updates)
                writer.add_scalar("loss/critic_2", critic_2_loss, updates)
                writer.add_scalar("loss/policy", policy_loss, updates)
                writer.add_scalar("loss/entropy_loss", ent_l, updates)
                writer.add_scalar("entropy_temprature/alpha", alph, updates)
                # print(f"updates is {updates}")
                updates += 1

        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        cost = 0
        if env.unsafe(next_state):
            total_unsafe_episodes += 1
            episode_reward -= 1000
            print("UNSAFE (outside testing)")
            done = True
            cost = 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon.
        # github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        if not (
            np.any(next_state <= env.observation_space.low)
            or np.any(next_state >= env.observation_space.high)
        ):
            # only add transition to the replay buffer when clip is not used
            # this is used to learn the dynamics using dso
            replay_buffer.push(state, action, reward, next_state, mask)

        # tmp_buffer.append((state, action, reward, next_state, mask, cost))

        # if env.unsafe(next_state):
        # episode_reward -= 10000

        # Don't add states to the training data if they hit the edge of
        # the state space, this seems to cause problems for the regression.
        # if not (np.any(next_state <= env.observation_space.low) or
        #         np.any(next_state >= env.observation_space.high)):
        #     real_buffer.append((state, action, reward, next_state, mask,
        #                         cost))

        state = next_state
    return episode_steps, episode_reward, updates, total_unsafe_episodes, total_numsteps


def eval_one_episode(
    args, episode, writer, env, agent, total_eval_unsafe_episodes, device, plot=True
):
    if episode % args.eval_epochs != 0:
        return None, None, None

    # adapted from spice code
    print("evaluating on real data")

    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    plt.cla()
    xs = [state[0]]
    ys = [state[1]]
    while not done:
        state_tensor = torch.tensor(np.array([state])).type(torch.float32).to(device)
        action = agent(state_tensor, evaluate=False)

        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        episode_reward += reward

        if env.unsafe(next_state):
            episode_reward -= 1000
            print("UNSAFE (during testing)")
            done = True
            total_eval_unsafe_episodes += 1

        state = next_state
        xs.append(state[0])
        ys.append(state[1])

    if plot is True:
        plt.plot(xs, ys)
        plt.savefig(f"eval_episode_{episode}.png")
        print(f"saving trajectory to eval_episode_{episode}.png")

    writer.add_scalar("reward/eval", episode_reward, episode)
    writer.add_scalar("reward/eval_unsafe", total_eval_unsafe_episodes, episode)
    return episode_steps, episode_reward, total_eval_unsafe_episodes


def compute_safe_agent(learned_env, neural_agent, eval_function, run_lagrange):
    # learner_env = GymEnv(learned_env)
    # learner = DynamicsLearner(learner_env, ProgPolicy(learner_env.spec, neural_agent))
    # low = np.array([-0.1, -0.1, -0.1, -0.1])
    # high = np.array([0.1, 0.1, 0.1, 0.1])
    # learner.run(low, high, 500, 1, "logs/")

    model = load_model("logs/")

    # eval_function({}, model, 0)
    _, theta, safe_sets = run_lagrange(None, None, args.env, 500, neural_agent)
    return safe_agent(neural_agent, theta, learned_env, safe_sets)


def train(args):
    loss_list = []
    best_epoch = -1
    best_reward = -math.inf

    # start time
    timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    loss_file_name = f"loss_figs/{args.env}/loss_{timestamp}.png"

    # define training
    # pdb.set_trace()
    traj_len = max(args.sample_len, args.sac_batch_size)
    if args.env == "cartpole":
        from environments.gymnasium_cartpole import CartPole

        pole = CartPole()
    elif args.env == "safe_inverted_pendulum":
        from environments.inverted_pendulum import SafeInvertedPendulum

        pole = SafeInvertedPendulum()
    elif args.env == "obstacle":
        from environments.obstacle import Obstacle

        pole = Obstacle()
    elif args.env == "obstacle_mid":
        from environments.obstacle_mid import ObstacleMid
        from verification.obstacle_mid.vel_one import (
            eval_controller_lagrange,
            run_lagrange,
        )

        pole = ObstacleMid()
    elif args.env == "acc":
        from environments.acc import ACC

        pole = ACC()
    elif args.env == "pendulum_spice":
        from environments.pendulum_spice import PendulumSpice

        pole = PendulumSpice()
    elif args.env == "mountain_car":
        from environments.mountain_car import MountainCar

        pole = MountainCar()
    elif args.env == "road":
        from environments.road import Road

        pole = Road()
    elif args.env == "noisy_road":
        from environments.noisy_road import NoisyRoad

        pole = NoisyRoad()
    elif args.env == "noisey_road_2d":
        from environments.noisy_road_2d import NoisyRoad2d

        pole = NoisyRoad2d()
    else:
        assert False

    learner_env = GymEnv(pole.gym_env)

    # init environment
    # env = GymENV(pole.env_name)
    env = gym.make(pole.env_name)

    state_dim = args.state_dim
    action_space = env.action_space
    replay_buffer = ReplayMemory(args)

    # init dso model (TODO)
    # dso = DSO(args)
    # env_dynamic = env.ground_truth_dynamic
    device = torch.device("cuda" if args.cuda else "cpu")

    # init optimal policy
    neural_policy = SAC(args, state_dim, action_space)
    neural_agent = SAC_agent(neural_policy, args.sac_batch_size)

    # logger
    if not os.path.exists(args.store_path):
        os.makedirs(args.store_path)
    logger = Logger(args.store_path)
    writer = SummaryWriter(
        f"runs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_SAC_{args.env}"
    )

    # log
    init_logging(args.store_path)

    updates = 0
    total_numsteps = 0
    total_unsafe_episodes = 0
    total_eval_unsafe_episodes = 0

    # here begins the new code

    if args.load_neural_model:
        f_name = f"sac_checkpoint_{args.env}_.pth"
        neural_agent.agent.load_checkpoint(f_name)
        print(f"loading neural agent from {f_name}")
    else:
        # in warm up episodes, no linear controllers are involved and only trained on real data
        for episode in range(0, args.warm_up_episodes):
            # train the neural policy for one episode
            (
                episode_steps,
                episode_reward,
                updates,
                total_unsafe_episodes,
                total_numsteps,
            ) = train_one_episode(
                args,
                writer,
                env,
                neural_agent,
                replay_buffer,
                updates,
                total_numsteps,
                total_unsafe_episodes,
                device,
            )
            writer.add_scalar("reward/train", episode_reward, episode)
            writer.add_scalar("reward/unsafe", total_unsafe_episodes, episode)
            print(
                f"Episode: {episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {episode_reward}"
            )

            eval_steps, eval_reward, total_eval_unsafe_episodes = eval_one_episode(
                args,
                episode,
                writer,
                env,
                neural_agent,
                total_eval_unsafe_episodes,
                device,
            )
            if eval_steps is not None:
                print(
                    f"Eval Episode: {episode}, eval_steps: {eval_steps}, eval_reward: {eval_reward}"
                )

        neural_agent.agent.save_checkpoint(args.env, "", "")

    # learn the dso model
    if args.load_dynamic_model:
        with open("learned_dynamics.txt", "r") as f:
            lines = f.readlines()
            learned_dynamic_model = [line[:-1] for line in lines]
            print("loading dyancmic model from learned_dynamics.txt")
    else:
        # learn a new model
        dso = DSO(args)
        X, y_list = replay_buffer.sample_for_dso(
            data_num=None, preprocess=args.preprocess
        )
        dso.train_dso(X, y_list)
        learned_dynamic_model = dso.get_learned_model()

        # save learned dynamics to be used by the verifier
        save_learned_dynamics(learned_dynamic_model)

    # import pdb
    # pdb.set_trace()
    # construct the learned environment
    learned_env = learned_model.LearnedModel(
        gym.make(pole.env_name), learned_dynamic_model, args.preprocess
    )
    pdb.set_trace()
    # Entering the second phase
    for episode in range(args.warm_up_episodes, args.total_episodes):
        if episode % 100 < 10:
            # train on real data for 10% of the eposides
            train_on_real_data = True
        else:
            train_on_real_data = False

        train_on_real_data = True
        if train_on_real_data:
            safe_agent = compute_safe_agent(
                learned_env, neural_agent.agent, eval_controller_lagrange
            )
        else:
            (
                episode_steps,
                episode_reward,
                updates,
                total_unsafe_episodes,
                total_numsteps,
            ) = train_one_episode(
                args,
                writer,
                learned_env,
                neural_agent,
                replay_buffer,
                updates,
                total_numsteps,
                total_unsafe_episodes,
                device,
                real_data=False,
            )
        writer.add_scalar("reward/train", episode_reward, episode)
        writer.add_scalar("reward/unsafe", total_unsafe_episodes, episode)
        print(
            f"Episode: {episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {episode_reward}"
        )

        eval_steps, eval_reward, total_eval_unsafe_episodes = eval_one_episode(
            args, episode, writer, env, neural_agent, total_eval_unsafe_episodes, device
        )
        if eval_steps is not None:
            print(
                f"Eval Episode: {episode}, eval_steps: {eval_steps}, eval_reward: {eval_reward}"
            )

    # do train
    # total_start_time = time.time()
    # for step in range(args.epochs):
    #     log_and_print("-------- current epoch: {} -----------".format(step))

    #     start_time = time.time()

    #     # sample trajectory (TODO: consider when to done)
    #     push_trajectory(env, replay_buffer, neural_policy, traj_len, False)

    #     # X, y_list = replay_buffer.sample_for_dso()
    #     # replay_buffer.write_CSV_for_dso(args.env, X, y_list)
    #     # replay_buffer.write_pickle_for_EQL(args.env, X, y_list)
    #     # pdb.set_trace()

    #     # dso.train_dso(X, y_list)

    #     sac_loss_dict = train_sac(args, neural_policy, replay_buffer, True)

    #     # evaluate
    #     if step % args.eval_epochs == 0:
    #         # if step % 1 == 0:
    #         rs, states, us = do_eval(env, neural_policy, 1000, num_rollouts=1)
    #         print(rs.shape)
    #         current_reward = np.mean(rs)
    #         log_and_print("avg reward: {}".format(current_reward))

    #         # loss_list.append(current_reward)
    #         loss_list.append(np.sum(rs))
    #         # print("loss list", loss_list)
    #         plot_loss(loss_list, loss_file_name)
    #         # logs
    #         # logger.store_trajectory(rs, states, us, Vs, enf_rs, enf_states, enf_us, enf_Vs)
    #         logger.draw_traj(rs, states, us, step)

    #         if current_reward > best_reward:
    #             best_reward = current_reward
    #             best_epoch = step

    #     # logs
    #     logger.store_loss(sac_loss_dict)

    #     if step % args.log_epochs == 0:
    #         # if step % 1 == 0:
    #         # s_policy.save_checkpoint(args.env, args.store_path, suffix="")
    #         neural_policy.save_checkpoint(args.env, args.store_path, suffix=f"{step}")
    #         logger.store_log()

    #     log_and_print("use time: {}".format(time.time() - start_time))

    #     log_and_print("----------------------")

    # # log_and_print("total use time: {}".format(time.time() - total_start_time))

    # # print(f"best policy at epoch {best_epoch} with average reward {best_reward}")
    # # loss_list.append("adding linear controllers")
    # best_epoch = 262
    # best_policy = SAC(args, state_dim, action_space)
    # best_policy.load_checkpoint(
    #     f"{args.store_path}/sac_checkpoint_{args.env}_{best_epoch}.pth"
    #     # f"experiments/cartpole_our/debug/sac_checkpoint_cartpole_our_{best_epoch}.pth"
    # )
    # # best_policy.load_checkpoint(f"experiments/debug/sac_checkpoint_Pendulum-v0_0_second_phase_this_time.pth")

    # total_start_time = time.time()
    # for step in range(args.epochs, 100000):
    #     start_time = time.time()
    #     log_and_print("-------- current epoch: {} -----------".format(step))

    #     # learn linear model here
    #     learner = DynamicsLearner(
    #         learner_env, ProgPolicy(learner_env.spec, best_policy), prior_type=None
    #     )
    #     # learner = DynamicsLearner(learner_env, best_policy, prior_type=None)
    #     # low = np.array([0.25, -0.95])
    #     # high = np.array([0.35, -0.85])
    #     high = np.array([0.0, 0.0, 0.0, 0.0])
    #     low = np.array([0.0, 0.0, 0.0, 0.0])
    #     learner.run(low, high, 2, 1, "logs/")
    #     model = load_model("logs/")
    #     # _, theta, safe_sets = run("", model)
    #     _, theta, safe_sets = run_lagrange("", model, env.env_name, 1, best_policy)
    #     print("safe sets length", len(safe_sets))
    #     linear_models = []
    #     for i in range(0, len(theta), 3):
    #         linear_models.append([theta[i], theta[i + 1], theta[i + 2]])

    #     push_trajectory_with_safe_sets(
    #         env, buffer, best_policy, traj_len, linear_models, safe_sets, False
    #     )

    #     # evaluate
    #     if step % args.eval_epochs == 0:
    #         rs, states, us = do_eval_with_linear(
    #             env,
    #             best_policy,
    #             1000,
    #             num_rollouts=1,
    #             linear_models=linear_models,
    #             safe_sets=safe_sets,
    #         )

    #         current_reward = np.mean(rs)
    #         log_and_print("avg reward: {}".format(np.mean(current_reward)))

    #         # logs
    #         logger.draw_traj(rs, states, us, step)

    #         loss_list.append(np.sum(rs))
    #         print("loss list", loss_list)
    #         plot_loss(loss_list, loss_file_name)

    #         if current_reward > best_reward:
    #             best_reward = current_reward
    #             best_epoch = step

    #     # logs
    #     # logger.store_loss(cbf_loss_dict)
    #     sac_loss_dict = train_sac(args, best_policy, buffer, True)
    #     logger.store_loss(sac_loss_dict)

    #     if step % args.log_epochs == 0:
    #         # if step % 1 == 0:
    #         # s_policy.save_checkpoint(args.env, args.store_path, suffix="")
    #         best_policy.save_checkpoint(
    #             args.env, args.store_path, suffix=f"{step}_second_phase_this_time"
    #         )
    #         logger.store_log()

    #     log_and_print("use time: {}".format(time.time() - start_time))

    #     log_and_print("----------------------")

    # log_and_print("total use time: {}".format(time.time() - total_start_time))


def plot_loss(loss_list, fname):
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.savefig(fname)
    plt.close()


def visualize_safe_set(safe_sets):
    plt.cla()
    for box in safe_sets:
        x1_low, x1_high, x2_low, x2_high = box
        plt.plot(
            [x1_low, x1_high, x1_high, x1_low, x1_low],
            [x2_high, x2_high, x2_low, x2_low, x2_high],
        )
    plt.savefig("safe_sets.png")


if __name__ == "__main__":
    # train
    args = train_args()
    train(args)
