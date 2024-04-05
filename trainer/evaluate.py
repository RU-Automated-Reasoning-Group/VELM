import numpy as np
import torch


def do_eval(env, policy, traj_len=100, num_rollouts=100):
    # init
    reward_list = []
    state_list = []
    action_list = []

    for _ in range(num_rollouts):
        # init state
        init_state = env.reset()
        state = init_state
        # print(state)

        # sample and store
        episode_reward_list = []
        episode_state_list = []
        episode_action_list = []
        for i in range(traj_len):
            # print(f"{i} in eval")
            u = policy(
                torch.tensor(state).type(torch.float32).unsqueeze(0), evaluate=True
            )
            next_state, r, done, _ = env.step(u)
            episode_reward_list.append(r)
            episode_state_list.append(state)
            episode_action_list.append(u)

            state = next_state

            if done:
                reward_list.append(episode_reward_list)
                state_list.append(episode_state_list)
                action_list.append(episode_action_list)
                break

    return np.array(reward_list), np.array(state_list), np.array(action_list)


def do_eval_with_linear(
    env, policy, traj_len=100, num_rollouts=100, linear_models=None, safe_sets=None
):
    # init
    reward_list = []
    state_list = []
    action_list = []

    total = 0
    original = 0
    for _ in range(num_rollouts):
        # sample and store
        index = 0
        # init state
        init_state = env.reset()
        state = init_state
        # print(state)

        episode_reward_list = []
        episode_state_list = []
        episode_action_list = []
        for _ in range(traj_len):
            u = policy(
                torch.tensor(state).type(torch.float64).unsqueeze(0), evaluate=True
            )

            next_state = env.simulate(u)
            if (
                next_state[0] > safe_sets[index][0]
                and next_state[0] < safe_sets[index][1]
                and next_state[1] > safe_sets[index][2]
                and next_state[1] < safe_sets[index][3]
            ):
                # take this step
                next_state, r, done, _ = env.step(u)
                # print("original step")
                original += 1
            else:
                # use the linear model
                model_idx = index // 10
                # print(model_idx)
                linear_model = linear_models[model_idx]
                next_state, r, done, _ = env.step(
                    np.array(
                        [
                            linear_model[0] * state[0]
                            + linear_model[1] * state[1]
                            + linear_model[2]
                        ]
                    )
                )
                if not (
                    next_state[0] > safe_sets[index][0]
                    and next_state[0] < safe_sets[index][1]
                    and next_state[1] > safe_sets[index][2]
                    and next_state[1] < safe_sets[index][3]
                ):
                    # print("linear step is not good")
                    pass
            episode_reward_list.append(r)
            episode_state_list.append(state)
            episode_action_list.append(u)
            state = next_state

            total += 1
            index += 1
            if done:
                reward_list.append(episode_reward_list)
                state_list.append(episode_state_list)
                action_list.append(episode_action_list)
                break

    print("intervention rate is", 1 - original / total)
    print("total step", total)

    return np.array(reward_list), np.array(state_list), np.array(action_list)
