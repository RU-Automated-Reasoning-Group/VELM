import numpy as np
from tqdm import trange

from utils.logs import log_and_print


def train_sac(args, sac_net, buffer, print_loss=False):
    # init
    agent = sac_net
    memory = buffer
    updates = 0

    # from argument
    batch_size = args.sac_batch_size
    updates_per_step = args.sac_updates_per_step

    # Train SAC
    loss_store = {
        "sac_critic_1_loss": [],
        "sac_critic_2_loss": [],
        "sac_policy_loss": [],
        "sac_entropy_loss": [],
        "sac_alpha": [],
    }
    for i in trange(updates_per_step):
        # Update parameters of all the networks
        (
            critic_1_loss,
            critic_2_loss,
            policy_loss,
            ent_loss,
            alpha,
        ) = agent.update_parameters(memory, batch_size, updates)
        updates += 1

        # store
        loss_store["sac_critic_1_loss"].append(critic_1_loss)
        loss_store["sac_critic_2_loss"].append(critic_2_loss)
        loss_store["sac_policy_loss"].append(policy_loss)
        loss_store["sac_entropy_loss"].append(ent_loss)
        loss_store["sac_alpha"].append(alpha)

    if print_loss:
        log_and_print(
            "critic_1 loss {}".format(np.mean(loss_store["sac_critic_1_loss"]))
        )
        log_and_print(
            "critic_2 loss {}".format(np.mean(loss_store["sac_critic_2_loss"]))
        )
        log_and_print("policy loss {}".format(np.mean(loss_store["sac_policy_loss"])))
        log_and_print("entropy loss {}".format(np.mean(loss_store["sac_entropy_loss"])))
        log_and_print("alpha {}".format(np.mean(loss_store["sac_alpha"])))

    return loss_store
