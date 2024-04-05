import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# V network
class VNet(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(VNet, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# Q network
# define 2 Q network follow SAC
class QNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNet, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


# policy with Guassian policy
class GuassPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GuassPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state):
        # print(state)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        # pdb.set_trace()
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GuassPolicy, self).to(device)


# directly transform parameters of source to target
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# update parameters from source to target with ratio tau
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# SAC
class SAC(nn.Module):
    def __init__(self, args, num_inputs, action_space):
        super(SAC, self).__init__()

        # init
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.n_hidden = args.sac_n_hidden
        self.lr = args.sac_learning_rate

        self.device = torch.device("cuda" if args.cuda else "cpu")

        # network
        self.critic = QNet(num_inputs, action_space.shape[0], self.n_hidden).to(
            device=self.device
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNet(num_inputs, action_space.shape[0], self.n_hidden).to(
            self.device
        )
        hard_update(self.critic_target, self.critic)

        # Guassian Policy
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(
                torch.Tensor(action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

        self.policy = GuassPolicy(
            num_inputs, action_space.shape[0], self.n_hidden, action_space
        ).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

    def forward(self, state, evaluate=True):
        return self.select_action(state, evaluate)

    def select_action(self, state, evaluate=False):
        state = state.to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            mask_batch,
        ) = memory.sample(batch_size=batch_size)

        state_batch = state_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        mask_batch = mask_batch.to(self.device)

        with torch.no_grad():
            # pdb.set_trace()
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )

    # Save model parameters
    def save_checkpoint(self, env_name, store_path, suffix=""):
        ckpt_path = os.path.join(
            store_path, "sac_checkpoint_{}_{}.pth".format(env_name, suffix)
        )
        print("Saving models to {}".format(ckpt_path))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            ckpt_path,
        )

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

# adapted from SPICE code
class SAC_agent:
    def __init__(self, sac_policy, batch_size):
        self.agent = sac_policy
        self.updates = 0
        self.batch_size = batch_size
    
    def __call__(self, state, evaluate):
        return self.agent.forward(state, evaluate)
    
    def train(self, replay_buffer):
        ret = self.agent.update_parameters(replay_buffer, self.batch_size, self.updates)
        self.updates += 1
        return ret