import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from stable_baselines3 import SAC

class ToraPolicy(nn.Module):
    def __init__(self, keras_model):
        super(ToraPolicy, self).__init__()

        self.dim_in = 4
        self.hidden = 100
        self.out = 1
        self.layer1 = nn.Linear(self.dim_in, self.hidden, bias=True)
        self.layer2 = nn.Linear(self.hidden, self.hidden, bias=True)
        self.layer3 = nn.Linear(self.hidden, self.hidden, bias=True)
        self.layer4 = nn.Linear(self.hidden, self.out, bias=True)

        # move weight from keras model to this model
        weights = keras_model.get_weights()
        self.layer1.weight.data = torch.from_numpy(np.transpose(weights[0]))
        self.layer1.bias.data = torch.from_numpy(weights[1])
        self.layer2.weight.data = torch.from_numpy(np.transpose(weights[2]))
        self.layer2.bias.data = torch.from_numpy(weights[3])
        self.layer3.weight.data = torch.from_numpy(np.transpose(weights[4]))
        self.layer3.bias.data = torch.from_numpy(weights[5])
        self.layer4.weight.data = torch.from_numpy(np.transpose(weights[6]))
        self.layer4.bias.data = torch.from_numpy(weights[7])

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x) - 10


class LinearProgNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(LinearProgNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.program = nn.Linear(self.obs_dim, self.act_dim, bias=True)

    def forward(self, x):
        # import pdb

        # pdb.set_trace()
        y = x.to(torch.float32)
        self.program.eval()
        return self.program(y)

    def discrete_forward(self, x):
        # import pdb

        # pdb.set_trace()
        return self.program(x)

    def interpret(self):
        code = (
            f"x * {self.program.weight.data.numpy()} + {self.program.bias.data.numpy()}"
        )
        return code


class ITELinearProgNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ITELinearProgNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # create a conditional predicate
        self.conditional = nn.Linear(self.obs_dim, 1, bias=True)
        self.thenp = nn.Linear(self.obs_dim, self.act_dim, bias=True)
        self.elsep = nn.Linear(self.obs_dim, self.act_dim, bias=True)
        self.beta = 1
        # self.program = nn.Linear(self.obs_dim, self.act_dim, bias=True)

    def forward(self, x):
        c = self.conditional(x)
        sc = torch.sigmoid(self.beta * c)
        return torch.add(
            torch.multiply(sc, self.thenp(x)),
            torch.multiply(torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep(x)),
        )
        # return self.program(x)

        # c = self.conditional (x)
        # sc = torch.sigmoid(self.beta * c)
        # c2 = self.conditional2 (x)
        # sc2 = torch.sigmoid(self.beta * c2)
        # return torch.add(torch.multiply(sc, self.thenp(x)),
        #         torch.add(
        #              torch.multiply(sc2, torch.multiply(torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep(x))),
        #              torch.multiply(torch.multiply(torch.add(sc2, -1.0), -1.0), torch.multiply(torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep2(x)))
        #              ))

    def discrete_forward(self, x):
        c = self.conditional(x)
        if c > 0:
            return self.thenp(x)
        else:
            return self.elsep(x)

    def interpret(self):
        code = f"if x * {self.conditional.weight.data.numpy()} + {self.conditional.bias.data.numpy()} > 0:\n"
        code += f"    then x * {self.thenp.weight.data.numpy()} + {self.thenp.bias.data.numpy()}\n"
        code += f"    else x * {self.elsep.weight.data.numpy()} + {self.elsep.bias.data.numpy()}"
        return code


class NestITELinearProgNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(NestITELinearProgNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # create a conditional predicate
        self.conditional = nn.Linear(self.obs_dim, 1, bias=True)
        self.conditional2 = nn.Linear(self.obs_dim, 1, bias=True)
        self.thenp = nn.Linear(self.obs_dim, self.act_dim, bias=True)
        self.elsep = nn.Linear(self.obs_dim, self.act_dim, bias=True)
        self.elsep2 = nn.Linear(self.obs_dim, self.act_dim, bias=True)
        self.beta = 1

    def forward(self, x):
        c = self.conditional(x)
        sc = torch.sigmoid(self.beta * c)
        c2 = self.conditional2(x)
        sc2 = torch.sigmoid(self.beta * c2)
        ac = torch.add(
            torch.multiply(sc, self.thenp(x)),
            torch.add(
                torch.multiply(
                    sc2,
                    torch.multiply(
                        torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep(x)
                    ),
                ),
                torch.multiply(
                    torch.multiply(torch.add(sc2, -1.0), -1.0),
                    torch.multiply(
                        torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep2(x)
                    ),
                ),
            ),
        )
        return ac

    def discrete_forward(self, x):
        c = self.conditional(x)
        if c > 0:
            return self.thenp(x)
        else:
            c2 = self.conditional2(x)
            if c2 > 0:
                return self.elsep(x)
            else:
                return self.elsep2(x)

    def interpret(self):
        code = f"if x * {self.conditional.weight.data.numpy()} + {self.conditional.bias.data.numpy()} > 0:\n"
        code += f"    then {self.thenp.weight.data.numpy()} + {self.thenp.bias.data.numpy()}\n"
        code += f"    elif x * {self.conditional2.weight.data.numpy()} + {self.conditional2.bias.data.numpy()} > 0:\n"
        code += f"        then {self.elsep.weight.data.numpy()} + {self.elsep.bias.data.numpy()}\n"
        code += f"        else {self.elsep2.weight.data.numpy()} + {self.elsep2.bias.data.numpy()}"
        return code


class Nest2ITELinearProgNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Nest2ITELinearProgNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # create a conditional predicate
        self.conditional = nn.Linear(self.obs_dim, 1, bias=True)
        self.conditional2 = nn.Linear(self.obs_dim, 1, bias=True)
        self.conditional3 = nn.Linear(self.obs_dim, 1, bias=True)
        self.thenp = nn.Linear(self.obs_dim, self.act_dim, bias=True)
        self.elsep = nn.Linear(self.obs_dim, self.act_dim, bias=True)
        self.elsep21 = nn.Linear(self.obs_dim, self.act_dim, bias=True)
        self.elsep22 = nn.Linear(self.obs_dim, self.act_dim, bias=True)
        self.beta = 1

    def forward(self, x):
        c = self.conditional(x)
        sc = torch.sigmoid(self.beta * c)
        c2 = self.conditional2(x)
        sc2 = torch.sigmoid(self.beta * c2)
        c3 = self.conditional3(x)
        sc3 = torch.sigmoid(self.beta * c3)
        ac = torch.add(
            torch.multiply(sc, self.thenp(x)),
            torch.add(
                torch.multiply(
                    sc2,
                    torch.multiply(
                        torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep(x)
                    ),
                ),
                torch.add(
                    torch.multiply(
                        sc3,
                        torch.multiply(
                            torch.multiply(torch.add(sc2, -1.0), -1.0),
                            torch.multiply(
                                torch.multiply(torch.add(sc, -1.0), -1.0),
                                self.elsep21(x),
                            ),
                        ),
                    ),
                    torch.multiply(
                        torch.multiply(torch.add(sc3, -1.0), -1.0),
                        torch.multiply(
                            torch.multiply(torch.add(sc2, -1.0), -1.0),
                            torch.multiply(
                                torch.multiply(torch.add(sc, -1.0), -1.0),
                                self.elsep22(x),
                            ),
                        ),
                    ),
                ),
            ),
        )
        return ac

    def discrete_forward(self, x):
        c = self.conditional(x)
        if c > 0:
            return self.thenp(x)
        else:
            c2 = self.conditional2(x)
            if c2 > 0:
                return self.elsep(x)
            else:
                c3 = self.conditional3(x)
                if c3 > 0:
                    return self.elsep21(x)
                else:
                    return self.elsep22(x)

    def interpret(self):
        code = f"if x * {self.conditional.weight.data.numpy()} + {self.conditional.bias.data.numpy()} > 0:\n"
        code += f"    then {self.thenp.weight.data.numpy()} + {self.thenp.bias.data.numpy()}\n"
        code += f"    elif x * {self.conditional2.weight.data.numpy()} + {self.conditional2.bias.data.numpy()} > 0:\n"
        code += f"        then {self.elsep.weight.data.numpy()} + {self.elsep.bias.data.numpy()}\n"
        code += f"        elif x * {self.conditional3.weight.data.numpy()} + {self.conditional3.bias.data.numpy()} > 0:\n"
        code += f"            then {self.elsep21.weight.data.numpy()} + {self.elsep21.bias.data.numpy()}"
        code += f"            else {self.elsep22.weight.data.numpy()} + {self.elsep22.bias.data.numpy()}"
        return code


# class BTreeITELinearProgNetwork(nn.Module):
#     def __init__(self, obs_dim, act_dim):
#         super(BTreeITELinearProgNetwork, self).__init__()
#
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#
#         #create a conditional predicate
#         self.conditional = nn.Linear(self.obs_dim, 1, bias=True)
#         self.conditional2 = nn.Linear(self.obs_dim, 1, bias=True)
#         self.conditional3 = nn.Linear(self.obs_dim, 1, bias=True)
#         self.thenp = nn.Linear(self.obs_dim, self.act_dim, bias=True)
#         self.thenp2 = nn.Linear(self.obs_dim, self.act_dim, bias=True)
#         self.elsep = nn.Linear(self.obs_dim, self.act_dim, bias=True)
#         self.elsep2 = nn.Linear(self.obs_dim, self.act_dim, bias=True)
#         self.beta = 1
#
#     def forward(self, x):
#         c = self.conditional (x)
#         sc = torch.sigmoid(self.beta * c)
#         c2 = self.conditional2 (x)
#         sc2 = torch.sigmoid(self.beta * c2)
#         c3 = self.conditional3 (x)
#         sc3 = torch.sigmoid(self.beta * c3)
#         ac = torch.add(#torch.multiply(sc, self.thenp(x))
#                 torch.add(
#                      torch.multiply(sc2, torch.multiply(sc, self.thenp(x))),
#                      torch.multiply(torch.multiply(torch.add(sc2, -1.0), -1.0), torch.multiply(sc, self.thenp2(x)))
#                      )
#                 ,
#                 torch.add(
#                      torch.multiply(sc3, torch.multiply(torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep(x))),
#                      torch.multiply(torch.multiply(torch.add(sc3, -1.0), -1.0), torch.multiply(torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep2(x)))
#                      ))
#         return ac
#
#     def discrete_forward(self, x):
#         c = self.conditional (x)
#         if c > 0:
#             c2 = self.conditional2 (x)
#             if c2 > 0:
#                 return self.thenp(x)
#             else:
#                 return self.thenp2(x)
#         else:
#             c3 = self.conditional3 (x)
#             if c3 > 0:
#                 return self.elsep(x)
#             else:
#                 return self.elsep2(x)
#
#     def interpret(self):
#         code  = f'if x * {self.conditional.weight.data.numpy()} + {self.conditional.bias.data.numpy()} > 0:\n'
#         code += f'    then {self.thenp.weight.data.numpy()} + {self.thenp.bias.data.numpy()}\n'
#         code += f'    elif x * {self.conditional2.weight.data.numpy()} + {self.conditional2.bias.data.numpy()} > 0:\n'
#         code += f'        then {self.elsep.weight.data.numpy()} + {self.elsep.bias.data.numpy()}\n'
#         code += f'        else {self.elsep2.weight.data.numpy()} + {self.elsep2.bias.data.numpy()}'
#         return code


class ITEConstantProgNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ITEConstantProgNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # create a conditional predicate
        self.conditional = nn.Linear(self.obs_dim, 1, bias=True)
        self.thenp = nn.Parameter(torch.zeros(1, self.act_dim))
        self.elsep = nn.Parameter(torch.zeros(1, self.act_dim))
        self.beta = 1
        # self.program = nn.Linear(self.obs_dim, self.act_dim, bias=True)

    def forward(self, x):
        c = self.conditional(x)
        sc = torch.sigmoid(self.beta * c)
        return torch.add(
            torch.multiply(sc, self.thenp),
            torch.multiply(torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep),
        )
        # return self.program(x)

    def discrete_forward(self, x):
        c = self.conditional(x)
        if c > 0:
            return self.thenp
        else:
            return self.elsep

    def interpret(self):
        code = f"if x * {self.conditional.weight.data.numpy()} + {self.conditional.bias.data.numpy()} > 0:\n"
        code += f"    then {self.thenp.data.numpy()}\n"
        code += f"    else {self.elsep.data.numpy()}"
        return code


class NestITEConstantProgNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(NestITEConstantProgNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # create a conditional predicate
        self.conditional = nn.Linear(self.obs_dim, 1, bias=True)
        self.conditional2 = nn.Linear(self.obs_dim, 1, bias=True)
        self.thenp = nn.Parameter(torch.zeros(1, self.act_dim))
        self.elsep = nn.Parameter(torch.zeros(1, self.act_dim))
        self.elsep2 = nn.Parameter(torch.zeros(1, self.act_dim))
        self.beta = 1
        # self.program = nn.Linear(self.obs_dim, self.act_dim, bias=True)

        # self.mu = np.zeros([self.obs_dim])
        # self.sigma_inv = np.ones([self.obs_dim])
        #
        # # Placeholders
        # # ------------------------
        # self.mu_var = Variable(torch.randn(self.obs_dim), requires_grad=False)
        # self.sigma_inv_var = Variable(torch.randn(self.obs_dim), requires_grad=False)

    def forward(self, x):
        # print (f'mu: {self.mu}')
        # print (f'sigma: {self.sigma_inv}')
        # print (f'raw x = {x}')
        # self.mu_var.data = torch.from_numpy(np.float32(self.mu.reshape(1, -1)))
        # self.sigma_inv_var.data = torch.from_numpy(np.float32(self.sigma_inv.reshape(1, -1)))
        # x = torch.multiply(torch.subtract(x, self.mu_var), self.sigma_inv_var)
        # print (f'norm x = {x}')
        # c = self.conditional (x)
        # sc = torch.sigmoid(self.beta * c)
        # ac = torch.add(torch.multiply(sc, self.thenp),
        #                  torch.multiply(torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep))
        # print (f'c {c}')
        # print (f'sc {sc}')
        # print (f'ac {ac}')
        # return ac
        # return self.program(x)
        c = self.conditional(x)
        sc = torch.sigmoid(self.beta * c)
        c2 = self.conditional2(x)
        sc2 = torch.sigmoid(self.beta * c2)
        ac = torch.add(
            torch.multiply(sc, self.thenp),
            torch.add(
                torch.multiply(
                    sc2,
                    torch.multiply(
                        torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep
                    ),
                ),
                torch.multiply(
                    torch.multiply(torch.add(sc2, -1.0), -1.0),
                    torch.multiply(
                        torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep2
                    ),
                ),
            ),
        )
        # print (f'sc {sc}')
        # print (f'sc2 {sc2}')
        # print (f'ac {ac}')
        # print (f'self.thenp {self.thenp}')
        # print (f'self.elsep {self.elsep}')
        # print (f'self.elsep2 {self.elsep2}')
        return ac

    def discrete_forward(self, x):
        c = self.conditional(x)
        if c > 0:
            return self.thenp
        else:
            c2 = self.conditional2(x)
            if c2 > 0:
                return self.elsep
            else:
                return self.elsep2

    def interpret(self):
        code = f"if x * {self.conditional.weight.data.numpy()} + {self.conditional.bias.data.numpy()} > 0:\n"
        code += f"    then {self.thenp.data.numpy()}\n"
        code += f"    elif x * {self.conditional2.weight.data.numpy()} + {self.conditional2.bias.data.numpy()} > 0:\n"
        code += f"        then {self.elsep.data.numpy()}\n"
        code += f"        else {self.elsep2.data.numpy()}"
        return code


class PDProgNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PDProgNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # PID parameters
        self.kp_alt = nn.Parameter(torch.zeros(1))
        self.kd_alt = nn.Parameter(torch.zeros(1))
        self.kp_ang = nn.Parameter(torch.zeros(1))
        self.kd_ang = nn.Parameter(torch.zeros(1))

        self.conditional = nn.Linear(self.obs_dim, 1, bias=True)
        self.elsep = nn.Parameter(torch.zeros(1, self.act_dim))
        self.beta = 1

    def forward(self, state):
        """calculates settings based on pid control"""
        c = self.conditional(state)
        sc = torch.sigmoid(self.beta * c)

        state0 = torch.index_select(state, 1, torch.tensor(0))
        state1 = torch.index_select(state, 1, torch.tensor(1))
        state2 = torch.index_select(state, 1, torch.tensor(2))
        state3 = torch.index_select(state, 1, torch.tensor(3))
        state4 = torch.index_select(state, 1, torch.tensor(4))
        state5 = torch.index_select(state, 1, torch.tensor(5))
        alt_tgt = torch.abs(state0)
        ang_tgt = (0.25 * np.pi) * (state0 + state2)

        # Calculate error values
        alt_error = alt_tgt - state1
        ang_error = ang_tgt - state4

        # Use PID to get adjustments
        alt_adj = self.kp_alt * alt_error + self.kd_alt * state3
        ang_adj = self.kp_ang * ang_error + self.kd_ang * state5

        a = torch.cat((alt_adj, ang_adj), 1)
        # a = torch.clip(a, min=-1., max=1.)
        # If the legs are on the ground we made it, kill engines
        # if(state[6] or state[7]):
        #     a[:] = 0
        return torch.add(
            torch.multiply(sc, a),
            torch.multiply(torch.multiply(torch.add(sc, -1.0), -1.0), self.elsep),
        )


class ProgPolicy:
    def __init__(self, env_spec, prog, min_log_std=-3, init_log_std=0, seed=None):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        self.model = prog  # FCNetwork(self.n, self.m, hidden_sizes=(), bias=bias)
        # make weights small
        # for param in list(self.model.parameters())[-2:]:  # only last layer
        #    param.data = 1e-2 * param.data
        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
        # self.trainable_params = list(
            # filter(lambda p: p.requires_grad, self.model.parameters())
        # ) + [self.log_std]
        self.trainable_params = [self.log_std]

        # Old Policy network
        # ------------------------
        # self.old_model = copy.deepcopy(prog) #FCNetwork(self.n, self.m, hidden_sizes=(), bias=bias)
        # self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
        # self.old_params = list(filter(lambda p: p.requires_grad, self.old_model.parameters())) + [self.old_log_std]
        # for idx, param in enumerate(self.old_params):
        # param.data = self.trainable_params[idx].data.clone()

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate(
            [p.contiguous().view(-1).data.numpy() for p in self.trainable_params]
        )
        return params.copy()

    def set_param_values(self, new_params, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            # print (f'curr_params {self.trainable_params}')
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx : current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.trainable_params[-1].data = torch.clamp(
                self.trainable_params[-1], self.min_log_std
            ).data
            # update log_std_val for sampling
            self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
            # print (f'new_params {self.trainable_params}')
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx : current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.old_params[-1].data = torch.clamp(
                self.old_params[-1], self.min_log_std
            ).data

    # Main functions
    # When phase_learning is True, sampler collects samples for learning.
    # ============================================
    def get_action(self, observation, discrete=False):
        # o = np.float64(observation.reshape(1, -1))
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        if discrete:
            mean = self.model.discrete_forward(self.obs_var).data.numpy().ravel()
        else:
            # import pdb
            # pdb.set_trace()
            if type(self.model) == SAC:
                mean, _ = self.model.predict(self.obs_var, deterministic=True)
                mean = mean[0]
            else:
                mean = self.model.model(self.obs_var).data.numpy().ravel()
        # np.exp(self.log_std_val) * np.random.randn(self.m)
        # mean = mean[0]
        action = mean  # + noise

        mean = torch.tensor(mean)
        action = torch.tensor(mean)
        # print(action, noise, self.log_std_val, np.exp(self.log_std_val))
        return [action, {"mean": mean, "log_std": self.log_std_val, "evaluation": mean}]

    def mean_LL(self, observations, actions, model=None, log_std=None):
        model = self.model if model is None else model
        log_std = self.log_std if log_std is None else log_std
        obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        mean = model(obs_var)
        zs = (act_var - mean) / torch.exp(log_std)
        LL = (
            -0.5 * torch.sum(zs ** 2, dim=1)
            + -torch.sum(log_std)
            + -0.5 * self.m * np.log(2 * np.pi)
        )
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.data.numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)
        return [LL, mean, self.old_log_std]

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)

    def param_size(self):
        return len(self.get_param_values())

    def evaluate(self, state, sign=None, direction=None, noise=None, step=None):
        state = torch.from_numpy(np.array([np.float32(state)]))
        if sign == "+":
            theta = self.get_param_values()
            theta1 = theta + noise * (direction)
            self.set_param_values(np.float32(theta1))
            with torch.no_grad():
                u = self.model(state)
            u = u[0].numpy()
            self.set_param_values(theta)
            return u
        elif sign == "-":
            theta = self.get_param_values()
            theta2 = theta - noise * (direction)
            self.set_param_values(np.float32(theta2))
            with torch.no_grad():
                u = self.model(state)
            u = u[0].numpy()
            self.set_param_values(theta)
            return u
        with torch.no_grad():
            u = self.model(state)
        u = u[0].numpy()
        return u

    def update(self, rollouts, reward_sigma, alpha, b):
        theta = self.get_param_values()
        update_step = np.zeros(theta.shape)
        for positive_reward, negative_reward, direction in rollouts[:b]:
            update_step += (positive_reward - negative_reward) * (direction)
        print(f"grad {1 / (b * reward_sigma) * update_step}")
        theta += alpha / (b * reward_sigma) * update_step
        self.set_param_values(np.float32(theta))
