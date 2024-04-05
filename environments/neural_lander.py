import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

# Neural lander dynamics. Thanks to Dawei for the code!!
n_dims = 6
n_controls = 3

rho = 1.225
gravity = 9.81
drone_height = 0.09
mass = 1.47  # mass

Sim_duration = 1000


class StateIndex:
    "list of static state indices"

    PX = 0
    PY = 1
    PZ = 2

    VX = 3
    VY = 4
    VZ = 5


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(12, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 3)

    def forward(self, x):
        if not x.is_cuda:
            self.cpu()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def read_weight(filename):
    model_weight = torch.load(filename, map_location=torch.device("cpu"))
    model = Network().double()
    model.load_state_dict(model_weight)
    model = model.float()
    # .cuda()
    return model


num_dim_x = 6
num_dim_control = 3

Fa_model = read_weight("models/data/Fa_net_12_3_full_Lip16.pth")


def Fa_func(z, vx, vy, vz):
    if next(Fa_model.parameters()).device != z.device:
        Fa_model.to(z.device)
    bs = z.shape[0]
    # use prediction from NN as ground truth
    state = torch.zeros([bs, 1, 12]).type(z.type())
    state[:, 0, 0] = z + drone_height
    state[:, 0, 1] = vx  # velocity
    state[:, 0, 2] = vy  # velocity
    state[:, 0, 3] = vz  # velocity
    state[:, 0, 7] = 1.0
    state[:, 0, 8:12] = 6508.0 / 8000
    state = state.float()

    Fa = Fa_model(state).squeeze(1) * torch.tensor([30.0, 15.0, 10.0]).reshape(
        1, 3
    ).type(z.type())
    return Fa.type(torch.FloatTensor)


def Fa_func_np(x):
    z = torch.tensor(x[2]).float().view(1, -1)
    vx = torch.tensor(x[3]).float().view(1, -1)
    vy = torch.tensor(x[4]).float().view(1, -1)
    vz = torch.tensor(x[5]).float().view(1, -1)
    Fa = Fa_func(z, vx, vy, vz).cpu().detach().numpy()
    return Fa


def f_func(x, mass=mass):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    x, y, z, vx, vy, vz = [x[:, i] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x).type(x.type())
    f[:, 0] = vx
    f[:, 1] = vy
    f[:, 2] = vz

    Fa = Fa_func(z, vx, vy, vz)
    f[:, 3] = Fa[:, 0] / mass
    f[:, 4] = Fa[:, 1] / mass
    f[:, 5] = Fa[:, 2] / mass - gravity
    return f


def g_func(x, mass=mass):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 3, 0] = 1 / mass
    B[:, 4, 1] = 1 / mass
    B[:, 5, 2] = 1 / mass
    return B


def control_affine_dynamics(x, **kwargs):
    """
    Return the control-affine dynamics evaluated at the given state

    x = [[x, z, theta, vx, vz, theta_dot]_1, ...]
    """
    return f_func(x, **kwargs), g_func(x, **kwargs)


# Define linearized matrices for LQR control (assuming no residual force)
A = np.zeros((n_dims, n_dims))
A[:3, 3:] = np.eye(3)
B = np.zeros((n_dims, n_controls))
B[3:, :] = np.eye(n_controls) / mass
# Define cost matrices as identity
Q = np.eye(n_dims)
R = np.eye(n_controls)
# Get feedback matrix

# domain
domain = [
    (-5, 5),  # x
    (-5, 5),  # y
    (-0.5, 2),  # z
    (-1, 1),  # vx
    (-1, 1),  # vy
    (-1, 1),  # vz
]
domain_near_origin = [
    (-1.0, 1.0),  # x
    (-1.0, 1.0),  # y
    (-0.5, 1.0),  # z
    (-1.0, 1.0),  # vx
    (-1.0, 1.0),  # vy
    (-1.0, 1.0),  # vz
]

# TODO (no use for now)
class NeuLandarEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.size = n_dims

        self.observation_space = spaces.Box(
            low=np.array(domain)[:, 0],
            high=np.array(domain)[:, 1],
            shape=(n_dims,),
            dtype=float,
        )
        self.action_space = spaces.Box(
            low=-np.infty, high=np.infty, shape=(n_controls,), dtype=float
        )

    def reset(self, seed=None, options=None):
        pass

    def _get_obs(self):
        pass

    def step(self, action):
        pass


class NeuLanderSampler:
    def __init__(self, mass_use=mass, dt=0.001):
        # init
        self.mass = mass_use
        self.n_dims = num_dim_x
        self.n_control = num_dim_control
        self.safe_z = -0.05
        self.unsafe_z = -0.2
        self.safe_radius = 3
        self.unsafe_radius = 3.5
        self.dt = dt
        # domain
        self.domain = domain
        self.domain_near_origin = domain_near_origin

    def check_safe(self, state):
        if type(state) != torch.Tensor:
            state = torch.tensor(state)

        if (
            state[StateIndex.PZ] >= self.safe_z
            and state[: StateIndex.PZ + 1].norm(dim=-1) <= self.safe_radius
        ):
            return True
        else:
            return False

    def get_gt_dynamic(self, state):
        return f_func(state, mass=self.mass), g_func(state, mass=self.mass)

    def get_gt_appro_step(self, state, u):
        # calculate next state
        f, g = self.get_gt_dynamic(state)
        xdot = f + torch.bmm(g, u).squeeze()
        x_next = state + self.dt * xdot

        return x_next

    def sample_state(self, sample_num):
        state_per_num = sample_num // 11

        # sample through domain
        state = torch.Tensor(state_per_num, self.n_dims).uniform_(0.0, 1.0)
        for i in range(self.n_dims):
            min_val, max_val = self.domain[i]
            state[:, i] = state[:, i] * (max_val - min_val) + min_val
        state_near_origin = torch.Tensor(10 * state_per_num, self.n_dims).uniform_(
            0.0, 1.0
        )
        for i in range(self.n_dims):
            min_val, max_val = self.domain[i]
            state_near_origin[:, i] = (
                state_near_origin[:, i] * (max_val - min_val) + min_val
            )
        state = torch.vstack((state, state_near_origin))

        # state = torch.tensor(state).type(torch.float64)
        state = state.type(torch.float64)
        safe_mask = torch.logical_and(
            state[:, StateIndex.PZ] >= self.safe_z,
            state[:, : StateIndex.PZ + 1].norm(dim=-1) <= self.safe_radius,
        )
        unsafe_mask = torch.logical_or(
            state[:, StateIndex.PZ] <= self.unsafe_z,
            state[:, : StateIndex.PZ + 1].norm(dim=-1) >= self.unsafe_radius,
        )

        return state, safe_mask, unsafe_mask

    def sample_single_state(self):
        state = torch.Tensor(self.n_dims).uniform_(0.0, 1.0)
        for i in range(self.n_dims):
            min_val, max_val = self.domain[i]
            state[i] = state[i] * (max_val - min_val) + min_val

        return state.tolist()
