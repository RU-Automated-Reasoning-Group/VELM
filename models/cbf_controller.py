import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def d_tanh_dx(tanh):
    return torch.diag_embed(1 - tanh ** 2)


class CBF_Net(nn.Module):
    """A neural network for simultaneously computing the Lyapunov function and the
    control input. The neural net makes the Lyapunov function, and the control input
    is computed by solving a QP.
    """

    def __init__(
        self,
        n_input,
        n_hidden,
        n_controls,
        cbf_lambda,
        control_affine_dynamics,
        device,
        action_space=None,
        approximate_dynamic_step=None,
    ):
        """
        Initialize the network

        args:
            n_input: number of states the system has
            n_hidden: number of hiddent layers to use
            n_controls: number of control outputs to use
            cbf_lambda: desired exponential convergence rate for the CLF
            control_affine_dynamics: a function that takes n_batch x n_dims and returns a tuple of:
                f_func: a function n_batch x n_dims -> n_batch x n_dims that returns the
                        state-dependent part of the control-affine dynamics
                g_func: a function n_batch x n_dims -> n_batch x n_dims x n_controls that returns
                        the input coefficient matrix for the control-affine dynamics
            scenarios: a list of dictionaries specifying the parameters to pass to f_func and g_func
        """
        super(CBF_Net, self).__init__()

        self.device = device

        # Save the dynamics and nominal controller functions
        self.dynamics = control_affine_dynamics
        self.approx_step = None
        if approximate_dynamic_step is not None:
            self.approx_step = approximate_dynamic_step
        # assert len(scenarios) > 0, "Must pass at least one scenario"
        # self.scenarios = scenarios

        # debug
        torch.manual_seed(0)

        # The network will have the following architecture
        #
        # n_input -> VFC1 (n_input x n_hidden) -> VFC2 (n_hidden, n_hidden)
        # -> VFC2 (n_hidden, n_hidden) -> V = x^T x --> QP -> u
        self.Vfc_layer_1 = nn.Linear(n_input, n_hidden).to(self.device)
        self.Vfc_layer_2 = nn.Linear(n_hidden, n_hidden).to(self.device)

        # We also train a controller to learn the nominal control input
        self.Ufc_layer_1 = nn.Linear(n_input, n_hidden).to(self.device)
        self.Ufc_layer_2 = nn.Linear(n_hidden, n_hidden).to(self.device)
        self.Ufc_layer_3 = nn.Linear(n_hidden, n_controls).to(self.device)

        self.n_controls = n_controls
        self.cbf_lambda = cbf_lambda

        # action rescaling
        if action_space is None:
            # self.action_scale = torch.tensor(1.).to(self.device)
            # self.action_bias = torch.tensor(0.).to(self.device)
            self.action_scale = None
            self.action_bias = None
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            ).to(self.device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            ).to(self.device)

    # freeze V net or unfreeze
    def grad_V_net(self, freeze=True):
        for layer in [self.Vfc_layer_1, self.Vfc_layer_2]:
            for param in layer.parameters():
                if freeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    # freeze U net
    def grad_U_net(self, freeze=True):
        for layer in [self.Ufc_layer_1, self.Ufc_layer_2, self.Ufc_layer_3]:
            for param in layer.parameters():
                if freeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def compute_controls(self, x):
        """
        Computes the control input (for use in the QP filter)

        args:
            x: the state at the current timestep [n_batch, n_dims]
        returns:
            u: the value of the barrier at each provided point x [n_batch, n_controls]
        """
        x = x.to(self.device)

        tanh = nn.Tanh()
        Ufc1_act = tanh(self.Ufc_layer_1(x))
        Ufc2_act = tanh(self.Ufc_layer_2(Ufc1_act))
        U = self.Ufc_layer_3(Ufc2_act)

        # scale action
        if self.action_bias is not None and self.action_scale is not None:
            U = tanh(U) * self.action_scale + self.action_bias

        return U

    def compute_lyapunov(self, x):
        """
        Computes the value and gradient of the Lyapunov function

        args:
            x: the state at the current timestep [n_batch, n_dims]
        returns:
            V: the value of the Lyapunov at each provided point x [n_batch, 1]
            grad_V: the gradient of V [n_batch, n_dims]
        """
        # Use the first two layers to compute the Lyapunov function
        x = x.to(self.device)

        tanh = nn.Tanh()

        Vfc1_act = tanh(self.Vfc_layer_1(x))
        Vfc2_act = tanh(self.Vfc_layer_2(Vfc1_act))
        # Compute the Lyapunov function as the square norm of the last layer activations
        V = 0.5 * (Vfc2_act * Vfc2_act).sum(1)

        # We also need to calculate the Lie derivative of V along f and g
        #
        # L_f V = \grad V * f
        # L_g V = \grad V * g
        #
        # Since V = tanh(w2 * tanh(w1*x + b1) + b1),
        # grad V = d_tanh_dx(V) * w2 * d_tanh_dx(tanh(w1*x + b1)) * w1

        # Jacobian of first layer wrt input (n_batch x n_hidden x n_input)
        DVfc1_act = torch.matmul(d_tanh_dx(Vfc1_act), self.Vfc_layer_1.weight)
        # Jacobian of second layer wrt input (n_batch x n_hidden x n_input)
        DVfc2_act = torch.bmm(
            torch.matmul(d_tanh_dx(Vfc2_act), self.Vfc_layer_2.weight), DVfc1_act
        )
        # Gradient of V wrt input (n_batch x 1 x n_input)
        grad_V = torch.bmm(Vfc2_act.unsqueeze(1), DVfc2_act)

        return V, grad_V

    def forward(self, x):
        """
        Compute the forward pass of the controller

        args:
            x: the state at the current timestep [n_batch, n_dims]
        returns:
            u: the input at the current state [n_batch, n_controls]
            r: the relaxation required to satisfy the CLF inequality
            V: the value of the Lyapunov function at a given point
            Vdot: the time derivative of the Lyapunov function plus self.cbf_lambda * V
        """
        x = x.to(self.device)

        # Compute the Lyapunov and barrier functions
        V, grad_V = self.compute_lyapunov(x)
        u_learned = self.compute_controls(x)

        # Compute lie derivatives for each scenario
        L_f_Vs = []
        L_g_Vs = []

        f, g = self.dynamics(x)

        # Lyapunov Lie derivatives
        L_f_Vs.append(torch.bmm(grad_V, f.unsqueeze(-1)).squeeze(-1))
        L_g_Vs.append(torch.bmm(grad_V, g).squeeze(1))

        # use approximate policy rather than QP
        u = u_learned

        # Accumulate across scenarios
        Vdot = F.relu(
            (
                L_f_Vs[0].unsqueeze(-1)
                + torch.bmm(L_g_Vs[0].unsqueeze(1), u.unsqueeze(-1))
            ).squeeze()
            + self.cbf_lambda * V
        )

        return u, V, Vdot

    # Save model parameters
    def save_checkpoint(self, env_name, store_path, suffix=""):
        ckpt_path = os.path.join(
            store_path, "cbf_checkpoint_{}_{}.pth".format(env_name, suffix)
        )
        print("Saving models to {}".format(ckpt_path))
        torch.save({"state_dict": self.state_dict()}, ckpt_path)
