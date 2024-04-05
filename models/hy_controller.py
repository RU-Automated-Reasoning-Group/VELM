import torch.nn as nn


class Hy_Net(nn.Module):
    def __init__(self, args, s_policy, o_policy):
        super(Hy_Net, self).__init__()

        self.s_policy = s_policy
        self.o_policy = o_policy
        self.safe_gap = args.cbf_safe_level * args.safe_tolerance
        self.device = self.s_policy.device

    def forward(self, state, evaluate, enforce_optimal=False):
        # calculate value from cbf
        V, _ = self.s_policy.compute_lyapunov(state)
        if V.item() < self.safe_gap or enforce_optimal:
            return self.o_policy.select_action(state, evaluate), V.detach().cpu()
        else:
            # pdb.set_trace()
            return (
                self.s_policy.compute_controls(state).detach().cpu().numpy()[0],
                V.detach().cpu(),
            )
