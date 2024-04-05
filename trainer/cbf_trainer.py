import torch
import torch.nn.functional as F
from tqdm import trange

from utils.logs import log_and_print

torch.set_default_dtype(torch.float64)


def lyapunov_loss(
    x,
    safe_mask,
    unsafe_mask,
    net,
    cbf_lambda,
    safe_level=1.0,
    timestep=0.001,
    print_loss=False,
):
    """
    Compute a loss to train the Lyapunov function

    args:
        x: the points at which to evaluate the loss
        x_goal: the origin
        safe_mask: the points in x marked safe
        unsafe_mask: the points in x marked unsafe
        net: a CLF_CBF_QP_Net instance
        cbf_lambda: the rate parameter in the CBF condition
        safe_level: defines the safe region as the sublevel set of the lyapunov function
        timestep: the timestep used to compute a finite-difference approximation of the
                  Lyapunov function
        print_loss: True to enable printing the values of component terms
    returns:
        loss: the loss for the given Lyapunov function
    """
    eps = 1e-2
    # Compute loss based on...
    loss = 0.0

    #   1.) term to encourage V <= safe_level in the safe region
    V_safe, _ = net.compute_lyapunov(x[safe_mask])
    safe_lyap_term = 100 * F.relu(eps + V_safe - safe_level)
    if safe_lyap_term.nelement() > 0:
        loss += safe_lyap_term.mean()

    #   2.) term to encourage V >= safe_level in the unsafe region
    V_unsafe, _ = net.compute_lyapunov(x[unsafe_mask])
    unsafe_lyap_term = 100 * F.relu(eps + safe_level - V_unsafe)
    if unsafe_lyap_term.nelement() > 0:
        loss += unsafe_lyap_term.mean()

    #   3.) A term to encourage satisfaction of CLF condition (TODO: delete during debug)
    u, V, lyap_descent_term_expected = net(x)
    # We compute the change in V in two ways: simulating x forward in time and check if V decreases
    # in each scenario, and using the expected decrease from Vdot
    lyap_descent_term_sim = 0.0
    # f, g = net.dynamics(x)
    # xdot = f + torch.bmm(g, u.unsqueeze(-1)).squeeze()
    # x_next = x + timestep * xdot
    if net.approx_step is not None:
        x_next = net.approx_step(x, u.unsqueeze(-1))
    else:
        f, g = net.dynamics(x)
        xdot = f + torch.bmm(g, u.unsqueeze(-1)).squeeze()
        x_next = x + timestep * xdot

    V_next, _ = net.compute_lyapunov(x_next)
    lyap_descent_term_sim += F.relu(
        eps + V_next - (1 - cbf_lambda * timestep) * V.squeeze()
    )
    loss += lyap_descent_term_sim.mean() + lyap_descent_term_expected.mean()
    # loss += lyap_descent_term_expected.mean()

    if print_loss:
        safe_pct_satisfied = (100.0 * (safe_lyap_term == 0)).mean().item()
        unsafe_pct_satisfied = (100.0 * (unsafe_lyap_term == 0)).mean().item()
        descent_pct_satisfied = (100.0 * (lyap_descent_term_sim == 0)).mean().item()
        log_and_print(
            f"           CLF safe region term: {safe_lyap_term.mean().item()}"
        )
        log_and_print(f"                  (% satisfied): {safe_pct_satisfied}")
        log_and_print(
            f"         CLF unsafe region term: {unsafe_lyap_term.mean().item()}"
        )
        log_and_print(f"                  (% satisfied): {unsafe_pct_satisfied}")
        log_and_print(
            f"               CLF descent term: {lyap_descent_term_sim.mean().item()}"
        )
        log_and_print(f"                  (% satisfied): {descent_pct_satisfied}")

    return loss


# TODO: consider whether needed
def controller_loss(x, net, print_loss=False, loss_coeff=1e-8):
    """
    Compute a loss to train the filtered controller

    args:
        x: the points at which to evaluate the loss
        net: a CLF_CBF_QP_Net instance
        print_loss: True to enable printing the values of component terms
    returns:
        loss: the loss for the given controller function
    """
    u_learned, _, _ = net(x)

    controller_squared_error = loss_coeff * u_learned.norm(dim=-1)
    loss = controller_squared_error.mean()

    if print_loss:
        log_and_print(
            f"                controller term: {controller_squared_error.mean().item()}"
        )

    return loss


def train_cbf(
    args,
    cbf_net,
    optimizer,
    x_train,
    safe_mask_train,
    unsafe_mask_train,
    x_test,
    safe_mask_test,
    unsafe_mask_test,
    epoch,
    print_loss=False,
):
    # put into device
    x_train = x_train.to(cbf_net.device)
    safe_mask_train = safe_mask_train.to(cbf_net.device)
    unsafe_mask_train = unsafe_mask_train.to(cbf_net.device)
    x_test = x_test.to(cbf_net.device)
    safe_mask_test = safe_mask_test.to(cbf_net.device)
    unsafe_mask_test = unsafe_mask_test.to(cbf_net.device)

    # First, sample training data uniformly from the state space
    N_train = x_train.shape[0]

    # Also get some testing data
    N_test = x_test.shape[0]

    # Define hyperparameters and define the learning rate and penalty schedule
    cbf_lambda = args.cbf_lambda
    safe_level = args.cbf_safe_level
    timestep = args.cbf_timestep
    learning_rate = args.cbf_learning_rate
    batch_size = args.cbf_batch_size
    init_controller_loss_coeff = args.cbf_init_controller_loss_coeff

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = learning_rate * (0.9 ** (epoch // 3))
        lr = learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = max(lr, 1e-4)

    # We penalize deviation from the nominal controller more heavily to start, then gradually relax
    def adjust_controller_penalty(epoch):
        # penalty = init_controller_loss_coeff * (0.1 ** (epoch // 1))
        penalty = init_controller_loss_coeff * (0.1 ** (epoch // 2))
        # return max(penalty, 1e-5)
        return max(penalty, 1e-3)

    # Initialize the optimizer
    # optimizer = optim.SGD(cbf_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train!
    loss_store = {
        "L_loss_train": [],
        "C_loss_train": [],
        "L_loss_test": [],
        "C_loss_test": [],
    }

    # Randomize presentation order
    permutation = torch.randperm(N_train)

    # Cool learning rate
    adjust_learning_rate(optimizer, epoch)
    # And reduce the reliance on the nominal controller loss
    controller_loss_coeff = adjust_controller_penalty(epoch)

    loss_acumulated = 0.0
    for i in trange(0, N_train, batch_size):
        # Get state from training data
        indices = permutation[i : i + batch_size]
        x = x_train[indices]
        safe_mask = safe_mask_train[indices]
        unsafe_mask = unsafe_mask_train[indices]

        # Zero parameter gradients before training
        optimizer.zero_grad()

        # Compute loss
        loss = 0.0
        l_loss = lyapunov_loss(
            x,
            safe_mask,
            unsafe_mask,
            cbf_net,
            cbf_lambda,
            safe_level,
            timestep,
            print_loss=print_loss,
        )
        c_loss = controller_loss(
            x, cbf_net, print_loss=print_loss, loss_coeff=controller_loss_coeff
        )

        loss += l_loss + c_loss

        # Accumulate loss from this epoch and do backprop
        loss.backward()
        loss_acumulated += loss.detach()

        # Update the parameters
        optimizer.step()

        # log_and_print('{}: {}'.format(i, loss))

        # store
        loss_store["L_loss_train"].append(l_loss.detach().cpu().item())
        loss_store["C_loss_train"].append(c_loss.detach().cpu().item())

    # Print progress on each epoch, then re-zero accumulated loss for the next epoch
    log_and_print(
        f"Epoch {epoch + 1} training loss: {loss_acumulated / (N_train / batch_size)}"
    )
    loss_acumulated = 0.0

    # Get loss on test set
    with torch.no_grad():
        # Compute loss
        loss = 0.0
        test_batch_size = 2 * batch_size
        for i in range(0, N_test, test_batch_size):
            l_loss = lyapunov_loss(
                x_test[i : i + test_batch_size],
                safe_mask_test[i : i + test_batch_size],
                unsafe_mask_test[i : i + test_batch_size],
                cbf_net,
                cbf_lambda,
                safe_level,
                timestep,
                print_loss=(i == 0),
            )
            c_loss = controller_loss(
                x_test[i : i + test_batch_size],
                cbf_net,
                print_loss=(i == 0),
                loss_coeff=controller_loss_coeff,
            )
            loss += l_loss + c_loss

            # store
            loss_store["L_loss_test"].append(l_loss.detach().cpu().item())
            loss_store["C_loss_test"].append(c_loss.detach().cpu().item())

        log_and_print(
            f"Epoch {epoch + 1}     test loss: {loss.item() / (N_test / test_batch_size)}"
        )

    return loss_store
