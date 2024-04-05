import errno
import os
import signal
from math import floor
from os import path
from timeit import default_timer as timer
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, cuda
from torch.nn import functional as F


def pp_time(duration: float) -> str:
    """
    :param duration: in seconds
    """
    m = floor(duration / 60)
    s = duration - m * 60
    return "%dm %ds (%.3f seconds)" % (m, s, duration)


def time_since(since, existing=None):
    t = timer() - since
    if existing is not None:
        t += existing
    return pp_time(t)


def valid_lb_ub(
    lb: Union[float, Tensor], ub: Union[float, Tensor], eps: float = 1e-5
) -> bool:
    """To be valid:
        (1) Size ==
        (2) LB <= UB
    :param eps: added for numerical instability.
    """
    if isinstance(lb, float) and isinstance(ub, float):
        return lb <= ub + eps

    if lb.size() != ub.size():
        return False

    # '<=' will return a uint8 tensor of 1 or 0 for each element, it should have all 1s.
    return (lb <= ub + eps).all()


def cat0(*ts: Tensor) -> Tensor:
    """Usage: simplify `torch.cat((ts1, ts2), dim=0)` to `cat0(ts1, ts2)`."""
    return torch.cat(ts, dim=0)


def divide_pos_neg(ws: Tensor) -> Tuple[Tensor, Tensor]:
    """
    :return: positive part and negative part of the original tensor, 0 filled elsewhere.
    """
    pos_weights = F.relu(ws)
    neg_weights = F.relu(ws * -1) * -1
    return pos_weights, neg_weights


def batch_area(lb: Tensor, ub: Tensor) -> Tensor:
    """Return the total area constrained by LB/UB for each one in batch. Area = \Sum_{batch}{ \Prod{Element} }.
    :param lb: <Batch x ...>
    :param ub: <Batch x ...>
    """
    assert valid_lb_ub(lb, ub)
    diff = ub - lb

    # replace those zeros (degenerated dimensions) with 1, otherwise the product will have product 0
    zero_bits = diff == 0.0
    diff = torch.where(zero_bits, torch.ones_like(diff), diff)

    while diff.dim() > 1:
        diff = diff.prod(dim=-1)
    return diff


def total_area(
    lb: Tensor, ub: Tensor, eps: float = 1e-8, by_batch: bool = False
) -> float:
    """Return the total area constrained by LB/UB. Area = \Sum_{batch}{ \Prod{Element} }.
    :param lb: <Batch x ...>
    :param ub: <Batch x ...>
    :param by_batch: if True, return the areas of individual abstractions
    """
    assert valid_lb_ub(lb, ub)
    diff = ub - lb
    diff += eps  # some dimensions may be degenerated, then * 0 becomes 0.

    while diff.dim() > 1:
        diff = diff.prod(dim=-1)

    if by_batch:
        return diff
    else:
        return diff.sum().item()


# def total_area(lb: Tensor, ub: Tensor) -> float:
#     """ Return the total area constrained by LB/UB. Area = \Sum_{batch}{ \Prod{Element} }.
#     :param lb: <Batch x ...>
#     :param ub: <Batch x ...>
#     """
#     assert valid_lb_ub(lb, ub)
#     diff = ub - lb
#
#     # replace those zeros (degenerated dimensions) with 1, otherwise the product will have product 0
#     zero_bits = diff == 0.
#     diff = torch.where(zero_bits, torch.ones_like(diff), diff)
#
#     while diff.dim() > 1:
#         diff = diff.prod(dim=-1)
#     return diff.sum().item()


def bisect_by(
    lb: Tensor, ub: Tensor, idxs: Tensor, extra: Tensor = None
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Bisect specific columns.
    :param idxs: <Batch>, as the indices from torch.max()
    :param extra: if not None, it contains the bit vector for each LB/UB piece showing which prop they should obey
    :return: <New LB, New UB> if extra is None, otherwise <New LB, New UB, New Extra>
    """
    # scatter_() to convert indices into one-hot encoding
    split_idxs = idxs.unsqueeze(dim=-1)  # Batch x 1
    onehot_idxs = torch.zeros_like(lb).byte().scatter_(-1, split_idxs, 1)

    # then bisect the specified cols only
    mid = (lb + ub) / 2.0
    lefts_lb = lb
    lefts_ub = torch.where(onehot_idxs, mid, ub)
    rights_lb = torch.where(onehot_idxs, mid, lb)
    rights_ub = ub

    newlb = torch.cat((lefts_lb, rights_lb), dim=0)
    newub = torch.cat((lefts_ub, rights_ub), dim=0)
    if extra is None:
        return newlb, newub

    newextra = torch.cat((extra, extra), dim=0)
    return newlb, newub, newextra


def sample_regions(lb: Tensor, ub: Tensor, K: int, depth: int) -> Tuple[Tensor, Tensor]:
    """Uniformly sample K sub-regions with fixed width boundaries for each sub-region.
    :param lb: Lower bounds, batched
    :param ub: Upper bounds, batched
    :param K: how many pieces to sample
    :param depth: bisecting original region width @depth times for sampling
    """
    assert valid_lb_ub(lb, ub)
    assert K >= 1 and depth >= 1

    repeat_dims = [1] * (len(lb.size()) - 1)
    base = lb.repeat(
        K, *repeat_dims
    )  # repeat K times in the batch, preserving the rest dimensions
    orig_width = ub - lb

    try:
        piece_width = orig_width / (2 ** depth)
        # print('Piece width:', piece_width)
        avail_width = orig_width - piece_width
    except RuntimeError as e:
        print("Numerical error at depth", depth)
        raise e

    piece_width = piece_width.repeat(K, *repeat_dims)
    avail_width = avail_width.repeat(K, *repeat_dims)

    coefs = torch.rand_like(base)
    lefts = base + coefs * avail_width
    rights = lefts + piece_width
    return lefts, rights


def sample_points(lb: Tensor, ub: Tensor, K: int) -> Tensor:
    """Uniformly sample K points for each region.
    :param lb: Lower bounds, batched
    :param ub: Upper bounds, batched
    :param K: how many pieces to sample
    """
    assert valid_lb_ub(lb, ub)
    assert K >= 1

    K = max(1, int(K / lb.size()[0]))

    repeat_dims = [1] * (len(lb.size()) - 1)
    base = lb.repeat(
        K, *repeat_dims
    )  # repeat K times in the batch, preserving the rest dimensions
    width = (ub - lb).repeat(K, *repeat_dims)

    coefs = torch.rand_like(base)
    pts = base + coefs * width
    return pts


def datadir(exp: str) -> str:
    """Prepare a specific directory storing data for experiment @param exp."""
    exp = path.splitext(exp)[0]  # it may pass in exp_acas.py, drops extension
    dirpath = path.join("data", exp)

    if path.isdir(dirpath):
        return dirpath

    try:
        os.makedirs(dirpath)
    except OSError as e:
        # Guard against race condition
        if e.errno != errno.EEXIST:
            raise e
    return dirpath


def lbub_intersect(
    lb1: Tensor, ub1: Tensor, lb2: Tensor, ub2: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Return intersected [lb1, ub1] /\ [lb2, ub2], or raise ValueError when they do not overlap.
    :param lb1, ub1, lb2, ub2: not batched
    :return: not batched tensors
    """
    assert lb1.size() == lb2.size() and ub1.size() == ub2.size()

    res_lb, _ = torch.max(torch.stack((lb1, lb2), dim=-1), dim=-1)
    res_ub, _ = torch.min(torch.stack((ub1, ub2), dim=-1), dim=-1)

    if not valid_lb_ub(res_lb, res_ub):
        raise ValueError("Intersection failed.")
    return res_lb, res_ub


def lbub_exclude(
    lb1: Tensor,
    ub1: Tensor,
    lb2: Tensor,
    ub2: Tensor,
    accu_lb=Tensor(),
    accu_ub=Tensor(),
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """
    Return set excluded [lb1, ub1] (-) [lb2, ub2].
    Assuming [lb2, ub2] is in [lb1, ub1].
    :param lb1, ub1, lb2, ub2: not batched
    :param accu_lb: accumulated LBs, batched
    :param accu_ub: accumulated UBs, batched
    :param eps: error bound epsilon, only diff larger than this are considered different. This is to handle numerical
                issues while boundary comparison. With 1e-6 it may get 4 pieces for network <1, 9>, while with 1e-7,
                it may get 70 pieces..
    :return: batched tensors
    """
    for i in range(len(lb1)):
        left_aligned = (lb1[i] - lb2[i]).abs() < eps
        right_aligned = (ub2[i] - ub1[i]).abs() < eps

        if left_aligned and right_aligned:
            continue

        if not left_aligned:
            # left piece
            assert lb1[i] < lb2[i]
            left_lb = lb1.clone()
            left_ub = ub1.clone()
            left_ub[i] = lb2[i]
            accu_lb = torch.cat((accu_lb, left_lb.unsqueeze(dim=0)), dim=0)
            accu_ub = torch.cat((accu_ub, left_ub.unsqueeze(dim=0)), dim=0)

        if not right_aligned:
            # right piece
            assert ub2[i] < ub1[i]
            right_lb = lb1.clone()
            right_ub = ub1.clone()
            right_lb[i] = ub2[i]
            accu_lb = torch.cat((accu_lb, right_lb.unsqueeze(dim=0)), dim=0)
            accu_ub = torch.cat((accu_ub, right_ub.unsqueeze(dim=0)), dim=0)

        lb1[i] = lb2[i]
        ub1[i] = ub2[i]
        return lbub_exclude(lb1, ub1, lb2, ub2, accu_lb, accu_ub)
    return accu_lb, accu_ub


def join_box(
    lb: Tensor, ub: Tensor, other_lb: Tensor, other_ub: Tensor
) -> Optional[Tuple[Tensor, Tensor]]:
    """Conjoin two boxes, return None if they do not overlap."""
    assert lb.dim() == ub.dim() == other_lb.dim() == other_ub.dim() == 1

    res_lb = torch.stack((lb, other_lb), dim=-1)
    res_ub = torch.stack((ub, other_ub), dim=-1)
    res_lb = res_lb.max(dim=-1)[0]
    res_ub = res_ub.min(dim=-1)[0]

    if not valid_lb_ub(res_lb, res_ub):
        return None
    if (res_lb == res_ub).any():
        # some dimensions are degenerated, in this case, they are "contiguous" but do not "overlap"
        return None
    return res_lb, res_ub


def prune_box(
    lb: Tensor, ub: Tensor, other_lb: Tensor, other_ub: Tensor
) -> Tuple[List[Tensor], List[Tensor]]:
    """Prune the box1 against the other box2 such that resulting boxes do not overlap with box2."""
    assert lb.dim() == ub.dim() == other_lb.dim() == other_ub.dim() == 1

    joined = join_box(lb, ub, other_lb, other_ub)
    if joined is None:
        return [lb], [ub]

    joined_lb, joined_ub = joined
    res_lbs, res_ubs = [], []
    curr_lb, curr_ub = lb.clone(), ub.clone()
    for dim in range(lb.shape[0]):
        if curr_lb[dim] == joined_lb[dim] and curr_ub[dim] == joined_ub[dim]:
            # there is nothing to prune on this dimension
            continue

        if curr_lb[dim] != joined_lb[dim]:
            new_lb = curr_lb.clone()
            new_ub = curr_ub.clone()
            new_ub[dim] = joined_lb[dim]
            curr_lb[dim] = joined_lb[dim]
            res_lbs.append(new_lb)
            res_ubs.append(new_ub)

        if curr_ub[dim] != joined_ub[dim]:
            new_lb = curr_lb.clone()
            new_ub = curr_ub.clone()
            new_lb[dim] = joined_ub[dim]
            curr_ub[dim] = joined_ub[dim]
            res_lbs.append(new_lb)
            res_ubs.append(new_ub)

    # at the end, only the joined box should remain
    assert torch.equal(curr_lb, joined_lb) and torch.equal(curr_ub, joined_ub)
    return res_lbs, res_ubs


class timeout:
    """Raise error when timeout. Following that in <https://stackoverflow.com/a/22348885>.
    Usage:
        try:
            with timeout(sec=1):
                ...
        except TimeoutError:
            ...
    """

    def __init__(self, sec):
        self.seconds = sec
        self.error_message = "Timeout after %d seconds" % sec
        return

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
        return

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
        return

    pass


def avg_results(res: Tensor, num_nids: int, col_num: int = -1) -> Tensor:
    """
    Process to get the average numbers of some experiment results for each network id.
    :param res: Assuming of shape N x K, where N = num_ids * runs and K is the columns of numbers.
                For each row, the first column must be the AcasNetID's index, all the rest can be customized.
    :param num_nids: how many AcasNetIDs are in this table tensor.
    :param col_num: which specific data column to average;
                    if -1, all data columns will be averaged.
    :return: averaged results tensor containing the index column and all data columns specified.
    """
    avgs = []
    for i in range(num_nids):
        row_idxs = res[..., 0] == i  # picking only those data for network i
        relevant = res[row_idxs]
        if len(relevant) == 0:
            # there is not any data for this one, just return -1 for all columns
            nan = torch.tensor([-1.0]).expand(res.size()[-1])
            avgs.append(nan)
        else:
            if col_num != -1:
                # should still preserving the 0-th index column
                relevant = relevant[..., [0, col_num]]

            avg = torch.mean(relevant, dim=0)
            avgs.append(avg)

    avgs = torch.stack(avgs, dim=0)
    return avgs


def last_success_exp(res: Tensor, num_nids: int) -> Tuple[int, int]:
    """
    The experiment may abort due to various reasons, to avoid lengthy re-running,
    we can read the last successful experiment id and resume from there.
    :param res:
    :return: <K, i> where K is the K-th running of all networks, i is the i-th AcasNetID in experiment, both zero-based.
    """
    if len(res) == 0:
        return -1, -1

    col_nids = res[..., 0]
    for n in range(len(col_nids)):
        assert col_nids[n] == n % num_nids
    return len(col_nids) // num_nids, int(col_nids[-1])


def pp_cuda_mem(stamp: str):
    def sizeof_fmt(num, suffix="B"):
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, "Yi", suffix)

    print("-----", stamp, "-----")
    if cuda.is_available():
        print("Allocated:", sizeof_fmt(cuda.memory_allocated()))
        print("Max Allocated:", sizeof_fmt(cuda.max_memory_allocated()))
        print("Cached:", sizeof_fmt(cuda.memory_cached()))
        print("Max Cached:", sizeof_fmt(cuda.max_memory_cached()))
    else:
        print("CUDA not available.")
    print("----- End of", stamp, "-----")
    print()
    return


# ===== Below are test cases for basic assurance. =====


def _tc1(nrow=10, ncol=10, K=1000):
    """Validate that sampled regions are of the correct size and within range."""
    t1t2 = torch.stack((torch.randn(nrow, ncol), torch.randn(nrow, ncol)), dim=-1)
    lb, _ = torch.min(t1t2, dim=-1)
    ub, _ = torch.max(t1t2, dim=-1)
    outs_lb, outs_ub = sample_regions(lb, ub, K, depth=3)

    assert len(outs_lb) == len(outs_ub) == nrow * K
    width = torch.zeros_like(outs_lb)
    for i in range(nrow * K):
        row = i % nrow
        for j in range(ncol):
            assert lb[row][j] <= outs_lb[i][j] <= outs_ub[i][j] <= ub[row][j]
            width[i][j] = outs_ub[i][j] - outs_lb[i][j]

            # check the width as well, all sampled regions should have the same width on same dimension
            diff = width[i][j] - width[row][j]
            assert diff.abs() < 1e-6
    print("TC1 -- All passed.")
    return


def _tc2(nrow=10, ncol=10, K=1000):
    """Validate that sampled points are within range."""
    t1t2 = torch.stack((torch.randn(nrow, ncol), torch.randn(nrow, ncol)), dim=-1)
    lb, _ = torch.min(t1t2, dim=-1)
    ub, _ = torch.max(t1t2, dim=-1)
    outs = sample_points(lb, ub, K)

    assert len(outs) == nrow * K
    for i in range(nrow * K):
        row = i % nrow
        for j in range(ncol):
            assert lb[row][j] <= outs[i][j] <= ub[row][j]
    print("TC2 -- All passed.")
    return


def _tc3():
    """Validate that LB/UB intersection/exclusion ops are correct."""
    lb1, ub1 = torch.Tensor([1, 1]), torch.Tensor([4, 4])
    lb2, ub2 = torch.Tensor([2, 2]), torch.Tensor([3, 3])
    res_lb, res_ub = lbub_exclude(lb1, ub1, lb2, ub2)
    assert (
        len(res_lb) == len(res_ub) == 2 * len(lb1)
    )  # each dimension adds 2 pieces (left & right), no overlapping

    lb1, ub1 = torch.Tensor([1, 1]), torch.Tensor([4, 4])
    lb2, ub2 = torch.Tensor([2, 2]), torch.Tensor([3, 4])
    res_lb, res_ub = lbub_exclude(lb1, ub1, lb2, ub2)
    assert (
        len(res_lb) == len(res_ub) == 2 * (len(lb1) - 1) + 1
    )  # overlapped on one dimension

    lb1, ub1 = torch.Tensor([1, 1, 1]), torch.Tensor([4, 4, 4])
    lb2, ub2 = torch.Tensor([2, 2, 2]), torch.Tensor([3, 3, 3])
    res_lb, res_ub = lbub_exclude(lb1, ub1, lb2, ub2)
    assert (
        len(res_lb) == len(res_ub) == 2 * len(lb1)
    )  # each dimension adds 2 pieces (left & right), no overlapping

    print("TC3 -- All passed.")
    return


def _tc4(eps=1e-4):
    """Validate that the experiment results average number processing is correct."""
    res = torch.tensor(
        [
            [0.0000, 10.0000, 0.4956, 0.0000, 0.0000],
            [1.0000, 10.0000, 0.4930, 0.0000, 0.0000],
            [2.0000, 10.0000, 0.6930, 0.0000, 0.0000],
            [0.0000, 10.0000, 0.5038, 0.0000, 0.0000],
            [1.0000, 10.0000, 0.5114, 0.0000, 0.0000],
            [2.0000, 10.0000, 0.7029, 0.0000, 0.0000],
            [0.0000, 10.0000, 0.5434, 0.0000, 0.0000],
            [1.0000, 10.0000, 0.5600, 0.0000, 0.0000],
            [2.0000, 10.0000, 0.6346, 0.0000, 0.0000],
            [0.0000, 10.0000, 0.5136, 0.0000, 0.0000],
            [1.0000, 10.0000, 0.5708, 0.0000, 0.0000],
            [2.0000, 10.0000, 0.6786, 0.0000, 0.0000],
        ]
    )
    num_ids = 3

    full_avgs = avg_results(res, num_ids, -1)
    diff = full_avgs - torch.tensor(
        [
            [0.0000, 10.0000, 0.5141, 0.0000, 0.0000],
            [1.0000, 10.0000, 0.5338, 0.0000, 0.0000],
            [2.0000, 10.0000, 0.6773, 0.0000, 0.0000],
        ]
    )
    assert (diff.abs() < eps).all()

    col_avgs = avg_results(res, num_ids, 2)
    diff = col_avgs - torch.tensor(
        [[0.0000, 0.5141], [1.0000, 0.5338], [2.0000, 0.6773]]
    )
    assert (diff.abs() < eps).all()

    print("TC4 -- All passed.")
    return


def _tc5():
    """Validate that the parsing of last successful experiment is correct."""
    res = torch.tensor(
        [
            [0.0000, 100.0000, 0.5476, 1.7492, 0.0000],
            [1.0000, 100.0000, 0.4470, 4.7865, 1.0000],
            [2.0000, 100.0000, 0.5575, 1.6244, 0.0000],
            [3.0000, 100.0000, 0.9974, 3.1598, 1.0000],
            [4.0000, 100.0000, 0.6937, 1.7041, 0.0000],
            [5.0000, 100.0000, 0.4999, 5.6897, 0.0000],
            [6.0000, 100.0000, 0.6649, 1.6291, 0.0000],
            [7.0000, 100.0000, 0.7925, 2.0651, 0.0000],
            [8.0000, 100.0000, 0.6448, 1.5997, 0.0000],
            [0.0000, 100.0000, 0.5840, 1.8720, 0.0000],
            [1.0000, 100.0000, 0.4834, 1.6855, 0.0000],
            [2.0000, 100.0000, 0.7130, 1.6555, 0.0000],
            [3.0000, 100.0000, 0.9974, 3.1424, 1.0000],
            [4.0000, 100.0000, 0.5195, 2.7596, 0.0000],
            [5.0000, 100.0000, 0.6167, 4.8724, 0.0000],
            [6.0000, 100.0000, 0.6777, 70.9276, 0.0000],
            [7.0000, 100.0000, 0.8092, 1.6329, 0.0000],
            [8.0000, 100.0000, 0.4390, 1.5701, 0.0000],
            [0.0000, 100.0000, 0.4908, 1.6934, 0.0000],
            [1.0000, 100.0000, 0.2702, 1.6165, 0.0000],
            [2.0000, 100.0000, 0.7386, 1.7650, 0.0000],
            [3.0000, 100.0000, 0.9974, 3.1764, 1.0000],
            [4.0000, 100.0000, 0.6819, 1.6390, 0.0000],
        ]
    )
    assert last_success_exp(res, 9) == (2, 4)
    assert last_success_exp(torch.tensor([]), 9) == (-1, -1)
    print("TC5 -- All passed.")
    return


if __name__ == "__main__":
    # _tc1()
    # _tc2()
    # _tc3()
    # _tc4()
    _tc5()
    pass
