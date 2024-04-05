import itertools
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .utils import *


class AbsEle(ABC):
    """The abstract element propagated throughout the layers.
    I tried to inherit Tensor so as to smoothly inject into existing PyTorch methods. But that triggers problem
    even when creating an instance. And inheriting FloatTensor doesn't even compile. So I'll stay with slower python
    re-implementations of all sorts of layers for now. FIXME maybe optimize using CPP extensions in the future.
    """

    @classmethod
    @abstractmethod
    def by_intvl(cls, lb: Tensor, ub: Tensor, *args, **kwargs) -> "AbsEle":
        """Transform the Lower Bounds and Upper Bounds to abstract elements propagated throughout the layers."""
        raise NotImplementedError()

    def __getitem__(self, key):
        """It may only need to compute some rows but not all in the abstract element. Select those rows from here."""
        raise NotImplementedError()

    @abstractmethod
    def size(self):
        """Return the size of any concretized data point from this abstract element."""
        raise NotImplementedError()

    @abstractmethod
    def dim(self):
        """Return the number of dimensions for any concretized data point from this abstract element."""
        raise NotImplementedError()

    @abstractmethod
    def lb(self) -> Tensor:
        """Lower Bound."""
        raise NotImplementedError()

    @abstractmethod
    def ub(self) -> Tensor:
        """Upper Bound."""
        raise NotImplementedError()

    def gamma(self) -> Tuple[Tensor, Tensor]:
        """Transform the abstract elements back into Lower Bounds and Upper Bounds."""
        lb = self.lb()
        ub = self.ub()
        assert valid_lb_ub(lb, ub)
        return lb, ub

    # ===== Below are pre-defined operations that every abstract element must support. =====

    @abstractmethod
    def view(self, *shape) -> "AbsEle":
        raise NotImplementedError()

    @abstractmethod
    def matmul(self, other: Tensor) -> "AbsEle":
        raise NotImplementedError()

    @abstractmethod
    def __add__(self, other) -> "AbsEle":
        raise NotImplementedError()

    def to_dense(self) -> "AbsEle":
        return self

    # ===== Below are pre-defined functions to compute distances of the abstract elements to certain property. =====

    # === For ACAS experiments ===

    def col_le_val(
        self, idx: int, threshold: float, mean: float = 0.0, range: float = 1.0
    ) -> Tensor:
        """Return a distance tensor for 'idx-th column value <= threshold'.
        @mean and @range are for de-normalization since it is about absolute value.
        """
        t = self.ub()[..., idx]
        threshold = (threshold - mean) / range
        d = t - threshold
        return F.relu(d)

    def col_ge_val(
        self, idx: int, threshold: float, mean: float = 0.0, range: float = 1.0
    ) -> Tensor:
        """Return a distance tensor for 'idx-th column value >= threshold'.
        @mean and @range are for de-normalization since it is about absolute value.
        """
        t = self.lb()[..., idx]
        threshold = (threshold - mean) / range
        d = threshold - t
        return F.relu(d)

    def cols_not_max(self, *idxs: int) -> Tensor:
        """Return a distance tensor for 'Forall idx-th column value is not maximal among all'.
        <Loss definition>: Intuitively, always-not-max => exists col . target < col is always true.
        Therefore, target_col.UB() - other_col.LB() should < 0, if not, that is the distance.
        As long as some of the others < 0, it's OK (i.e., min).
        """
        raise NotImplementedError()

    def cols_is_max(self, *idxs: int) -> Tensor:
        """Return a distance tensor for 'Exists idx-th column value is the maximal among all.
        <Loss definition>: Intuitively, some-is-max => exists target . target > all_others is always true.
        Therefore, other_col.UB() - target_col.LB() should < 0, if not, that is the distance.
        All of the others should be accounted (i.e., max).
        """
        raise NotImplementedError()

    def cols_not_min(self, *idxs: int) -> Tensor:
        """Return a distance tensor for 'Forall idx-th column value is not minimal among all'.
        <Loss definition>: Intuitively, always-not-min => exists col . col < target is always true.
        Therefore, other_col.UB() - target_col.LB() should < 0, if not, that is the distance.
        As long as some of the others < 0, it's OK (i.e., min).
        """
        raise NotImplementedError()

    def cols_is_min(self, *idxs: int) -> Tensor:
        """Return a distance tensor for 'Exists idx-th column value is the minimal among all.
        <Loss definition>: Intuitively, some-is-min => exists target . target < all_others is always true.
        Therefore, target_col.UB() - other_col.LB() should < 0, if not, that is the distance.
        All of the others should be accounted (i.e., max).
        """
        raise NotImplementedError()

    def worst_of_labels_predicted(self, labels: Tensor) -> Tensor:
        """Return the worst case output tensor for 'Forall batched input, their prediction should match the corresponding label'.
            <Loss definition>: Intuitively, this is specifying a label_is_max for every input abstraction.
        :param label: same number of batches as self
        """
        raise NotImplementedError()

    def worst_of_labels_not_predicted(self, labels: Tensor) -> Tensor:
        """Return the worst case output tensor for 'Forall batched input, none of their prediction matches the corresponding label'.
            <Loss definition>: Intuitively, this is specifying a label_not_max for every input abstraction.
        :param label: same number of batches as self
        """
        raise NotImplementedError()

    # ===== Finally, some utility functions shared by different domains. =====

    def _idxs_not(self, *idxs: int) -> List[int]:
        """Validate and get other column indices that are not specified."""
        col_size = self.size()[-1]
        assert len(idxs) > 0 and all([0 <= i < col_size for i in idxs])
        assert len(set(idxs)) == len(idxs)  # no duplications
        others = [i for i in range(col_size) if i not in idxs]
        assert len(others) > 0
        return others

    pass


class PtwiseEle(AbsEle):
    """'Ptwise' if the abstract domain is non-Relational in each field.
    As a consequence, their loss functions are purely based on LB/UB tensors.
    """

    def cols_not_max(self, *idxs: int) -> Tensor:
        # FIXME Not considering corner case when target == col?
        others = self._idxs_not(*idxs)
        others = self.lb()[..., others]

        res = []
        for i in idxs:
            target = self.ub()[..., [i]]
            diff = target - others  # will broadcast
            diff = F.relu(diff)
            mins, _ = torch.min(diff, dim=-1)
            res.append(mins)
        return sum(res)

    def cols_is_max(self, *idxs: int) -> Tensor:
        # FIXME Not considering corner case when target == col?
        others = self._idxs_not(*idxs)
        others = self.ub()[..., others]

        res = []
        for i in idxs:
            target = self.lb()[..., [i]]
            diffs = others - target  # will broadcast
            diffs = F.relu(diffs)
            res.append(diffs)

        if len(idxs) == 1:
            all_diffs = res[0]
        else:
            all_diffs = torch.stack(res, dim=-1)
            all_diffs, _ = torch.min(
                all_diffs, dim=-1
            )  # it's OK to have either one to be max, thus use torch.min()

        # then it needs to surpass everybody else, thus use torch.max() for maximum distance
        diffs, _ = torch.max(all_diffs, dim=-1)
        return diffs

    def cols_not_min(self, *idxs: int) -> Tensor:
        # FIXME Not considering corner case when target == col?
        others = self._idxs_not(*idxs)
        others = self.ub()[..., others]

        res = []
        for i in idxs:
            target = self.lb()[..., [i]]
            diffs = others - target  # will broadcast
            diffs = F.relu(diffs)
            mins, _ = torch.min(diffs, dim=-1)
            res.append(mins)
        return sum(res)

    def cols_is_min(self, *idxs: int) -> Tensor:
        # FIXME Not considering corner case when target == col?
        others = self._idxs_not(*idxs)
        others = self.lb()[..., others]

        res = []
        for i in idxs:
            target = self.ub()[..., [i]]
            diffs = target - others  # will broadcast
            diffs = F.relu(diffs)
            res.append(diffs)

        if len(idxs) == 1:
            all_diffs = res[0]
        else:
            all_diffs = torch.stack(res, dim=-1)
            all_diffs, _ = torch.min(
                all_diffs, dim=-1
            )  # it's OK to have either one to be min, thus use torch.min()

        # then it needs to surpass everybody else, thus use torch.max() for maximum distance
        diffs, _ = torch.max(all_diffs, dim=-1)
        return diffs

    def worst_of_labels_predicted(self, labels: Tensor) -> Tensor:
        full_lb = self.lb()
        full_ub = self.ub()
        res = []
        for i in range(len(labels)):
            cat = labels[i]
            piece_outs_lb = full_lb[[i]]
            piece_outs_ub = full_ub[[i]]

            # default lb-ub or ub-lb doesn't know that target domain has distance 0, so specify that explicitly
            lefts = piece_outs_ub[..., :cat]
            rights = piece_outs_ub[..., cat + 1 :]
            target = piece_outs_lb[..., [cat]]

            full = torch.cat((lefts, target, rights), dim=-1)
            diffs = full - target  # will broadcast
            # no need to ReLU here, negative values are also useful
            res.append(diffs)

        res = torch.cat(res, dim=0)
        return res

    def worst_of_labels_not_predicted(self, labels: Tensor) -> Tensor:
        full_lb = self.lb()
        full_ub = self.ub()
        res = []
        for i in range(len(labels)):
            cat = labels[i]
            piece_outs_lb = full_lb[[i]]
            piece_outs_ub = full_ub[[i]]

            # default lb-ub or ub-lb doesn't know that target domain has distance 0, so specify that explicitly
            lefts = piece_outs_lb[..., :cat]
            rights = piece_outs_lb[..., cat + 1 :]
            target = piece_outs_ub[..., [cat]]

            full = torch.cat((lefts, target, rights), dim=-1)
            diffs = target - full  # will broadcast
            # no need to ReLU here, negative values are also useful
            res.append(diffs)

        res = torch.cat(res, dim=0)
        raise NotImplementedError(
            "To use this as distance, it has to have target category not being max, "
            + "thus use torch.min(dim=-1) then ReLU()."
        )
        return res

    pass


# ==========
# Above are abstract elements definitions, below are NN modules that take these abstract elements as inputs.
# Note that activation layers are defined by each abstract domain in individual. It may have different approximations.
# ==========


class Linear(nn.Linear):
    """Linear layer with the ability to take approximations rather than concrete inputs."""

    def forward(
        self, e: Union[AbsEle, Tensor], *args, **kwargs
    ) -> Union[AbsEle, Tensor]:
        """I have to implement the forward computation by myself, because F.linear() may apply optimization using
        torch.addmm() which requires inputs to be tensors.
        """
        if not isinstance(e, AbsEle):
            return super().forward(e)

        output = e.matmul(self.weight.t())
        if self.bias is not None:
            output += self.bias
        return output

    pass


class Normalize(nn.Module):
    """Normalize following a fixed mean/variance.
    This class was originall written to comply with the BatchNorm trained networks in PLDI'19 experiments.
    However, it turned out the BatchNorm mean/variance collected from there is problematic. So we end up not using
    the extracted parameters, but directly call TF networks as oracles.
    """

    def __init__(self, beta, gamma, mean, variance, epsilon=1e-5):
        """
        :param epsilon: 1e-5 is the default value in tflearn implementation,
                        this value is used to avoid devide 0, and does not change in training.
        """
        assert (variance >= 0).all() and epsilon > 0
        super().__init__()
        self.beta = torch.from_numpy(beta)
        self.gamma = torch.from_numpy(gamma)
        self.mean = torch.from_numpy(mean)

        # somehow it needs to change variance first, otherwise it becomes ndarray again
        self.variance = variance + epsilon
        self.variance = torch.from_numpy(variance)
        return

    def forward(self, x: Union[AbsEle, Tensor]) -> Union[AbsEle, Tensor]:
        x_hat = (x - self.mean) / torch.sqrt(self.variance)
        return x_hat * self.gamma + self.beta

    pass


# ===== Below are the encoded safety properties to be considered. =====


class AbsProp(ABC):
    """All encoded properties should provide safe distance function and violation distance function.
    This distance function can be used to further compute losses in training and verification.
    It means: safe(violation) dist=0 means safe(violation) proved by over-approximation.
    Moreover, dist means how much until it becomes safe(violation).
    """

    def __init__(self, name: str):
        self.name = name
        return

    @abstractmethod
    def lbub(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def safe_dist(self, outs: AbsEle, *args, **kwargs):
        # TODO eventually replace this method with safe_worst
        raise NotImplementedError()

    def safe_worst(self, outs: AbsEle, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def viol_dist(self, outs: AbsEle, *args, **kwargs):
        # TODO eventually replace this method with safe_worst
        raise NotImplementedError()

    def viol_worst(self, outs: AbsEle, *args, **kwargs) -> Tensor:
        raise NotImplementedError()

    def safe_dist_conc(self, outs: Tensor, *args, **kwargs):
        raise NotImplementedError()

    def viol_dist_conc(self, outs: Tensor, *args, **kwargs):
        raise NotImplementedError()

    def tex(self) -> str:
        """
        :return: the property name in tex format
        """
        raise NotImplementedError()

    pass


class AndProp(AbsProp):
    """Conjunction of a collection of AbsProps."""

    def __init__(self, props: List[AbsProp]):
        assert len(props) > 0
        super().__init__("&".join([p.name for p in props]))

        self.props = props
        self.lb, self.ub, self.labels = self.join_all(props)
        return

    def tex(self) -> str:
        names = [p.tex() for p in self.props]
        unique_names = []
        for n in names:
            if n not in unique_names:
                unique_names.append(n)
        return " \\land ".join(unique_names)

    def join_all(self, props: List[AbsProp]):
        """
        Conjoin multiple properties altogether. Now that each property may have different input space and different
        safety / violation distance functions. This method will re-arrange and determine the boundaries of sub-regions
        and which properties they should satisfy.
        """
        nprops = len(props)
        assert nprops > 0

        # initialize for 1st prop
        orig_label = torch.eye(
            nprops
        ).byte()  # showing each input region which properties they should obey
        lbs, ubs = props[0].lbub()
        labels = orig_label[[0]].expand(len(lbs), nprops)

        for i, prop in enumerate(props):
            if i == 0:
                continue

            new_lbs, new_ubs = prop.lbub()
            assert valid_lb_ub(new_lbs, new_ubs)
            new_labels = orig_label[[i]].expand(len(new_lbs), nprops)

            lbs, ubs, labels = self._join(
                lbs, ubs, labels, new_lbs, new_ubs, new_labels
            )
        return lbs, ubs, labels

    @staticmethod
    def _join(
        x_lbs: Tensor,
        x_ubs: Tensor,
        x_labels: Tensor,
        y_lbs: Tensor,
        y_ubs: Tensor,
        y_labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Algorithm: Keep searching for "new" intersections, and refine by it, until none is found.
        We assume X and Y are mutually exclusive within themselves. Here shared_xxx keeps the intersected ones from
        X and Y. Therefore, shared_xxx won't intersect with anyone from X or Y anymore. Because x1 /\ y1 /\ x2 = empty.
        All arguments are assumed to be batched tensors.
        """
        shared_lbs, shared_ubs, shared_labels = (
            [],
            [],
            [],
        )  # intersected ones from X and Y

        def _covered(new_lb: Tensor, new_ub: Tensor, new_label: Tensor) -> bool:
            """
            Returns True if the new LB/UB is already covered by some intersected piece. Assuming new_lb/new_ub is
            from X or Y. So there won't be intersection, thus just check subset? is sufficient.
            Assuming all params are not-batched.
            """
            for i in range(len(shared_lbs)):
                shared_lb, shared_ub, shared_label = (
                    shared_lbs[i],
                    shared_ubs[i],
                    shared_labels[i],
                )
                if valid_lb_ub(shared_lb, new_lb) and valid_lb_ub(new_ub, shared_ub):
                    assert torch.equal(
                        new_label | shared_label, shared_label
                    ), "New intersected cube got more props?!"
                    return True
            return False

        while True:
            found_new_shared = False
            for i, j in itertools.product(range(len(x_lbs)), range(len(y_lbs))):
                xlb, xub, xlabel = x_lbs[i], x_ubs[i], x_labels[i]
                ylb, yub, ylabel = y_lbs[j], y_ubs[j], y_labels[j]
                try:
                    new_shared_lb, new_shared_ub = lbub_intersect(xlb, xub, ylb, yub)
                    new_shared_label = xlabel | ylabel
                except ValueError:
                    continue

                if _covered(new_shared_lb, new_shared_ub, new_shared_label):
                    # Has been found before.
                    # Possible when a sub-piece x11 from X (due to x1 intersects y1) is comparing with y1 again.
                    continue

                # save new intersected cube
                found_new_shared = True
                shared_lbs.append(new_shared_lb)
                shared_ubs.append(new_shared_ub)
                shared_labels.append(new_shared_label)

                # replace x by split non-intersected boxes in the X list
                rest_x_lbs, rest_x_ubs = lbub_exclude(
                    xlb, xub, new_shared_lb, new_shared_ub
                )
                rest_x_labels = xlabel.unsqueeze(dim=0).expand(
                    len(rest_x_lbs), *xlabel.size()
                )
                x_lbs = torch.cat((x_lbs[:i], rest_x_lbs, x_lbs[i + 1 :]), dim=0)
                x_ubs = torch.cat((x_ubs[:i], rest_x_ubs, x_ubs[i + 1 :]), dim=0)
                x_labels = torch.cat(
                    (x_labels[:i], rest_x_labels, x_labels[i + 1 :]), dim=0
                )

                # replace y by split non-intersected boxes in the Y list
                rest_y_lbs, rest_y_ubs = lbub_exclude(
                    ylb, yub, new_shared_lb, new_shared_ub
                )
                rest_y_labels = ylabel.unsqueeze(dim=0).expand(
                    len(rest_y_lbs), *ylabel.size()
                )
                y_lbs = torch.cat((y_lbs[:j], rest_y_lbs, y_lbs[j + 1 :]), dim=0)
                y_ubs = torch.cat((y_lbs[:j], rest_y_ubs, y_ubs[j + 1 :]), dim=0)
                y_labels = torch.cat(
                    (y_labels[:j], rest_y_labels, y_labels[j + 1 :]), dim=0
                )
                break

            if not found_new_shared:
                break

        shared_lbs = torch.stack(shared_lbs, dim=0) if len(shared_lbs) > 0 else Tensor()
        shared_ubs = torch.stack(shared_ubs, dim=0) if len(shared_ubs) > 0 else Tensor()
        shared_labels = (
            torch.stack(shared_labels, dim=0)
            if len(shared_labels) > 0
            else Tensor().byte()
        )

        all_lbs = torch.cat((shared_lbs, x_lbs, y_lbs), dim=0)
        all_ubs = torch.cat((shared_ubs, x_ubs, y_ubs), dim=0)
        all_labels = torch.cat((shared_labels, x_labels, y_labels), dim=0)
        return all_lbs, all_ubs, all_labels

    def lbub(self) -> Tuple[Tensor, Tensor]:
        """
        :return: Tensor on CPU, need to move to GPU if necessary.
        """
        return self.lb, self.ub

    def rule(self) -> Tensor:
        """Return the bit tensor corresponding to default LB/UB, showing which properties they should satisfy."""
        return self.labels

    def safe_dist(self, outs: AbsEle, rules: Tensor, *args, **kwargs):
        """\sum every prop's safe_dists
        :param rules: the bit-vectors corresponding to outputs, showing what rules they should obey
        """
        if len(self.props) == 1:
            assert torch.equal(rules, torch.ones_like(rules))
            dists = self.props[0].safe_dist(outs, *args, **kwargs)
            return dists

        res = []
        for i, prop in enumerate(self.props):
            bits = rules[..., i].nonzero().squeeze(dim=-1)
            if len(bits) == 0:
                # no one here needs to obey this property
                continue

            piece_outs = outs[bits]
            piece_dists = prop.safe_dist(piece_outs, *args, **kwargs)
            full_dists = torch.zeros(
                len(rules), *piece_dists.size()[1:], device=piece_dists.device
            )
            full_dists.scatter_(0, bits, piece_dists)
            res.append(full_dists)

        res = torch.stack(res, dim=-1)  # Batch x nprops
        return torch.sum(res, dim=-1)

    def viol_dist(self, outs: AbsEle, rules: Tensor, *args, **kwargs):
        """\min every prop's viol_dists
        :param rules: the bit-vectors corresponding to outputs, showing what rules they should obey
        """
        res = []
        for i, prop in enumerate(self.props):
            bits = rules[..., i].nonzero().squeeze(dim=-1)
            if len(bits) == 0:
                # no one here needs to obey this property
                continue

            piece_outs = outs[bits]
            piece_dists = prop.viol_dist(piece_outs, *args, **kwargs)
            full_dists = torch.full(
                (len(rules), *piece_dists.size()[1:]),
                float("inf"),
                device=piece_dists.device,
            )
            full_dists.scatter_(0, bits, piece_dists)
            res.append(full_dists)

        res = torch.stack(res, dim=-1)  # Batch x nprops
        mins, _ = torch.min(res, dim=-1)
        return mins

    pass


# ===== Below are test cases for basic assurance. =====


def _tc1():
    """Validate (manually..) that the AndProp is correct."""
    from acas import AcasNetID

    def _go(id):
        props = id.applicable_props()
        ap = AndProp(props)

        print("-- For network", id)
        for p in props:
            print("-- Has", p.name)
            lb, ub = p.lbub()
            print("   LB:", lb)
            print("   UB:", ub)

        lb, ub = ap.lbub()
        print("-- All conjoined,", ap.name)
        print("   LB:", lb)
        print("   UB:", ub)
        print("   Labels:", ap.labels)
        print("Cnt:", len(lb))
        for i in range(len(lb)):
            print(
                "  ",
                i,
                "th piece, width:",
                ub[i] - lb[i],
                f"area: {total_area(lb[[i]], ub[[i]]) :E}",
            )
        print()
        return

    """ <1, 1> is tricky, as it has many props;
        <1, 9> is special, as it is different from many others;
        Many others have prop1, prop2, prop3, prop4 would generate 3 pieces, in which prop1 and prop2 merged.
    """
    # _go(AcasNetID(1, 1))
    # _go(AcasNetID(1, 9))
    # exit(0)

    for id in AcasNetID.all_ids():
        _go(id)

    print("XL: Go manually check the outputs..")
    return


if __name__ == "__main__":
    _tc1()
    pass
