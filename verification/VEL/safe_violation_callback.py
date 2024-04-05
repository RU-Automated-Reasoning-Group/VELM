from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard.writer import SummaryWriter


class safety_violation_tracker:
    def __init__(self, logdir: str):
        self.writer = SummaryWriter(logdir)
        self.num_violation = 0
        self.real_timestpes = 0

    def add_step(self, safe: bool):
        if safe == False:
            self.num_violation += 1
        self.real_timestpes += 1
        self.writer.add_scalar(
            "safe_violation", self.num_violation, self.real_timestpes
        )

class SafeViolationCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, tracker: safety_violation_tracker, verbose=0):
        super(SafeViolationCallback, self).__init__(verbose)
        self.tracker = tracker

    def _on_step(self) -> bool:
        safe = not self.training_env.envs[0].unsafe()
        self.tracker.add_step(safe)
        return True
