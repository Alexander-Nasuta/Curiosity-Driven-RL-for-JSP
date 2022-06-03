from stable_baselines3.common.callbacks import BaseCallback


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def _on_step(self) -> bool:
        return True

    def __init__(self, progress, task, total_steps):
        super(ProgressBarCallback, self).__init__()
        self.progress = progress
        self.task = task
        self.total_steps = total_steps

    def _on_rollout_end(self) -> None:
        n = self.num_timesteps
        self.progress.update(self.task, completed=n, description=f"[yellow] [{n:,} / {self.total_steps:,}]")

    def _on_training_start(self) -> None:
        self.progress.update(
            self.task,
            completed=0,
            description=""
        )

    def _on_training_end(self) -> None:
        self.progress.update(self.task, completed=self.total_steps, description="done")