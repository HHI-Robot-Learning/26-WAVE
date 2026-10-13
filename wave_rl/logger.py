import json
import os
import time
from pathlib import Path

import torch
import wandb


class Means:
    """Weighted means accumulated on the device. Adding never waits for the GPU; reading them
    out does, once for all keys."""

    def __init__(self, device) -> None:
        self.device = device
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, torch.Tensor] = {}

    def add(self, key: str, value, n=1.0) -> None:
        """Add `value` over `n` samples; a tensor value or weight is summed."""
        if key not in self._sums:
            self._sums[key] = torch.zeros((), device=self.device)
            self._counts[key] = torch.zeros((), device=self.device)
        self._sums[key] += value.detach().float().sum() if torch.is_tensor(value) else value
        self._counts[key] += n.detach().float().sum() if torch.is_tensor(n) else n

    def means(self) -> dict[str, float]:
        """Every key's mean since the last call, and a fresh start."""
        if not self._sums:
            return {}
        keys = list(self._sums)
        values = (torch.stack([self._sums[k] for k in keys])
                  / torch.stack([self._counts[k] for k in keys]).clamp(min=1)).tolist()
        self._sums, self._counts = {}, {}
        return dict(zip(keys, values))


class Logger(Means):
    """One run's metrics. Learners `add` to it; the harness `flush`es it once per block, which
    writes one record to ``data/runs/<run>_metrics.jsonl`` and, when the run has one, to wandb.
    Every flush also times the block since the previous one, so each learner reports the same
    `time/*` without measuring anything itself."""

    def __init__(self, run_name: str, device, wandb_cfg: dict | None) -> None:
        super().__init__(device)
        self.wandb = None if wandb_cfg is None else wandb.init(name=run_name, **wandb_cfg)
        if self.wandb is not None:
            self.wandb.define_metric("sweep/objects")
            self.wandb.define_metric("sweep/*", step_metric="sweep/objects")
        self.path = Path("data/runs") / f"{run_name}_metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self._started = self._last = time.perf_counter()
        self._last_step = 0

    def _record(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def config(self, config: dict) -> None:
        self._record({"config": config})
        if self.wandb is not None:
            self.wandb.config.update(config, allow_val_change=True)

    def flush(self, step: int, metrics: dict) -> dict:
        """The means since the last flush, `metrics` and the block's timing, logged at `step`."""
        record = {**self.means(), **{k: float(v) for k, v in metrics.items()}}
        now = time.perf_counter()
        record["time/block_seconds"] = now - self._last
        record["time/env_steps_per_second"] = (step - self._last_step) / (now - self._last)
        record["time/elapsed_hours"] = (now - self._started) / 3600.
        self._last, self._last_step = now, step
        self._record({"step": step, **record})
        if self.wandb is not None:
            self.wandb.log(record, step=step)
        return record

    def sweep(self, objects: int, metrics: dict) -> None:
        """One count of the final sweep, on the `sweep/objects` axis."""
        self._record({"sweep/objects": objects, **metrics})
        if self.wandb is not None:
            self.wandb.log({"sweep/objects": objects, **metrics})

    def video(self, key: str, path: str, step: int) -> None:
        """An encoded mp4. With wandb it is uploaded and deleted; without, it stays on disk."""
        if self.wandb is not None:
            self.wandb.log({key: wandb.Video(path, format="mp4")}, step=step)
            os.remove(path)

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
