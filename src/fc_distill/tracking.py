"""Experiment tracking with Weights & Biases, shared by every trainer and by `fc-eval --wandb`.

Logging is off unless a run asks for it (`enabled=True`), and then wandb is imported only inside
`start_run`, so tests, CI and offline runs never need an account or network. `WANDB_MODE=offline`
or `disabled` still applies to an enabled run: no `mode=` is passed to `wandb.init`.
Findings behind these choices: docs/notes/wandb.md.
"""

import hashlib
import os
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from fc_distill.data import DATASET_REVISION, DEFAULT_SPLITS_PATH, SYSTEM_PROMPT
from fc_distill.scoring import METRICS

if TYPE_CHECKING:
    import wandb

# The public project (README). WANDB_ENTITY / WANDB_PROJECT override them (e.g. a scratch project).
ENTITY = "joaomigueloliveira30"
PROJECT = "qwen-function-calling-distillation"
JobType = Literal["train", "eval"]
Role = Literal["student", "teacher"]
# The x-axis of #2's accuracy-per-GPU-hour curves: cumulative GPU-hours, the teacher's included.
GPU_HOURS = "compute/gpu_hours"
# Validation scores used for checkpoint selection, logged by trainers in the same row as GPU_HOURS.
VAL_METRICS = tuple(f"val/{m}" for m in METRICS)
STUDENT_GPU_HOURS = "compute/student_gpu_hours"
TEACHER_GPU_HOURS = "compute/teacher_gpu_hours"
# Logged next to the GPU-hours: running non-padding tokens per GPU-second, and peak reserved GiB.
TOKENS_PER_S = "compute/tokens_per_s"
PEAK_GPU_MEM_GB = "compute/peak_gpu_mem_gb"
RUN_ID_FILE = "wandb_run_id"


def git_state() -> dict[str, Any]:
    def git(*cmd: str) -> str:
        return subprocess.run(["git", *cmd], capture_output=True, text=True).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def data_provenance() -> dict[str, Any]:
    """The dataset, split and prompt versions a run used."""
    return {
        "xlam_revision": DATASET_REVISION,
        "splits_sha256": _sha256(DEFAULT_SPLITS_PATH.read_bytes()),
        "prompt_sha256": _sha256(SYSTEM_PROMPT.encode()),
    }


def hub_revision(model: str, revision: str | None = None) -> str | None:
    """The commit sha a Hub model id resolves to (None for a local path). Needs network."""
    from huggingface_hub import model_info

    return None if Path(model).exists() else model_info(model, revision=revision).sha


def train_provenance(
    *,
    base: str,
    base_revision: str | None,
    teacher: str | None = None,
    teacher_revision: str | None = None,
) -> dict[str, Any]:
    """What produced a training run: code commit and dirty flag, data and prompt versions, and the
    base and teacher models with the revisions `hub_revision` resolved them to."""
    return (
        git_state()
        | data_provenance()
        | {
            "base_model": base,
            "base_revision": base_revision,
            "teacher_model": teacher,
            "teacher_revision": teacher_revision,
        }
    )


class GpuClock:
    """GPU time of one training run, split into student and teacher compute.

    Time the GPU work with `measure`; add this run's share of an offline teacher job (generated
    SFT data, precomputed logits) with `add_teacher_hours`. Save `state()` with each checkpoint
    and restore it with `from_state` on resume, so the axis continues instead of restarting at 0.
    """

    def __init__(
        self,
        now: Callable[[], float] = time.perf_counter,
        student_seconds: float = 0.0,
        teacher_seconds: float = 0.0,
    ) -> None:
        self._now = now
        self._seconds: dict[Role, float] = {"student": student_seconds, "teacher": teacher_seconds}

    @contextmanager
    def measure(self, role: Role) -> Iterator[None]:
        start = self._now()
        try:
            yield
        finally:
            self._seconds[role] += self._now() - start

    def add_teacher_hours(self, hours: float) -> None:
        self._seconds["teacher"] += hours * 3600

    def metrics(self) -> dict[str, float]:
        student, teacher = self._seconds["student"] / 3600, self._seconds["teacher"] / 3600
        return {
            STUDENT_GPU_HOURS: student,
            TEACHER_GPU_HOURS: teacher,
            GPU_HOURS: student + teacher,
        }

    def state(self) -> dict[str, float]:
        return {f"{role}_seconds": s for role, s in self._seconds.items()}

    @classmethod
    def from_state(
        cls, state: Mapping[str, float], now: Callable[[], float] = time.perf_counter
    ) -> "GpuClock":
        return cls(now, state["student_seconds"], state["teacher_seconds"])


class Tracker(Protocol):
    def config(self, values: Mapping[str, Any]) -> None:
        """Merge fixed inputs (settings, data, provenance) into the run's config."""

    def log(self, metrics: Mapping[str, float]) -> None:
        """Append one history row."""

    def summary(self, values: Mapping[str, Any]) -> None:
        """Set one-shot results (an eval's scores) without adding a history row."""

    def table(self, key: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        """Log a table, e.g. per-slice scores."""

    def finish(self) -> None:
        """Close the run; nothing is logged after this."""


class NullTracker:
    """What a run gets when logging is off: every call does nothing."""

    def config(self, values: Mapping[str, Any]) -> None:
        pass

    def log(self, metrics: Mapping[str, float]) -> None:
        pass

    def summary(self, values: Mapping[str, Any]) -> None:
        pass

    def table(self, key: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbTracker:
    """A Tracker backed by one wandb run."""

    def __init__(self, run: "wandb.Run") -> None:
        self._run = run

    def config(self, values: Mapping[str, Any]) -> None:
        # A resumed run may legitimately change a value (e.g. the step count it trains to).
        # wandb.Config.update is the one call we use that ships without type hints.
        self._run.config.update(dict(values), allow_val_change=True)  # type: ignore[no-untyped-call]

    def log(self, metrics: Mapping[str, float]) -> None:
        self._run.log(dict(metrics))

    def summary(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            self._run.summary[key] = value

    def table(self, key: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        import wandb

        self._run.log({key: wandb.Table(columns=list(columns), data=[list(r) for r in rows])})

    def finish(self) -> None:
        self._run.finish()


def start_run(
    *,
    enabled: bool,
    job_type: JobType,
    group: str | None,
    run_dir: Path,
    name: str | None = None,
    tags: Sequence[str] = (),
) -> Tracker:
    """Open a tracked run, or a NullTracker when logging is off.

    The W&B run id is kept in `run_dir/wandb_run_id`, so starting again from the same directory
    (a resumed training run, a re-logged eval) continues that run instead of making a new one.
    W&B never reuses a deleted run's id, so deleting a run in the UI means deleting that file too.

    A trainer opens its run before `trainer.train()` with `report_to="wandb"`: the HF callback
    reuses the open run and adds its own arguments to the config.
    """
    if not enabled:
        return NullTracker()
    import wandb

    run_dir.mkdir(parents=True, exist_ok=True)
    id_file = run_dir / RUN_ID_FILE
    run_id = id_file.read_text().strip() if id_file.exists() else None
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", ENTITY),
        project=os.environ.get("WANDB_PROJECT", PROJECT),
        dir=run_dir,
        job_type=job_type,
        group=group,
        name=name,
        tags=list(tags),
        id=run_id,
        resume="allow" if run_id else None,
    )
    if not run.disabled:
        id_file.write_text(run.id)
    # By exact name, never by glob: the HF callback defines "*" against train/global_step, and
    # wandb picks among matching globs in random order (docs/notes/wandb.md section 6).
    run.define_metric(GPU_HOURS)
    for metric in VAL_METRICS:
        run.define_metric(metric, step_metric=GPU_HOURS, summary="max,last")
    return WandbTracker(run)
