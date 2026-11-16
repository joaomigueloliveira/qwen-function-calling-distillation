# Weights & Biases: what #23 needs and how to use it

Research notes for issue #23. Facts checked on 2026-11-06 against `wandb` 0.31.0, `transformers` 5.22.2
and TRL `main`. Each claim cites its source. Claims marked *(verified locally)* were also run on the
training machine (WSL2, RTX 3060) with `wandb==0.31.0` in a scratch environment, with no account
and no network calls.

**Where the docs live.** `docs.wandb.ai` now 301-redirects to `docs.coreweave.com`, which asks for
an access code. The docs' source repo [wandb/docs](https://github.com/wandb/docs) was archived on
2026-09-30, and its last commit (`86eefc3`) is the snapshot cited below as **D**. SDK source is cited
at tag `v0.31.0` as **W**.

- D = `https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/`
- W = `https://github.com/wandb/wandb/blob/v0.31.0/`
- HF = `https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/`
- TRL = `https://github.com/huggingface/trl/blob/cf1bfd464c3308466f9746b9fec17cbc94f97af9/trl/`

## 1. SDK version and recent changes

| Fact | Source |
|---|---|
| The latest release is **0.31.0** (2026-10-16). It requires Python >=3.10, and its classifiers cover 3.10 to 3.15 | [PyPI JSON](https://pypi.org/pypi/wandb/json) |
| **The default server is now CoreWeave Forge** (`https://api.forge.coreweave.com`). Run links point to `https://forge.coreweave.com/wandb`, and API keys stored for `api.wandb.ai` keep working | [W/CHANGELOG.md](https://github.com/wandb/wandb/blob/v0.31.0/CHANGELOG.md) 0.31.0 "Notable Changes"; `DEFAULT_BASE_URL` in W/wandb/sdk/lib/urls.py L7 |
| The package ships `py.typed` and a typed `wandb/__init__.pyi`, so `wandb.init`, `log`, `Run`, `Settings` and the rest are typed. A small script calling `init`, `define_metric`, `log`, `summary`, `Table`, `Artifact`, `log_artifact` and `finish` passes `mypy --strict` (mypy 2.4.1) with no override *(verified locally)* | wheel contents; W/wandb/\_\_init\_\_.pyi |
| Use `run.log` / `run.finish` on the object `wandb.init()` returns. The docs and docstrings now write every example as `with wandb.init() as run: run.log(...)`. `wandb.log` still exists, but it acts on the global `wandb.run` | W/wandb/\_\_init\_\_.pyi L206-L241 (signature), L242-L254 (context manager) |
| `reinit` takes `"default"`, `"return_previous"`, `"finish_previous"` and `"create_new"`. Boolean values are deprecated. With `"create_new"`, `wandb.run` and `wandb.log` are left unchanged, *"so some older integrations that rely on the global run will not work"*. That includes the HF callback (§8) | W/wandb/\_\_init\_\_.pyi init docstring (`reinit`); CHANGELOG 0.19.9 (bools deprecated), 0.19.10 (`create_new`) |
| Anonymous mode is deprecated and does nothing (`anonymous: DoNotSet`, which only warns) | CHANGELOG 0.23.1 "Deprecated"; W/wandb/sdk/wandb_settings.py L1099-L1108 |
| Recent removals: `define_metric(summary="best")` and its `goal` argument (use `"min"`/`"max"`), `finish(quiet=...)`, `run.get_url()` (use `run.url`), `wandb.InternalApi`/`PublicApi` (use `wandb.Api()`). `wandb.api` is deprecated | CHANGELOG 0.29.0 and 0.30.0 "Removed"/"Deprecated" |
| `GitPython` is no longer a dependency. Git metadata comes from running the `git` binary | CHANGELOG 0.28.1 "Removed" |
| Runtime dependencies include pydantic>=2.6, protobuf, opentelemetry-*, `cwevals` and others. It is CPU-only, and the wheel bundles the Go `wandb-core` binary (19 MB) | PyPI `requires_dist`; wheel size |
| `import wandb` takes about 0.16 s *(verified locally)* | — |

## 2. `wandb.init` options and modes

Signature (W/wandb/\_\_init\_\_.pyi L206-L241): `entity, project, dir, id, name, notes, tags,
config, config_exclude_keys, config_include_keys, allow_val_change, group, job_type, mode, force,
reinit, resume, resume_from, fork_from, save_code, tensorboard, sync_tensorboard, monitor_gym,
settings`. The docstring (same file, L256-L398) says:

- `project`: if it is unset, the SDK guesses one from the git root or the program, and falls back to
  `"uncategorized"`. Always pass it.
- `dir`: where run files go, `./wandb` by default. It doesn't control where artifact downloads go.
  The repo's `.gitignore` already ignores `wandb/`.
- `config`: a dict, `argparse.Namespace` or YAML path. Keys must not contain `.`, values should
  stay under 10 MB, and nested dicts are flattened with dots ([D/models/track/config.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/config.mdx),
  "If you pass a nested dictionary"; [D/models/track/limits.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/limits.mdx) "Config size").
  Setting a config value a second time raises an error unless `allow_val_change=True`.
- `group` / `job_type`: for grouping runs, e.g. `job_type` `"train"` / `"eval"` inside a group
  ([D/models/runs/grouping.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/runs/grouping.mdx)).
- `tags`: when resuming, the tags passed replace the existing ones.
- `mode`: `"online"` (default), `"offline"` (saved locally, synced later with `wandb sync`),
  `"disabled"` (*"Disables all W&B functionality, making the run's methods no-ops. Typically used
  in testing"*), or `"shared"` (experimental).
- `force`: if it is False (the default) and nobody is logged in, the run *switches to offline mode*
  instead of failing.

**What `mode="disabled"` / `WANDB_MODE=disabled` does** (W/wandb/sdk/wandb_init.py L748-L775 and
L1515-L1516; W/wandb/sdk/lib/noop_run.py):

- `init` returns a `NoopRun`, a subclass of `wandb.Run`, and sets it as the global `wandb.run`.
  `project` defaults to `"dummy"`, `name` to `dummy-<id>` and `url` to `None`. No login and no
  `wandb-core` process.
- `log(d)` only does `summary.update(d)`. `define_metric` returns a dummy and does nothing, so
  summary aggregation (`"max"`) **does not run in disabled mode**. `config` behaves like a real
  config. `use_artifact`, `link_artifact`, `watch`, `alert` and `restore` return `None`.
  `log_artifact` returns an object that accepts any call.
- *(verified locally)*: disabled mode wrote no files to the working directory or to `$HOME`.
  `run.summary` and `run.config` could be read back as dicts.
- **An explicit `mode=` argument overrides `WANDB_MODE`.** `wandb_init.py` L298-L302 copies the
  global settings (env included) and then applies the `init()` arguments. *(verified locally)*:
  `WANDB_MODE=disabled` with `wandb.init(mode="offline")` gave a real `Run`. So a helper that hard-codes
  `mode="online"` can't be switched off by the environment.
- [D/models/track/environment-variables.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/environment-variables.mdx):
  *"If you set this to `disabled` wandb will turn off completely."* The same page lists
  `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_DIR`, `WANDB_RUN_GROUP`, `WANDB_JOB_TYPE`, `WANDB_TAGS`,
  `WANDB_RUN_ID`, `WANDB_RESUME`, `WANDB_SILENT` and `WANDB_DISABLE_GIT`.

**Offline mode** *(verified locally)* writes `wandb/offline-run-<ts>-<id>/` (a `.wandb` record file,
`files/requirements.txt` and `logs/`) under `dir`, plus `~/.cache/wandb/logs/*.log`. It starts
`wandb-core` and the system monitor, but makes no network calls. `wandb leet inspect --summary
--json <file>.wandb` (new in 0.31.0, per the CHANGELOG) reads the summary, config and git record
back without a server. That is useful for one integration test.

**Testing.** Neither the docs nor the SDK name a preferred testing pattern beyond the "typically
used in testing" note on `disabled`. The SDK's own `tests/` use a mocked backend that isn't part of
the public API (unverified whether it is meant for users). For this repo:

- Unit tests should go through the helper's own seam: a fake tracker, or a `Tracker` protocol with
  a null implementation. Then most tests don't need `wandb` installed and never start a process.
- Use `mode="disabled"` only to check that the real adapter calls the API without crashing. It
  can't check `define_metric` or summary aggregation, because those are no-ops in that mode.
- If one test should check what a real run records (summary `max`, config), run it in
  `mode="offline"` with `dir=tmp_path` and read the result back with `wandb leet inspect --json`. It
  still needs no account or network, but it starts `wandb-core`, which makes it slower.

## 3. Git commit and dirty flag

- **What wandb records automatically:** the git `commit` and `remote_url` of the repo at `root_dir`
  (`git_root`), in the run metadata (W/wandb/sdk/wandb_run.py L1579-L1595, `_populate_git_info`).
  *(verified locally)*: an offline run in a repo with an uncommitted change recorded
  `"git":{"commit":"995e…","remote_url":"https://github.com/example/x.git"}`.
- **No dirty flag is recorded.** `GitRepo.dirty` exists in W/wandb/sdk/lib/gitlib.py L257, but
  nothing in the SDK reads it. The metadata record has only `commit` and `remote_url` *(verified
  locally)*.
- **`diff.patch` is saved only when `save_code` is on.** wandb-core calls `handlePatchSave()` only
  under `if h.settings.IsSaveCode()` ([core/internal/stream/handler.go L506-L512, L612-L660 at
  v0.31.0](https://github.com/wandb/wandb/blob/v0.31.0/core/internal/stream/handler.go#L506-L512)).
  It then writes `diff.patch` (against `HEAD`) and `diff_<upstream>.patch`. `save_code` is off by
  default (init docstring). `disable_git` / `WANDB_DISABLE_GIT` turn all of this off
  (W/wandb/sdk/wandb_settings.py L263). The env-vars page says `WANDB_DISABLE_CODE` keeps the
  commit but drops diffs.
- **Recommendation:** pass the repo's own `git_state()` (`commit` + `dirty`, as `fc_distill.evaluate`
  already computes it) in `config["provenance"]`. Then the dirty flag sits in a field we control, and
  runs can be filtered by it. Leave wandb's own git capture on: it costs nothing and adds the
  remote URL. `fc-eval` already refuses to start on a dirty tree without `--allow-dirty`, and the same
  rule should apply to training.

## 4. `run.summary` vs `run.log`; Tables; how eval runs sit next to training runs

- `run.log(d)` appends a history row at a new step and sets each key's summary to its last value,
  unless `define_metric(summary=...)` says otherwise. `run.summary[k] = v` sets a single value
  without adding a history row ([D/models/track/log/log-summary.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/log/log-summary.mdx), intro).
  **One-shot eval numbers belong in `run.summary`.** A one-row history makes empty line charts, and
  summary values are what the runs table and scatter or bar panels compare across runs.
- Nested keys are flattened to dotted names, and every distinct key adds to the project's "metric
  cardinality", the documented main cause of slow workspaces
  ([D/models/track/limits.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/limits.mdx) "Metric cardinality").
  xLAM's slices (3 call patterns + 3 distractor buckets + 2 argument sources, × 5 metrics × CIs)
  come to about 100 keys per split, which is fine. Don't log per-row data as keys.
- **Per-slice results** fit a `wandb.Table(columns=[...], data=[...])` or `wandb.Table(dataframe=df)`,
  up to 200,000 rows. Logging the table again under the same key makes a new version, and tables
  under the same key can be compared across runs
  ([D/models/track/log/log-tables.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/log/log-tables.mdx) "Create tables", "Save tables" L135).
  Tables are logged with `run.log({"slices": table})`. Suggestion: put the headline numbers and
  each slice's `exact` in the summary as scalars (for comparing runs), and the full slice block as
  one Table (for reading).
- **Placement:** use the same `project` for everything, `job_type="train"` or `"eval"`, and a
  `group` per method (e.g. `sft-teacher-outputs`), so the runs table can group by method and filter by job
  type ([D/models/runs/grouping.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/runs/grouping.mdx) "Organize runs by job type").
  Baselines use `group="baseline"`. Tags (`split:test`, `split:bfcl`) make filtering easy.

## 5. Artifacts: linking eval runs to checkpoints

- `run.log_artifact(artifact, aliases=...)` records an artifact as a run's output, and
  `run.use_artifact("name:alias")` records it as an input. W&B draws the lineage graph from these
  (W/wandb/sdk/wandb_run.py L3055-L3090 `use_artifact`, L3179 `log_artifact`).
- `Artifact.add_reference(uri, checksum=True)`: *"Unlike files or directories that you add to an
  artifact, references are not uploaded to W&B."* With `file://`, the checksum and size come from the
  file system (W/wandb/sdk/artifacts/artifact.py L1662-L1700;
  [D/models/artifacts/track-external-files.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/artifacts/track-external-files.mdx)).
- Options for the LoRA adapter, which is a few tens of MB:
  1. **No artifact.** Put the checkpoint path and its `adapter_model.safetensors` sha256 in the
     eval run's config, together with the training run's id. This is the simplest option and enough
     to link the two runs by hand.
  2. **Reference artifact** (`file://…/runs/<run>/checkpoint-N`). It gives the lineage graph and a
     checksum but uploads nothing. The link only resolves on this machine, which is fine for a
     single-GPU project.
  3. **Uploaded artifact.** It makes adapters downloadable from the public project, but uses storage.
     Free-tier storage limits under Forge are **unverified**.
- The HF callback can upload checkpoints itself through `WANDB_LOG_MODEL=checkpoint|end`. The default
  is `"false"` (HF/integrations/integration_utils.py L709, L902-L925). Leave it off and decide on
  purpose (open question below).
- In disabled mode `use_artifact` returns `None` (§2), so code must not call methods on its result
  without a check.

## 6. `define_metric`: custom x-axis and summary aggregation

- `run.define_metric(name, step_metric=None, step_sync=None, hidden=None, summary=None,
  overwrite=None)`. `summary` takes a comma-separated list of `min,max,mean,last,first,none`
  (`copy` is deprecated). Globs are allowed only as a suffix (`"train/*"`)
  (W/wandb/sdk/wandb_run.py L2818-L2908).
- To use a custom x-axis, log the axis value in the same `run.log` call as the metric
  ([D/models/track/log/customize-logging-axes.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/log/customize-logging-axes.mdx)).
  With `step_sync=True` (the default when `step_metric` is set), the SDK fills in the axis's last
  value if it is left out.
- *(verified locally)*: with `summary="max"` the summary holds **only** `{"val/exact": {"max":
  0.7}}`, and the last value is dropped. Use `summary="max,last"` to keep both.
- **Glob precedence:** wandb-core looks up an exact definition first. If there is none, it takes
  the *first matching glob in Go map iteration order*, which is random
  ([core/internal/runmetric/runmetric.go L39-L50, L206-L219 at v0.31.0](https://github.com/wandb/wandb/blob/v0.31.0/core/internal/runmetric/runmetric.go#L39-L50)).
  The HF callback defines `"*"` → `train/global_step` (§8), so our own axes must be **defined by
  exact name, not by glob**, or the x-axis a metric gets is left to chance.
- Accuracy-per-GPU-hour across runs: within a run, a line chart of `val/exact` against
  `compute/gpu_hours` overlays every training run that uses the same x key. Final test numbers come
  from separate eval runs, so plot them as a scatter of summary values: `exact` (eval run) against
  `compute/gpu_hours` (copied into the eval run's config from the checkpoint's training run).

## 7. System metrics and GPU time

- wandb-core samples system metrics every **15 s** by default (`x_stats_sampling_interval`,
  W/wandb/sdk/wandb_settings.py L908). The NVIDIA metrics are `gpu.N.gpu` (utilization %),
  `memory`, `memoryAllocated(Bytes)`, `temp`, `powerWatts`, `powerPercent`, the clocks and the ECC
  errors ([D/models/ref/python/experiments/system-metrics.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/ref/python/experiments/system-metrics.mdx) "NVIDIA GPU").
  Newer versions add `gpu.N.energyJoules` (cumulative energy, NVIDIA and AMD) and
  `gpu.process.N.*` for the calling process (CHANGELOG 0.31.0 "Added"/"Fixed").
- *(verified locally on WSL2)*: NVML works and the RTX 3060 shows up with `gpu`, `memory*`, `temp`,
  `powerWatts`, `energyJoules`, `pcieRx/TxBytes` and others. `energyJoules` was already 1.16 MJ on
  a 0.2 s run, so it counts from driver or boot time, not from run start.
- **There is no GPU-time total.** The summary has only `_runtime`, the run's wall-clock seconds
  (verified in the offline-run summary). GPU-hours have to be logged by hand.
- **Manual accounting (suggested):** time GPU work with `time.perf_counter()` around each phase, and
  log the running totals:
  - `compute/student_gpu_hours`: the trainer's own forward and backward passes.
  - `compute/teacher_gpu_hours`: teacher forward passes inside the loop (on-policy scoring, online
    logit KD), plus the **amortized share of any offline teacher job**. That covers SFT data
    generation and precomputed top-k logits, whose GPU-hours come from those jobs' own logged
    `_runtime` or timers and are recorded in the training run's config as
    `teacher_precompute_gpu_hours`.
  - `compute/gpu_hours`: the sum of the two, the x-axis for #2's curves.

  With a single GPU, wall-clock training time (excluding eval and checkpoint saves) is a fair
  measure of GPU time. Whether validation evals count toward the budget is a project decision
  (open question).

## 8. HF Transformers Trainer and TRL

Source: [HF/integrations/integration_utils.py](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/integrations/integration_utils.py#L695-L944)
(`WandbCallback`; identical on `main` 04b0d42).

- `TrainingArguments.report_to` **defaults to `"none"`** in v5 (HF/training_args.py L395, L1059 on
  `main`). Logging to W&B needs `report_to="wandb"` (or `["wandb"]`). `"none"` or `[]` turns it
  off.
- **Reusing an active run:** `setup()` calls `wandb.init(project=os.getenv("WANDB_PROJECT",
  "huggingface"), name=args.run_name)` **only if `wandb.run is None`** (L761-L765). Otherwise it
  uses the run we started and merges `args.to_dict()`, the model config and `peft_config` into it
  with `config.update(..., allow_val_change=True)` (L767). So: start the run ourselves (project,
  group, job_type, our config) **before** `trainer.train()`, and the callback adds its arguments.
  Its keys are flat (`learning_rate`, `lora_…` under `peft_config`), so keep ours under prefixes
  (`fc/…`) to avoid collisions.
- It defines `train/global_step` and **`define_metric("*", step_metric="train/global_step",
  step_sync=True)`** (L770-L772). It logs through the global `wandb.log`, adding
  `train/global_step` (L900). It renames keys to `train/…` / `eval/…` (`rewrite_logs`), and sends
  `train_runtime`, `train_loss`, `total_flos` and similar to the summary (L882-L898).
- `WANDB_WATCH` (`gradients|parameters|all|false`, default `false`) and `WANDB_LOG_MODEL`
  (`end|checkpoint|false`, default `false`) are read from the environment (L709, L775).
  `run_name` becomes the run name (L755-L757).
- Since it uses the global `wandb.run`, `reinit="create_new"` would break it (§1). Use one run per
  process.
- **TRL**: its trainers are `Trainer` subclasses and use the same callback. In addition, when
  `"wandb" in report_to` and `wandb.run` is set:
  - `GRPOConfig.log_completions` logs a `completions` Table of (prompt, completion) samples
    ([TRL/trainer/grpo_config.py L364-L367, L993](https://github.com/huggingface/trl/blob/cf1bfd464c3308466f9746b9fec17cbc94f97af9/trl/trainer/grpo_config.py#L364-L367);
    grpo_trainer.py L3458).
  - The new `DistillationTrainer` does the same (distillation_trainer.py L2028).
  - `trl.extras.profiling` sends `wandb.log` timings (extras/profiling.py L122-L123), which add
    history steps and metric keys. They are another reason to define our axes by exact name.

## 9. Resuming runs

- `wandb.init(id=..., resume="allow"|"must"|"never"|"auto")`
  ([D/models/runs/resuming.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/runs/resuming.mdx) table).
  W&B recommends **`"allow"` with an explicit id** over `"auto"`, which depends on restarting from the
  same directory. `"must"` raises an error if the id doesn't exist. Since 0.31.0 it also works for
  offline runs (CHANGELOG).
- History resumes from the run's last step, and the step must always increase. The HF callback's
  x-axis is `train/global_step`, which restarts from the checkpoint's step, so a resumed run may log
  a few duplicate `global_step` values (unverified how charts show them). A cumulative
  `compute/gpu_hours` must be **restored from the checkpoint** (save it in `trainer_state` or a
  sidecar file) and not restarted at zero.
- `resume_from="{id}?_step=N"` (beta) rewinds history to step N. That suits resuming from an
  earlier checkpoint than the last logged step (init docstring).
- What this means here: on a single consumer GPU, OOMs and WSL restarts are likely. Store the run id
  in the run's output directory (`runs/<run>/wandb_run_id`), and on `--resume-from-checkpoint` call
  `init(id=..., resume="allow")`.

## 10. Public project and anonymous viewing

- Project visibility can be **Open, Public, Team or Restricted**, set from the icon next to the
  project name → **Project Visibility**. **Public** means *"Anyone who knows about the project can
  view it. Only your team can submit runs or reports."* Org admins can turn Public off
  ([D/snippets/_includes/project-visibility-scopes-table.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/snippets/_includes/project-visibility-scopes-table.mdx),
  [project-visibility-settings.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/snippets/_includes/project-visibility-settings.mdx)).
  Use **Public**, not Open.
- **Unverified:** whether a Public project can be viewed without logging in under the Forge domain,
  whether personal (free) accounts can still create Public projects after the Forge move, and the
  final URL shape (`https://forge.coreweave.com/wandb/<entity>/<project>`?). Check by opening the
  link in a private window before adding it to the README.
- Anonymous *logging* is gone (§1). That has no effect on public viewing.

## Recommendations for #23

> Written before the implementation. What was built differs in places, and
> `src/fc_distill/tracking.py` is the reference. Decisions taken with the maintainer: `wandb` is a
> base dependency (it is never imported while logging is off); no adapter uploads, so eval runs
> link to checkpoints by path; validation time and shared teacher-generation time count toward
> the GPU-hour budget, with the shared time split across the runs that use it; and the #10
> baselines are logged from `results/baselines.json` with `fc-report --wandb` (one run per model,
> see docs/notes/wandb-dashboard.md), not from the saved runs, which a later rerun overwrote.

**Minimal feature set:** `init` (project / group / job_type / tags / config), `run.log` + exact-name
`define_metric` for the GPU-hours axis, `run.summary` for one-shot evals, one `wandb.Table` for
slices, `finish`. Resume by id. No sweeps, no `watch`, no `save_code`, and no uploaded artifacts at
first.

**Dependency:** add `wandb` with `uv add --optional tracking wandb` (or include it in the training
extra). It is CPU-only, but CI doesn't need it if unit tests go through a fake tracker. Import it
inside functions, like the GPU libraries. `wandb` ships type hints, so no mypy override is needed.

**Module sketch:** `src/fc_distill/tracking.py`

```python
class Tracker(Protocol):
    def log(self, metrics: Mapping[str, float], *, gpu_hours: float | None = None) -> None: ...
    def summary(self, values: Mapping[str, Any]) -> None: ...
    def table(self, key: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None: ...
    def finish(self, exit_code: int = 0) -> None: ...

def start_run(
    *, enabled: bool, job_type: Literal["train", "eval"], group: str, name: str | None,
    config: Mapping[str, Any], run_dir: Path, resume_id: str | None = None,
    tags: Sequence[str] = (),
) -> Tracker:
    """NullTracker when not enabled (wandb never imported). Otherwise wandb.init(project=PROJECT,
    entity=ENTITY or None, dir=run_dir, id=resume_id, resume="allow" if resume_id else None, ...),
    with no mode= argument so WANDB_MODE still works, followed by the define_metric calls below."""

def provenance_config(...) -> dict[str, Any]:
    """git_state() (commit, dirty) + base model id/revision sha + teacher id/revision sha +
    xlam_revision, splits_sha256, prompt_sha256, library versions. Shared with fc-eval."""

class GpuClock:
    """Accumulates student/teacher seconds via context managers; state saved with checkpoints."""
```

- **Config** (fixed inputs): `method`, `variant`, LoRA (`r`, `alpha`, `dropout`, `target_modules`), LR,
  schedule, steps or epochs, batch and sequence lengths, data (`split`, filters, `n_rows`,
  teacher-output source), `provenance` (as above, dirty flag included), `teacher_precompute_gpu_hours`.
  The eval run also gets `checkpoint` (path + adapter sha256), `train_run_id` and the checkpoint's
  `gpu_hours`.
- **History** (training): the trainer's own `train/*` and `eval/*` (from the HF callback), plus our
  `val/exact`, `val/parse_ok`, …, and `compute/{student,teacher,total}_gpu_hours`, all in the same
  `log` call.
- **Summary**: on eval runs, the whole `summary.json` (overall metrics, slice scalars, `seconds`,
  `output_tokens`, `provenance`). On training runs, `val/exact` gets `max,last` through
  `define_metric`, plus the final `compute/gpu_hours`.
- **define_metric**, by exact name and never by glob: `compute/gpu_hours`; then for each validation
  metric `m`: `run.define_metric(m, step_metric="compute/gpu_hours", summary="max,last")`.
- **`fc-eval --wandb`**: after `write_summary`, open `start_run(enabled=True, job_type="eval",
  group=<method or "baseline">, config={"model", "split", "provenance": ...})`, put the summary in
  `run.summary` and the slices in a Table, then finish. `summary.json` stays the source of truth and
  W&B only mirrors it. `--rescore` could re-log through the same path (maintainer's call). Re-logging
  #10's baselines is then just running `fc-eval --rescore … --wandb` on the saved runs, with no GPU
  needed.
- **Tests:** (1) a fake `Tracker` records calls, and trainer and eval code are tested against it.
  (2) `start_run(enabled=False)` returns the null tracker and never imports wandb (check with
  `sys.modules`). (3) If wandb is installed in dev: `WANDB_MODE=disabled` +
  `start_run(enabled=True)` runs without network or files (`tmp_path` stays empty). (4) Optional and
  slower: an offline run in `tmp_path`, read back with `wandb leet inspect --json`, checks
  `val/exact.max` and that `provenance.dirty` is in the config.

**Open questions for the maintainer**

1. Entity and project name. Personal account or team, and is Public visibility available on that
   account under Forge?
2. Upload LoRA adapters as artifacts (public download), use reference artifacts, or neither?
3. Does validation eval time count toward the GPU-hour budget? Does SFT's teacher data generation
   count fully toward every method that reuses it, or is it shared out?
4. Should `wandb` be a base dependency, a `tracking` extra, or part of a training extra, and should
   CI install it for the disabled-mode test?
5. Should `fc-eval --rescore … --wandb` be the way to re-log #10's baselines (no GPU), even though
   it logs `rescored_by` provenance alongside the original?
