# Weights & Biases: metrics, keys and a dashboard to show the work

Follow-up to [`wandb.md`](wandb.md) (SDK behaviour, `define_metric`, resuming, artifacts, the HF
callback's run reuse), which this page doesn't repeat. This page covers what to log, under which
keys, so runs and experiments compare cleanly, and how to build and screenshot the public
dashboard. Facts checked on 2026-11-06. Claims marked *(unverified)* were not confirmed in a source
or by running them. *(code reading)* means a conclusion drawn from the source but not run.

Sources, pinned:

- D = `https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/`, the archived
  docs (`docs.wandb.ai` redirects to `docs.coreweave.com`, which asks for an access code)
- W = `https://github.com/wandb/wandb/blob/v0.31.0/`, the SDK
- WS = `https://github.com/wandb/wandb-workspaces/blob/b3a9729130d6a8d5303f00458adc8ac9d8472488/`,
  `main` on 2026-10-21, one dependency bump after the 0.4.13 release
- HF = `https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/`
- TRL = `https://github.com/huggingface/trl/blob/cf1bfd464c3308466f9746b9fec17cbc94f97af9/trl/`
- EX = `https://github.com/wandb/examples/blob/a03988fc6b6a41576956822bf241f5ae84422391/`

State of the project on 2026-11-06: `wandb.Api().runs("joaomigueloliveira30/qwen-function-calling-distillation")`
returned **0 runs**, so nothing has been logged for good yet and the key schema can still change
without leftover runs.

## 1. Metric names: namespaces and what HF / TRL already log

### What W&B does with names

- **Sections come from the first `/` prefix.** By default the workspace groups panels by the
  metric name's first prefix (`a/b/c/d` and `a/e/f` both go in section `a`). This can be switched
  to the last prefix ([D/models/track/workspaces.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/workspaces.mdx)
  "Section organization"; `WorkspaceSettings.group_by_prefix`, WS/wandb_workspaces/workspaces/interface.py
  L638). So `train/`, `val/`, `compute/` and so on each become one workspace section.
- **Nested dicts become dotted names**, and every distinct key counts toward the project's metric
  cardinality ([D/models/track/limits.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/limits.mdx)
  "Metric cardinality"). Dots don't make sections, so for metrics use flat `/` keys, not nested
  dicts.
- **Character rules.** The docs say metric names should match `/^[_a-zA-Z][_a-zA-Z0-9]*$/`, and
  warn that commas, spaces, hyphens (`"loss-train"`) and a leading digit (`"5_fold_cv"`) *"may
  cause problems with sorting, querying, or display"*
  ([D/snippets/\_includes/metric-naming-rules.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/snippets/_includes/metric-naming-rules.mdx),
  [metric-naming-examples.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/snippets/_includes/metric-naming-examples.mdx);
  [D/models/track/log.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/log.mdx) L68-L76).
  Read literally, that also rules out `/`, but the HF callback logs every metric as `train/…` and
  the docs build sections from `/`, so `/` is the de facto separator. Read the rule as applying to
  each segment. **This affects our names:** the split `test-clean` and the buckets `1-2`, `3+` and
  `0` break it (hyphen, `+`, leading digit). Use `test_clean`, `d1_2`, `d3plus` and `d0` in keys,
  and keep the original names as values in Tables and config.
- Recommended limits (multi-tenant cloud): 100,000 metric keys per project, 10,000 runs, 1 MB per
  value, 10 MB per config (limits.mdx table L112-L119, "Value size", "Config size"). The schema
  below needs about 100 summary keys per eval run, shared by every eval run.

### What `transformers` 5.22.2 logs (`report_to="wandb"`)

`WandbCallback.on_log` sends **five keys to the summary only**: `train_runtime`,
`train_samples_per_second`, `train_steps_per_second`, `train_loss` and `total_flos`
(HF/integrations/integration_utils.py L882-L897). Everything else goes through `rewrite_logs`:
`eval_x` → `eval/x`, `test_x` → `test/x`, and **any other key → `train/x`** (L550-L563). Each row
also gets `train/global_step` (L900).

| Key in W&B | Source | Notes |
|---|---|---|
| `train/loss`, `train/grad_norm`, `train/learning_rate` | HF/trainer.py L2234-L2240 | every `logging_steps` |
| `train/epoch` | trainer.py L4172 (`Trainer.log`) | |
| `train/num_input_tokens_seen`, `train/train_tokens_per_second` | trainer.py L4173-L4177 | only with `include_num_input_tokens_seen="all"\|"non_padding"` (default `"no"`, training_args.py L373, L1006) |
| `train/global_step` | integration_utils.py L770-L772, L900 | defined as the x-axis of `"*"` |
| summary `train_runtime`, `train_loss`, `total_flos`, `train_samples_per_second`, `train_steps_per_second` | trainer.py L1980-L1993; `speed_metrics`, trainer_utils.py L535-L560 | end of training |
| `train/train_mem_gpu_peaked_delta` and similar | `TrainerMemoryTracker`, trainer_utils.py L625-L810 | only with `skip_memory_metrics=False` (default `True`, "slows down training", training_args.py L1453-L1456) |
| `eval/loss`, `eval/<compute_metrics key>`, `eval/runtime` … | `evaluate()` with `metric_key_prefix="eval"` | teacher-forced, on `eval_dataset` |
| `test/…` | `on_predict` → `rewrite_logs` (L925-L931) | **only if `trainer.predict()` is called**. It would land in the same `test/` namespace as our official test scores, so trainers must not call `predict` (or must pass a different `metric_key_prefix`) |

Because of the `train/` rewrite, **a key we add to the Trainer's `logs` dict comes out as
`train/<key>`**. Use that on purpose for training-loop metrics (below), and log the `compute/` and
`val/` keys through our own `Tracker.log` instead.

### What TRL (`main` cf1bfd4) adds

TRL trainers fill `self._metrics[mode][name]` and average them into `logs` in `log()`, adding
`eval_` in eval mode (TRL/trainer/sft_trainer.py L1737-L1748; distillation_trainer.py L1995-L2014).
So a TRL metric `x` shows up as **`train/x` and `eval/x`**.

| Trainer | Metrics (each becomes `train/…` and `eval/…`) | Source |
|---|---|---|
| `SFTTrainer` | `entropy`, `mean_token_accuracy`, `num_tokens` (cumulative), `aux_loss` (MoE) | TRL/trainer/sft_trainer.py L1685-L1705 |
| `GKDTrainer` (experimental) | none of its own: it subclasses `SFTTrainer` and overrides `compute_loss`, so only `loss` | TRL/experimental/gkd/gkd_trainer.py L54, L324 |
| `DistillationTrainer` (new, on-policy, `beta` picks forward KL / JSD / reverse KL) | `num_tokens`, `completions/{mean,min,max}_length`, `completions/clipped_ratio`, `completions/{mean,min,max}_terminated_length`, `tools/call_frequency`, `tools/failure_frequency`, `entropy`, `step_time`; `log_completions` adds a `completions` Table | distillation_trainer.py L1484-L1511, L1865, L1982, L2017-L2035; distillation_config.py L359-L389 |
| `GRPOTrainer` | `reward`, `reward_std`, `rewards/<func>/{mean,std}`, `frac_reward_zero_std`, `kl` (when `beta > 0`), `entropy`, `policy_loss`, `clip_ratio/{low_mean,low_min,high_mean,high_max,region_mean}`, `completions/*`, `num_tokens`, `step_time`, `sampling/*` (vLLM importance sampling) | grpo_trainer.py L1586, L2388-L2415, L2942-L3013, L3335-L3410 |

The divergence itself is the `loss` in both distillation trainers. **Neither TRL trainer logs the
forward and reverse KL separately, teacher accuracy, or student-teacher agreement.**

### Worth adding for this project

The idiomatic way in a TRL subclass is to append to `self._metrics[mode]` inside `compute_loss`.
The keys then get the `train/` / `eval/` twins for free. Under a `distill/` sub-prefix:

| Key | Meaning | Why |
|---|---|---|
| `train/distill/kl_fwd`, `train/distill/kl_rev` | KL(teacher‖student) and KL(student‖teacher) per completion token, whatever the loss mixes | compares logit KD and on-policy runs on one scale |
| `train/distill/teacher_token_acc` | teacher argmax == label token (SFT and logit KD on gold) | shows how noisy the teacher signal is |
| `train/distill/top1_agree` | student argmax == teacher argmax | the simplest "is it learning the teacher" curve |
| `train/distill/teacher_entropy` | mean teacher entropy on the batch | next to TRL's student `entropy` |
| `train/num_tokens` | TRL's own (SFT, distillation, GRPO). A custom loop logs the same key | one tokens axis across methods |
| `train/examples_seen` | `global_step × effective batch size` | x-axis that ignores packing and length |

Logged through `Tracker.log` (our keys, not rewritten by HF):

| Key | Meaning |
|---|---|
| `compute/student_gpu_hours`, `compute/teacher_gpu_hours`, `compute/gpu_hours` | already in `tracking.py`, the x-axis of the curves |
| `compute/peak_gpu_mem_gb` | `torch.cuda.max_memory_allocated()` since the last log, then `reset_peak_memory_stats()`. It doesn't see vLLM's preallocated pool; the system metric `gpu.0.memoryAllocatedBytes` (every 15 s, [`wandb.md`](wandb.md) §7) covers the whole process |
| `compute/tokens_per_s` | training tokens per GPU-second, for the cost story |
| `val/<metric>`, `val/n` | generation-based validation scores, already in `tracking.py`, in the same row as `compute/gpu_hours` |

**`system/` is reserved:** W&B puts its own system metrics there. Don't log our own keys under it.

**Getting `train/loss` on the GPU-hours axis.** HF rows don't carry `compute/gpu_hours`. With
`define_metric("train/loss", step_metric="compute/gpu_hours")` **by exact name** (an exact
definition wins over HF's `"*"` glob, [`wandb.md`](wandb.md) §6), `step_sync` fills in the last
logged `compute/gpu_hours` ([D/models/track/log/customize-logging-axes.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/log/customize-logging-axes.mdx)).
To keep that value fresh, log `compute/*` from a callback's `on_step_end` whenever
`control.should_log` is set. `on_step_end` runs just before `_maybe_log_save_evaluate`
(HF/trainer.py L1929-L1930), and `DefaultFlowCallback`, which sets `should_log`, runs before user
callbacks (L586, L595). `train/global_step` stays in every HF row, so the step axis is still
available in the UI. *(code reading; unverified that the chart shows both axes as expected)*

## 2. Eval keys that compare across splits

### What we log today

`log_eval` copies `summary.json` into the run summary unchanged (`src/fc_distill/evaluate.py`
L197-L207). The two eval families don't line up:

| | xLAM (`test`, `validation`) | BFCL |
|---|---|---|
| headline | `exact` | `overall.exact` |
| slices | `slices.call_pattern.single.exact` … | `slices.simple.exact`, `categories.parallel.exact` |
| intervals | `exact_ci: [lo, hi]` (a list) | `overall.exact_ci` |
| test-clean | not in any eval run: `fc-report` computes it from the test predictions (`report.py` L5, L47-L68) | — |

Problems:

1. A bar or scatter panel picks **one key**, so `exact` and `overall.exact` can't share a panel.
2. One run per split with the same key `exact` means every panel needs a split filter, or test and
   BFCL numbers mix in one chart.
3. `exact_ci` is a list. Bar, scatter and parallel-coordinates panels plot numbers, so the
   intervals can't be plotted *(unverified how the UI shows a list-valued summary)*.
4. Nested dicts give dotted names (`slices.call_pattern.single.exact`), which don't make sections,
   and the bucket names `1-2` and `3+` break the naming rule.
5. test-clean, the split the success criterion uses, doesn't exist in W&B at all.

### How panels compare runs

- **Runs table:** one row per run with config and summary columns. It can be grouped and sorted,
  and exported to CSV ([D/models/runs/customize-run-display.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/runs/customize-run-display.mdx)
  "Export runs table to CSV"). It shows at most 500 columns (WS/…/reports/v2/interface.py,
  `Runset` "Column limit").
- **Bar plot:** one bar per run for a summary metric. Runs can be grouped by any config key, and
  grouping enables box and violin plots
  ([D/models/app/features/panels/bar-plot.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/app/features/panels/bar-plot.mdx)).
  `wr.BarPlot(metrics=[...])` takes several metrics (WS/…/reports/v2/interface.py L2399-L2430).
- **Scatter plot:** each point is a run, and x, y and z are summary **or config** values
  (`SummaryOrConfigOnlyMetric`, WS/…/interface.py L2289-L2320). It has reference lines and a
  frontier line ([D/models/app/features/panels/scatter-plot.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/app/features/panels/scatter-plot.mdx)
  L18-L19, L46). `wandb-workspaces` exposes no reference-line field, so those are set in the UI
  *(code reading: no match for "reference" in reports/v2)*.
- **Parallel coordinates:** each line is a run, and the axes are config keys and metrics
  ([D/models/app/features/panels/parallel-coordinates.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/app/features/panels/parallel-coordinates.mdx)).

All four compare **values of one run across columns**. A scatter of test-clean against BFCL (does
the gain carry over to unseen tools?) or a parallel-coordinates line through test → test-clean →
BFCL → each slice only works **if one run holds every split**.

### One run per split, or one run per checkpoint?

| | One run per (model, split): today's `fc-eval --wandb` | **One run per model/checkpoint, all official evals** |
|---|---|---|
| Runs table | 3 rows per model, read across rows | 1 row per model: test, test-clean, BFCL side by side |
| Bar per eval | works with a split filter | works directly (`test_clean/exact`) |
| Scatter test-clean vs BFCL, or vs `compute.gpu_hours` | **impossible** (x and y come from one run) | works |
| Parallel coordinates across splits and slices | impossible | works |
| test-clean | needs a fourth, made-up run | computed from the test predictions in the same run, as `fc-report` does |
| Fits `fc-eval` | yes: one process, one directory, one run | needs an aggregation step after the per-split `fc-eval` runs (they stay the source of truth) |
| Re-logging | per directory | per model. A run id **can't be reused, even after the run is deleted** ([D/models/runs/delete-runs.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/runs/delete-runs.mdx) L91), so after a UI delete, the saved `wandb_run_id` file must be removed too (today's `start_run` would reuse it and fail) |
| Provenance | one `provenance` block | one block per split (`provenance.test.*`, `provenance.bfcl.*`) |

**Recommendation:** one W&B eval run per model or checkpoint. Keep `fc-eval` exactly as it is per
split (directories, `summary.json`, `predictions.jsonl`). Do the W&B logging in an aggregation
step: for the baselines, `fc-report --wandb`, from `results/baselines.json`, which already holds
`test`, `test-clean` and `bfcl` per model (`report.py` L55-L68). For trained checkpoints, the same
code over that checkpoint's run directories. Validation stays in the **training** run's history
(`val/*` against `compute/gpu_hours`). Eval runs log only `test`, `test_clean` and `bfcl`, so the
training-run summary columns `val/exact.max` / `.last` never mix with eval scalars.

### Key schema for eval runs (summary)

`<eval>` ∈ {`test`, `test_clean`, `bfcl`}. xLAM metrics are `parse_ok`, `exact`, `lenient_exact`,
`names` and `arg_f1`; BFCL's are `parse_ok`, `exact`, `bfcl_exact` and `names`.

| Key | Example |
|---|---|
| `<eval>/<metric>` | `test_clean/exact = 0.551` |
| `<eval>/<metric>_lo`, `<eval>/<metric>_hi` | the 95% interval as two scalars |
| `<eval>/n` | `1625` |
| `<eval>/slice/<dim>/<bucket>/exact`, `…/n` | `test_clean/slice/call_pattern/parallel/exact`; `test/slice/distractors/d3plus/n` |
| `bfcl/slice/<slice>/exact`, `bfcl/category/<cat>/exact` | `bfcl/slice/multiple_parallel/exact` |
| `<eval>/seconds`, `<eval>/output_tokens`, `<eval>/output_tokens_per_s` | inference cost, for the serving story (test-clean has none: it's a subset of the test run) |
| `test_clean/gap_recovered` (+ `_lo` / `_hi` once #2's paired bootstrap exists) | `(x − student) / (teacher − student)`, trained checkpoints only |

That is about 35 keys per xLAM eval and 28 for BFCL, so roughly 100 in all. Every slice × metric ×
interval goes in **one `slices` Table** with columns `eval, group, slice, n, too_small, <metric>,
<metric>_lo, <metric>_hi`. The original bucket names stay there as values. Keep only `exact` and
`n` per slice as summary scalars, since those are what panels compare.

## 3. Run metadata: config, group, job type, tags, names

### What the HF callback writes into the config

`setup()` merges `{"peft_config": model.peft_config, **model.config.to_dict(), **args.to_dict()}`
into the config with `allow_val_change=True` (HF/integrations/integration_utils.py L739-L767), and
sets `model/num_parameters` (L782). Consequences:

- **About 105 flat top-level keys** from `TrainingArguments` (counted `field(` declarations) (`learning_rate`, `seed`,
  `per_device_train_batch_size`, …), plus the **base model's config, also flat at the top level**
  (`vocab_size`, `hidden_size`, `model_type`, …). The arguments overwrite model-config keys with
  the same name. *(unverified how a `LoraConfig` object under `peft_config` serialises)*
- It can't be turned off without dropping `report_to="wandb"`: `config_exclude_keys` only applies
  to `wandb.init`'s own `config`, and the run is already open. The alternative is our own
  `TrainerCallback` that forwards `logs` to `Tracker`. That gives clean config and naming, but
  it's non-standard and we would have to maintain it. **Keep the standard callback**, put our keys
  under a few nested top-level dicts that HF never uses, and hide the noise in the dashboard by
  locking the visible runs-table columns (`Runset.visible_columns` + `lock_columns`, WS/…/interface.py
  `Runset` docstring).
- Nested dicts become dotted columns (`exp.lora.r`), and config keys must not contain `.` themselves
  ([`wandb.md`](wandb.md) §2). In `wandb-workspaces` filters and columns they are addressed as
  `Config("exp.method")` and `config:exp.method.value` (WS/…/interface.py `Runset` "Column format").
- HF's `seed`, `learning_rate` and `num_train_epochs` collide with the obvious names. Ours live
  under `exp.*` even if that repeats a value.

### Conventions

- **`group`** = the method: `baseline`, `sft-gold`, `sft-teacher`, `opd`, `grpo`, … W&B describes a
  group as *"a collection of runs that share a common purpose"*
  ([D/models/runs/grouping.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/runs/grouping.mdx)).
  Seeds and checkpoints of one method share it, so line plots can average a group with a min-max band
  (`groupby_aggfunc`, `groupby_rangefunc`, WS/…/interface.py L2097-L2135).
- **`job_type`**: `train`, `eval`, `data` (split statistics), `generate` (teacher outputs and
  precomputed logits). `tracking.JobType` has only the first two today.
- **`tags`**: status rather than facts: `official` (shown in the README dashboard), `smoke`,
  `superseded`. Facts that also live in config (split, method) make tags redundant.
- **`name`**: `train-<method>-<variant>-s<seed>` (`train-sft-teacher-r16-s0`) and
  `eval-<model-or-run-name>` (`eval-qwen3.5-0.8b`, `eval-train-sft-teacher-r16-s0-ckpt1200`).
  Names don't have to be unique ([D/models/runs/run-identifiers.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/runs/run-identifiers.mdx)),
  but ours should be. Pass `TrainingArguments.run_name` the same name, so HF doesn't warn about
  `run_name == output_dir` (integration_utils.py L755-L760).
- W&B's own LLM fine-tuning example does the same thing: `wandb.init(project, entity,
  job_type="training")` before a `Trainer` with `report_to="wandb"`, then separate runs with
  `job_type="inference"` for a prompt/completion Table
  ([EX/colabs/huggingface/LLM_Finetuning_Notebook.ipynb](https://github.com/wandb/examples/blob/a03988fc6b6a41576956822bf241f5ae84422391/colabs/huggingface/LLM_Finetuning_Notebook.ipynb)).

## 4. Dataset metrics

W&B's model is **inputs and outputs as Artifacts**. Its table lists *"Model Training: Dataset
(training and validation data) → Trained Model"* and *"Model Evaluation: Model + Dataset (test data)
→ W&B Table"* ([D/models/artifacts.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/artifacts.mdx)
"Use cases"). Artifacts take a `metadata` dict
([D/models/artifacts/construct-an-artifact.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/artifacts/construct-an-artifact.mdx)
L42-L53). A dataset logged once and referenced by later runs is what *artifact tables* are for
([D/models/track/log/log-tables.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/log/log-tables.mdx)
L161-L176). Artifact metadata shows on the artifact page, not in the runs table *(unverified)*, so
any number a panel should compare must also go in config or the summary.

| What | Where | Why |
|---|---|---|
| Split statistics: rows per split, rows removed by each train filter, prompt and answer token length p50 / p95 / max, call-pattern and distractor mix | One `job_type="data"` run: summary `data/<split>/…` scalars, plus a `data/filters` Table, plus an artifact `xlam-splits` (type `dataset`) holding `data/splits.json` with these stats as metadata | Logged once per `splits_sha256`, and shows the data story in the dashboard |
| What a training run actually used: `exp.data.n_rows` after the method's filters, `exp.data.source` (gold / teacher), `exp.data.filters`, `splits_sha256` | **Config** of the training run, plus `run.use_artifact("xlam-splits:<sha8>")` for lineage | These are inputs, so they go in config. They differ by method or variant (teacher-output filtering) and runs get filtered by them |
| Teacher-generated data: rows generated, teacher `parse_ok`, rows kept, GPU-hours | A `job_type="generate"` run whose output is a `teacher-outputs` artifact, used by the SFT-on-teacher run | Its `compute/gpu_hours` becomes the training run's `teacher_precompute_gpu_hours` ([`wandb.md`](wandb.md) §7) |

`splits.json` holds only ids, so it's small. Uploading row *text* (queries, tools, answers) to a
public project redistributes the gated xLAM data (CC-BY-4.0, gated behind accepting its terms,
`docs/DATASET.md` L12-L14). That is a maintainer decision (open questions).

## 5. Dashboards: workspaces, saved views, reports; as code; screenshots

### The three objects

- **Personal workspace**: every user has one. Only its owner can edit it, and teammates can view
  it. **Saved view**: a shared snapshot of a workspace that the team can edit
  ([D/models/track/workspaces.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/workspaces.mdx)
  "Workspace types"). The automatic workspace makes a panel for every key, and the docs recommend
  a **manual** workspace for focused views and speed (limits.mdx "Panel count", "Many metrics per
  run").
- **Report**: a document with text and live panel grids, each over a run set with its own filters.
  It can be published or saved as a draft (`Report.save(draft=False)`, WS/…/reports/v2/interface.py
  L3671-L3698), **exported as PDF or LaTeX** (action menu → Download,
  [D/models/reports/clone-and-export-reports.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/reports/clone-and-export-reports.mdx)),
  and embedded with an iframe, but only from public projects or view-only links
  ([embed-reports.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/reports/embed-reports.mdx)).
- **Public visibility:** view-only "magic" links let *"anyone … view the report without logging in
  first"*. For Public projects such links don't apply, because *"The project is already public, so
  the report is already available to anyone with the link"*
  ([D/models/reports/cross-project-reports.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/reports/cross-project-reports.mdx)
  L27-L38). By implication a Public project's report opens logged out. That this holds on the
  Forge domain is still *(unverified)* ([`wandb.md`](wandb.md) §10). Check in a private window.

**For the story, the Report is the artifact to link from the README.** It has prose, fixed run
sets (baselines plus `official` runs), it can't be changed by accident the way a workspace can,
and it exports to PDF. The workspace (one saved view) is for day-to-day work.

### Dashboard as code: `wandb-workspaces`

| Fact | Source |
|---|---|
| Latest **0.4.13** (2026-10-06). Monthly releases since 0.4.4 (2026-06). Python >=3.10 | [PyPI JSON](https://pypi.org/pypi/wandb-workspaces/json) |
| *"This feature is in **Public Preview**"*. Docs: *"W&B Report and Workspace API is in Public Preview"* | WS/README.md L11; clone-and-export-reports.mdx L7 |
| Dependencies: only `pydantic>=2.7` and `wandb>=0.28.2`. Both are already in `uv.lock` (pydantic 2.14.3, wandb 0.31.0), so **it adds no new transitive packages**. It ships `py.typed`. Also available as the `wandb[workspaces]` extra | WS/pyproject.toml L23-L26; W/pyproject.toml L112 |
| Workspaces: `ws.Workspace(entity, project, name, sections=[ws.Section(name, panels=[...])], settings=ws.WorkspaceSettings(x_axis=…), runset_settings=ws.RunsetSettings(filters=…, groupby=…))`, then `.save()` (a new saved view) or `.save_as_new_view()` | WS/wandb_workspaces/workspaces/interface.py L459-L1373; D/models/track/workspaces.mdx "Programmatically create workspaces" |
| Reports: `wr.Report(entity, project, title, blocks=[wr.H1, wr.P, wr.PanelGrid(runsets=[wr.Runset(filters=…)], panels=[…])])`, then `.save()`. `enable_share_link()` makes a view-only link | WS/README.md; WS/examples/reports/05_share_link.py |
| Panels: `LinePlot` (any metric as `x`; `groupby`, `groupby_aggfunc`, `groupby_rangefunc`), `ScatterPlot` (summary or config axes), `BarPlot(metrics=[…])`, `ParallelCoordinatesPlot`, `RunComparer`, `ScalarChart`, `WeavePanelSummaryTable(table_name="slices")` for a logged Table, `MarkdownPanel`, `CustomChart` (Vega) | WS/…/reports/v2/interface.py L2097-L3299 |
| Filters: string expressions (`"Config('exp.method') = 'baseline' and State = 'finished'"`) or `ws.Config`, `ws.Summary`, `ws.Tags`, `ws.Metric` objects | WS/examples/workspaces/03_working_with_workspace_filters.py; `Runset` docstring |
| **Auth:** it calls `wandb.Api()` (the stored API key, and since 0.31.0 the Forge host by default) and talks GraphQL | WS/wandb_workspaces/\_graphql.py; WS/…/workspaces/internal.py L147, L189; W/wandb/sdk/lib/urls.py L7 |
| User (personal) views can't be loaded: *"Workspace API does not currently support user views"* | WS/wandb_workspaces/utils/validators.py L50-L54 |

**Forge caveats** *(code reading, not run)*:

- The Forge app URL is `https://forge.coreweave.com/wandb/<entity>/<project>`. The SDK's
  `api_to_app_url` returns `https://forge.coreweave.com/wandb` (W/wandb/util.py L210-L216,
  `FORGE_APP_PATH = "/wandb"` in W/wandb/sdk/lib/urls.py L10).
- `Workspace.url` and `Report.url` build `scheme://netloc/<entity>/<project>…` from that app URL
  and **drop its `/wandb` path** (WS/…/workspaces/interface.py L935-L952; reports/v2/interface.py
  L3645-L3669). So the URLs they print are probably wrong on Forge.
- `Workspace.from_url` / `validate_url` split the path as `/<entity>/<project>`
  (WS/…/utils/validators.py L33-L46; workspaces/interface.py L1343-L1356), and `Report.from_url`
  as `/<entity>/<project>/reports/<name>` (reports/v2/interface.py L4019-L4023). **A Forge URL,
  with its extra `/wandb` segment, fails to parse.** Workaround: strip `/wandb` from the URL before
  calling `from_url` (only the path and the `nw` query are read, and the call goes through
  `wandb.Api()`).
- `save()` goes through `wandb.Api()` GraphQL, so creating views and reports should work. Nothing
  in the repo mentions Forge (`gh search` for "forge" in wandb/wandb-workspaces found nothing).
  **Try it on a scratch project first.**

### Panels for our story

| Section / report block | Panel | Run set |
|---|---|---|
| Headline | `BarPlot(metrics=["test_clean/exact"])`, and the same for `test/exact` and `bfcl/exact`, colored by `group` | `job_type = eval`, `official` |
| Accuracy per GPU-hour (the core claim, #2) | `LinePlot(x="compute/gpu_hours", y=["val/exact"], groupby="group", groupby_rangefunc="minmax")` | `job_type = train`, `official` |
| Final accuracy vs compute | `ScatterPlot(x=Config("compute.gpu_hours"), y=SummaryMetric("test_clean/exact"))`. Baselines sit at x = 0. Reference lines (set in the UI) at the teacher's 76.8 and the 65.8 target | eval runs |
| Does it carry over to unseen tools? | `ScatterPlot(x="test_clean/exact", y="bfcl/exact")` | eval runs |
| Where the gap closes | `ParallelCoordinatesPlot` over `test_clean/slice/*/*/exact`, plus `WeavePanelSummaryTable(table_name="slices")` | eval runs |
| Training | `LinePlot` of `train/loss`, `train/mean_token_accuracy`, `train/distill/kl_rev`, `train/grad_norm`, `train/learning_rate` (x = `train/global_step` or `compute/gpu_hours`) | train runs |
| Cost | `LinePlot` of `compute/peak_gpu_mem_gb` and `compute/tokens_per_s`; `BarPlot` of `test/output_tokens_per_s` (serving) | both |
| Runs table | `visible_columns` locked to `config:exp.method.value`, `config:exp.variant.value`, `summary:test_clean/exact`, `summary:bfcl/exact`, `config:compute.gpu_hours.value` | — |

### Screenshots for the repo

- **Report → Download → PDF** is the only documented export of panels
  (clone-and-export-reports.mdx). `pdftoppm -png` turns pages into images.
- A per-panel PNG/SVG export from the panel menu isn't in the docs snapshot *(unverified; there may
  be one in the UI)*. The documented data exports are CSV: runs table
  (customize-run-display.mdx L91-L93), Tables
  ([D/models/tables/tables-download.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/tables/tables-download.mdx)),
  and the Public API ([D/models/track/public-api-guide.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/public-api-guide.mdx)
  L223, L266).
- **Suggestion:** draw the figures committed to `results/` with matplotlib (already a dev
  dependency) from `results/*.json` and the runs' saved files, since `results/` is the source of
  truth. That keeps them reproducible, diffable and themed. Then add **one** screenshot of the
  public report or workspace to the README as evidence that the tracking exists, taken in a
  logged-out private window, which also checks that it's public.

## 6. What eval runs could log now: a per-example predictions Table

- Size is no problem. A run's Table goes through `MAX_ARTIFACT_ROWS = 200000`
  (W/wandb/sdk/data_types/table.py L223-L224, L847; log-tables.mdx L135-L140). 2,000 test rows
  plus 1,000 BFCL rows is 1.5% of that. test-clean is a subset of test, so it's an
  `in_test_clean` column and not more rows. Today's `predictions.jsonl` for the 0.8B test run is
  0.86 MB. Every value is under the 1 MB per-value guideline, unless whole prompts with tool
  specs are added (then a few MB per run *(estimate)*). Storage quotas on Forge are
  *(unverified)*.
- What it's good for: Tables logged under the same key in different runs can be compared in a
  **merged view joined on a key column** (`id`) or side by side
  ([D/models/tables/visualize-tables.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/tables/visualize-tables.mdx)
  L26-L75; [evaluate-models.mdx](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/tables/evaluate-models.mdx)
  L130). Student against teacher, row by row and filtered to `parallel`, is the clearest way to
  show *what* distillation fixed. The README's format-vs-skill question (#2, open question 1) is
  one filter there: `parse_ok = false`.
- Cost and risk: it publishes gold answers and model outputs for the **test** split in a public
  project, and with prompts, the gated dataset's text too. The test labels are already public on
  the Hub behind a terms click, so this is about the dataset's terms, not about leaking test labels
  to training. Decide before logging (open questions).
- Suggested columns: `eval, id, in_test_clean, call_pattern, distractors, argument_source,
  parse_ok, exact, lenient_exact, names, arg_f1, prediction, gold` (BFCL: `category` and
  `bfcl_exact`, without `lenient_exact` and `arg_f1`). No query or tools unless the maintainer
  accepts redistributing them; `id` links back to the dataset.

## Recommendations

> Decided with the maintainer (2026-11-06): one W&B eval run per model or checkpoint, holding
> every eval; the baselines are logged from `results/baselines.json` with `fc-report --wandb`;
> no per-example predictions table (xLAM is gated); README figures are drawn with matplotlib
> from `results/`, plus one W&B screenshot. The dashboard-as-code script and the figures are a
> follow-up issue, done after the first training run. Implemented in #23: the eval key schema
> (`src/fc_distill/eval_tracking.py`). The trainer checklist (c) is for the training tickets.

### (a) Metric and key schema

| Key | Logged by | When | Where |
|---|---|---|---|
| `exp.{method, variant, student, teacher, seed}`, `exp.lora.{r, alpha, dropout, target_modules}`, `exp.optim.{lr, schedule, warmup_ratio, effective_batch, max_length}` | trainer, before `trainer.train()` | init | config |
| `exp.data.{source, filters, n_rows, splits_sha256}` | trainer | init | config |
| `compute.teacher_precompute_gpu_hours` | trainer (from the generate run) | init | config |
| `provenance.*` (`train_provenance()`) | trainer | init | config |
| HF's flat `TrainingArguments` and model config, `peft_config`, `model/num_parameters` | HF callback | `on_train_begin` | config (noise; hide in the dashboard) |
| `train/{loss, grad_norm, learning_rate, epoch, global_step}` | HF | every `logging_steps` | history |
| `train/{entropy, mean_token_accuracy, num_tokens}` | TRL `SFTTrainer` | `logging_steps` | history |
| `train/completions/*`, `train/step_time`, `train/reward*`, `train/kl`, `train/clip_ratio/*` | TRL `DistillationTrainer` / `GRPOTrainer` | `logging_steps` | history |
| `train/distill/{kl_fwd, kl_rev, teacher_token_acc, top1_agree, teacher_entropy}`, `train/examples_seen` | our trainer subclasses (`self._metrics`) | `logging_steps` | history |
| `eval/{loss, mean_token_accuracy, …}` | HF/TRL `evaluate()` on validation, teacher-forced | `eval_steps` | history |
| `compute/{student_gpu_hours, teacher_gpu_hours, gpu_hours, peak_gpu_mem_gb, tokens_per_s}` | `Tracker.log` (`GpuClock`) from `on_step_end` | each logging step | history; summary = last |
| `val/{parse_ok, exact, lenient_exact, names, arg_f1, n}` | trainer's generation eval on validation | each checkpoint | history (x = `compute/gpu_hours`); summary `max,last` |
| `train_runtime`, `train_loss`, `total_flos`, `train_samples_per_second`, `train_steps_per_second` | HF | end of training | summary |
| `best/{checkpoint_step, val_exact, gpu_hours}` | trainer | end of training | summary |
| `model`, `model_revision`, `role`, `exp.*` (copied), `train.{run_id, checkpoint_step, adapter_sha256}`, `compute.gpu_hours` (0 for baselines), `provenance.<eval>.*` | eval aggregation | init | config |
| `<eval>/{metric, metric_lo, metric_hi, n}`, `<eval>/slice/<dim>/<bucket>/{exact, n}`, `bfcl/category/<cat>/{exact, n}`, `<eval>/{seconds, output_tokens, output_tokens_per_s}`, `test_clean/gap_recovered*` | eval aggregation | once | summary |
| `slices` Table (every eval × slice × metric with intervals); optional `predictions` Table | eval aggregation | once | Table |
| `data/<split>/{rows, prompt_tokens_p50, prompt_tokens_p95, prompt_tokens_max, answer_tokens_p95, call_pattern/<b>, distractors/<b>}`, `data/train/removed/<filter>` | `job_type=data` run | once per `splits_sha256` | summary, plus `xlam-splits` artifact metadata |
| `compute/gpu_hours`, `gen/{rows, teacher_parse_ok, kept}` | `job_type=generate` run | once | summary, plus `teacher-outputs` artifact |

`<eval>` ∈ {`test`, `test_clean`, `bfcl`}. Buckets in keys: `single`, `multiple`, `parallel`,
`d0`, `d1_2`, `d3plus`, `from_query`, `not_in_query`. No hyphens, `+` or leading digits in any key
segment.

### (b) Change now, before the baselines are logged for good

1. **Change `log_eval` from a mirror into a mapper:** a pure function `eval_summary(eval_name,
   summary) -> dict[str, float]` that turns an xLAM or BFCL `summary.json` block into the flat
   `<eval>/…` scalars above (intervals split into `_lo` / `_hi`, slice buckets renamed, `test-clean`
   → `test_clean`). It's easy to unit-test, and no list or nested dict reaches the W&B summary.
2. **One W&B run per model:** add `fc-report --wandb` (or `fc-eval --relog DIR …` taking several
   directories) that opens one `job_type="eval"`, `group="baseline"`, `tags=["official"]` run per
   model, named `eval-<model-slug>`. It logs test, test-clean (from `xlam_block` over the test
   predictions, as `report.py` already does) and BFCL together. Config: `model`, `model_revision`,
   `role`, `exp.method="baseline"`, `compute.gpu_hours=0.0`,
   `provenance.<eval>.*`.
3. **Drop per-split W&B logging from `fc-eval --wandb`**, or keep it only for validation and smoke
   runs under `tags=["smoke"]`, so official per-split runs don't sit beside per-model runs and
   duplicate them.
4. Give the `slices` Table an `eval` column, so a run's one table covers all three evals.
5. In `start_run`: store the run id where the aggregated run lives (e.g. `runs/wandb/eval-<slug>/`),
   and document that **deleting a run in the UI means deleting its `wandb_run_id` file** (run ids
   can't be reused). Add `"data"` and `"generate"` to `JobType`. Accept `config=` at `wandb.init`,
   so a run has its config from the start.
6. Optional, if the maintainer accepts what it publishes: the `predictions` Table (§6), without
   prompts.

### (c) Checklist for every training ticket

- [ ] Open the run with `start_run(job_type="train", group=<method>, name="train-<method>-<variant>-s<seed>",
      tags=[…])` and the full `exp.*` / `provenance.*` / `exp.data.*` config **before**
      `trainer.train()`. Use `report_to="wandb"` and set `run_name` to the same name.
- [ ] `include_num_input_tokens_seen="non_padding"` (or rely on TRL's `train/num_tokens`); leave
      `skip_memory_metrics=True`.
- [ ] `GpuClock` around student and teacher GPU work, and `add_teacher_hours` for the amortised
      share of offline teacher jobs. `compute/*` logged from `on_step_end` when `control.should_log`.
      `GpuClock.state()` saved with every checkpoint and restored on resume.
- [ ] `define_metric` **by exact name** for `train/loss` (and any other curve wanted on the
      GPU-hours axis) with `step_metric="compute/gpu_hours"`; `val/*` already have it.
- [ ] Distillation diagnostics appended to `self._metrics[mode]["distill/…"]` in `compute_loss`.
- [ ] Generation-based validation per checkpoint: `val/*` and `val/n` in the **same** `Tracker.log`
      call as `compute/*`.
- [ ] Never call `trainer.predict()` with the default `test` prefix. Official test numbers come
      only from eval runs.
- [ ] Summary at the end: `best/checkpoint_step`, `best/val_exact`, `best/gpu_hours`.
- [ ] `use_artifact` for `xlam-splits` (and `teacher-outputs` for SFT-on-teacher), once those
      artifacts exist.
- [ ] Afterwards, the eval aggregation for the chosen checkpoint, with `train.run_id` and
      `compute.gpu_hours` copied into the eval run's config.
- [ ] TRL's `log_completions=True` for on-policy and GRPO runs (a `completions` Table), with
      `num_completions_to_print` kept small.

### (d) Dashboard as code

- **Where:** `src/fc_distill/dashboard.py` with an `fc-dashboard` entry point (package code lives
  in `src/`; there is no `scripts/` directory). Two builders, `build_workspace(entity, project) ->
  ws.Workspace` and `build_report(...) -> wr.Report`, are pure and return pydantic objects. `main()`
  saves them and writes the returned URLs to `results/wandb-dashboard.json`. On the next run it
  loads them with `from_url` (after stripping Forge's `/wandb` prefix) and updates in place
  instead of creating duplicates.
- **Contents:** one saved view, "Overview", with the sections in §5 and a locked runs-table column
  set. One published Report, "Distilling function calling into Qwen3.5-0.8B": intro text, then
  the headline bars, the accuracy-per-GPU-hour curves, the compute scatter, the slices, the
  training curves and the runs table, each grid with its own run set (`job_type`, `official`).
- **Dependency:** `uv add --optional dashboard wandb-workspaces` (0.4.13). It adds **no new
  transitive packages** (pydantic and wandb are already locked), but it is Public Preview. Import it
  inside functions, as with the GPU libraries.
- **Tests:** build the objects on CPU without saving, and check panel keys against the schema
  constants (so a renamed metric breaks a test, not the dashboard) *(unverified that building
  without `save()` makes no network call; `Report` defaults `entity` from `wandb.Api()`, so always
  pass it)*.
- **Before relying on it:** run a `save()` on a scratch Forge project, check the printed URLs, then
  open the report logged out.
- **Fallback:** build the view and report in the UI, and keep only the matplotlib figures and one
  screenshot in the repo.

### (e) Open questions for the maintainer

1. One W&B eval run per model with all evals (recommended), or keep one per split?
2. Publish per-example predictions for test and BFCL in the public project? With or without the
   query and tool text (xLAM is CC-BY-4.0 but gated)? Same question for an `xlam-splits` artifact
   with only ids.
3. Accept the HF callback's ~105 flat argument keys plus the model config (and hide them), or write our own callback for
   a clean config?
4. Add `wandb-workspaces` (Public Preview, Forge URL handling looks broken) for dashboard as code,
   or build the dashboard by hand once?
5. README figures from matplotlib over `results/` (reproducible) plus one W&B screenshot, or
   screenshots of W&B panels only?
6. Is `test_clean/gap_recovered` with its paired-bootstrap interval the headline number in W&B
   (it depends on #2's open question 2)?
