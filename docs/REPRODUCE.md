# Reproducing the experiments

Every experiment in the [essay](https://joaomigueloliveira.github.io/qwen-function-calling-distillation/),
from the split to the site, in the order it ran. Each command is complete: run it from the
repository root on one NVIDIA GPU (the runs here used an RTX 3060, 12 GB). GPU-hours are the
measured cost of each step on that card.

## Setup

```bash
uv sync --extra inference --extra train --group site   # vLLM, TRL/PEFT and Quarto
uv run --no-sync hf auth login                         # after accepting xLAM's terms on the Hub
uv run --no-sync wandb login                           # required for the method runs, see below
export WANDB_ENTITY=<you> WANDB_PROJECT=<your project>   # defaults point at the maintainer's
```

Weights & Biases is optional for evaluating a Hub model, but not for the methods: `fc-eval --adapter`
reads each checkpoint's GPU-hours back from its training run's W&B history, so every training run
below is launched with `--wandb`, as the committed runs were.

Run everything with `--no-sync`: plain `uv run` or `uv sync` re-syncs the environment to the CPU
tools and uninstalls vLLM. Outputs go to `runs/`, which is gitignored; in this repo it is a symlink to
`$FC_RUNS_DIR` (default `~/ml-runs`). Trainers and `fc-eval` refuse to start from a tree with
uncommitted changes, so every run records the commit that produced it (`--allow-dirty` overrides).

## Repository structure

```
src/fc_distill/
  data.py           xLAM at a pinned revision, the committed splits, the prompt every model sees
  quality.py        detectors for label and schema problems in the data
  splits.py         train / validation / test construction and the train filters
  eda.py            statistics and plots for the dataset notebook
  scoring.py        the strict, deterministic scorer (exact, lenient exact, names, arg F1)
  slices.py         per-slice scores: call pattern, distractor tools, argument source
  bfcl.py           BFCL's unseen-tools subset: loading, spec conversion, scoring
  evaluate.py       fc-eval: greedy eval in vLLM of a model or a LoRA checkpoint, on any split
  audit.py          fc-audit: the test label audit and the test-clean ids
  sft.py            fc-sft: methods A, B and C (LoRA SFT on gold, teacher outputs, or top-20 log-probs)
  teacher_topk.py   fc-teacher-topk: method C's teacher pass, the 2B's top-20 log-probs
  kd.py             method C's loss: forward KL over the stored top 20 plus a rest entry
  opd.py            fc-opd: method D, on-policy distillation with TRL's DistillationTrainer
  checkpoints.py    reading training runs and their checkpoints
  report.py         fc-report: the baselines report, intervals and McNemar tests
  methods.py        fc-report --methods: checkpoint selection, official scores, the paired claims
  stats.py          Wilson intervals, McNemar, Holm
  serving.py        fc-serve: serving cost in vLLM, the student against the teacher
  figures.py        fc-figures: the figures' data and SVGs, from results/methods.json
  examples.py       fc-examples: the essay's worked examples, picked by rule
  probe.py          fc-probe: the essay's measured token figures
  story.py          the numbers the essay quotes, read from results/ when it builds
  tracking.py, eval_tracking.py   Weights & Biases logging and GPU-hour accounting
notebooks/01_dataset.ipynb   the dataset story and the split
data/               split ids (splits.json), the audit labels (audit/), the token probes (probe/)
results/            results worth keeping: baselines, methods, label audit, serving, examples
docs/               dataset card, ADRs (adr/), the audit rubric, BFCL and W&B notes, figures
site/               the essay and its reference pages (Quarto)
tests/              unit tests, CPU only
```

## 1. Data and the split

```bash
uv run --no-sync jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset.ipynb
                                                   # data/splits.json: train / validation / test
```

The notebook applies the train filters and groups rows by query ([ADR 0001](adr/0001-train-validation-test-split.md),
[dataset card](DATASET.md)). `data/splits.json` is committed, so this step only re-derives it.

## 2. Baselines and the label audit

```bash
for m in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B; do
  for s in validation test bfcl; do uv run --no-sync fc-eval --model $m --split $s; done
done
for s in test bfcl; do uv run --no-sync fc-eval --model Qwen/Qwen3.5-4B --split $s --max-num-seqs 24; done

uv run --no-sync fc-audit prepare   # data/audit/test_labels.csv: the rows any baseline gets wrong
                                    # (verdicts by hand, by the rubric in docs/label-audit.md)
uv run --no-sync fc-audit build     # the test-clean ids, into data/splits.json
uv run --no-sync fc-report          # results/baselines.{json,md}
```

Test-clean is scored from the test run, restricted to its ids; a separate `--split test-clean` run
of the same model can differ by up to 3 rows (batch nondeterminism in vLLM), so the reported numbers
always come from the test run. The audit's verdicts are committed in
`data/audit/test_labels.csv` ([ADR 0002](adr/0002-test-labels-never-edited.md)).

## 3. The teacher's answers (B, C and their variants train on them)

```bash
uv run --no-sync fc-eval --model Qwen/Qwen3.5-2B --split train --max-model-len 5120   # 1.43 GPU-h
uv run --no-sync fc-teacher-topk --teacher-run runs/eval-train-Qwen_Qwen3.5-2B       # C only, 1.56 GPU-h
```

## 4. The methods

Every method shares one setup ([ADR 0006](adr/0006-shared-training-setup.md)): LoRA r16/α32, lr 2e-4,
4 rows × 4 accumulation, one epoch, seed 0, a checkpoint every 500 steps. The optimiser settings are
the trainers' defaults; the SFT commands add length grouping and the checkpoint interval.

| Method | Command | Pick | GPU-h |
|---|---|---:|---:|
| A, SFT on gold labels | `uv run --no-sync fc-sft --group-by-length --save-steps 500 --wandb` | 3411 | 3.8 |
| B, SFT on teacher outputs | `uv run --no-sync fc-sft --targets teacher --teacher-run runs/eval-train-Qwen_Qwen3.5-2B --group-by-length --save-steps 500 --wandb` | 1500 | 3.1 |
| Filtered B | B's command `--filter lenient` | 1000 | 2.5 |
| Grounded B | B's command `--filter grounded` | 1500 | 3.0 |
| C, logit distillation | `uv run --no-sync fc-sft --targets teacher-topk --teacher-run runs/eval-train-Qwen_Qwen3.5-2B --teacher-topk runs/topk-20-eval-train-Qwen_Qwen3.5-2B --group-by-length --save-steps 500 --wandb` | 1500 | 4.7 |
| Filtered C | C's command `--filter lenient` | 1000 | 3.6 |
| D, on-policy distillation | `uv run --no-sync fc-opd --save-steps 500 --wandb` | 3000 | 13.5 |
| B then D | `uv run --no-sync fc-opd --init-adapter runs/train-sft-teacher-r16-b4x4-n52424-gl-s0/checkpoint-1500 --lr 5e-5 --save-at 100,200,300,400,500,1000 --stop-at 1000 --wandb` | 400 | 4.8 |

GPU-hours are at the pick and include the teacher's passes each method needs ([ADR 0008](adr/0008-method-d-setup.md)
for D's accounting). Each run writes its adapter checkpoints and `cost.json` to `runs/train-*`.

Then, for each run, score every checkpoint on validation, let the selection rule pick one (best
strict exact, ties to the earlier step), and run the pick once on test and BFCL:

```bash
RUN=runs/train-sft-teacher-r16-b4x4-n52424-gl-s0     # e.g. B
for c in $RUN/checkpoint-*; do uv run --no-sync fc-eval --adapter $c --split validation --wandb; done
uv run --no-sync fc-report --methods                  # results/methods.{json,md}: every pick
uv run --no-sync fc-eval --adapter $RUN/checkpoint-1500 --split test --wandb
uv run --no-sync fc-eval --adapter $RUN/checkpoint-1500 --split bfcl --wandb
uv run --no-sync fc-report --methods                  # the official scores and the paired claims
```

Two more evals feed the reported claims:

- **D at B's budget**: D's only checkpoint within B's GPU-hours, step 500, is also run on test and
  BFCL, for the claims "D@500 against B" ([ADR 0008](adr/0008-method-d-setup.md)):
  `fc-eval --adapter runs/train-opd-r16-b4x4-n54561-s0/checkpoint-500 --split test --wandb`, then
  `--split bfcl`.
- **The adapter check on BFCL**: `fc-eval --adapter` fails if the adapter changes fewer than 25% of
  32 seeded answers against the base model, to catch an adapter that never loaded. Four BFCL evals
  ran with `--adapter-min-changed 0.15`, because these adapters change fewer BFCL answers (6 or 7 of
  32) while loading correctly on every other split (#64, #76): D at steps 3000 and 500, B then D at
  step 400, filtered B at step 1000. Each summary records the threshold it used.

BFCL along the whole run (B, D, filtered B and filtered C, explanation only) is the same `--split bfcl --wandb`
on every checkpoint ([ADR 0009](adr/0009-selection-stays-methods-compared-on-test-clean.md)).

## 5. Serving

```bash
for m in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B; do
  for k in 16 64 256; do uv run --no-sync fc-serve --model $m --max-num-seqs $k; done
done
uv run --no-sync fc-serve --report   # results/serving.{json,md}
```

## 6. The write-up

```bash
uv run --no-sync fc-figures          # docs/figures/: the figures' data and SVGs
uv run --no-sync fc-examples         # results/examples.json: the worked examples
uv run --no-sync fc-probe            # data/probe/: the measured token figures (GPU)
uv run --no-sync quarto render site  # site/_site/: the essay, every number from results/ and data/
```

`uv run --no-sync pytest` and `uv run --no-sync pre-commit run --all-files` check the code on CPU.
