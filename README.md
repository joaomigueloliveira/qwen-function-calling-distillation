# qwen-function-calling-distillation

Can a small Qwen3.5 model learn function calling from a bigger one, **without labels**? In practice
distillation is what you reach for when there are queries but no gold answers, and a model better
than the one you want to ship. This project trains Qwen3.5-0.8B from a Qwen3.5-2B teacher on one
consumer GPU, and compares the label-free methods with SFT on gold labels, the reference for what
labels buy (ADR 0007).

| Method | | Training uses labels |
|---|---|---|
| **A** | SFT on gold labels | yes: the reference |
| **B** | SFT on teacher outputs (sequence-level distillation) | no |
| **D** | On-policy distillation from the raw student: it writes, the teacher scores every token ([Thinking Machines, 2025](https://thinkingmachines.ai/blog/on-policy-distillation/); [GKD](https://arxiv.org/abs/2306.13649)) | no |
| **C** | Logit distillation on the teacher's top-k token probabilities | no |
| **grounded B** | B on the teacher outputs whose values all appear in the query: a label-free filter | no |
| **filtered B** | B on the teacher outputs that match gold: what the teacher's mistakes cost | yes, as a filter |
| **E** | GRPO, reward checked against gold: does RL add anything on top of SFT? | yes, in the reward |

A, B, D, grounded B and filtered B are done, and B then D (a pipeline) was tried and dropped; C and
E are next. Every result is
reported next to the untrained baselines and A, with two numbers on `test-clean`: the gap recovered
from the raw 0.8B towards the 2B, and the share of A's gain reached without labels.

## Task

[Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k):
given a user query and a set of tool definitions, output the JSON list of calls that answers it.
Scoring is deterministic: no LLM judge.

| Metric | Meaning |
|---|---|
| `parse_ok` | output parses as JSON calls `{name, arguments}`: a list, or a single call object; a missing `arguments` counts as `{}`. If the output has a code fence, only the fence's contents are read, so text around a fence is ignored; without one, any text around the JSON fails. `<think>` blocks are removed first |
| `exact` | same calls as the gold answer (order-insensitive; `5 == 5.0`, `true != 1`) |
| `lenient_exact` | `exact`, but ignoring optional arguments passed at their declared default or `null`, on either side. Many xLAM defaults look like example values copied from API docs, so such calls are the same by the tool spec, not necessarily for the real API. `exact` stays the headline |
| `names` | the right functions were called |
| `arg_f1` | F1 over `(function, argument, value)` triples, as partial credit |

Each xLAM summary also breaks every metric down by call pattern (single / multiple / parallel),
distractor tools (0 / 1-2 / 3+) and argument source (every value written in the query, or not), with
row counts and 95% intervals. Buckets under 300 rows are flagged `too_small` (ADR 0001).

Splits: train (54,561), validation (1,000) and test (2,000), grouped so no query text appears in
two splits, with train filtered of duplicates, code-valued arguments, contradicting labels and gold
answers that also appear in validation or test. The data story, from the quality findings to the label audit and `test-clean`, is in the dataset card:
[`docs/DATASET.md`](docs/DATASET.md).

A second eval, on tools the student never trained on, uses the non-live single-turn part of
[BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (1,000
rows), with specs converted to xLAM's format: [`docs/bfcl.md`](docs/bfcl.md).

## Results

Strict exact match with 95% intervals. Baselines (#10): [`results/baselines.md`](results/baselines.md).
Methods, with every checkpoint, McNemar tests and the teacher-output counts:
[`results/methods.md`](results/methods.md). Checkpoints are chosen on validation; test, test-clean
and BFCL are run once, for the chosen one. Test-clean scores are read from the test run, restricted
to its rows; a separate test-clean run of the same checkpoint can differ by up to 3 rows (batch
nondeterminism in vLLM).

| Model | Training uses labels | GPU-h | Test | Test-clean | BFCL (unseen tools) |
|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B (student) | | 0 | 46.2 [44.0, 48.4] | 54.8 [52.3, 57.2] | 64.5 [61.5, 67.4] |
| Qwen3.5-2B (teacher) | | 0 | 65.1 [63.0, 67.2] | 76.8 [74.7, 78.8] | 82.7 [80.2, 84.9] |
| Qwen3.5-4B (reference) | | 0 | 68.8 [66.7, 70.7] | 81.8 [79.8, 83.6] | 82.7 [80.2, 84.9] |
| **B**: 0.8B, SFT on teacher outputs | no | 3.12 | 71.2 [69.2, 73.1] | 83.4 [81.5, 85.1] | 79.1 [76.5, 81.5] |
| **D**: 0.8B, on-policy distillation | no | 13.54 | 70.5 [68.5, 72.5] | 83.3 [81.4, 85.0] | 82.5 [80.0, 84.7] |
| D at B's budget (step 500) | no | 2.22 | 69.0 [67.0, 71.0] | 81.0 [79.1, 82.9] | 78.7 [76.1, 81.1] |
| B then D (pipeline, step 400) | no | 4.77 | 69.8 [67.8, 71.8] | 81.7 [79.7, 83.5] | 78.9 [76.3, 81.3] |
| **Grounded B**: 0.8B, SFT on teacher outputs whose values are all in the query | no | 3.02 | 71.0 [69.0, 73.0] | 82.9 [81.0, 84.7] | 79.8 [77.2, 82.2] |
| **C**: 0.8B, logit distillation on the teacher's top 20 | no | 4.70 | 70.7 [68.7, 72.7] | 82.5 [80.6, 84.3] | 81.6 [79.1, 83.9] |
| **Filtered B**: 0.8B, SFT on teacher outputs that match gold | yes, as a filter | 2.50 | 74.8 [72.8, 76.6] | 85.7 [83.9, 87.3] | 80.8 [78.2, 83.1] |
| **Filtered C**: 0.8B, logit distillation on teacher outputs that match gold | yes, as a filter | 3.63 | 73.5 [71.5, 75.3] | 84.9 [83.1, 86.6] | 84.1 [81.7, 86.2] |
| **A**: 0.8B, SFT on gold labels | yes | 3.83 | 84.8 [83.1, 86.3] | 93.2 [91.9, 94.3] | 84.5 [82.1, 86.6] |

Each claim below compares two models on the same rows with a paired McNemar test; "Holm" is that
p-value adjusted over every claim listed in [`results/methods.md`](results/methods.md#readme-claims),
which also gives each one under lenient exact. "Not significantly different" means the test can't
separate them, not that they are equal: a paired difference carries about ±2 points.
The Holm values move as claims are added: each new claim widens the family.

- **Without labels, B beats its own teacher on xLAM test-clean.** It recovers 130% of the 22.0-point
  gap from the 0.8B to the 2B (159 rows only B gets right against 52, Holm p < 0.0001) and reaches
  74% of what training on gold labels gains. #2's success criterion, at least 50% of the gap, is
  met. On validation the lead is smaller (70.8 vs 68.0, p = 0.006), and on the validation rows
  where both outputs parse it is not significant (71.8 vs 70.2, p = 0.12). Little of it is output
  format: 43 of B's 159 rows are 2B outputs that don't parse, but closing their brackets turns 29 of
  them into answers with fewer calls than gold, a first call cut short where a second was due; 13
  are real bracket slips. Across all 58 unparsed 2B test-clean outputs, the repair makes 45 parse and
  only one of them right. Dropping the teacher's
  unparsed outputs from B's training data acted as a label-free filter: of the 2,137 dropped, the
  1,210 that parse after the same repair match gold only 34 times. It isn't memorisation either: no
  test-clean query appears in train (9 do if punctuation is ignored; 237 share a template with other
  values), and B's lead is larger on the 109 rows whose tool set train never saw (83.5 vs 68.8).
- **On tools it never trained on, it doesn't beat the 2B, and trails it, though not robustly.** B
  gains 14.6 points on BFCL over the raw 0.8B but stays below the 2B (54 vs 90, p = 0.003), though
  not after Holm (0.054) since C's claims joined the family. Its format holds (parse_ok 98.8 vs 98.9); it loses
  on argument values, in every BFCL category.
- **On xLAM, D and B are not significantly different under strict exact; they differ in kind.** On
  test-clean D scores 83.3 and B 83.4 (60 vs 62 rows, p = 0.93): 129% of the gap to the 2B, 74% of
  A's gain. That tie is two effects cancelling. D is better at choosing functions and completing
  every call: under lenient exact, which ignores optional arguments at their default, it leads
  (68 vs 44, p = 0.03, unadjusted). B is better at xLAM's habit of writing out or leaving out
  defaults (on the 123 test-clean rows that turn on it, B gets 52 and D 40). D also stays closer to
  the teacher: it repeats 43% of the 2B's wrong test-clean answers, B 35%.
- **On BFCL, D reaches 82.5, not significantly different from the 2B (p = 0.93).** It is above B's
  selected checkpoint (62 vs 28, Holm p = 0.009), from argument values and function choice, not
  format. That holds for B's selected checkpoint only: B's BFCL keeps rising after it, to 81.1 at the
  end of its epoch (5.1 GPU-hours), which D doesn't significantly beat (p = 0.12; #70). D took
  13.5 GPU-hours, and D's own BFCL rises with training (78.7 at step 500, p = 0.0002 against step
  3000), so with one seed the data can't tell on-policy training from more training.
- **Per GPU-hour, D shows no advantage.** At B's budget (step 500, 2.2 GPU-hours) D trails on
  test-clean, 81.0 vs 83.4 (p = 0.006), but not after Holm (0.085), and the gap comes from the default
  habit (p = 0.14 without those 123 rows); on BFCL it is not significantly different. Validation, on
  1,000 rows, called it a tie. Along the runs, B's BFCL at the end of its epoch (81.1, 5.1 GPU-hours)
  is above D's at 4.5 and 6.8 GPU-hours (77.8 and 78.2; #70, explanation only). B's GPU-hours are charged for the teacher answering all
  54,561 train rows (1.43 h), though by its selected step it had trained on about 24,000 (≈0.63 h).
- **D's loss on top of B doesn't help.** B then D continues B's selected checkpoint with D's loss
  (lr 5e-5, a fourth of D's, stopped at step 1000 of the epoch, #58). No checkpoint beat B on
  validation. Its pick, step 400 (1.65 GPU-hours on top of B's), is behind B on test-clean under
  strict exact (81.7 vs 83.4, 42 vs 70, p = 0.01), but not after Holm (0.12), and level under lenient (43 vs 47,
  p = 0.75): it moves B towards the teacher's conventions. Of the 70 test-clean rows it turns wrong,
  29 differ only by a default and 31 now give the 2B's answer. On BFCL it is not significantly
  different from B (78.9 vs 79.1, p = 0.91) and below D (Holm p = 0.002). One learning rate, one
  seed: the run wasn't resumed.
- **Labels still buy a lot, partly by learning gold's quirks.** A beats B and D on test and
  test-clean (Holm p < 0.0001); the teacher's outputs agree with gold on only 67.6% of the train rows
  B learns from. But on the 287 test rows whose gold is one reasonable answer among several, A
  matches gold's choice 56% of the time, against 17% for the 2B and 22% for B (ADR 0005's addendum).
  That is why methods are compared on test-clean. On BFCL, A is not significantly different from the
  2B (p = 0.20) or D (p = 0.11). It leaves out required arguments whose description mentions a
  default (51 arguments, about 23 rows), an xLAM gold habit.
- **Training only on the teacher outputs that match gold gains about 2 points, not robustly.**
  Filtered B keeps the 37,696 teacher outputs that lenient-match gold (#60). It leads B on test-clean
  (85.7 vs 83.4, p = 0.007; 140% of the gap, 80% of A's gain), but not after Holm (p = 0.09), and its
  lead shrinks from validation (+5.8) to test (+3.6) to test-clean (+2.3): part of it is gold's
  quirks, which the filter lets through (34% on ambiguous rows, B 22%). It uses fewer rows and fewer
  GPU-hours (2.50 vs 3.12), and is not significantly different from B on BFCL (80.8 vs 79.1,
  p = 0.18), where it chooses functions worse. A is still 7.5 points ahead on test-clean and 3.7 on
  BFCL (Holm p < 0.0001, 0.02); under lenient exact the test-clean gap is 5.2, so about a third is
  the optional-argument convention. The rest likely comes from the 17,000 dropped rows, which hold
  most of the train rows whose values aren't written in the query. Filtered B also differs from B in
  its schedule length (2,356 steps vs 3,277), and each is one seed.
- **A label-free filter on the teacher's outputs doesn't do what filtering against gold does.**
  Grounded B keeps only the 33,918 teacher outputs that call offered tools with their required
  arguments and write every value as it appears in the query (#74): they agree with gold 86.8% of the
  time under lenient exact, against 71.9% for all of B's data. Its selected step, 1500, is not
  significantly different from B on test-clean (82.9 vs 83.4, 89 vs 97, p = 0.61) or BFCL (79.8 vs
  79.1, p = 0.58); under lenient exact B leads (97 vs 69, p = 0.04, unadjusted). Filtered B beats it
  on test-clean (125 vs 80, Holm p = 0.03). The tie is two effects cancelling (test-clean, computed
  from the saved predictions in #75). On the 1,261 rows whose values are all in the query it gains on
  B (90.5 vs 88.4, p = 0.01), level with filtered B (89.8). On the 364 rows that need a value the
  query doesn't spell out it falls behind (56.6 vs 65.9, p = 0.0004; filtered B 71.2), since the rule
  dropped 91% of B's training rows of that kind. Its epoch is shorter too (2,120 steps vs 3,277), and
  each is one seed. It doesn't become a separate headline candidate: it ties B at the same cost
  (3.02 vs 3.12 GPU-hours) and trails it under lenient, so B stays the label-free reference, and
  Phase 4 chooses among the methods test-clean can't separate (ADR 0009).
- **Soft targets don't beat hard ones on xLAM; on unseen tools C leads B, not robustly.** C trains on
  B's own 52,424 rows and text, with the 2B's top-20 log-probs at every token instead of the one
  token it wrote (forward KL, ADR 0010, #87). Its selected step, 1500, is not significantly different
  from B on test-clean (82.5 vs 83.4, 50 vs 64, p = 0.22; 126% of the gap, 72% of A's gain), under
  lenient (47 vs 55, p = 0.49), or from D (p = 0.32). On BFCL it leads B's selected checkpoint (81.6
  vs 79.1, 54 vs 29, p = 0.008), but not after Holm (0.10), B's own BFCL reaches 81.1 by the end of
  its epoch (#70), and C's BFCL along its own run wasn't measured; it is level with the 2B (82.7,
  55 vs 66, p = 0.36). It copies the 2B's BFCL errors no more than B (63 vs 64 of 162); it agrees
  with the 2B more (74.7% vs 71.7%), as accuracy alone would make it, as with D.
  The soft targets carry the 2B's habits where it hedges. C writes fewer defaults out (37.4% on the
  123 rows that turn on it, B 42.3%, the 2B 20.3%), and it fences 5.5% of its test-clean answers in
  ``` (B 0%, the 2B 3.6%), a habit B's re-serialized targets drop on the 6,617 rows where the 2B
  had one (#85). The fence costs C rows: its training text never goes on after a fence, and 7 of its
  90 fenced answers open a call object but close a list, so they don't parse. C fails to parse 13
  test-clean answers to B's 6 (parse_ok 99.2 vs 99.6), and 10 of its 64 losses are those; on the
  rows where both parse it is 49 vs 54 (p = 0.69). Its pick is also its worst-parsing checkpoint on
  validation (98.3, the others 98.9 to 99.3). It costs more: the 2B's top-20 pass adds 1.56
  GPU-hours to the 1.43 B pays for the teacher's answers, so its pick sits at 4.70 GPU-hours and no
  checkpoint falls within B's 3.12. It doesn't become a separate headline candidate: it ties B on
  test-clean at higher cost (ADR 0009).
- **On clean rows, soft targets lift unseen tools robustly.** Filtered C is C's loss on filtered B's
  37,696 rows (#88). Its selected step, 1000, beats filtered B on BFCL (84.1 vs 80.8, 64 vs 31,
  Holm p = 0.02), also on the rows where both parse (62 vs 30, p = 0.001), and is level there with
  A (84.5, p = 0.78) and the 2B (82.7, p = 0.28): the only robust soft-against-hard gain in the
  project, and it needs labels to pick the rows. On test-clean it is not significantly different
  from filtered B (84.9 vs 85.7, 70 vs 82, p = 0.37). Its fences cost it more than C's: 24 of its 47
  fenced test-clean answers don't parse, it fails to parse 29 rows to filtered B's 3, and on the
  rows where both parse it is ahead (69 vs 57, p = 0.33). It copies a few more of the 2B's BFCL
  errors (42 vs 35 of 162). Its pick is again its worst-parsing checkpoint on validation (97.2,
  the others 98.7 to 99.3). It costs 3.63 GPU-hours to filtered B's 2.50.

B's, grounded B's, filtered B's, C's and filtered C's GPU-hours include the 1.43 h the teacher took
to answer the train split, and C's also the 1.56 h of the 2B's top-20 scoring pass (#85), filtered
C's the 1.12 h share of it its rows needed; D's
include its own sampling and the teacher's grading, inside every step; B then D's include B's.
Checkpoints are each run's best on validation (ADR 0006); the pick within a run is noisy, so a claim
that rests on it names the checkpoint (ADR 0009). Next: GRPO, E, with labels.

Training cost (#36): one epoch of LoRA SFT over the full train split takes about 3.9 GPU-hours on the
RTX 3060 (0.072 GPU-hours per 1,000 rows, 4.6 GiB peak), after length grouping removed most of the
padding. The settings and the sweeps behind them: [`results/sft-cost.md`](results/sft-cost.md).

Serving cost (#63): in vLLM on the same card, **the 0.8B answers 2.4× faster than the 2B one
request at a time** (median latency 0.27 s against 0.64 s), **and serves 1.9× cheaper with every
test-clean prompt batched** (46.7 against 89.3 GPU-seconds per 1,000 requests, each at its best cap
on concurrent requests, vLLM's default scheduler). Batching narrows the gap because the prompts are
long and the answers short. Every method's model is the
0.8B with a LoRA merged in, so this should hold for each of them if its answers are as long; the
final student is re-run as a check. Settings and every run: [`results/serving.md`](results/serving.md).

Training curves and eval runs, the baselines included, are logged to the public W&B project
[joaomigueloliveira30/qwen-function-calling-distillation](https://forge.coreweave.com/wandb/joaomigueloliveira30/qwen-function-calling-distillation).
`results/` stays the source of truth; W&B mirrors it.

## Limitations

Shortcuts taken on purpose, why, and what would make us revisit them.

- **The LoRA settings and learning rate are untuned.** Every SFT method shares one config (r=16,
  α=32, lr 2e-4), chosen as common defaults for the cost run. A sweep costs about 4 GPU-hours per
  point and would have to be repeated for each method to stay fair (ADR 0006). Revisit if a method's
  validation curve looks learning-rate-limited, e.g. its best checkpoint is an early one.
- **One seed per method.** Comparisons rest on paired McNemar tests over the same test rows, not on
  seed-to-seed variance (ADR 0006). Revisit if two methods land within a point or two on validation.
- **Method D's learning rate is untuned.** It reuses the SFT config (lr 2e-4) for a different loss,
  (ADR 0008). Revisit if its validation curve collapses or is still climbing at the end of the
  epoch. B then D tried one rate, 5e-5, for 1,000 of the epoch's 3,410 steps.
- **"Label-free" is about training only.** Checkpoints are chosen on labelled validation and scored on
  labelled test, as in practice, where a small labelled eval set is the norm (ADR 0007).
- **One small teacher.** The 2B is right on about 65% of train rows; a stronger teacher would likely
  lift every label-free method. The results are about this pair (ADR 0004).
- **Validation is not audited, and the pick within a run is noisy.** Checkpoints are chosen on raw
  validation labels (ADR 0005). 1,000 rows detect a difference of about 2.4 points at 80% power,
  so they can't separate most checkpoints of a run, nor B from D. Test numbers stay fair for the
  checkpoint picked, so methods are compared on `test-clean`, never on validation (ADR 0009).
- **Methods learn gold's quirks at different rates.** On the 287 ambiguous test rows A matches
  gold's arbitrary choice 56% of the time, filtered B 34%, B 22%, the 2B 17%. Raw test rewards that;
  `test-clean` leaves those rows out (ADR 0005).
- **B's lead over its teacher is unexplained in size.** It is +2.8 points on validation and +6.1 on
  test, about 2.5 standard errors apart, though both are random samples of the same data.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/), an NVIDIA GPU for inference/training, and access to the
gated dataset (accept the terms on the dataset page, then `uv run hf auth login`).

```bash
uv sync --extra inference         # base + vLLM
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split validation
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split test-clean   # official numbers: test and test-clean
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split bfcl         # unseen tools
uv run fc-eval --rescore runs/eval-test-Qwen_Qwen3.5-0.8B       # re-score a saved run, no GPU
uv run fc-report                  # results/baselines.{json,md} from the test and BFCL runs
uv run --no-sync fc-report --methods   # results/methods.{json,md}: each method's checkpoints
uv run fc-serve --model Qwen/Qwen3.5-0.8B --max-num-seqs 256   # serving cost -> runs/serve-*
uv run --no-sync fc-serve --report     # results/serving.{json,md}: the 0.8B against the 2B

uv sync --extra inference --extra train   # + TRL, PEFT, flash-linear-attention
uv run --no-sync fc-sft --n-rows 1000     # method A: LoRA SFT on gold labels -> runs/train-sft-gold-*
uv run --no-sync fc-teacher-topk --teacher-run runs/eval-train-Qwen_Qwen3.5-2B   # C's teacher pass
```

### Experiment tracking

Logging to [Weights & Biases](https://forge.coreweave.com/wandb) is off unless asked for, so tests,
CI and offline runs need no account. To turn it on, log in once with `uv run wandb login`, then pass
`--wandb`:

```bash
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split validation --wandb   # one eval run per split
uv run fc-report --wandb          # the official baselines, one run per model, from results/
```

The official baseline runs (`summary.json` and `predictions.jsonl`) are backed up in a private
W&B project, because they hold gated xLAM rows:
`joaomigueloliveira30/qwen-function-calling-distillation-backup`, artifact `official-baseline-runs`.

`WANDB_MODE=offline` keeps the run local (`wandb sync` uploads it later), and `WANDB_PROJECT` /
`WANDB_ENTITY` point it at another project. Trainers log through the same helper,
`fc_distill.tracking`, with GPU-hours (the teacher's included) as the x-axis. Every eval run uses
the same flat keys (`test_clean/exact`, `bfcl/slice/simple/exact`, ...), so runs compare in one
table. W&B findings, the key schema and what trainers log: [`docs/wandb.md`](docs/wandb.md),
[`docs/wandb-dashboard.md`](docs/wandb-dashboard.md).

## Development

```bash
uv sync                           # base + dev tools, no GPU needed
uv run pre-commit install
uv run pytest

# re-run the dataset notebook (writes data/splits.json; needs the HF login above)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset.ipynb
```

Linting and formatting use ruff, type checking uses mypy (strict), and both run in pre-commit and CI.

## Layout

```
src/fc_distill/
  data.py       dataset loading (pinned revision), committed splits, prompt
  quality.py    detectors for label and schema problems
  splits.py     train / validation / test construction and train filters
  eda.py        statistics and plots for the dataset notebook
  scoring.py    strict, deterministic scorer
  evaluate.py   vLLM eval CLI (fc-eval)
  audit.py      test label audit and the test-clean subset (fc-audit)
  bfcl.py       BFCL unseen-tools subset: loading, spec conversion, scoring
  slices.py     per-slice scores: call pattern, distractor tools, argument source
  report.py     baselines report with intervals and McNemar tests (fc-report)
  methods.py    method results: checkpoint selection, official scores, teacher-output counts (fc-report --methods)
  sft.py        methods A, B and C: LoRA training on gold labels, teacher outputs or the teacher's top 20 (fc-sft)
  teacher_topk.py  method C's teacher pass: the 2B's top-20 log-probs on B's training tokens (fc-teacher-topk)
  kd.py         method C's loss: forward KL over the stored top 20 plus a rest entry
  opd.py        method D: on-policy distillation with TRL's DistillationTrainer (fc-opd)
  serving.py    serving cost in vLLM: throughput, latency, memory; student vs teacher (fc-serve)
  tracking.py   W&B tracking shared by trainers and evals; GPU-hour accounting, provenance
  eval_tracking.py  how eval results are laid out in W&B (flat keys, slices table)
notebooks/      01_dataset.ipynb: the dataset story and the split
tests/          unit tests (CPU only)
data/           committed split ids and the audit labels
docs/           dataset card, ADRs, the label-audit rubric, the BFCL and W&B notes
results/        results worth keeping (label audit, baselines, methods, serving)
```

## License

Apache-2.0. Qwen3.5 models are Apache-2.0; the xLAM dataset is CC-BY-4.0.
