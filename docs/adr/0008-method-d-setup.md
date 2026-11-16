# 0008. Method D: on-policy distillation from the raw student, one epoch

Date: 2026-11-08 · Status: accepted · Amended 2026-11-09 (D at B's budget; B then D)

## Context

Method D is the second label-free method (ADR 0007). The student samples its own answers, and the
teacher grades every token: the loss is the reverse KL, KL(student ‖ teacher), on the student's
samples ([Thinking Machines, 2025](https://thinkingmachines.ai/blog/on-policy-distillation/);
[GKD](https://arxiv.org/abs/2306.13649)).

Phase 2 left B at 83.4 on `test-clean`, above its 2B teacher, but below it on BFCL (79.1 vs 82.7),
where it gets argument values wrong (`results/methods.md`). TRL 1.17, which the project pins, ships
`trl.DistillationTrainer`: always on-policy, generalized JSD with `beta=1.0` for reverse KL, LoRA
through PEFT, student sampling in vLLM colocated with training, and the loss computed chunk by
chunk so neither model's full logits are ever held at once.

## Decision

| Setting | Value |
|---|---|
| Trainer | TRL's `DistillationTrainer`, in a new `fc-opd` CLI (#53) |
| Loss | reverse KL (`beta=1.0`), exact over the vocabulary at every answer position |
| Start | the raw 0.8B with a fresh LoRA, r16/α32 all-linear (ADR 0006) |
| Teacher | Qwen3.5-2B, bf16, in the same process |
| Data | all train-split prompts, never gold (a seed-0 subset only under the cap); one sample per prompt |
| Sampling | temperature 1.0, thinking off, the eval prompt, at most 1,024 tokens |
| Optimiser | lr 2e-4, cosine, 3% warmup, 16 rows per step (ADR 0006, untuned) |
| Length | one epoch over the train prompts, a checkpoint every 500 steps, as A and B; capped at about 12 GPU-hours |
| Sampling engine | vLLM colocated with sleep mode: it fitted next to training and the teacher (#53) |

- **GPU-hours start at 0** and every step counts the student's sampling, the teacher's forward pass
  and the update, so the teacher's cost is inside D's curve.
- **The cap.** If the smoke run (#53) puts an epoch above about 12 GPU-hours (one night), D trains
  on a seed-0 subset sized to 12 GPU-hours instead, with the cosine schedule over that subset. The cap
  is fixed before the run, from throughput, never from validation scores.
- **Selection** follows ADR 0006: every checkpoint scored on validation with strict exact after the
  run, the best wins, ties to the earlier, then test, `test-clean` and BFCL for that one only. One
  seed.
- **At B's budget** (amended 2026-11-09): each label-free method after B is also reported at its
  best checkpoint on validation within B's GPU-hours, with test, `test-clean` and BFCL for that
  checkpoint too. It replaces "the checkpoint nearest B's 3.1 GPU-hours, on validation only",
  where 1,000 validation rows called D and B a tie (p = 0.50). The rule was set after the
  selected checkpoint's test and `test-clean` runs, and before step 500's test runs and any BFCL
  run of D. It had nothing to choose: step 500 is D's only checkpoint within 3.12 GPU-hours, the
  checkpoint the old rule already named. Its result, behind B on `test-clean` (p = 0.006), is in
  `results/methods.md`. D is still reported as its selected checkpoint.
- **Monitoring during the run**, watch only: the loss, the parse rate of the student's rollouts
  and, at each checkpoint, a probe: greedy answers on a fixed 200-row validation subset, and on all
  of validation once, at step 1000. It never stops the run or chooses the checkpoint; the test split
  is never touched. The probe uses HF `generate`, not the colocated vLLM engine, which samples at
  the training temperature and holds no weights while asleep, so its scores can differ from
  `fc-eval`'s by a few rows.
- **B→D**, continuing B's adapter with D's loss for one epoch, is a **pipeline**, reported as
  "B then D", not as method D. It becomes the next run if D doesn't beat B on validation; otherwise
  it stays a stretch goal. D didn't, so it ran (#58; see Consequences). This amends ADR 0007, where
  D started from B.

## Considered options

- **Start from B, as Thinking Machines start from an SFT checkpoint.** Rejected as method D: A, B
  and C start from the raw student, and from B the curve can't show what D does on its own or per
  GPU-hour. Kept as the B→D pipeline.
- **Thinking Machines' sampled-token loss**, one teacher log-probability per token. Rejected: a
  hand-written sampling loop and update, where TRL computes the exact per-position KL in chunks.
- **A GPU-hour budget (3.8, one SFT epoch's cost) instead of an epoch.** Rejected: the cosine
  schedule needs the total step count up front, so a budget means guessing it from throughput; and
  the per-checkpoint GPU-hours already give the matched-cost point. An epoch keeps D on A's and B's
  steps and rows.
- **Early stopping on validation** (stop at 1500 if no better than 1000). Rejected: it would have
  stopped A at 80.8, outside the interval of its final 84.7; adjacent checkpoints rarely differ
  significantly on 1,000 rows; and it needs scoring during training. A patience rule (two
  checkpoints without a new best) would not have stopped A, but still needs in-run scoring.
- **Half the learning rate.** Only argued for when refining B's adapter; from a fresh adapter the
  shared config holds.

## Consequences

- The smoke run (#53) put an epoch at about 12.3 GPU-hours, so D got the full epoch. It ran slower
  than the smoke run (about 15 s a step against 13) and cost 15.4 GPU-hours (#54); the cap was
  never meant to cut a run short once started.
- On BFCL's first 32 rows, the ones fc-eval reruns without the adapter, D's changes 6 answers
  (B's: 16), under the 25% load check. Over all 1,000 BFCL rows it changes 515 (B: 528), and 69%
  of validation and test rows, so it is loaded; its BFCL runs pass an explicit
  `--adapter-min-changed 0.15` (#64), set before any of their scores existed. B then D (6 of 32) and filtered B
  (7 of 32; ADR 0007) failed the same check on BFCL only, with 19–25 of 32 changed on the other
  splits, and took the same 0.15, chosen before any of their BFCL scores existed (B then D's test
  scores did). Over all 1,000 BFCL rows they change 509 and 545 answers from the raw 0.8B's.
- The learning rate is untuned for a loss it wasn't chosen for. The README lists it.
- **D's validation dip at step 1000 (65.7, against 70.0 at step 500) is missing calls**, not a
  broken run: 52 validation outputs stop before a call gold makes, against 16 at step 1500. It is
  temporary (42 of the 76 rows lost from step 500 are back by step 1500), comes near the peak
  learning rate (about 1.9e-4), and doesn't show in the training metrics: the loss keeps falling and
  rollout parse_ok stays above 97% (external review 2026-11-09, s6 and s7).
- Temperature-1.0 samples from the raw student break the JSON more often than greedy decoding
  (parse_ok 84.5% on validation, greedy); those tokens are still graded by the teacher.
- **B then D** (#58) ran at lr 5e-5 (LoRA continuing an SFT adapter; on-policy rates in the
  literature sit well below the SFT ones), with the cosine schedule over the full epoch and a stop at
  step 1000, saving at 100–500 and 1000 to see whether it gained fast. No checkpoint beat B on
  validation (best: step 400, 70.4 vs 70.8, p = 0.69; the other five significantly worse). Step 400,
  fixed as its checkpoint before any test run (official numbers were run anyway, to report the
  pipeline), is behind B on `test-clean` (81.7 vs 83.4,
  p = 0.01) and level on BFCL (78.9 vs 79.1). Not resumed. The probe's full-validation score at
  step 1000 (70.3) was 19 rows above fc-eval's (68.4), more than the few rows expected above.
