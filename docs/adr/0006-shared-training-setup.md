# 0006. One untuned training setup shared by the SFT methods

Date: 2026-11-07 · Status: accepted · Amended 2026-11-08 by ADR 0007 (filtered B always runs)

## Context

Methods A (SFT on gold labels) and B (SFT on teacher outputs) differ only in the targets. The
comparison is fair only if everything else is held fixed. The first LoRA run (#36,
[results](../../results/sft-cost.md)) measured one epoch over the 54,561 train rows at about
3.9 GPU-hours on the RTX 3060, so each extra configuration tried costs an evening of the one GPU.
The 2B teacher answered 1,000 rows in 95 s in the ADR 0004 scouting run, so teacher outputs for the
whole train split cost about 1.5 GPU-hours.

## Decision

**Training config, frozen and untuned.** Every SFT method uses the settings of the first full method
A run (`train-sft-gold-r16-b4x4-n54561-gl-s0`), which is the official method A run:

| Setting | Value |
|---|---|
| LoRA | r=16, α=32, dropout 0, `all-linear` |
| Optimiser | AdamW, lr 2e-4, cosine, 3% warmup |
| Batch | 4 rows × 4 accumulation = 16 rows per step |
| Length | `--max-length 4096`, `--group-by-length` |
| Length of training | one epoch |
| Seed | 0, one run per method |

These are common LoRA defaults picked for the cost run, not tuned. A sweep would cost GPU-hours, and
to stay fair every method would need its own.

**Checkpoint selection.** An adapter is saved every 500 steps and at the end. Every checkpoint is
scored on validation with strict exact against gold, whatever the method trained on. The best one,
with ties going to the earlier checkpoint, is the method's model on test, `test-clean` and BFCL.
The rule stays as is despite validation's noise, and methods are compared on `test-clean`, not
picked on validation (ADR 0009).

**GPU-hours.** A method's cost is its optimizer-step time plus, for methods that use the teacher,
the teacher's generation time (and, for C, its scoring pass: ADR 0010). Validation scoring is reported separately as selection overhead:
every method pays about the same.

**Teacher outputs (method B).** The 2B answers all train rows once:
- with `fc-eval`'s decoding: greedy, thinking off (ADR 0003), the eval prompt, the same 1,024-token cap as
  the baselines, through vLLM;
- written to `runs/` with provenance (model, revision, settings) and backed up only to the private
  W&B project, because the outputs contain the gated xLAM prompts.

**Method B variants.**
- **Unfiltered B** trains on every teacher output that parses, right or wrong. Outputs that don't
  parse are dropped and counted.
- **Filtered B** keeps only outputs that lenient-match gold (ADR 0005). It uses the labels, so it is
  an analysis run in the with-labels regime, run after the label-free methods (ADR 0007; it first
  ran only if unfiltered B didn't beat A). The teacher outputs are shared, so it costs only its
  training run.

## Considered options

- **Sweep rank and learning rate on validation first.** Rejected for now: about 4 GPU-hours per
  point, repeated per method, before the first comparison exists.
- **Several seeds per method.** Rejected for now: A-vs-B claims use paired McNemar on the same test
  rows. Revisit if A and B land within a point or two on validation.
- **Count validation scoring in a method's GPU-hours.** Rejected: it is the same for every method and
  only adds noise to the curves.
- **Run filtered B unconditionally.** First deferred; adopted in ADR 0007, as an analysis run.

## Consequences

- Results are conditional on one untuned config and one seed. The README lists both as limitations.
- A method that is sensitive to the learning rate may look worse than it is.
- C and D reuse the LoRA config, with their own settings in ADR 0010 and ADR 0008. E, a stretch goal
  (ADR 0007), would too.
