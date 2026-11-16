# 0007. Label-free distillation is the main question; SFT on labels is the reference

Date: 2026-11-08 · Status: accepted · Amended 2026-11-08 by ADR 0008 (D starts from the raw student); 2026-11-09 (filtered B before C; grounded B); 2026-11-13 by ADR 0010 (filtered C); 2026-11-16 (E becomes a stretch goal); 2026-11-16 by ADR 0009 (the claim reads "past the teacher")

## Context

The first two Phase 2 runs (#42, #46) changed what the comparison can say:

| Validation, strict exact | Score | GPU-h |
|---|---:|---:|
| raw 0.8B (student) | 46.6 | 0 |
| raw 2B (teacher) | 68.0 | 0 |
| B, SFT on teacher outputs, best checkpoint (1500) | 70.8 | 3.12 |
| A, SFT on gold labels, best checkpoint (3411) | 84.7 | 3.84 |

A beats its own teacher by 17 points, so "gap recovered" towards the 2B can't separate A from
anything. B lands near its teacher, as sequence-level distillation students usually do: the teacher
is right on 64.9% of train rows (#44). Comparing A and B head to head mostly measures label quality.

In practice distillation is used when there are no labels: there are queries, and a model better
than the student. That is the question worth answering here.

## Decision

**Two regimes.** Every method is labelled by whether its training uses gold labels.

| Regime | Methods | Role |
|---|---|---|
| Label-free | B (unfiltered), grounded B, D, C | The main question and the README's headline claim |
| With labels | A, filtered B, filtered C, E | The reference: what labels buy |

- **Order:** B → D → C → filtered B → E. The label-free methods run first; the with-labels runs follow.
  Amended 2026-11-09: filtered B ran before C (#60), since it needed no new code and C's design
  wasn't settled; the results table, not the run order, tells the story.
- **D starts from the raw student**, like A, B and C (ADR 0008; first decided as "from B's
  selected checkpoint", which replaced #35's "the better of A and B"). Starting from A would bring
  labels into a label-free method. B then D is run as a pipeline, not as method D.
- **C, logit distillation, is a main method**, run after D. It is the off-policy, dense-signal point
  between B and D, which tells whether D's gain comes from the teacher's probabilities or from
  training on the student's own outputs.
- **Filtered B always runs** (amends ADR 0006, where it ran only if B lost to A). It is an analysis
  run: filtered B − B is what the teacher's mistakes cost, A − filtered B is what is left (fewer rows,
  the teacher's style).
- **E, GRPO, counts as with labels**: its reward is checked against gold. It answers whether RL adds
  anything on top of SFT. Its start point and reward are settled if it runs.
  Amended 2026-11-16: **E is a stretch goal after Phase 4.** The destination claim is about the best
  label-free method, and E, with labels, fills none of its blanks (objective 3, KISS). Phase 3
  closes with C and filtered C (#87, #88); the write-up comes first, and E is run, if at all, once
  the repo tells the finished story.
- **Grounded B is a label-free variant of B** (amended 2026-11-09, #74, from the external review
  on #2): it keeps the teacher outputs that pass the validity check below and whose every argument
  value is written in the query. The rule never reads gold, so it counts as label-free, and it is a
  candidate for the headline. It keeps 33,918 of 54,561 train rows at 86.8% lenient agreement with
  gold (B's data: 71.9%), but only 1,360 of the 16,316 rows whose gold values aren't all in the
  query, so it trains on far fewer answers that need a value worked out.
- **Filtered C** (amended 2026-11-13, ADR 0010): C's loss on filtered B's rows, a with-labels
  analysis run after C. Filtered C − filtered B is soft against hard targets on clean rows.
- **A label-free validity check** (every called function exists in the offered tools, required
  arguments present) is reported as a count of the teacher outputs it would drop, not trained on.

**Headline numbers.** For the best label-free method, two numbers on `test-clean`:
- **gap recovered**, (method − raw student) / (teacher − raw student). The ≥ 50% success criterion
  (#35) stays unchanged;
- **share of A's gain**, (method − raw student) / (A − raw student): how much of what labels buy is
  reached without them.

Amended 2026-11-16 (ADR 0009, #95): every label-free pick beats the 2B on `test-clean` (B 83.4
against 76.8), so its gap recovered is over 100% (B 130%). The README states "past the teacher's
76.8%" instead; both numbers stay in `results/methods.md`.

Every result is shown next to the baselines and A.

## Considered options

- **Keep A vs B as the core comparison.** Rejected: it measures label quality against a weak
  teacher, not distillation.
- **D from the better of A and B.** Rejected, see above.
- **Keep C as a stretch goal.** Rejected: without it the README covers two of the three classic kinds
  of distillation, and C separates D's two ingredients.
- **A label-free filtered B variant trained.** Deferred to the count: if the validity check drops
  only a handful of rows, a run would show nothing.

## Consequences

- "Label-free" is about training. Checkpoints are still chosen on labelled validation and scored on
  labelled test, as in practice, where a small labelled eval set is the norm.
- The teacher is a 2B. With a stronger teacher the label-free numbers would likely be higher; the
  results are about this pair (ADR 0004).
- C uses a stored pass of the teacher's top-k probabilities and a KL loss in `fc-sft` (ADR 0010).
