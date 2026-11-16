# 0010. Method C: logit distillation from stored top-k teacher log-probs, on the teacher's text

Date: 2026-11-13 · Status: accepted (#59) · Amended 2026-11-13 from the teacher pass (#85)

## Context

C is the third label-free method (ADR 0007): off-policy like B, with the teacher's distribution at
every token like D. B trains on the one token the teacher wrote, which is forward KL against a
one-hot teacher. C puts the rest of the teacher's distribution back.

Facts the design rests on, checked 2026-11-13:

- **The vocabularies line up.** The 0.8B and the 2B ship the same `tokenizer.json` and the same
  248,320-entry vocabulary, so entry *i* is the same token in both and positions match one for one.
- **The teacher is usually sure.** On three made-up tool-call queries, the 2B put more than 99% on
  its top token at 91 of 103 answer positions. There soft and hard targets give the same gradient.
  The rest are the choices: which tool, which value, whether to write an optional argument (one
  position closed the call early, leaving the default out, 8% of the time). The external review
  expects C within about 2 points of B and D, which validation can't separate.
- **Top 20 is nearly all of it at T = 1.** Over all 2.78M answer positions of B's rows (#85), the
  top 20 hold a median of 100.00% of the probability (1st percentile 99.71%, minimum 27.6%); the
  tail averages 0.039%, is over 0.12% at 2.0% of positions and over 1% at 0.41%. At T = 2 the top
  20 held 78–92% at the three probe positions measured in full: the 248,000-token tail grows fast.
- **No trainer in TRL 1.17 fits.** `GKDTrainer` with `lmbda=0` trains on dataset text but holds
  both models' full logits for the whole sequence, prompt included: over 12 GB at 4 rows per
  micro-batch, by estimate. `DistillationTrainer` chunks its loss but is on-policy only.
- **vLLM scores fixed text.** `prompt_logprobs=k` returns, at every position of given token ids, the
  top k log-probs plus the actual token's (k ≤ `max_logprobs`, 20 by default).

## Decision

**C** trains on B's 52,424 parsed teacher outputs, the same rows, order and seed as B. Gold is
never read, so C is label-free, and C − B isolates soft against hard targets on identical text.

**Filtered C** trains on the 37,696 of those outputs that lenient-match gold, filtered B's rows.
It uses labels as a filter, so it is a with-labels analysis run (amends ADR 0007's regime table),
and filtered C − filtered B is the same contrast on clean rows. It runs after C.

| Setting | Value |
|---|---|
| Teacher distribution | stored: one vLLM pass of the 2B over B's text, fed fc-sft's own token ids, `prompt_logprobs=20` |
| Stored per answer position | the top-20 token ids and log-probs |
| Loss | forward KL(teacher ‖ student) at every answer position, over the 20 entries plus one rest entry (1 − their sum, against the student's mass outside them) |
| Temperature | T = 1 |
| Hard term | none (α = 0): the written token is the teacher's top-1 at 99.55% of answer positions, so it is already in the target (see the format note below) |
| Start | the raw 0.8B with a fresh LoRA, r16/α32 all-linear |
| Training | ADR 0006's frozen config: lr 2e-4, cosine, 16 rows per step, one epoch, `--group-by-length`, seed 0; untuned for this loss |
| Selection | ADR 0006 and ADR 0009: every 500 steps on validation, strict exact, ties to the earlier; test, `test-clean` and BFCL for the pick |

- **GPU-hours** start at 0 and include the 2B's generation pass (1.43 h, as for B), the scoring
  pass and training. Filtered C counts the same two passes, as filtered B counts the generation.
  Each is also reported at its best checkpoint within B's GPU-hours (ADR 0008).
- **Claims** added to `fc-report --methods` before training: (C, B) and (filtered C, filtered B),
  each on `test-clean` and BFCL. The Holm family grows to 25; a claim that crosses 0.05 because of
  it is reported as such.
- **Explanation readouts**, fixed now and never used for selection: BFCL, agreement with the
  teacher and copying its errors, the default-made-explicit breakdown, where C would show the
  teacher's habit of leaving defaults out, and the share of outputs that are fenced or span several
  lines (format note below).
- **Format** (amended from #85's eval-risk review): B's text is the teacher's parsed calls written
  back with `json.dumps`, so its targets differ from gold in content only. On 45,807 of the 52,424
  rows that is byte for byte what the 2B wrote, and the written token is its top-1 at 99.90% of
  positions. On the other 6,617 the 2B had fenced its answer (2,178) or spread it over several lines
  (3,987); there it is top-1 at 97.78%, and at the first answer tokens it prefers a fence or a
  newline. C's soft targets keep that preference, B's hard targets drop it. The design stays as it
  is (same rows and text as B, α = 0), and the format readout shows whether C − B moves format:
  fc-eval parses fenced and multi-line answers alike, so it would show there, not in exact match.
- **The stored file** is derived from the gated xLAM prompts: `runs/` only, backed up to the
  private W&B project, like the teacher outputs (ADR 0006).
- **Scoring** uses fc-eval's adapter path and #76's check: C's adapter has B's layout, so the
  one-time comparison #79 asks of a new layout doesn't apply.

## Considered options

- **Gold text, scored by the teacher** (α = 0.5 against the gold token). Rejected: it reads labels,
  so it can't answer the label-free question, and on gold text the teacher's top-1 often disagrees with
  the written token (the 2B is wrong on about a third of train rows), which forces an untuned α. Filtered C asks the
  with-labels version more cheaply.
- **A live teacher in the training process**, the full vocabulary at any T. Rejected: `GKDTrainer`
  doesn't fit at B's micro-batch, a chunked off-policy loss would be new code, and it adds 3.51 GiB
  of teacher weights for a tail that averages 0.039% at T = 1.
- **A hard term to pin B's format** (α > 0), **C on only the 45,807 rows where B's text is the
  teacher's own**, or **scoring the teacher's raw text**. Rejected after #85: an untuned α changes
  what C − B measures; the other two break "the same rows and text as B". The format readout is
  cheaper and keeps the comparison.
- **Re-generating with `logprobs=20`** instead of scoring the fixed text. Rejected: batched inference
  isn't always deterministic, and any row that changed would break "the same text as B".
- **Renormalising the top 20 to sum to 1.** Rejected: it tells the student every other token should
  get zero, a target sharper than the teacher's, the top-k bias Anshumann et al. describe
  ([Sparse Logit Sampling, ACL 2025](https://arxiv.org/abs/2503.16870)). The rest entry costs the
  student's log-sum-exp at answer positions only.
- **T > 1, with Hinton's T² factor.** Rejected: stored top-20 can't recover the tail, which at T = 2
  already holds 8–22%.

## Consequences

- C is bound to T = 1 and top 20; a softer target needs a new teacher pass.
- C inherits the teacher's uncertainty where it hedges, including about writing optional
  arguments and about fencing its answer; the default-made-explicit and format readouts are there
  to see it.
- The teacher pass cost 1.56 GPU-h (#85), so C starts at 2.99 GPU-h before training. Filtered C
  is charged the share of that pass its 37,696 rows needed (1.12 h, as fc-sft computes it since
  #86), so it starts at 2.55: it could have scored only those rows, while the generation pass
  has to answer every row before the filter can choose.
- One untuned config and one seed, as for every method; the learning rate wasn't chosen for a KL
  loss, though at T = 1 its gradient (student − teacher probabilities) has cross-entropy's scale.
- Build and run tickets, in order: the teacher top-k pass (#85), the KL loss in fc-sft (#86), the
  C run (#87), the filtered C run (#88).
