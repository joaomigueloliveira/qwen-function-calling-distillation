# Function-calling distillation

A small Qwen3.5 student learns single-turn tool calling on xLAM-60k from a larger Qwen3.5 teacher.

## Models

**Student**:
Qwen3.5-0.8B, the model being trained.

**Teacher**:
Qwen3.5-2B, the model whose outputs or logits the student learns from.

## Experiments

**Method**:
One way of training the student, compared with the others per GPU-hour: A, SFT on gold labels; B,
SFT on teacher outputs; C, logit distillation; D, on-policy distillation; E, GRPO (a stretch goal,
not run).
_Avoid_: arm

**Label-free method**:
A method whose training never sees gold labels, only queries and the teacher: B (unfiltered),
grounded B, C, D.
The main question (ADR 0007).

**With-labels method**:
A method whose training uses gold labels, as targets, as a filter or in a reward: A, filtered B,
filtered C, E.
The reference for what labels buy.

**Pipeline**:
Two methods run one after the other on the same student, e.g. B then D. Reported as a combination,
never as either method.

**Probe**:
A score taken while a run trains, on validation, to watch it. Never used to stop the run or to
choose its checkpoint.

**Rollout**:
An answer the student samples during training, to be graded: by the teacher in D, by a reward in E.

**Variant**:
A named change within a method, e.g. method B filtered to the teacher outputs that match gold.

**Filtered B**:
Method B trained only on the teacher outputs that lenient-match gold. Uses labels, as a filter.

**Grounded B**:
Method B trained only on the teacher outputs that call offered tools with their required arguments
and whose every argument value is written in the query. Label-free: the rule reads the query and
the tools, never gold.

**Filtered C**:
Method C trained only on the teacher outputs that lenient-match gold, the rows of filtered B. Uses
labels, as a filter.

**Hard target / soft target**:
What the student is trained toward at one position: the one token written in the text (hard; A and
B), or the teacher's probabilities over the vocabulary there (soft; C and D).

**Run**:
One training job of a method or variant, with its own checkpoints and its own W&B run.

**Teacher output**:
The teacher's answer to a train row's query, parsed and written back as JSON calls. B's training
text; C trains on the same text.

**Teacher pass**:
A run of the teacher over train, done once before training and counted in the methods' GPU-hours:
generating teacher outputs (B, C), or storing its top 20 token probabilities on them (C).

**Selected checkpoint**:
The checkpoint of a run with the best strict exact on validation, a tie going to the earlier
(ADR 0006). The only one scored on test, test-clean and BFCL. Informally, the pick.
_Avoid_: best model (alone)

**GPU-hours**:
A method's cost on the one RTX 3060, from zero up to a checkpoint: its teacher passes plus training.
Validation scoring isn't counted.

**At B's budget**:
A label-free method's best validation checkpoint within B's GPU-hours, scored like a selected
checkpoint (ADR 0008).

**Headline method**:
The label-free method the README's claim names, chosen by ADR 0009's rule among the methods
test-clean can't separate.

**Readout**:
A measurement reported to explain a result and never used to choose: BFCL along a run, agreement
with the teacher, the default-made-explicit rows, the share of fenced or multi-line answers.

**Baseline**:
An untrained model used as a reference point: the student, the teacher, the 4B.

## Data

**Row**:
One xLAM example: a query, the offered tools and the gold answer.
_Avoid_: sample, example (when a split row is meant)

**Gold answer**:
The list of calls xLAM publishes as correct for a row.
_Avoid_: label (alone), ground truth

**Offered tools**:
The tools a row's prompt lists. Some are distractors that the gold answer doesn't call.

**Query template**:
The same query sentence with different values, as xLAM's generator produced many per tool.

**Explicit default**:
An optional argument that a call passes at its declared default or as `null`.

**Made-up value**:
A gold argument value that the query doesn't give and the gold answer invents.

## Splits

**Train / validation / test**:
The three splits of xLAM-60k. Validation chooses checkpoints and settings; test only produces
official numbers.

**test-clean**:
Test minus the rows whose gold answer the label audit marks ambiguous or wrong.
_Avoid_: clean test, filtered test

**Train filter**:
A documented rule that removes rows from train. It never rewrites a gold answer.

## Label audit

**Audited row**:
A test row that at least one of the 0.8B, 2B and 4B baselines gets wrong.

**Sound label**:
A gold answer that a careful person, reading only the query and the tool specs, would write.
_Avoid_: model wrong (the old verdict name), correct label

**Ambiguous label**:
A gold answer that is one of several reasonable calls for the query and specs.

**Wrong label**:
A gold answer that contradicts the query or specs, or contains a made-up value.

**Default made explicit**:
An audited row whose gold and a baseline's answer differ only by an optional argument at its declared
default: one side writes it, the other leaves it out. The label is sound; which way gold goes is
xLAM's habit.

**Gold quirk**:
An arbitrary choice in gold that a sound alternative could replace: an ambiguous label's pick, or the
habit of writing or omitting a default. A model can score by learning it without calling tools any
better.

**Ceiling**:
The share of test rows with a sound label: the best strict exact score an always-right model could
reach on test.

## Scoring

**McNemar test**:
The paired test used to compare two models: it counts only the rows where exactly one of them is
right.

**Holm**:
The adjustment applied to the p-values of all the README's claims together, so that testing many
claims doesn't make one look significant by chance.

**BFCL**:
1,000 single-turn rows of the Berkeley Function-Calling Leaderboard, with tools the student never
trained on, converted to xLAM's prompt. An eval only: nothing is chosen on it.

**Strict exact**:
A prediction matches when its multiset of calls equals the gold answer's. The headline metric.
_Avoid_: accuracy (alone)

**Lenient exact**:
Strict exact, except that explicit defaults on either side are ignored.

**Quirk-learning**:
A model gaining on raw test by reproducing gold quirks. Measured as strict exact on the
ambiguous test rows, compared with the teacher's.

**Missing call**:
An answer that stops before a call the gold answer makes, e.g. the first of two calls and nothing
after it.

**Agreement with the teacher**:
The share of rows where a model makes exactly the teacher's calls, both parsed. It rises with
accuracy alone, since two right answers agree.

**Copying the teacher's errors**:
On rows where the teacher's parsed answer is wrong, how often a model makes the same calls. Separates
imitating the teacher from getting better.

**Gap recovered**:
How much of the distance from the raw student to the teacher a trained student closes, in strict
exact on the same eval: (student − raw student) / (teacher − raw student).

**Share of A's gain**:
How much of method A's improvement a label-free method reaches, in strict exact on the same eval:
(method − raw student) / (A − raw student).
