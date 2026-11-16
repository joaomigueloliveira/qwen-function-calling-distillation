# Test label audit: rubric

Issue #8, [ADR 0002](adr/0002-test-labels-never-edited.md). The audit decides which test rows have a
sound gold label. `test-clean` is test minus the rows labelled `label_ambiguous` or `label_wrong`. The
test split itself is never edited.

## Which rows are audited

Every test row that **at least one** of Qwen3.5-0.8B, 2B and 4B gets wrong (strict exact match).
Auditing only the teacher's (2B) errors would remove the bad labels that trip the teacher and keep the
ones that trip only the student, which would bias the student/teacher gap on `test-clean`. Rows that every
model gets right are assumed sound. A row keeps its verdict once it has one, even if a later run no
longer misses it: the verdict is about the label, not about which model erred.

The labeller sees the query, the offered tools, the gold label and the wrong predictions, with the
models anonymised and shuffled, so a verdict can't favour one model.

## The question

For each row: **would a careful person, reading only the query and the tool specs, write this gold
call (up to call order and arguments left at their default)?**

| Verdict | When |
|---|---|
| `label_sound` | Yes. Every differing prediction is a model mistake. This includes a prediction that only adds or drops an optional argument at its default: the label is still sound. |
| `label_ambiguous` | The query or specs allow two or more reasonable calls, and the label is only one of them (a value the query doesn't fix, unclear units, a choice between two fitting tools, a number of calls the query doesn't pin down). |
| `label_wrong` | No. The label contradicts the query or the specs: a wrong value, a wrong or missing call, a hallucinated argument, a required argument left out, a Python expression where a value is declared, a value of the wrong type. |

When in doubt between `label_sound` and `label_ambiguous`, ask whether a prediction is *also* defensible.
If yes, the row is ambiguous; if every prediction has a clear fault, the label is sound. Judge each
prediction on its own: one defensible prediction makes the row ambiguous even when the others are
plainly wrong.

Three rules settle the cases the review disagreed on:

- **Format the spec leaves open.** If a prediction differs from gold only in a format choice that the spec
  doesn't settle (casing, `UK` vs `GB`, a phone number with or without dashes, a `www.` prefix), the
  row is ambiguous.
- **Outside knowledge with one answer.** A value that needs real-world knowledge but has exactly one
  correct answer (Inception's IMDb id `tt1375666`) is sound. Values with many acceptable answers
  (a city's coordinates, to some precision) are ambiguous.
- **Made-up values.** If the query doesn't give a value and gold makes one up (`zuid: "jkl012"` for
  "agent D", `"https://www.example.com"` for an unnamed URL), the label is wrong (`wrong value`),
  not ambiguous.

## Categories

`category` is optional and names a recurring reason, so the write-up can count it.

| Category | Verdict | Meaning |
|---|---|---|
| `default made explicit` | `label_sound` | The gold call passes an optional argument at its declared default or as `null`, and the model leaves it out (or the reverse). It's the same call by the spec, so the label counts as sound. #15 adds a lenient exact match that scores this. Set automatically when every wrong prediction differs only in this way. |
| `convention not stated in spec` | `label_sound` | The label follows a convention the tool always uses (e.g. "5%" passed as `0.05`) that train teaches but the tool description doesn't state. Consistent within each tool across the dataset (#14), so it counts as sound. |
| `code-valued argument` | `label_wrong` | A gold argument is a Python expression (`"math.pi / 2"`) where a value is declared. Set automatically. |
| `wrong value` | `label_wrong` | A gold value contradicts the query, or is made up where the query gives none. |
| `wrong calls` | `label_wrong` | A gold call is missing, extra, or calls the wrong tool. |
| `spec violation` | `label_wrong` | A required argument is missing, an argument has the wrong type, or the value is outside what the spec allows. |
| `underspecified query` | `label_ambiguous` | The query doesn't fix a value the label picks, and several values are reasonable (a date range for "last week", a search wording). |
| `unclear units or format` | `label_ambiguous` | The value's unit or format (string vs number, list vs single value) can be read two ways, and the spec doesn't settle it. |
| `several fitting tools` | `label_ambiguous` | Two offered tools fit the request and the label picks one. |

## Pre-tags

`fc-audit prepare` adds deterministic hints in the `tags` column: `code_argument`, `type_mismatch`,
`missing_required`, `explicit_default`, `percentage`. They point the labeller at a likely issue. They
aren't verdicts on their own: a `percentage` row is often a sound convention, and a `type_mismatch`
can be a loose spec (a `str` declared where a list is plainly meant) rather than a bad label.

## Review

The maintainer relabels 30 random audited rows blind, without seeing the verdicts. `uv run fc-audit
review` draws them (seed 0) and writes the sheet `docs/notes/label-audit-review.md` plus
`data/audit/review.csv` to fill in. `uv run fc-audit build` then reports two kinds of disagreement:
any change of verdict, and changes of soundness (sound vs ambiguous/wrong), which are the ones that
move rows in or out of `test-clean`. More than 3 soundness disagreements means the rubric is too
loose: tighten it and relabel. The outcome is recorded in
[`results/label-audit.md`](../results/label-audit.md).
