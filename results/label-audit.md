# Test label audit

Issue #8. Rubric: [`docs/label-audit.md`](../docs/label-audit.md). Per-row labels:
[`data/audit/test_labels.csv`](../data/audit/test_labels.csv). `test-clean` ids: the `test-clean`
key in `data/splits.json` (`fc-eval --split test-clean`).

## Ceiling

**1,625 of the 2,000 test rows (81.25%) have a sound label.** That's the highest strict exact match a
model that's always right could score on raw test. The other 375 rows have a gold label that is
ambiguous (287) or wrong (88), and `test-clean` leaves them out.

| | Rows |
|---|---:|
| Test | 2,000 |
| Every model right (assumed sound) | 742 |
| Audited (some model wrong in some run) | 1,258 |
| Label sound | 883 |
| Label ambiguous | 287 |
| Label wrong | 88 |
| **`test-clean`** | **1,625** |

## Scores

The official baseline runs (#10, [`baselines.md`](baselines.md)), strict exact match:

| Model | Test | `test-clean` |
|---|---:|---:|
| Qwen3.5-0.8B (student) | 46.2 | 54.8 |
| Qwen3.5-2B (teacher) | 65.1 | 76.8 |
| Qwen3.5-4B | 68.8 | 81.8 |

Removing the unsound rows raises every model by 9 to 13 points, and the 2B-to-4B gap grows from 3.6 to
5.0 points. The wrong-argument errors that stayed flat from 0.8B to 4B in scouting ([ADR 0002](../docs/adr/0002-test-labels-never-edited.md)) were mostly label noise: 4B gets 242
of the 287 ambiguous rows wrong, and 87 of the 88 wrong ones.

The audit picked its rows from an earlier run of these baselines, whose outputs were lost; the
`models_wrong` column of `data/audit/test_labels.csv` comes from that run. The scores above are from
the official runs, which differ from it by at most 0.4 points on test-clean (batched greedy decoding
is not exactly deterministic). Because only rows some model got wrong were audited, `test-clean`
depends on that earlier run. In the official runs 740 rows have every model right (742 above), and
25 rows that some model gets wrong were never audited and count as sound (the 0.8B gets 18 of them
wrong, the 2B 3, the 4B 4). Read by hand, they are run-to-run slips with sound labels: malformed
JSON, a missing `"name"`, a dropped second call. Two hinge on argument types (6299: an id as a
string; 16520: a CNPJ without punctuation). Dropping all 25 would move the test-clean gap from
22.0 to 21.4 points, so ±0.6 bounds their effect.

## What makes a label unsound

| Category | Rows | Example |
|---|---:|---|
| Unclear units or format | 189 | country `UK` vs `GB`; `.jpg` vs `jpg`; casing when the spec example is lowercase; `9:30` as `930` or `9.5` |
| Underspecified query | 90 | "last week" with no reference date; coordinates of a landmark; a search string that could be worded several ways |
| Wrong value | 73 | placeholders (`cursor_value`, `<is_id>`, `YOUR_API_KEY`); ids and URLs made up where the query gives none (`zuid: "jkl012"`); wrong timestamps; `x**3` where the query and spec say `x^3`; missing `-` for descending order |
| Spec violation | 8 | a string where `int`/`bool` is declared; a dict where a list is declared |
| Several fitting tools | 8 | two offered tools that both validate an email |
| Code-valued argument | 6 | `"math.pi / 2"` where a float is declared (set automatically) |
| Wrong calls | 1 | a required argument given in the query left out |

Sound labels that a strict scorer still marks wrong for an equivalent call:

| Category | Rows | |
|---|---:|---|
| Default made explicit | 123 | gold passes an optional argument at its default, or the model does; 71 set automatically. #15 adds a lenient score for these |
| Convention not stated in spec | 6 | interest rates passed as decimals (`0.04` for 4%), consistent per tool across the dataset |

## Method

- The audit covers every row that **any** of 0.8B, 2B and 4B gets wrong, not only the teacher's
  errors as #8 first proposed. Auditing only 2B and 4B errors would have kept 18 unsound rows that only
  the student misses. That would have lowered the student's `test-clean` score by 0.5 points and
  widened the teacher-student gap by 0.8.
- Automatic verdicts are used only where the outcome doesn't depend on which model erred:
  code-valued gold arguments (label wrong), and rows where every wrong prediction differs from gold
  only by optional defaults (sound label).
- The audited rows follow the official baseline runs (#10). Batched greedy decoding is not
  perfectly deterministic, and the 4B's errors changed by a few rows between runs. Each new error on an
  unaudited row was labelled (13 rows: 12 sound, 1 ambiguous), and a row keeps its verdict once it has
  one, even when a later run no longer misses it. So the audit only grows and `test-clean` doesn't
  move with run-to-run noise.
- The remaining 1,181 rows were labelled by hand against the rubric. Model identities were hidden
  and predictions were shown in a fixed shuffled order.
- Repeated cases get the same verdict: a format choice the spec leaves open (casing, prefixes,
  separators, off-by-one page numbering when the spec is silent) is ambiguous every time it comes up.

## Review

The maintainer relabels 30 random audited rows blind (`uv run fc-audit review` writes
[the sheet](../docs/notes/label-audit-review.md) and `data/audit/review.csv`, seed 0). More than 3 disagreements
that change soundness means the rubric is too loose: tighten it and relabel.

**Outcome: 9 of 30 rows changed soundness, over the threshold of 3. The rubric was tightened, and
under the tightened rubric every one of them keeps its audit verdict.** One more row (53165) changed
only its unsound verdict. By cause:

| Cause | Rows | Resolution |
|---|---|---|
| The verdict name misled | 22044, 31706, 36037 | The reviewer wrote "model filled the default value, still correct" and picked ambiguous. That is the sound case, but the verdict was called `model_wrong`. Renamed to `label_sound`. |
| Bug in the review page | 59212 | The page parsed gold with JavaScript, which rounded `max_cursor: 1234567890123456789` to `…800`, so gold looked wrong. In the data gold matches the query. |
| Format the spec leaves open | 4334, 31012, 38510, 47684 | `UK` vs `GB`, `WTA` vs `wta`, `Moscow` vs `Moscow, Russia`, dashes in a phone number. Rule added: one prediction that differs only by such a choice makes the row ambiguous. The reviewer agreed and had applied it on 48573 and 57548. |
| Outside knowledge | 9238 | Inception's IMDb id. Rule added: a value with exactly one real-world answer is sound. |
| Made-up values | 53165 (verdict only) | Gold makes up `zuid: "jkl012"`. The reviewer called it wrong where the audit said ambiguous. Rule added, and the audit was rescanned: 13 rows with made-up values moved from ambiguous to wrong. `test-clean` is unchanged. |

The rules were settled with the reviewer after comparing the disagreements, so this isn't a second
independent check. A fresh blind sample under the tightened rubric would be one.

## What this means for training

Train very likely has a similar share of unsound labels and keeps them. Why, and how scoring and
checkpoint selection account for it: [ADR 0005](../docs/adr/0005-train-and-validation-keep-label-noise.md).

## Caveats

- One labeller. The 30-row review is the only check on consistency, and its disagreements were
  settled by clarifying the rubric (see Review).
- Rows that every model gets right are assumed sound. An ambiguous label that all three models happen
  to match stays in `test-clean`, so 81.25% is an upper bound on the sound share.
- "Ambiguous" is applied generously to format choices the spec leaves open. A lenient scorer (#15)
  that normalises casing and `www.`/`https://` prefixes could recover some of these rows.
