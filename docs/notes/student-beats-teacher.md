# Why B beats its teacher on test-clean

Issue #97, from #95. B, SFT on the 2B's own outputs, scores 83.4 on `test-clean` against the 2B's
76.8 (159 rows only B gets right, 52 only the 2B; p = 8e-14). The guess was format: B's targets are
the teacher's parsed calls written back as plain JSON, so B never fails to parse where the 2B does.
The check below, from the saved eval runs (no GPU), supports that only in part.

| `test-clean`, 1,625 rows | 2B (teacher) | B, step 1500 |
|---|---:|---:|
| Strict exact, every row | 76.8 | 83.4 |
| Answers that parse | 96.4% | 99.6% |
| Strict exact on the 1,563 rows where both parse | 79.8 | 84.0 |
| Lenient exact on those rows | 85.1 | 87.7 |

- **Format explains about a quarter.** 43 of the 159 rows only B gets right are rows where the 2B's
  answer doesn't parse.
- **On rows where both parse, B still leads**: 116 rows only B gets right against 51 (McNemar
  p = 5e-7).
- **Defaults explain about 40% of what is left.** Under lenient exact, which ignores optional
  arguments at their default, the gap on those rows falls from 4.2 to 2.6 points.
- **The rest is untested.** The likeliest cause is that B trained on 52,424 xLAM queries from the
  same distribution as the test rows, and the 2B never did: B learns xLAM's tools and conventions
  from its teacher's answers to them (self-training). A run of the 2B on its own outputs would test
  it; it is a stretch goal on map #2.

So the README says B beats its teacher partly because it always writes parseable JSON, and mostly
for reasons not measured here.

## Reproduce

```python
import json

clean = set(json.load(open("data/splits.json"))["test-clean"])


def load(run):
    rows = map(json.loads, open(f"runs/{run}/predictions.jsonl"))
    return {r["id"]: r for r in rows if r["id"] in clean}


t = load("eval-test-Qwen_Qwen3.5-2B")
b = load("eval-test-train-sft-teacher-r16-b4x4-n52424-gl-s0-checkpoint-1500")
both = [i for i in t if t[i]["parse_ok"] and b[i]["parse_ok"]]
only_b = [i for i in t if b[i]["exact"] and not t[i]["exact"]]
print(sum(not t[i]["parse_ok"] for i in only_b), "of", len(only_b))  # 43 of 159
print(
    sum(b[i]["exact"] and not t[i]["exact"] for i in both),  # 116
    sum(t[i]["exact"] and not b[i]["exact"] for i in both),
)  # 51
```
