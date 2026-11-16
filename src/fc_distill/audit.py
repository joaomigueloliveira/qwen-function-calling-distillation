"""Label audit of the test split (issue #8, ADR 0002). Rubric: docs/label-audit.md.

Every test row that at least one model gets wrong is audited. Each audited row gets a verdict on its
gold label: `label_sound`, `label_ambiguous` or `label_wrong`. `test-clean` is
test minus the ambiguous and wrong rows.

uv run fc-audit prepare   # write data/audit/test_labels.csv: rows to audit, pre-tags, auto verdicts
uv run fc-audit build     # write the test-clean ids into data/splits.json and print the ceiling
uv run fc-audit review    # sample 30 audited rows for the maintainer's blind review
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypedDict, get_args

from fc_distill.data import DEFAULT_SPLITS_PATH, Call, Example, load_split
from fc_distill.quality import (
    code_arguments,
    explicit_defaults,
    missing_required,
    type_mismatches,
)
from fc_distill.scoring import lenient_match, parse_calls

Verdict = Literal["label_sound", "label_ambiguous", "label_wrong"]
VERDICTS: tuple[Verdict, ...] = get_args(Verdict)
UNSOUND: frozenset[Verdict] = frozenset({"label_ambiguous", "label_wrong"})

DEFAULT_LABELS_PATH = Path("data/audit/test_labels.csv")
DEFAULT_REVIEW_PATH = Path("data/audit/review.csv")
DEFAULT_SHEET_PATH = Path("docs/notes/label-audit-review.md")
REVIEW_SIZE = 30
DEFAULT_MODELS = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B")
FIELDS = ("id", "verdict", "category", "reason", "models_wrong", "tags")

AUTO = "auto: "  # reason prefix of verdicts set by `auto_verdict`, recomputed on every prepare
_PERCENT = re.compile(r"\d\s*%|percent", re.IGNORECASE)


class AuditRow(TypedDict):
    id: int
    verdict: Verdict | Literal[""]
    category: str
    reason: str
    models_wrong: str  # ';'-joined model ids
    tags: str  # ';'-joined pre-tags


def is_auto(row: AuditRow) -> bool:
    return row["reason"].startswith(AUTO)


def pre_tags(example: Example) -> list[str]:
    """Deterministic hints about the gold label, shown to the labeller and kept in the CSV."""
    one = [example]
    found = {
        "code_argument": bool(code_arguments(one)),
        "type_mismatch": bool(type_mismatches(one)),
        "missing_required": bool(missing_required(one)),
        "explicit_default": bool(explicit_defaults(one)),
        "percentage": bool(_PERCENT.search(example["query"])),
    }
    return [tag for tag, hit in found.items() if hit]


def only_defaults_differ(prediction: Sequence[Call] | None, example: Example) -> bool:
    """The prediction is the gold call once optional arguments at their default are dropped."""
    if prediction is None:
        return False
    return lenient_match(prediction, example["answers"], example["tools"])


def auto_verdict(
    example: Example, predictions: Sequence[Sequence[Call] | None]
) -> tuple[Verdict, str, str] | None:
    """(verdict, category, reason) when the outcome doesn't depend on which model erred, else None.

    `predictions` are the parsed outputs of the models that got the row wrong.
    """
    if code_arguments([example]):
        return "label_wrong", "code-valued argument", AUTO + "gold argument is a Python expression"
    if predictions and all(only_defaults_differ(p, example) for p in predictions):
        return "label_sound", "default made explicit", AUTO + "differs only by an optional default"
    return None


def load_predictions(run_dir: Path) -> dict[int, tuple[bool, str]]:
    """id -> (exact, raw prediction) from an `fc-eval` run."""
    with (run_dir / "predictions.jsonl").open() as f:
        records = [json.loads(line) for line in f]
    return {r["id"]: (r["exact"], r["prediction"]) for r in records}


def prepare(
    test: Sequence[Example],
    predictions: Mapping[str, Mapping[int, tuple[bool, str]]],
    existing: Mapping[int, AuditRow] | None = None,
) -> list[AuditRow]:
    """One row per test example that at least one model gets wrong, plus every row that already
    has a verdict. Verdicts already in `existing` are kept, so re-running after new model runs
    doesn't lose labelling work, and the audited set only grows.
    Automatic verdicts are always recomputed, so a rule change reaches every row."""
    existing = existing or {}
    rows: list[AuditRow] = []
    for ex in test:
        wrong = [m for m, preds in predictions.items() if not preds[ex["id"]][0]]
        if not wrong:
            # A verdict is about the label, not about who erred: keep it when a later run no
            # longer misses the row, so test-clean doesn't change with run-to-run noise.
            if (old := existing.get(ex["id"])) and old["verdict"]:
                rows.append(old | {"models_wrong": ""})
            continue
        row: AuditRow = {
            "id": ex["id"],
            "verdict": "",
            "category": "",
            "reason": "",
            "models_wrong": ";".join(wrong),
            "tags": ";".join(pre_tags(ex)),
        }
        auto = auto_verdict(ex, [parse_calls(predictions[m][ex["id"]][1]) for m in wrong])
        if auto:
            row["verdict"], row["category"], row["reason"] = auto
        elif (
            (old := existing.get(ex["id"]))
            and old["verdict"]
            and not old["reason"].startswith(AUTO)
        ):
            row |= {"verdict": old["verdict"], "category": old["category"], "reason": old["reason"]}
        rows.append(row)
    return rows


def read_labels(path: Path = DEFAULT_LABELS_PATH) -> dict[int, AuditRow]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[int, AuditRow] = {}
    for r in rows:
        if r["verdict"] and r["verdict"] not in VERDICTS:
            raise ValueError(f"row {r['id']}: unknown verdict {r['verdict']!r}")
        out[int(r["id"])] = {**r, "id": int(r["id"])}  # type: ignore[typeddict-item]
    return out


def write_labels(rows: Sequence[AuditRow], path: Path = DEFAULT_LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clean_ids(test_ids: Sequence[int], labels: Mapping[int, AuditRow]) -> list[int]:
    """Test minus rows labelled ambiguous or wrong. Every audited row must have a verdict."""
    unlabelled = [i for i, r in labels.items() if not r["verdict"]]
    if unlabelled:
        raise ValueError(f"{len(unlabelled)} audited rows have no verdict, e.g. {unlabelled[:5]}")
    drop = {i for i, r in labels.items() if r["verdict"] in UNSOUND}
    return sorted(set(test_ids) - drop)


def summary(test_ids: Sequence[int], labels: Mapping[int, AuditRow]) -> dict[str, object]:
    clean = clean_ids(test_ids, labels)
    return {
        "test_rows": len(test_ids),
        "audited_rows": len(labels),
        "verdicts": dict(Counter(r["verdict"] for r in labels.values())),
        "categories": dict(Counter(r["category"] for r in labels.values() if r["category"])),
        "test_clean_rows": len(clean),
        "ceiling": round(len(clean) / len(test_ids), 4),
    }


def review_sample(labels: Mapping[int, AuditRow], n: int = REVIEW_SIZE, seed: int = 0) -> list[int]:
    """The audited rows the maintainer relabels blind: a fixed random sample."""
    return sorted(random.Random(seed).sample(sorted(labels), n))


def agreement(labels: Mapping[int, AuditRow], review: Mapping[int, str]) -> dict[str, object]:
    """Compare the maintainer's verdicts with the audit's. `soundness` counts only disagreements
    that change test-clean (sound vs ambiguous/wrong)."""
    done = {i: v for i, v in review.items() if v}
    verdict = [i for i, v in done.items() if v != labels[i]["verdict"]]
    soundness = [i for i in verdict if (done[i] in UNSOUND) != (labels[i]["verdict"] in UNSOUND)]
    return {
        "reviewed": len(done),
        "verdict_disagreements": sorted(verdict),
        "soundness_disagreements": sorted(soundness),
    }


def read_review(path: Path = DEFAULT_REVIEW_PATH) -> dict[int, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["verdict"] and r["verdict"] not in VERDICTS:
            raise ValueError(f"review row {r['id']}: unknown verdict {r['verdict']!r}")
    return {int(r["id"]): r["verdict"] for r in rows}


def review_sheet(
    ids: Sequence[int],
    test: Mapping[int, Example],
    predictions: Mapping[str, Mapping[int, tuple[bool, str]]],
) -> str:
    """Markdown sheet for the blind review: query, the tools involved, gold label and the wrong
    predictions in a fixed shuffled order. The audit's verdicts are deliberately left out."""
    out = [
        "# Label audit: blind review sheet",
        "",
        "Relabel each row against [the rubric](../label-audit.md) and write the verdict "
        "(`label_sound`, `label_ambiguous` or `label_wrong`) into `data/audit/review.csv`. "
        "Then run `uv run fc-audit build`, which reports the disagreements.",
    ]
    for i in ids:
        ex = test[i]
        wrong = [preds[i][1].strip() for preds in predictions.values() if not preds[i][0]]
        random.Random(i).shuffle(wrong)
        called = {c["name"] for c in ex["answers"]}
        out += ["", f"## {i}", "", f"**Query:** {ex['query']}", "", "**Tools:**", ""]
        out += [
            f"- `{t['name']}`: {json.dumps(t.get('parameters') or {})}"
            for t in ex["tools"]
            if t["name"] in called
        ]
        out += ["", f"**Gold:** `{json.dumps(ex['answers'])}`", "", "**Wrong predictions:**", ""]
        for p in wrong:
            out += ["~~~", p[:600], "~~~"]
    return "\n".join(out) + "\n"


def write_test_clean(ids: Sequence[int], splits_path: Path = DEFAULT_SPLITS_PATH) -> None:
    splits = json.loads(splits_path.read_text())
    splits["test-clean"] = sorted(ids)
    splits_path.write_text(json.dumps(splits, indent=0) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=("prepare", "build", "review"))
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    args = ap.parse_args()

    test = load_split("test")
    if args.command in ("prepare", "review"):
        predictions = {
            m: load_predictions(args.runs / f"eval-test-{m.replace('/', '_')}") for m in args.models
        }
    if args.command == "review":
        ids = review_sample(read_labels(args.labels))
        if not DEFAULT_REVIEW_PATH.exists():
            with DEFAULT_REVIEW_PATH.open("w", newline="") as f:
                f.write("id,verdict,note\n" + "".join(f"{i},,\n" for i in ids))
        DEFAULT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_SHEET_PATH.write_text(review_sheet(ids, {ex["id"]: ex for ex in test}, predictions))
        print(f"wrote {DEFAULT_SHEET_PATH} and {DEFAULT_REVIEW_PATH} ({len(ids)} rows)")
    elif args.command == "prepare":
        existing = read_labels(args.labels) if args.labels.exists() else {}
        rows = prepare(test, predictions, existing)
        write_labels(rows, args.labels)
        todo = sum(not r["verdict"] for r in rows)
        print(f"wrote {args.labels}: {len(rows)} rows to audit, {todo} still need a verdict")
    else:
        labels = read_labels(args.labels)
        ids = [ex["id"] for ex in test]
        write_test_clean(clean_ids(ids, labels))
        result = summary(ids, labels)
        if DEFAULT_REVIEW_PATH.exists():
            result["review"] = agreement(labels, read_review())
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
