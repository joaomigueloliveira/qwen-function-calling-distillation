"""Worked examples for the story site (#107): real rows that show what the numbers mean.

uv run --no-sync fc-examples

One row per category, chosen by a fixed rule so no example is picked by hand: among the rows that
qualify, the one with the shortest prompt (query plus tool specs), ties to the lowest id.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

Record = Mapping[str, Any]  # one row of an eval run's predictions.jsonl
Runs = Mapping[str, Mapping[str, Mapping[Any, Record]]]  # split -> model -> row id -> record


class Answer(TypedDict):
    model: str
    prediction: str
    exact: bool


class Example(TypedDict, total=False):
    key: str
    split: str
    id: Any
    query: str
    tools: list[dict[str, Any]]
    gold: Any
    answers: list[Answer]
    note: str  # the audit's reason, where gold is wrong


class Category(NamedTuple):
    key: str
    split: str
    models: tuple[str, ...]
    # The row's record for each model, and the audit's reason if its gold is wrong -> qualifies.
    keep: Callable[[Mapping[str, Record], str | None], bool]


CATEGORIES = (
    Category(
        "beats-teacher",
        "test-clean",
        ("B", "raw 2B"),
        lambda r, _: bool(r["B"]["exact"]) and not r["raw 2B"]["exact"],
    ),
    Category(
        "unseen-tools",
        "bfcl",
        ("D", "B"),
        lambda r, _: bool(r["D"]["exact"]) and not r["B"]["exact"],
    ),
    # On test, since test-clean leaves these rows out; B "wrong" only because gold is.
    Category(
        "gold-wrong",
        "test",
        ("B", "raw 2B"),
        lambda r, wrong: wrong is not None and not r["B"]["exact"],
    ),
    # Soft targets carry the teacher's habit of fencing its answer in a code block.
    Category(
        "fenced",
        "test-clean",
        ("C", "B"),
        lambda r, _: str(r["C"]["prediction"]).strip().startswith("```"),
    ),
)


def _id_order(i: Any) -> tuple[str, int]:
    """xLAM ids are ints, BFCL's "<category>_<n>": both in number order."""
    if isinstance(i, int):
        return ("", i)
    head, _, n = str(i).rpartition("_")
    return (head, int(n)) if n.isdigit() else (str(i), 0)


def _prompt_length(row: Mapping[str, Any]) -> int:
    return len(row["query"]) + len(json.dumps(row["tools"]))


def worked_examples(
    xlam: Sequence[Mapping[str, Any]],
    bfcl: Sequence[Mapping[str, Any]],
    runs: Runs,
    wrong_labels: Mapping[Any, str],
) -> list[Example]:
    """One example per category that has a qualifying row, in CATEGORIES order."""
    out: list[Example] = []
    for cat in CATEGORIES:
        by_model = runs.get(cat.split)
        if by_model is None or any(m not in by_model for m in cat.models):
            continue
        rows = [
            r
            for r in (bfcl if cat.split == "bfcl" else xlam)
            if all(r["id"] in by_model[m] for m in cat.models)
            and cat.keep({m: by_model[m][r["id"]] for m in cat.models}, wrong_labels.get(r["id"]))
        ]
        if not rows:
            continue
        best = min(rows, key=lambda r: (_prompt_length(r), _id_order(r["id"])))
        example: Example = {
            "key": cat.key,
            "split": cat.split,
            "id": best["id"],
            "query": best["query"],
            "tools": best["tools"],
            "gold": best["gold"],
            "answers": [
                {
                    "model": m,
                    "prediction": by_model[m][best["id"]]["prediction"],
                    "exact": bool(by_model[m][best["id"]]["exact"]),
                }
                for m in cat.models
            ],
        }
        if best["id"] in wrong_labels:
            example["note"] = wrong_labels[best["id"]]
        out.append(example)
    return out


DEFAULT_OUT = Path("results/examples.json")
MODELS = ("B", "C", "D", "raw 2B")


def _run_name(model: str, methods: Mapping[str, Any]) -> str:
    """A model's eval run name: a method's selected checkpoint, or a baseline's model id."""
    from fc_distill.methods import METHODS, TEACHER

    if model == "raw 2B":
        return TEACHER
    train_run = next(m.train_run for m in METHODS if m.key == model)
    return f"{train_run}-checkpoint-{methods['methods'][model]['selected']}"


def load_runs(runs_dir: Path, methods: Mapping[str, Any], clean: set[int]) -> dict[str, Any]:
    """split -> model -> row id -> record, for every model the categories show."""
    from fc_distill.evaluate import read_run
    from fc_distill.methods import _eval_dir

    out: dict[str, dict[str, dict[Any, Record]]] = {}
    for split in ("test-clean", "test", "bfcl"):
        out[split] = {}
        for model in MODELS:
            _, records = read_run(runs_dir / _eval_dir(split, _run_name(model, methods)))
            keep = [r for r in records if split != "test-clean" or r["id"] in clean]
            out[split][model] = {r["id"]: r for r in keep}
    return out


def main() -> None:
    import argparse
    import csv

    from fc_distill import bfcl
    from fc_distill.data import DEFAULT_SPLITS_PATH, load_split
    from fc_distill.methods import DEFAULT_AUDIT, DEFAULT_JSON

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    methods = json.loads(DEFAULT_JSON.read_text())
    clean = set(json.loads(DEFAULT_SPLITS_PATH.read_text())["test-clean"])
    with DEFAULT_AUDIT.open() as f:
        wrong = {
            int(r["id"]): r["reason"] for r in csv.DictReader(f) if r["verdict"] == "label_wrong"
        }
    xlam = [{**ex, "gold": ex["answers"]} for ex in load_split("test")]
    rows = [{**r, "gold": r["possible"]} for r in bfcl.load()]
    examples = worked_examples(xlam, rows, load_runs(args.runs, methods, clean), wrong)
    args.out.write_text(
        json.dumps(
            {
                "about": (
                    "Worked examples for the story site (#107), one per category, each the "
                    "qualifying row with the shortest prompt (ties to the lowest id). Rows from "
                    "xLAM-60k (Salesforce, CC-BY-4.0) and BFCL (Apache-2.0); answers are each "
                    "model's selected checkpoint's greedy output. Built by fc-examples."
                ),
                "examples": examples,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"wrote {args.out}: {', '.join(e['key'] for e in examples)}")
