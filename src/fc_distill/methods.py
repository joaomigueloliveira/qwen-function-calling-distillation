"""Method results (#46): results/methods.json and results/methods.md from the runs.

uv run --no-sync fc-report --methods

For each method: every checkpoint's validation score against its GPU-hours, the selected checkpoint
(ADR 0006: best strict exact, ties to the earlier), and the selected checkpoint's official test,
test-clean and BFCL scores once they exist. Test-clean is scored from the test run restricted to
its ids, as in the baselines report. Label-free methods are the main question; methods with labels
are the reference (ADR 0007). Each label-free method after B is also reported at B's budget: its
best checkpoint on validation within B's GPU-hours, with that checkpoint's official scores once
they exist (ADR 0008). A pipeline (B→D) continues another method's selected checkpoint, and its
GPU-hours start from that checkpoint's.

The README's pairwise claims are listed again under strict exact (Holm-adjusted together), lenient
exact and, on test-clean, without the rows the audit tags "default made explicit" (gold and a
baseline differed only by an optional argument at its default, either way); each method's score on
those rows is reported apart (#68).
"""

import csv
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from fc_distill.data import DEFAULT_SPLITS_PATH, Example, load_split
from fc_distill.evaluate import read_run
from fc_distill.scoring import call_multiset, parse_calls, validity
from fc_distill.slices import argument_source, grounded
from fc_distill.stats import holm, mcnemar, metric_block


class Method(NamedTuple):
    key: str
    name: str
    train_run: str
    labels: bool
    after: str | None = None  # a pipeline: the method whose selected checkpoint it continues


METHODS = (
    Method("A", "SFT on gold labels", "train-sft-gold-r16-b4x4-n54561-gl-s0", labels=True),
    Method("B", "SFT on teacher outputs", "train-sft-teacher-r16-b4x4-n52424-gl-s0", labels=False),
    Method(
        "Bf",
        "SFT on teacher outputs that match gold",
        "train-sft-teacher-lenient-r16-b4x4-n37696-gl-s0",
        labels=True,
    ),
    Method(
        "Bg",
        "SFT on grounded teacher outputs",
        "train-sft-teacher-grounded-r16-b4x4-n33918-gl-s0",
        labels=False,
    ),
    Method(
        "C",
        "logit distillation (teacher's top 20)",
        "train-sft-teacher-topk-r16-b4x4-n52424-gl-s0",
        labels=False,
    ),
    Method(
        "Cf",
        "logit distillation on teacher outputs that match gold",
        "train-sft-teacher-topk-lenient-r16-b4x4-n37696-gl-s0",
        labels=True,
    ),
    Method("D", "on-policy distillation", "train-opd-r16-b4x4-n54561-s0", labels=False),
    Method(
        "B→D",
        "B then on-policy distillation",
        "train-opd-from-sft-teacher-r16-b4x4-n52424-gl-s0-1500-r16-b4x4-n54561-s0",
        labels=False,
        after="B",
    ),
)
STUDENT, TEACHER = "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"
BASELINES = {STUDENT: "raw 0.8B", TEACHER: "raw 2B"}
OFFICIAL = ("test", "test-clean", "bfcl")
METRICS = ("parse_ok", "exact", "lenient_exact")
DEFAULT_JSON = Path("results/methods.json")
DEFAULT_MD = Path("results/methods.md")
_STEP = re.compile(r"-checkpoint-(\d+)$")
TEACHER_RUN = "eval-train-Qwen_Qwen3.5-2B"
DEFAULT_AUDIT = Path("data/audit/test_labels.csv")
DEFAULTS_CATEGORY = "default made explicit"
# The pairwise claims the README states (#68): Holm-adjusted together, and each shown under strict
# and lenient exact and, on test-clean, without the "default made explicit" rows.
CLAIMS = (
    ("B", "raw 2B", "test-clean"),
    ("B", "raw 2B", "bfcl"),
    ("A", "raw 2B", "bfcl"),
    ("D", "B", "test-clean"),
    ("D", "raw 2B", "bfcl"),
    ("D", "B", "bfcl"),
    ("D@500", "B", "test-clean"),
    ("D@500", "B", "bfcl"),
    ("B→D", "B", "test-clean"),
    ("B→D", "B", "bfcl"),
    ("B→D", "D", "bfcl"),
    ("A", "B", "test-clean"),
    ("A", "D", "test-clean"),
    ("A", "D", "bfcl"),
    ("Bf", "B", "test-clean"),
    ("Bf", "B", "bfcl"),
    ("A", "Bf", "test-clean"),
    ("A", "Bf", "bfcl"),
    # Grounded B (#75), set before its run: does a label-free filter do what filtered B's does?
    ("Bg", "B", "test-clean"),
    ("Bg", "B", "bfcl"),
    ("Bf", "Bg", "test-clean"),
    # C and filtered C (ADR 0010), set before their runs: soft against hard targets on the same
    # text, without labels and with them.
    ("C", "B", "test-clean"),
    ("C", "B", "bfcl"),
    ("Cf", "Bf", "test-clean"),
    ("Cf", "Bf", "bfcl"),
)


def select(checkpoints: Sequence[Mapping[str, Any]]) -> tuple[int, list[int]]:
    """ADR 0006: the best strict exact wins, an exact tie goes to the earlier step. Also the steps
    whose score lies inside the winner's 95% interval: statistically tied with it."""
    best = max(checkpoints, key=lambda c: (c["exact"], -c["step"]))
    low, high = best["exact_ci"]
    tied = [c["step"] for c in checkpoints if c is not best and low <= c["exact"] <= high]
    return best["step"], sorted(tied)


def gains(score: float, raw: float, teacher: float, a: float) -> dict[str, float]:
    """Gap recovered towards the teacher, and share of A's gain (GLOSSARY)."""
    return {
        "gap_recovered": round((score - raw) / (teacher - raw), 4),
        "share_of_a": round((score - raw) / (a - raw), 4),
    }


def teacher_counts(
    records: Sequence[Mapping[str, Any]], examples: Mapping[int, Example]
) -> dict[str, Any]:
    """How many teacher outputs each B variant keeps, how many of those match gold, and how much
    of the train rows whose gold values aren't all in the query each keeps."""
    out: dict[str, Any] = {"rows": len(records)}
    kept: dict[str, list[Mapping[str, Any]]] = {
        "unfiltered": [],
        "valid": [],
        "grounded": [],
        "lenient": [],
    }
    for r in records:
        ex = examples[r["id"]]
        verdict = validity(r["prediction"], ex["tools"])
        out[verdict] = out.get(verdict, 0) + 1
        if verdict != "unparsed":
            kept["unfiltered"].append(r)
        if verdict == "valid":
            kept["valid"].append(r)
            calls = parse_calls(r["prediction"])
            if calls is not None and grounded(calls, ex):
                kept["grounded"].append(r)
        if r["lenient_exact"]:
            kept["lenient"].append(r)
    # The same check on gold: how often it rejects a correct answer, since xLAM doesn't always mark
    # its optional arguments as optional.
    out["gold_not_valid"] = sum(
        validity(json.dumps(examples[r["id"]]["answers"]), examples[r["id"]]["tools"]) != "valid"
        for r in records
    )
    not_in_query = {
        r["id"] for r in records if argument_source(examples[r["id"]]) == "not_in_query"
    }
    out["not_in_query"] = len(not_in_query)

    def share(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        return round(float(sum(r[key] for r in rows)) / len(rows), 4)

    out["variants"] = {
        name: {
            "kept": len(rows),
            "exact": share(rows, "exact"),
            "lenient": share(rows, "lenient_exact"),
            "not_in_query_kept": sum(r["id"] in not_in_query for r in rows),
        }
        for name, rows in kept.items()
        if rows
    }
    return out


def _eval_dir(split: str, run: str) -> str:
    """The eval run an official split is read from: test-clean comes from the test run."""
    return f"eval-{'test' if split == 'test-clean' else split}-{run.replace('/', '_')}"


def _records(runs: Path, run: str, split: str, clean: set[int]) -> list[dict[str, Any]] | None:
    """A run's records for an official split (test-clean from the test run), or None if absent."""
    path = runs / run
    if not (path / "summary.json").exists():
        return None
    _, records = read_run(path)
    return [r for r in records if r["id"] in clean] if split == "test-clean" else records


def _paired(
    a: Mapping[Any, bool] | None, b: Mapping[Any, bool] | None, ids: Sequence[Any] | None = None
) -> dict[str, Any] | None:
    """McNemar of a against b on the rows of b (or on `ids`), or None if either is missing."""
    if a is None or b is None:
        return None
    order = sorted(b if ids is None else ids, key=str)
    return mcnemar([a[i] for i in order], [b[i] for i in order])


def _calls(records: Sequence[Mapping[str, Any]]) -> dict[Any, Any]:
    """Each row's calls in the form exact compares, or None where the output doesn't parse."""
    out: dict[Any, Any] = {}
    for r in records:
        calls = parse_calls(r["prediction"])
        out[r["id"]] = None if calls is None else call_multiset(calls)
    return out


def _bfcl_along(
    records: Sequence[Mapping[str, Any]], teacher: Sequence[Mapping[str, Any]] | None
) -> dict[str, Any]:
    """A checkpoint's BFCL score, and how often it makes the same calls as the teacher (#70):
    on all rows, and on the rows where the teacher's parsed answer is wrong (copying its errors).
    Agreement on all rows rises with accuracy alone; the second count is the one that tells
    imitation from getting better."""
    block = metric_block(records, ("exact",))
    if teacher is None:
        return block
    mine, theirs = _calls(records), _calls(teacher)
    errors = [r["id"] for r in teacher if not r["exact"] and theirs[r["id"]] is not None]
    same = {i for i, c in mine.items() if c is not None and c == theirs.get(i)}
    return block | {
        "agrees_with_teacher": round(len(same) / len(mine), 4),
        "copies_teacher_errors": len(same.intersection(errors)),
        "teacher_errors": len(errors),
    }


def build(
    runs: Path, splits_path: Path = DEFAULT_SPLITS_PATH, audit_path: Path = DEFAULT_AUDIT
) -> dict[str, Any]:
    clean = set(json.loads(splits_path.read_text())["test-clean"])
    with audit_path.open() as f:
        defaults_rows = {
            int(r["id"]) for r in csv.DictReader(f) if r["category"] == DEFAULTS_CATEGORY
        } & clean
    exact: dict[tuple[str, str], dict[Any, bool]] = {}
    lenient: dict[tuple[str, str], dict[Any, bool]] = {}

    def keep(who: str, split: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        exact[who, split] = {r["id"]: r["exact"] for r in records}
        if split != "bfcl":
            lenient[who, split] = {r["id"]: r["lenient_exact"] for r in records}
        return metric_block(records, ("exact",) if split == "bfcl" else METRICS)

    validation = {
        label: keep(
            model, "validation", read_run(runs / f"eval-validation-{model.replace('/', '_')}")[1]
        )
        for model, label in BASELINES.items()
    }
    official_baselines: dict[str, dict[str, Any]] = {}
    for model, label in BASELINES.items():
        official_baselines[label] = {}
        for split in OFFICIAL:
            records = _records(runs, _eval_dir(split, model), split, clean)
            if records is not None:
                official_baselines[label][split] = keep(model, split, records)

    teacher_bfcl = _records(runs, _eval_dir("bfcl", TEACHER), "bfcl", clean)
    methods: dict[str, Any] = {}
    for m in METHODS:
        start = 0.0
        if m.after is not None:
            if m.after not in methods:
                continue
            first = methods[m.after]
            start = next(c for c in first["checkpoints"] if c["step"] == first["selected"])[
                "gpu_hours"
            ]
        checkpoints = []
        for path in sorted(runs.glob(f"eval-validation-{m.train_run}-checkpoint-*")):
            match = _STEP.search(path.name)
            if match is None:
                continue
            summary, records = read_run(path)
            step = int(match.group(1))
            block = keep(f"{m.key}@{step}", "validation", records)
            # BFCL at every checkpoint scored there: explanation only, never selection (#70).
            bfcl = _records(
                runs, _eval_dir("bfcl", f"{m.train_run}-checkpoint-{step}"), "bfcl", clean
            )
            checkpoints.append(
                {"step": step, "gpu_hours": round(start + summary["adapter"]["gpu_hours"], 3)}
                | block
                | {"bfcl": None if bfcl is None else _bfcl_along(bfcl, teacher_bfcl)}
            )
        if not checkpoints:
            continue
        checkpoints.sort(key=lambda c: c["step"])
        step, tied = select(checkpoints)
        exact[m.key, "validation"] = exact[f"{m.key}@{step}", "validation"]
        lenient[m.key, "validation"] = lenient[f"{m.key}@{step}", "validation"]
        official = {}
        for split in OFFICIAL:
            run = (
                f"eval-{'test' if split == 'test-clean' else split}-{m.train_run}-checkpoint-{step}"
            )
            records = _records(runs, run, split, clean)
            if records is not None:
                official[split] = keep(m.key, split, records)
        methods[m.key] = {
            "name": m.name,
            "labels": m.labels,
            "after": m.after,
            "train_run": m.train_run,
            "checkpoints": checkpoints,
            "selected": step,
            "tied": tied,
            "official": official,
        }

    def score(who: str, split: str) -> float | None:
        if who in BASELINES.values():
            block = validation[who] if split == "validation" else official_baselines[who].get(split)
        else:
            m = methods.get(who)
            if m is None:
                return None
            block = (
                next(c for c in m["checkpoints"] if c["step"] == m["selected"])
                if split == "validation"
                else m["official"].get(split)
            )
        return None if block is None else float(block["exact"])

    at_b_budget: dict[str, Any] = {}
    if "B" in methods:
        b = methods["B"]
        budget = next(c for c in b["checkpoints"] if c["step"] == b["selected"])["gpu_hours"]
        for m in METHODS:
            if m.labels or m.key == "B" or m.key not in methods:
                continue
            within = [c for c in methods[m.key]["checkpoints"] if c["gpu_hours"] <= budget]
            # A pick that already fits B's budget is in the headline; listing it again adds nothing.
            if not within or any(c["step"] == methods[m.key]["selected"] for c in within):
                continue
            step, _ = select(within)
            point = next(c for c in within if c["step"] == step)
            who = f"{m.key}@{step}"
            official = {}
            for split in OFFICIAL:
                records = _records(
                    runs, _eval_dir(split, f"{m.train_run}-checkpoint-{step}"), split, clean
                )
                if records is not None:
                    official[split] = keep(who, split, records)
            vs_b = {}
            for split in ("validation", *official):
                strict = _paired(exact.get(("B", split)), exact[who, split])
                if strict is None:
                    continue
                vs_b[split] = strict | {
                    "lenient": _paired(lenient.get(("B", split)), lenient.get((who, split)))
                }
            at_b_budget[m.key] = point | {"official": official, "vs_b": vs_b}

    headline = {}
    for key in methods:
        for split in ("validation", "test-clean"):
            values = [score(w, split) for w in (key, "raw 0.8B", "raw 2B", "A")]
            if all(v is not None for v in values):
                score_m, raw, t2b, a = (float(v) for v in values if v is not None)
                headline[f"{key}/{split}"] = gains(score_m, raw, t2b, a)

    ids = {"raw 0.8B": STUDENT, "raw 2B": TEACHER}
    comparisons = []
    for key in methods:
        for other in ("raw 0.8B", "raw 2B", *(k for k in methods if k < key)):
            for split in ("validation", *OFFICIAL):
                first = ids.get(other, other)
                result = _paired(exact.get((first, split)), exact.get((key, split)))
                if result is None:
                    continue
                loose = _paired(lenient.get((first, split)), lenient.get((key, split)))
                comparisons.append(
                    {"a": other, "b": key, "eval": split} | result | {"lenient": loose}
                )

    claims: list[dict[str, Any]] = []
    claims_missing: list[str] = []
    for one, two, split in CLAIMS:
        ka, kb = ids.get(one, one), ids.get(two, two)
        result = _paired(exact.get((ka, split)), exact.get((kb, split)))
        if result is None:
            # Left out of Holm's family too; the report lists it so the family can't shrink unseen.
            claims_missing.append(f"{one} vs {two}, {split}")
            continue
        without = None
        if split == "test-clean":
            rest = sorted(clean - defaults_rows)
            without = _paired(exact[ka, split], exact[kb, split], rest)
        loose = _paired(lenient.get((ka, split)), lenient.get((kb, split)))
        claims.append(
            {"a": one, "b": two, "eval": split}
            | result
            | {"lenient": loose, "without_defaults": without}
        )
    for claim, adjusted in zip(claims, holm([c["p"] for c in claims]), strict=True):
        claim["holm_p"] = adjusted

    defaults: dict[str, Any] = {}
    whos = [*BASELINES.values(), *methods, *(f"{k}@{c['step']}" for k, c in at_b_budget.items())]
    for who in whos:
        rows = exact.get((ids.get(who, who), "test-clean"))
        if rows is None or not defaults_rows:
            continue
        others = clean - defaults_rows
        defaults[who] = {
            "rows": len(defaults_rows),
            "default_made_explicit": round(
                sum(rows[i] for i in defaults_rows) / len(defaults_rows), 4
            ),
            "rest": round(sum(rows[i] for i in others) / len(others), 4),
        }
    teacher: dict[str, Any] | None = None
    if (runs / TEACHER_RUN / "summary.json").exists():
        t_summary, t_records = read_run(runs / TEACHER_RUN)
        train = {ex["id"]: ex for ex in load_split("train", splits_path)}
        teacher = teacher_counts(t_records, train) | {
            "gpu_hours": round(t_summary["seconds"] / 3600, 3)
        }
    return {
        "teacher": teacher,
        "baselines": {"validation": validation, "official": official_baselines},
        "methods": methods,
        "headline": headline,
        "at_b_budget": at_b_budget,
        "mcnemar": comparisons,
        "claims": claims,
        "claims_missing": claims_missing,
        "defaults": defaults,
    }


def _pct(block: Mapping[str, Any] | None, metric: str = "exact") -> str:
    if block is None or block.get(metric) is None:
        return "-"
    low, high = block[f"{metric}_ci"]
    return f"{100 * block[metric]:.1f} [{100 * low:.1f}, {100 * high:.1f}]"


def _p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def _pair(result: Mapping[str, Any] | None) -> str:
    """'only a / only b, p' for a McNemar result, or '-'."""
    if result is None:
        return "-"
    return f"{result['only_a_right']} / {result['only_b_right']}, {_p(result['p'])}"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    align = ["---"] + ["---:"] * (len(head) - 1)
    return [
        "| " + " | ".join(head) + " |",
        "|" + "|".join(align) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def _teacher_cells(bfcl: Mapping[str, Any] | None) -> list[str]:
    if bfcl is None or "agrees_with_teacher" not in bfcl:
        return ["-", "-"]
    return [
        f"{100 * bfcl['agrees_with_teacher']:.1f}",
        f"{bfcl['copies_teacher_errors']} / {bfcl['teacher_errors']}",
    ]


def render(data: Mapping[str, Any]) -> str:
    methods = data["methods"]
    base = data["baselines"]
    out = [
        "# Method results",
        "",
        "Issues #42, #46, #54, #58, #60, #75. Generated by `uv run --no-sync fc-report --methods` "
        "from the runs in `runs/`; numbers: [`methods.json`](methods.json). Strict exact with "
        "95% intervals. Checkpoints are chosen on validation (ADR 0006); test, test-clean and "
        "BFCL are run only for the selected one (B's and D's other checkpoints also have BFCL, "
        "for explanation only: BFCL along the run). Label-free methods are the main question; "
        "methods with labels are the reference (ADR 0007). The baseline test and BFCL runs "
        "predate #30, so their `splits_sha256` differs from the method runs'; #30 changed only "
        "train, and the validation, test and test-clean ids are identical.",
        "",
        "## Headline",
        "",
    ]
    head = ["Model", "Labels", "GPU-h", "Validation", *OFFICIAL]
    rows = [
        [
            label,
            "-",
            "0",
            _pct(base["validation"][label]),
            *(_pct(base["official"][label].get(s)) for s in OFFICIAL),
        ]
        for label in BASELINES.values()
    ]
    for key, m in methods.items():
        sel = next(c for c in m["checkpoints"] if c["step"] == m["selected"])
        rows.append(
            [
                f"**{key}**, {m['name']} (step {m['selected']})",
                "yes" if m["labels"] else "no",
                f"{sel['gpu_hours']:.2f}",
                _pct(sel),
                *(_pct(m["official"].get(s)) for s in OFFICIAL),
            ]
        )
    out += _table(head, rows)
    teacher_h = f" ({data['teacher']['gpu_hours']:.2f} h)" if data["teacher"] else ""
    out += [
        "",
        "GPU-hours are training step time plus, for B, Bf and Bg, the teacher's generation over "
        f"the train split{teacher_h}; D's steps include its own sampling and the teacher's forward "
        "pass. B→D is a pipeline, not a method: B's selected checkpoint continued with D's loss "
        "(lr 5e-5, stopped at step 1000 of the epoch), its GPU-hours counted on top of B's. "
        "Validation scoring is not included (ADR 0006, ADR 0008).",
        "",
        "Gap recovered = (method - raw 0.8B) / (raw 2B - raw 0.8B); share of A's gain = "
        "(method - raw 0.8B) / (A - raw 0.8B). #2's success criterion is a gap recovered of at "
        "least 50% on test-clean for the best label-free method.",
        "",
    ]
    out += _table(
        ["Method", "Eval", "Gap recovered", "Share of A's gain"],
        [
            [
                k.split("/")[0],
                k.split("/")[1],
                f"{100 * v['gap_recovered']:.0f}%",
                f"{100 * v['share_of_a']:.0f}%",
            ]
            for k, v in data["headline"].items()
        ],
    )
    if data["claims"]:
        out += [
            "",
            "## README claims",
            "",
            "Each pairwise comparison the README states, as first / second right on the rows where "
            "only one is (#68). Strict exact is the headline; Holm adjusts its p-values over the "
            "claims listed. Lenient exact ignores optional arguments passed at their declared "
            "default or null, on either side (BFCL has no lenient score). On test-clean, the last "
            'column leaves out the rows the audit tags "default made explicit" (see below). The '
            "lenient and last columns give unadjusted p-values.",
            "",
        ]
        if data["claims_missing"]:
            out += [f"Not yet measured: {'; '.join(data['claims_missing'])}.", ""]
        out += _table(
            ["First", "Second", "Eval", "Strict", "Holm p", "Lenient", "Strict, no default rows"],
            [
                [
                    c["a"],
                    c["b"],
                    c["eval"],
                    _pair(c),
                    _p(c["holm_p"]),
                    _pair(c["lenient"]),
                    _pair(c["without_defaults"]),
                ]
                for c in data["claims"]
            ],
        )
    if data["defaults"]:
        n = next(iter(data["defaults"].values()))["rows"]
        out += [
            "",
            "## Default made explicit",
            "",
            f'Strict exact on the {n} test-clean rows the audit tags "default made explicit", '
            "and on the rest. On these rows gold and a baseline differed only by an optional "
            "argument at its default, in either direction: gold wrote it and the model left it "
            "out, or the reverse. Which way gold goes is xLAM's labelling habit, so a method "
            "can gain or lose here by matching that habit, though some of these rows also hold "
            "content errors. The audit only looked at rows a baseline got wrong, so the baselines "
            "score low here by construction, and the rest still holds rows where gold passes a "
            "default that no audited model disagreed with.",
            "",
        ]
        out += _table(
            ["Model", f"Default made explicit ({n})", "Rest"],
            [
                [who, f"{100 * v['default_made_explicit']:.1f}", f"{100 * v['rest']:.1f}"]
                for who, v in data["defaults"].items()
            ],
        )
    if data["at_b_budget"]:
        b = methods["B"]
        b_h = next(c for c in b["checkpoints"] if c["step"] == b["selected"])["gpu_hours"]
        out += [
            "",
            "## At B's budget",
            "",
            f"Each label-free method at its best checkpoint on validation within B's {b_h:.2f} "
            "GPU-hours (ADR 0008), with its official scores, against B's selected checkpoint. "
            "Only B / only it right are the McNemar counts on each eval.",
            "",
        ]
        out += _table(
            ["Method", "Step", "GPU-h", "Validation", *OFFICIAL],
            [
                [key, str(c["step"]), f"{c['gpu_hours']:.2f}", _pct(c)]
                + [_pct(c["official"].get(s)) for s in OFFICIAL]
                for key, c in data["at_b_budget"].items()
            ],
        )
        out += [""]
        out += _table(
            [
                "Method",
                "Eval",
                "Only B right",
                "Only it right",
                "p",
                "Lenient: only B / only it, p",
            ],
            [
                [
                    key,
                    split,
                    str(v["only_a_right"]),
                    str(v["only_b_right"]),
                    _p(v["p"]),
                    _pair(v.get("lenient")),
                ]
                for key, c in data["at_b_budget"].items()
                for split, v in c["vs_b"].items()
            ],
        )
    out += ["", "## Validation curves", ""]
    for key, m in methods.items():
        out += [f"### {key}, {m['name']} (`{m['train_run']}`)", ""]
        out += _table(
            ["Step", "GPU-h", *METRICS],
            [
                [
                    f"**{c['step']}**" if c["step"] == m["selected"] else str(c["step"]),
                    f"{c['gpu_hours']:.2f}",
                    *(_pct(c, x) for x in METRICS),
                ]
                for c in m["checkpoints"]
            ],
        )
        tied = ", ".join(str(s) for s in m["tied"]) or "none"
        out += [
            "",
            f"Selected: step {m['selected']}. Inside its interval (statistically tied): {tied}.",
            "",
        ]
    # A curve needs two points: a method with only its selected checkpoint on BFCL is left out.
    along = {
        k: m for k, m in methods.items() if sum(c["bfcl"] is not None for c in m["checkpoints"]) > 1
    }
    if along:
        out += [
            "## BFCL along the run",
            "",
            "Explanation only (#70): these scores never choose a checkpoint or a method, and the "
            "selected checkpoints (in bold) keep their official numbers. Agreement with the 2B is "
            "the share of BFCL rows where the checkpoint makes the same calls as the raw 2B, both "
            "parsed; it rises with accuracy alone, since two right answers agree. Copying the "
            "2B's errors counts the rows where the 2B's parsed answer is wrong and the checkpoint "
            "makes the same calls: that one separates imitating the teacher from getting better. "
            "The intervals overlap almost everywhere; compare two checkpoints with a paired test, "
            "not by overlap.",
            "",
        ]
        for key, m in along.items():
            out += [f"### {key}, {m['name']}", ""]
            out += _table(
                [
                    "Step",
                    "GPU-h",
                    "Validation exact",
                    "BFCL exact",
                    "Agrees with the 2B",
                    "Copies the 2B's errors",
                ],
                [
                    [
                        f"**{c['step']}**" if c["step"] == m["selected"] else str(c["step"]),
                        f"{c['gpu_hours']:.2f}",
                        _pct(c),
                        _pct(c["bfcl"]),
                        *_teacher_cells(c["bfcl"]),
                    ]
                    for c in m["checkpoints"]
                ],
            )
            out += [""]
    out += ["## Paired comparisons", "", "Exact McNemar test on strict exact, row by row.", ""]
    out += _table(
        [
            "A",
            "B",
            "Eval",
            "Rows",
            "Only A right",
            "Only B right",
            "p",
            "Lenient: only A / only B, p",
        ],
        [
            [
                c["a"],
                c["b"],
                c["eval"],
                f"{c['n']:,}",
                str(c["only_a_right"]),
                str(c["only_b_right"]),
                _p(c["p"]),
                _pair(c.get("lenient")),
            ]
            for c in data["mcnemar"]
        ],
    )
    teacher = data["teacher"]
    if teacher:
        variants = teacher["variants"]
        out += [
            "",
            "## Teacher outputs (method B)",
            "",
            f"The 2B answered all {teacher['rows']:,} train rows in {teacher['gpu_hours']:.2f} "
            "GPU-hours (#44). Rows each variant would train on, how many of them match gold, and "
            f"how many of the {teacher['not_in_query']:,} train rows whose gold values aren't all "
            "written in the query each keeps:",
            "",
            *_table(
                [
                    "Variant",
                    "Uses labels",
                    "Rows kept",
                    "Exact vs gold",
                    "Lenient vs gold",
                    "Values not in query, kept",
                ],
                [
                    [
                        label,
                        labels,
                        f"{variants[key]['kept']:,}",
                        f"{100 * variants[key]['exact']:.1f}",
                        f"{100 * variants[key]['lenient']:.1f}",
                        f"{variants[key]['not_in_query_kept']:,}",
                    ]
                    for key, label, labels in (
                        ("unfiltered", "unfiltered (parses), method B", "no"),
                        (
                            "valid",
                            "validity check (tools offered, required arguments present)",
                            "no",
                        ),
                        (
                            "grounded",
                            "validity check and every value in the query, grounded B",
                            "no",
                        ),
                        ("lenient", "lenient match with gold, filtered B", "yes"),
                    )
                    if key in variants
                ],
            ),
            "",
            f"The validity check drops {teacher.get('unknown function', 0):,} outputs calling a "
            f"tool that wasn't offered and {teacher.get('missing required', 0):,} missing an "
            "argument not marked optional. The same rule rejects "
            f"{teacher['gold_not_valid']:,} gold answers, because xLAM doesn't always mark "
            "optional arguments, so only the unknown-function count is a clean signal. It moves "
            "the kept rows' agreement with gold from "
            f"{100 * variants['unfiltered']['exact']:.1f}% "
            f"to {100 * variants['valid']['exact']:.1f}%, "
            "so it is reported, not trained (ADR 0007). Grounded B adds that every argument value "
            "is written in the query: still label-free, but by construction it drops most rows "
            "whose answer needs a value the query doesn't spell out (#74).",
        ]
    return "\n".join(out) + "\n"
