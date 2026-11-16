"""Official baselines report (issue #10): results/baselines.json and results/baselines.md from runs.

uv run fc-report    # reads runs/eval-{test,bfcl}-Qwen_Qwen3.5-{0.8B,2B,4B}
uv run fc-report --wandb    # logs the committed results/baselines.json to W&B, one run per model
uv run fc-report --methods  # results/methods.{json,md}: method checkpoints vs baselines (#46)

Test-clean is scored from the test runs restricted to the test-clean ids, not from separate runs:
the predictions are the same, and a second vLLM run could differ slightly because batched greedy
decoding isn't perfectly deterministic.
"""

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from fc_distill import bfcl
from fc_distill.data import DEFAULT_SPLITS_PATH, Example, load_split
from fc_distill.eval_tracking import log_evals
from fc_distill.evaluate import read_run
from fc_distill.scoring import METRICS
from fc_distill.slices import slice_scores
from fc_distill.stats import mcnemar, metric_block
from fc_distill.tracking import start_run


class Model(NamedTuple):
    id: str
    short: str
    role: str


STUDENT = Model("Qwen/Qwen3.5-0.8B", "0.8B", "student")
TEACHER = Model("Qwen/Qwen3.5-2B", "2B", "teacher")
REFERENCE = Model("Qwen/Qwen3.5-4B", "4B", "reference")
MODELS = (STUDENT, TEACHER, REFERENCE)
# (a, b, eval): paired comparisons reported with McNemar's test.
COMPARISONS = (
    (STUDENT, TEACHER, "test-clean"),
    (STUDENT, TEACHER, "test"),
    (STUDENT, TEACHER, "bfcl"),
    (TEACHER, REFERENCE, "test-clean"),
    (TEACHER, REFERENCE, "test"),
    (TEACHER, REFERENCE, "bfcl"),
)
DEFAULT_JSON = Path("results/baselines.json")
DEFAULT_MD = Path("results/baselines.md")


def xlam_block(
    records: Sequence[Mapping[str, Any]], examples: Mapping[int, Example]
) -> dict[str, Any]:
    """Every metric with its 95% interval, and the slices, for a set of scored rows."""
    rows = [examples[r["id"]] for r in records]
    return metric_block(records, METRICS) | {"slices": slice_scores(rows, records, METRICS)}


def build(runs: Path, splits_path: Path = DEFAULT_SPLITS_PATH) -> dict[str, Any]:
    test = {ex["id"]: ex for ex in load_split("test", splits_path)}
    clean = set(json.loads(splits_path.read_text())["test-clean"])
    models: dict[str, Any] = {}
    exact: dict[tuple[str, str], dict[Any, bool]] = {}
    for model in MODELS:
        tag = model.id.replace("/", "_")
        t_summary, t_records = read_run(runs / f"eval-test-{tag}")
        b_summary, b_records = read_run(runs / f"eval-bfcl-{tag}")
        clean_records = [r for r in t_records if r["id"] in clean]
        models[model.id] = {
            "role": model.role,
            "test": xlam_block(t_records, test) | {"seconds": t_summary["seconds"]},
            "test-clean": xlam_block(clean_records, test),
            "bfcl": {k: b_summary[k] for k in ("overall", "slices", "categories")}
            | {
                "seconds": b_summary["seconds"],
                "rows_calling_a_train_function_name": b_summary[
                    "rows_calling_a_train_function_name"
                ],
            },
            "provenance": {
                "test": t_summary.get("provenance"),
                "bfcl": b_summary.get("provenance"),
            },
        }
        exact[model.id, "test"] = {r["id"]: r["exact"] for r in t_records}
        exact[model.id, "test-clean"] = {r["id"]: r["exact"] for r in clean_records}
        exact[model.id, "bfcl"] = {r["id"]: r["exact"] for r in b_records}
    comparisons = []
    for a, b, split in COMPARISONS:
        ids = sorted(exact[a.id, split], key=str)
        result = mcnemar([exact[a.id, split][i] for i in ids], [exact[b.id, split][i] for i in ids])
        comparisons.append({"a": a.id, "b": b.id, "eval": split} | result)
    return {"models": models, "mcnemar": comparisons}


def _pct(block: Mapping[str, Any], metric: str) -> str:
    x = block.get(metric)
    if x is None:
        return "-"
    low, high = block[f"{metric}_ci"]
    return f"{100 * x:.1f} [{100 * low:.1f}, {100 * high:.1f}]"


def _p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def _table(head: Sequence[str], align: str, rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "|".join(align.split()) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def _headline(models: Mapping[str, Any]) -> list[str]:
    rows = [
        [
            m.id,
            m.role,
            _pct(models[m.id]["test"], "exact"),
            _pct(models[m.id]["test-clean"], "exact"),
            _pct(models[m.id]["bfcl"]["overall"], "exact"),
            _pct(models[m.id]["test-clean"], "lenient_exact"),
        ]
        for m in MODELS
    ]
    head = ["Model", "Role", "Test", "Test-clean", "BFCL", "Test-clean, lenient"]
    return _table(head, "--- --- ---: ---: ---: ---:", rows)


def _starting_line(models: Mapping[str, Any]) -> list[str]:
    s = models[STUDENT.id]["test-clean"]["exact"]
    t = models[TEACHER.id]["test-clean"]["exact"]
    gap = t - s
    return [
        f"On test-clean the student scores **{100 * s:.1f}** and the teacher **{100 * t:.1f}**: "
        f"a gap of **{100 * gap:.1f} points**. #2's success criterion is to recover at least half "
        f"of it, a distilled 0.8B at **{100 * (s + gap / 2):.1f}** or better on test-clean, with "
        "the recovered fraction significant at p < 0.05. #35 settled the test: the point estimate, "
        "with a paired McNemar test against the raw 0.8B.",
    ]


def _mcnemar_table(comparisons: Sequence[Mapping[str, Any]]) -> list[str]:
    short = {m.id: m.short for m in MODELS}
    rows = [
        [
            short[c["a"]],
            short[c["b"]],
            c["eval"],
            f"{c['n']:,}",
            str(c["only_a_right"]),
            str(c["only_b_right"]),
            _p(c["p"]),
        ]
        for c in comparisons
    ]
    head = ["A", "B", "Eval", "Rows", "Only A right", "Only B right", "p"]
    return _table(head, "--- --- --- ---: ---: ---: ---:", rows)


def _metrics_table(blocks: Mapping[str, Mapping[str, Any]], metrics: Sequence[str]) -> list[str]:
    rows = [
        [m.short, f"{blocks[m.id]['n']:,}", *(_pct(blocks[m.id], x) for x in metrics)]
        for m in MODELS
    ]
    return _table(["Model", "Rows", *metrics], "--- ---: " + "---: " * len(metrics), rows)


def _slices_table(models: Mapping[str, Any]) -> list[str]:
    slices = {m.id: models[m.id]["test-clean"]["slices"] for m in MODELS}
    rows = []
    for dim, buckets in slices[STUDENT.id].items():
        for bucket, info in buckets.items():
            cells = [_pct(slices[m.id][dim][bucket], "exact") for m in MODELS]
            s, t = (slices[m.id][dim][bucket]["exact"] for m in (STUDENT, TEACHER))
            diff = "-" if s is None or t is None else f"{100 * (t - s):.1f}"
            star = "*" if info["too_small"] else ""
            rows.append([dim, bucket + star, f"{info['n']:,}", *cells, diff])
    head = ["Slice", "Bucket", "Rows", *(m.short for m in MODELS), "2B - 0.8B"]
    return _table(head, "--- --- ---: " + "---: " * (len(MODELS) + 1), rows)


def _bfcl_groups(models: Mapping[str, Any]) -> list[str]:
    rows = [
        [
            m.short,
            _pct(models[m.id]["bfcl"]["slices"]["simple"], "exact"),
            _pct(models[m.id]["bfcl"]["slices"]["multiple_parallel"], "exact"),
        ]
        for m in MODELS
    ]
    return _table(["Model", "simple", "multiple + parallel"], "--- ---: ---:", rows)


def _provenance(models: Mapping[str, Any]) -> list[str]:
    rows = []
    for m in MODELS:
        r = models[m.id]
        prov = r["provenance"]["test"] or {}
        commit = (prov.get("commit") or "-")[:7] + (" (dirty)" if prov.get("dirty") else "")
        rows.append(
            [
                m.short,
                str(r["test"]["seconds"]),
                str(r["bfcl"]["seconds"]),
                str(prov.get("max_num_seqs") or "default"),
                f"`{(prov.get('model_revision') or '-')[:12]}`",
                f"`{commit}`",
                str(prov.get("vllm") or "-"),
            ]
        )
    head = ["Model", "Test (s)", "BFCL (s)", "max_num_seqs", "Model revision", "Commit", "vLLM"]
    return _table(head, "--- ---: ---: ---: --- --- ---", rows)


def render(data: Mapping[str, Any]) -> str:
    models = data["models"]
    n_clean = models[STUDENT.id]["test-clean"]["n"]
    rows_seen = models[STUDENT.id]["bfcl"]["rows_calling_a_train_function_name"]
    out = [
        "# Baselines",
        "",
        "Issue #10. Generated by `uv run fc-report` from the runs in `runs/`; numbers: "
        "[`baselines.json`](baselines.json). Greedy decoding, thinking off (ADR 0003). Every "
        "number is a percentage with its 95% interval (Wilson for right/wrong metrics). "
        f"Test-clean is scored from the test runs, restricted to its {n_clean:,} rows.",
        "",
        "## Headline",
        "",
        *_headline(models),
        "",
        "Strict exact match is the headline metric. Lenient exact also accepts optional arguments "
        "at their default (#15).",
        "",
        "## Starting line for Phase 2",
        "",
        *_starting_line(models),
        "",
        "## Paired comparisons",
        "",
        "Exact McNemar test on strict exact match, row by row on the same eval.",
        "",
        *_mcnemar_table(data["mcnemar"]),
        "",
        "## All metrics",
    ]
    for split in ("test", "test-clean"):
        blocks = {m.id: models[m.id][split] for m in MODELS}
        out += ["", f"### {split}", "", *_metrics_table(blocks, METRICS)]
    bfcl_blocks = {m.id: models[m.id]["bfcl"]["overall"] for m in MODELS}
    out += [
        "",
        "### BFCL (unseen tools)",
        "",
        *_metrics_table(bfcl_blocks, bfcl.METRICS),
        "",
        f"{rows_seen} of the BFCL rows call a function name that some xLAM train answer calls. "
        "BFCL has no `lenient_exact` (its answers already mark which arguments may be left out) "
        "and no `arg_f1` (its acceptable-value lists have no single gold value per argument). "
        "`bfcl_exact` reimplements BFCL's string normalisation only; see "
        "[`docs/bfcl.md`](../docs/bfcl.md).",
        "",
        "## Slices (strict exact)",
        "",
        "Test-clean. Buckets under 300 rows are marked *.",
        "",
        *_slices_table(models),
        "",
        "BFCL, by category group (simple has 400 rows; the other three are merged to 600, "
        "ADR 0001):",
        "",
        *_bfcl_groups(models),
        "",
        "## Cost and provenance",
        "",
        "Wall-clock time for the full eval on one RTX 3060 (12 GB), vLLM, all rows in one batch. "
        "The 4B runs cap concurrency (`max_num_seqs` 24) to fit in memory, so its time is not "
        "comparable with the others. Later runs compared against these must keep the vLLM "
        "version and `max_num_seqs` fixed, because batched greedy decoding is not perfectly "
        "deterministic.",
        "",
        *_provenance(models),
    ]
    return "\n".join(out) + "\n"


def log_baselines(data: Mapping[str, Any], runs: Path, source_sha256: str) -> None:
    """One W&B eval run per model, holding its test, test-clean and BFCL results as committed, so
    training runs are compared against the official numbers. `source_sha256` (of the results file)
    shows whether a logged run still matches the committed file."""
    for model, r in data["models"].items():
        short = model.split("/")[-1]
        tracker = start_run(
            enabled=True,
            job_type="eval",
            group="baseline",
            run_dir=runs / f"wandb-baseline-{short}",
            name=f"baseline-{short}",
            tags=["baseline", r["role"]],
        )
        tracker.config(
            {
                "model": model,
                "role": r["role"],
                "source": str(DEFAULT_JSON),
                "source_sha256": source_sha256,
                "provenance": r["provenance"],
                "compute": {"gpu_hours": 0.0},
            }
        )
        log_evals(tracker, {e: r[e] for e in ("test", "test-clean", "bfcl")})
        tracker.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument(
        "--wandb",
        action="store_true",
        help=f"log {DEFAULT_JSON} to Weights & Biases instead of rebuilding it",
    )
    ap.add_argument(
        "--methods",
        action="store_true",
        help="write results/methods.{json,md} (fc_distill.methods) instead of the baselines",
    )
    args = ap.parse_args()
    if args.methods:
        from fc_distill import methods

        data = methods.build(args.runs)
        methods.DEFAULT_JSON.write_text(json.dumps(data, indent=1) + "\n")
        methods.DEFAULT_MD.write_text(methods.render(data))
        print(f"wrote {methods.DEFAULT_JSON} and {methods.DEFAULT_MD}")
        return
    if args.wandb:
        raw = DEFAULT_JSON.read_bytes()
        log_baselines(json.loads(raw), args.runs, hashlib.sha256(raw).hexdigest())
        return
    data = build(args.runs)
    DEFAULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_JSON.write_text(json.dumps(data, indent=1) + "\n")
    DEFAULT_MD.write_text(render(data))
    print(f"wrote {DEFAULT_JSON} and {DEFAULT_MD}")


if __name__ == "__main__":
    main()
