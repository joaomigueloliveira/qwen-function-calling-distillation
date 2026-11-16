import json
from pathlib import Path
from typing import Any

import pytest

from fc_distill import methods
from fc_distill.methods import (
    METHODS,
    build,
    gains,
    output_format,
    render,
    select,
    teacher_counts,
    validity,
)
from tests.factories import make_example

TOOL = {
    "name": "weather",
    "description": "",
    "parameters": {
        "city": {"type": "str", "description": ""},
        "units": {"type": "str, optional", "description": "", "default": "C"},
    },
}


def _ckpt(step: int, exact: float, ci: tuple[float, float]) -> dict[str, Any]:
    return {"step": step, "exact": exact, "exact_ci": list(ci)}


def test_select_takes_the_best_and_breaks_an_exact_tie_towards_the_earlier_step() -> None:
    checkpoints = [
        _ckpt(500, 0.60, (0.57, 0.63)),
        _ckpt(1000, 0.70, (0.67, 0.73)),
        _ckpt(1500, 0.70, (0.67, 0.73)),
        _ckpt(2000, 0.68, (0.65, 0.71)),
    ]
    assert select(checkpoints) == (1000, [1500, 2000])


def test_gains_against_the_teacher_gap_and_against_method_a() -> None:
    # raw 50, teacher 70, A 90: a method at 80 recovers 150% of the gap and 75% of A's gain
    assert gains(0.80, 0.50, 0.70, 0.90) == {"gap_recovered": 1.5, "share_of_a": 0.75}


@pytest.mark.parametrize(
    ("prediction", "verdict"),
    [
        ('[{"name": "weather", "arguments": {"city": "Porto"}}]', "valid"),
        ('[{"name": "weather", "arguments": {"units": "F"}}]', "missing required"),
        ('[{"name": "forecast", "arguments": {"city": "Porto"}}]', "unknown function"),
        ("Sure! The weather in Porto", "unparsed"),
    ],
)
def test_validity_checks_teacher_calls_without_gold(prediction: str, verdict: str) -> None:
    assert validity(prediction, [TOOL]) == verdict


def _write_run(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def _rows(ids: range, right: set[int]) -> list[dict[str, Any]]:
    return [
        {"id": i, "prediction": "[]", "parse_ok": True, "exact": i in right}
        | {"lenient_exact": i in right}
        for i in ids
    ]


def test_build_selects_per_method_compares_and_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test-clean": [0, 1, 2, 3]}))
    val = range(10)
    # validation: raw 0.8B gets 2 right, the 2B 6, A 10, B's best checkpoint 7
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-0.8B", _rows(val, {0, 1}), {})
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-2B", _rows(val, set(range(6))), {})
    a, b, *_ = METHODS
    for method, scores in ((a, {500: 8, 1000: 10}), (b, {500: 7, 1000: 6})):
        for step, k in scores.items():
            _write_run(
                tmp_path / f"eval-validation-{method.train_run}-checkpoint-{step}",
                _rows(val, set(range(k))),
                {"adapter": {"gpu_hours": step / 1000}},
            )
    # official runs exist only for A's selected checkpoint and the raw 0.8B
    test = range(6)
    _write_run(tmp_path / f"eval-test-{a.train_run}-checkpoint-1000", _rows(test, set(test)), {})
    _write_run(tmp_path / "eval-test-Qwen_Qwen3.5-0.8B", _rows(test, {0}), {})
    teacher = [
        {"id": 0, "prediction": '[{"name": "weather", "arguments": {"city": "P"}}]'}
        | {"exact": True, "lenient_exact": True},
        {"id": 1, "prediction": "no", "exact": False, "lenient_exact": False},
    ]
    _write_run(tmp_path / "eval-train-Qwen_Qwen3.5-2B", teacher, {"seconds": 3600.0})
    examples = [make_example(i) | {"tools": [TOOL]} for i in range(2)]
    monkeypatch.setattr(methods, "load_split", lambda split, path: examples)

    data = build(tmp_path, splits)

    assert data["methods"]["A"]["selected"] == 1000
    assert data["methods"]["B"]["selected"] == 500
    assert data["methods"]["A"]["official"]["test-clean"]["n"] == 4
    assert "test" not in data["methods"]["B"]["official"]
    # B at 70: (70 - 20) / (60 - 20) of the gap, (70 - 20) / (100 - 20) of A's gain
    assert data["headline"]["B/validation"] == {"gap_recovered": 1.25, "share_of_a": 0.625}
    a_vs_b = next(c for c in data["mcnemar"] if (c["a"], c["b"]) == ("A", "B"))
    assert (a_vs_b["only_a_right"], a_vs_b["only_b_right"]) == (3, 0)
    assert data["teacher"]["variants"]["unfiltered"] == {
        "kept": 1,
        "exact": 1.0,
        "lenient": 1.0,
        "not_in_query_kept": 1,
    }
    assert data["teacher"]["unparsed"] == 1
    assert set(data["formats"]) == {"raw 0.8B", "A"}  # the ones with a test run
    md = render(data)
    assert "## Output format" in md
    assert "**B**, SFT on teacher outputs (step 500)" in md
    assert "Inside its interval" in md


def test_at_bs_budget_takes_the_best_checkpoint_within_bs_gpu_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test-clean": [0, 1, 2, 3]}))
    val = range(10)
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-0.8B", _rows(val, {0}), {})
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-2B", _rows(val, set(range(5))), {})
    by_key = {m.key: m for m in METHODS}
    # B's best is step 1000 at 3.1 GPU-h. D's checkpoints: 1.0, 2.0 (inside B's budget), then 4.0
    # (nearer to 3.1 than 1.0, but over it) and its best, 6.0. The budget rule takes the best of
    # the first two on validation, step 500, never the nearest or the overall best.
    runs = {
        "B": {500: (1.6, 5), 1000: (3.1, 7)},
        "D": {250: (1.0, 4), 500: (2.0, 6), 1000: (4.0, 7), 1500: (6.0, 8)},
    }
    for key, checkpoints in runs.items():
        for step, (hours, k) in checkpoints.items():
            _write_run(
                tmp_path / f"eval-validation-{by_key[key].train_run}-checkpoint-{step}",
                _rows(val, set(range(k))),
                {"adapter": {"gpu_hours": hours}},
            )
    # Official runs: B's selected checkpoint and D's at-budget one. Test-clean is ids 0-3.
    test = range(6)
    _write_run(
        tmp_path / f"eval-test-{by_key['B'].train_run}-checkpoint-1000", _rows(test, {0, 1, 2}), {}
    )
    _write_run(
        tmp_path / f"eval-test-{by_key['D'].train_run}-checkpoint-500", _rows(test, {0, 4}), {}
    )
    monkeypatch.setattr(methods, "load_split", lambda split, path: [])

    data = build(tmp_path, splits)

    assert data["methods"]["D"]["selected"] == 1500
    budget = data["at_b_budget"]["D"]
    assert (budget["step"], budget["gpu_hours"], budget["exact"]) == (500, 2.0, 0.6)
    assert (
        budget["vs_b"]["validation"]["only_a_right"],
        budget["vs_b"]["validation"]["only_b_right"],
    ) == (1, 0)
    # test-clean: B right on 0, 1, 2; D on 0 only
    assert budget["official"]["test-clean"]["n"] == 4
    assert (
        budget["vs_b"]["test-clean"]["only_a_right"],
        budget["vs_b"]["test-clean"]["only_b_right"],
    ) == (2, 0)
    assert "B" not in data["at_b_budget"]
    md = render(data)
    assert "**D**, on-policy distillation (step 1500)" in md
    assert "| D | 500 | 2.00 |" in md


def test_a_method_selected_within_bs_budget_gets_no_second_entry_at_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test-clean": [0, 1, 2, 3]}))
    val = range(10)
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-0.8B", _rows(val, {0}), {})
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-2B", _rows(val, set(range(5))), {})
    by_key = {m.key: m for m in METHODS}
    # Grounded B's selected checkpoint (1.5 GPU-h) already fits B's 3.1: the headline shows it.
    runs = {"B": {1000: (3.1, 7)}, "Bg": {500: (1.0, 4), 1000: (1.5, 6)}}
    for key, checkpoints in runs.items():
        for step, (hours, k) in checkpoints.items():
            _write_run(
                tmp_path / f"eval-validation-{by_key[key].train_run}-checkpoint-{step}",
                _rows(val, set(range(k))),
                {"adapter": {"gpu_hours": hours}},
            )
    monkeypatch.setattr(methods, "load_split", lambda split, path: [])

    data = build(tmp_path, splits)

    assert data["methods"]["Bg"]["selected"] == 1000
    assert "Bg" not in data["at_b_budget"]


def test_a_method_with_no_checkpoint_within_bs_budget_is_named_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test-clean": [0, 1, 2, 3]}))
    val = range(10)
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-0.8B", _rows(val, {0}), {})
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-2B", _rows(val, set(range(5))), {})
    by_key = {m.key: m for m in METHODS}
    # C's teacher passes alone cost more than B's whole pick: its first checkpoint is past it.
    runs = {"B": {1000: (3.1, 7)}, "C": {500: (3.6, 6), 1000: (4.1, 7)}}
    for key, checkpoints in runs.items():
        for step, (hours, k) in checkpoints.items():
            _write_run(
                tmp_path / f"eval-validation-{by_key[key].train_run}-checkpoint-{step}",
                _rows(val, set(range(k))),
                {"adapter": {"gpu_hours": hours}},
            )
    monkeypatch.setattr(methods, "load_split", lambda split, path: [])

    data = build(tmp_path, splits)

    assert "C" not in data["at_b_budget"]
    assert data["over_b_budget"] == ["C"]
    assert "No checkpoint of C falls within it." in render(data)


def test_a_pipeline_counts_the_gpu_hours_of_the_checkpoint_it_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test-clean": []}))
    val = range(10)
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-0.8B", _rows(val, {0}), {})
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-2B", _rows(val, set(range(5))), {})
    by_key = {m.key: m for m in METHODS}
    # B then D starts from B's selected checkpoint (step 1000, 3.1 GPU-h): its own 0.4 and 1.6
    # GPU-h come on top, so neither of its checkpoints falls inside B's budget.
    runs = {"B": {500: (1.6, 5), 1000: (3.1, 7)}, "B→D": {100: (0.4, 6), 400: (1.6, 7)}}
    for key, checkpoints in runs.items():
        for step, (hours, k) in checkpoints.items():
            _write_run(
                tmp_path / f"eval-validation-{by_key[key].train_run}-checkpoint-{step}",
                _rows(val, set(range(k))),
                {"adapter": {"gpu_hours": hours}},
            )
    monkeypatch.setattr(methods, "load_split", lambda split, path: [])

    data = build(tmp_path, splits)

    pipeline = data["methods"]["B→D"]
    assert [c["gpu_hours"] for c in pipeline["checkpoints"]] == [3.5, 4.7]
    assert pipeline["selected"] == 400
    assert "B→D" not in data["at_b_budget"]
    assert "**B→D**, B then on-policy distillation (step 400)" in render(data)


def _scored(ids: range, exact: set[int], lenient: set[int]) -> list[dict[str, Any]]:
    return [
        {"id": i, "prediction": "[]", "parse_ok": True, "exact": i in exact}
        | {"lenient_exact": i in lenient}
        for i in ids
    ]


def test_claims_are_reported_under_strict_lenient_and_without_the_default_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test-clean": [0, 1, 2, 3]}))
    # The audit calls row 0 sound, but its gold writes out an optional argument at its default.
    audit = tmp_path / "test_labels.csv"
    audit.write_text("id,verdict,category\n0,label_sound,default made explicit\n1,label_sound,\n")
    val = range(10)
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-0.8B", _rows(val, {0}), {})
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-2B", _rows(val, set(range(5))), {})
    by_key = {m.key: m for m in METHODS}
    for key, step in (("B", 1000), ("D", 1000)):
        _write_run(
            tmp_path / f"eval-validation-{by_key[key].train_run}-checkpoint-{step}",
            _rows(val, set(range(6))),
            {"adapter": {"gpu_hours": 5.0}},
        )
    test = range(6)
    # B is right on 0-2. D misses row 0 strictly (it left the default out) but is lenient-right
    # there, and is right on 1, 2 and 4.
    _write_run(
        tmp_path / f"eval-test-{by_key['B'].train_run}-checkpoint-1000",
        _scored(test, {0, 1, 2}, {0, 1, 2}),
        {},
    )
    _write_run(
        tmp_path / f"eval-test-{by_key['D'].train_run}-checkpoint-1000",
        _scored(test, {1, 2, 4}, {0, 1, 2, 4}),
        {},
    )
    monkeypatch.setattr(methods, "load_split", lambda split, path: [])

    data = build(tmp_path, splits, audit)

    claim = next(
        c for c in data["claims"] if (c["a"], c["b"], c["eval"]) == ("D", "B", "test-clean")
    )
    assert (claim["only_a_right"], claim["only_b_right"]) == (0, 1)
    assert (claim["lenient"]["only_a_right"], claim["lenient"]["only_b_right"]) == (0, 0)
    assert (claim["without_defaults"]["n"], claim["without_defaults"]["only_b_right"]) == (3, 0)
    assert claim["holm_p"] >= claim["p"]
    pair = next(c for c in data["mcnemar"] if (c["a"], c["b"], c["eval"]) == ("B", "D", "test"))
    assert (pair["lenient"]["only_a_right"], pair["lenient"]["only_b_right"]) == (0, 1)
    assert data["defaults"]["D"] == {
        "rows": 1,
        "default_made_explicit": 0.0,
        "rest": round(2 / 3, 4),
    }
    # Claims whose runs don't exist here are listed, not silently dropped from Holm's family.
    assert "A vs B, test-clean" in data["claims_missing"]
    md = render(data)
    assert "## README claims" in md
    assert "## Default made explicit" in md


def test_teacher_counts_keep_grounded_outputs_without_looking_at_gold() -> None:
    # Gold says Porto for both rows; the teacher wrote Lisbon, which only row 2's query mentions.
    gold = [{"name": "weather", "arguments": {"city": "Porto"}}]
    examples = {
        i: make_example(i, query=q, answers=gold) | {"tools": [TOOL]}
        for i, q in ((1, "Weather in Porto?"), (2, "Weather in Lisbon?"))
    }
    out = '[{"name": "weather", "arguments": {"city": "Lisbon"}}]'
    records = [{"id": i, "prediction": out, "exact": False, "lenient_exact": False} for i in (1, 2)]
    grounded = teacher_counts(records, examples)["variants"]["grounded"]
    # Kept on row 2 although it's wrong: the rule never reads gold.
    assert (grounded["kept"], grounded["exact"]) == (1, 0.0)


def _bfcl_rows(ids: range, calls: dict[int, str], right: set[int]) -> list[dict[str, Any]]:
    return [
        {"id": i, "parse_ok": True, "exact": i in right, "prediction": calls.get(i, "no call")}
        for i in ids
    ]


def test_bfcl_along_the_run_scores_each_checkpoint_and_its_agreement_with_the_2b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test-clean": []}))
    audit = tmp_path / "test_labels.csv"
    audit.write_text("id,verdict,category\n")
    val, bfcl = range(10), range(4)
    porto = '[{"name": "weather", "arguments": {"city": "Porto"}}]'
    # The same call with its arguments reordered agrees; a different value doesn't.
    porto_again = '[{"arguments": {"city": "Porto"}, "name": "weather"}]'
    lisbon = '[{"name": "weather", "arguments": {"city": "Lisbon"}}]'
    teacher = {0: porto, 1: porto, 2: porto}
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-0.8B", _rows(val, {0}), {})
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-2B", _rows(val, set(range(5))), {})
    _write_run(tmp_path / "eval-bfcl-Qwen_Qwen3.5-2B", _bfcl_rows(bfcl, teacher, {0, 1}), {})
    run = next(m for m in METHODS if m.key == "B").train_run
    for step in (500, 1000, 1500):
        _write_run(
            tmp_path / f"eval-validation-{run}-checkpoint-{step}",
            _rows(val, set(range(7 if step == 1000 else 5))),
            {"adapter": {"gpu_hours": step / 1000}},
        )
    # Step 500: agrees with the 2B on row 0 only. Step 1500: on rows 0-2. Step 1000 has no run.
    _write_run(tmp_path / f"eval-bfcl-{run}-checkpoint-500", _bfcl_rows(bfcl, {0: porto}, {0}), {})
    _write_run(
        tmp_path / f"eval-bfcl-{run}-checkpoint-1500",
        _bfcl_rows(bfcl, {0: porto, 1: porto_again, 2: porto, 3: lisbon}, {0, 1, 3}),
        {},
    )
    monkeypatch.setattr(methods, "load_split", lambda split, path: [])

    data = build(tmp_path, splits, audit)

    along = {c["step"]: c["bfcl"] for c in data["methods"]["B"]["checkpoints"]}
    # The 2B is wrong on row 2 with a parsed answer (its only copyable error); row 3 doesn't parse.
    assert (along[500]["exact"], along[500]["agrees_with_teacher"]) == (0.25, 0.25)
    assert (along[500]["copies_teacher_errors"], along[500]["teacher_errors"]) == (0, 1)
    assert (along[1500]["exact"], along[1500]["agrees_with_teacher"]) == (0.75, 0.75)
    assert along[1500]["copies_teacher_errors"] == 1
    assert along[1000] is None
    md = render(data)
    section = md[md.index("## BFCL along the run") : md.index("## Paired comparisons")]
    assert "| 1500 | 1.50 |" in section
    assert "| 75.0 | 1 / 1 |" in section
    assert "| **1000** | 1.00 |" in section
    assert "| - | - | - |" in section
    # Explanation only: the selected checkpoint is still validation's best.
    assert data["methods"]["B"]["selected"] == 1000


def test_output_format_counts_fenced_and_multi_line_answers_apart() -> None:
    records = [
        {"prediction": '[{"name": "f", "arguments": {}}]'},  # one line, as B's targets
        {"prediction": '```json\n[{"name": "f", "arguments": {}}]\n```'},  # fenced
        {"prediction": '[\n  {"name": "f", "arguments": {}}\n]'},  # pretty-printed
        {"prediction": '  [{"name": "f"}]\n'},  # a trailing newline isn't layout
    ]

    assert output_format(records) == {"n": 4, "fenced": 0.25, "multi_line": 0.25}
