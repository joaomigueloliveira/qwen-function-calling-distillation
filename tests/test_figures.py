import json
from pathlib import Path
from typing import Any

import pytest

from fc_distill.figures import bfcl_along, cost_accuracy, main, readouts, wandb_links


def _ckpt(step: int, gpu_hours: float, exact: float, bfcl: float | None = None) -> dict[str, Any]:
    return {
        "step": step,
        "gpu_hours": gpu_hours,
        "exact": exact,
        "bfcl": None
        if bfcl is None
        else {"exact": bfcl, "exact_ci": [round(bfcl - 0.03, 4), round(bfcl + 0.03, 4)]},
    }


def _method(
    name: str, labels: bool, checkpoints: list[dict[str, Any]], selected: int, test_clean: float
) -> dict[str, Any]:
    return {
        "name": name,
        "labels": labels,
        "train_run": f"train-{name}",
        "wandb_run": f"id-{name}",
        "checkpoints": checkpoints,
        "selected": selected,
        "official": {
            "test-clean": {
                "exact": test_clean,
                "exact_ci": [round(test_clean - 0.02, 4), round(test_clean + 0.02, 4)],
            }
        },
    }


DATA: dict[str, Any] = {
    "defaults": {
        "raw 0.8B": {"rows": 123, "default_made_explicit": 0.17},
        "A": {"rows": 123, "default_made_explicit": 0.79},
        "B": {"rows": 123, "default_made_explicit": 0.42},
        "D@500": {"rows": 123, "default_made_explicit": 0.27},
    },
    "formats": {
        "raw 0.8B": {"n": 1625, "fenced": 0.002, "multi_line": 0.31},
        "A": {"n": 1625, "fenced": 0.0, "multi_line": 0.0},
        "B": {"n": 1625, "fenced": 0.0, "multi_line": 0.0},
    },
    "baselines": {
        "validation": {"raw 0.8B": {"exact": 0.47}, "raw 2B": {"exact": 0.68}},
        "official": {
            "raw 0.8B": {"test-clean": {"exact": 0.55}, "bfcl": {"exact": 0.645}},
            "raw 2B": {"test-clean": {"exact": 0.77}, "bfcl": {"exact": 0.827}},
        },
    },
    "methods": {
        "A": _method(
            "gold", True, [_ckpt(500, 0.5, 0.79), _ckpt(1000, 1.1, 0.85, 0.84)], 1000, 0.93
        ),
        "B": _method(
            "teacher",
            False,
            [
                _ckpt(500, 2.0, 0.62, 0.73),
                _ckpt(1000, 2.5, 0.70, 0.78),
                _ckpt(1500, 3.1, 0.69, 0.79),
            ],
            1000,
            0.83,
        ),
    },
}


def test_cost_accuracy_puts_every_checkpoint_on_validation_and_each_pick_on_test_clean() -> None:
    fig = cost_accuracy(DATA)

    b = [
        (r["step"], r["gpu_hours"], r["exact"], r["pick"])
        for r in fig["checkpoints"]
        if r["method"] == "B"
    ]
    assert b == [(500, 2.0, 0.62, False), (1000, 2.5, 0.70, True), (1500, 3.1, 0.69, False)]
    assert {r["method"]: r["labels"] for r in fig["checkpoints"]} == {"A": True, "B": False}
    picks = {r["method"]: (r["step"], r["gpu_hours"], r["exact"], r["ci"]) for r in fig["picks"]}
    assert picks == {"A": (1000, 1.1, 0.93, [0.91, 0.95]), "B": (1000, 2.5, 0.83, [0.81, 0.85])}
    assert {(r["model"], r["split"], r["exact"]) for r in fig["baselines"]} == {
        ("raw 0.8B", "validation", 0.47),
        ("raw 2B", "validation", 0.68),
        ("raw 0.8B", "test-clean", 0.55),
        ("raw 2B", "test-clean", 0.77),
    }


def test_bfcl_along_keeps_the_checkpoints_scored_on_bfcl_with_the_baselines() -> None:
    fig = bfcl_along(DATA, ("B", "A"))

    rows = [(r["method"], r["step"], r["gpu_hours"], r["exact"], r["pick"]) for r in fig["runs"]]
    assert rows == [
        ("B", 500, 2.0, 0.73, False),
        ("B", 1000, 2.5, 0.78, True),
        ("B", 1500, 3.1, 0.79, False),
        ("A", 1000, 1.1, 0.84, True),  # A's step 500 wasn't scored on BFCL
    ]
    assert fig["runs"][0]["ci"] == [0.70, 0.76]
    assert {(r["model"], r["exact"]) for r in fig["baselines"]} == {
        ("raw 0.8B", 0.645),
        ("raw 2B", 0.827),
    }


def test_readouts_join_defaults_and_format_per_model_in_report_order() -> None:
    rows = readouts(DATA)

    assert rows == [
        {"model": "raw 0.8B", "default_made_explicit": 0.17, "fenced": 0.002, "multi_line": 0.31},
        {"model": "raw 2B", "default_made_explicit": None, "fenced": None, "multi_line": None},
        {"model": "A", "default_made_explicit": 0.79, "fenced": 0.0, "multi_line": 0.0},
        {"model": "B", "default_made_explicit": 0.42, "fenced": 0.0, "multi_line": 0.0},
    ]  # D@500 is a checkpoint, not a method: left out


def test_wandb_links_point_at_each_training_run_in_the_public_project() -> None:
    data = {"methods": DATA["methods"] | {"X": DATA["methods"]["A"] | {"wandb_run": None}}}

    assert wandb_links(data, ("B", "X")) == {
        "B": "https://forge.coreweave.com/wandb/joaomigueloliveira30/"
        "qwen-function-calling-distillation/runs/id-teacher"
    }  # a run never logged to W&B gets no link


def test_fc_figures_writes_each_figure_as_json_and_light_and_dark_svg(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    results = tmp_path / "methods.json"
    results.write_text(json.dumps(DATA))

    main(["--results", str(results), "--out", str(tmp_path / "figures")])

    for name in ("cost-accuracy", "bfcl-along", "readouts"):
        figure = json.loads((tmp_path / "figures" / f"{name}.json").read_text())
        assert figure["source"] == "results/methods.json"
        assert figure["wandb"]  # every figure links the runs it draws
        for svg in (f"{name}.svg", f"{name}-dark.svg"):
            assert (tmp_path / "figures" / svg).read_text().lstrip().startswith("<?xml")
