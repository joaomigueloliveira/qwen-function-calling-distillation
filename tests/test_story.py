from typing import Any

from fc_distill.story import Story

METHODS: dict[str, Any] = {
    "baselines": {
        "validation": {"raw 0.8B": {"exact": 0.466}, "raw 2B": {"exact": 0.68}},
        "official": {
            "raw 0.8B": {"test-clean": {"exact": 0.5477}, "bfcl": {"exact": 0.645}},
            "raw 2B": {"test-clean": {"exact": 0.768}, "bfcl": {"exact": 0.827}},
        },
    },
    "methods": {
        "A": {
            "checkpoints": [{"step": 3411, "gpu_hours": 3.835, "exact": 0.847}],
            "selected": 3411,
            "official": {"test-clean": {"exact": 0.9323}, "bfcl": {"exact": 0.845}},
        },
        "B": {
            "name": "SFT on teacher outputs",
            "labels": False,
            "wandb_run": "z8e95dog",
            "checkpoints": [
                {"step": 1000, "gpu_hours": 2.554, "exact": 0.7},
                {"step": 1500, "gpu_hours": 3.119, "exact": 0.708},
            ],
            "selected": 1500,
            "official": {
                "test": {"exact": 0.712},
                "test-clean": {"exact": 0.8338},
                "bfcl": {"exact": 0.791},
            },
        },
    },
}
METHODS["claims"] = [
    {
        "a": "B",
        "b": "raw 2B",
        "eval": "test-clean",
        "only_a_right": 159,
        "only_b_right": 52,
        "p": 8.4e-14,
        "holm_p": 1.85e-12,
    },
    {
        "a": "D",
        "b": "B",
        "eval": "bfcl",
        "only_a_right": 62,
        "only_b_right": 28,
        "p": 0.000438,
        "holm_p": 0.00876,
    },
    {
        "a": "C",
        "b": "B",
        "eval": "bfcl",
        "only_a_right": 54,
        "only_b_right": 29,
        "p": 0.00804,
        "holm_p": 0.104,
    },
]
METHODS["headline"] = {"B/test-clean": {"gap_recovered": 1.2987, "share_of_a": 0.7439}}
SERVING: dict[str, Any] = {"speedup": 1.91, "latency_speedup": 2.39}


def test_score_prints_a_models_exact_match_on_a_split_as_a_percentage() -> None:
    story = Story(METHODS, SERVING)

    assert story.score("B") == "83.4%"  # test-clean by default
    assert story.score("raw 2B", "bfcl") == "82.7%"
    assert story.score("B", "validation") == "70.8%"  # its selected checkpoint
    assert story.score("raw 0.8B", "validation") == "46.6%"


def test_claim_prints_a_paired_comparison_with_its_holm_p_value() -> None:
    story = Story(METHODS, SERVING)

    assert story.claim("B", "raw 2B", "test-clean") == "159 vs 52 rows, Holm p < 0.0001"
    assert story.claim("D", "B", "bfcl") == "62 vs 28 rows, Holm p = 0.009"
    assert story.claim("C", "B", "bfcl") == "54 vs 29 rows, Holm p = 0.10"


def test_headline_gives_the_claims_numbers_for_the_headline_method_b() -> None:
    h = Story(METHODS, SERVING).headline()

    assert h == {
        "method": "B",
        "before": "54.8%",
        "after": "83.4%",
        "teacher": "76.8%",
        "with_labels": "93.2%",
        "share_of_a": "74%",
        "gap_recovered": "130%",
        "gpu_hours": "3.1",
    }


def test_serving_gives_the_students_speed_up_over_the_teacher() -> None:
    assert Story(METHODS, SERVING).serving() == {
        "batched": "1.9\u00d7",
        "one_at_a_time": "2.4\u00d7",
    }


def test_facts_gives_a_methods_row_of_the_results_table() -> None:
    assert Story(METHODS, SERVING).facts("B") == {
        "name": "SFT on teacher outputs",
        "labels": "no",
        "step": "1500",
        "gpu_hours": "3.1",
        "validation": "70.8%",
        "test": "71.2%",
        "test_clean": "83.4%",
        "bfcl": "79.1%",
        "wandb": "https://forge.coreweave.com/wandb/joaomigueloliveira30/"
        "qwen-function-calling-distillation/runs/z8e95dog",
    }
