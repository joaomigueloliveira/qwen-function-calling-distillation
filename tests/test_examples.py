from typing import Any

from fc_distill.examples import worked_examples


def row(i: Any, query: str) -> dict[str, Any]:
    return {
        "id": i,
        "query": query,
        "tools": [{"name": "f"}],
        "gold": [{"name": "f", "arguments": {}}],
    }


def rec(prediction: str, exact: bool) -> dict[str, Any]:
    return {"prediction": prediction, "exact": exact}


def runs(split: str, by_model: dict[str, dict[Any, dict[str, Any]]]) -> dict[str, Any]:
    return {split: by_model}


def test_the_student_beats_its_teacher_on_the_shortest_row_where_it_does() -> None:
    xlam = [row(1, "a long query about the weather"), row(2, "short"), row(3, "x")]
    records = runs(
        "test-clean",
        {
            # Row 3 is shortest, but there both are right: it doesn't qualify.
            "B": {1: rec("b1", True), 2: rec("b2", True), 3: rec("b3", True)},
            "raw 2B": {1: rec("t1", False), 2: rec("t2", False), 3: rec("t3", True)},
        },
    )

    (example,) = worked_examples(xlam, [], records, {})

    assert example["key"] == "beats-teacher"
    assert example["id"] == 2
    assert example["answers"] == [
        {"model": "B", "prediction": "b2", "exact": True},
        {"model": "raw 2B", "prediction": "t2", "exact": False},
    ]


def test_on_unseen_tools_d_is_right_where_b_is_wrong_and_ties_go_to_the_lowest_id() -> None:
    bfcl = [row("simple_4", "same"), row("simple_3", "same"), row("simple_1", "longer query")]
    records = runs(
        "bfcl",
        {
            "D": {
                "simple_4": rec("d4", True),
                "simple_3": rec("d3", True),
                "simple_1": rec("d1", True),
            },
            "B": {
                "simple_4": rec("b4", False),
                "simple_3": rec("b3", False),
                "simple_1": rec("b1", False),
            },
        },
    )

    (example,) = worked_examples([], bfcl, records, {})

    assert (example["key"], example["split"], example["id"]) == ("unseen-tools", "bfcl", "simple_3")
    assert [a["model"] for a in example["answers"]] == ["D", "B"]


def test_ids_tie_in_number_order_not_string_order() -> None:
    xlam = [row(10, "same"), row(9, "same")]
    records = runs(
        "test-clean",
        {
            "B": {9: rec("b", True), 10: rec("b", True)},
            "raw 2B": {9: rec("t", False), 10: rec("t", False)},
        },
    )
    bfcl = [row("simple_10", "same"), row("simple_9", "same")]
    records |= runs(
        "bfcl",
        {
            "D": {i["id"]: rec("d", True) for i in bfcl},
            "B": {i["id"]: rec("b", False) for i in bfcl},
        },
    )

    assert [e["id"] for e in worked_examples(xlam, bfcl, records, {})] == [9, "simple_9"]


def test_a_wrong_gold_label_comes_with_the_audits_reason() -> None:
    xlam = [row(5, "q"), row(6, "q"), row(7, "a longer query")]
    records = runs(
        "test",
        {
            # Row 5's gold is sound; row 6's is wrong but B matches it: neither shows the point.
            "B": {5: rec("b5", False), 6: rec("b6", True), 7: rec("b7", False)},
            "raw 2B": {5: rec("t5", False), 6: rec("t6", False), 7: rec("t7", False)},
        },
    )

    (example,) = worked_examples(xlam, [], records, {6: "made-up value", 7: "3% is 0.03, not 3"})

    assert (example["key"], example["id"], example["note"]) == (
        "gold-wrong",
        7,
        "3% is 0.03, not 3",
    )


def test_c_fences_its_answer_like_its_teacher() -> None:
    xlam = [row(1, "q"), row(2, "a longer query")]
    records = runs(
        "test-clean",
        {
            "C": {1: rec('[{"name": "f"}]', True), 2: rec('```json\n[{"name": "f"}]\n```', True)},
            "B": {1: rec("b1", True), 2: rec("b2", True)},
        },
    )

    (example,) = worked_examples(xlam, [], records, {})

    assert (example["key"], example["id"]) == ("fenced", 2)
    assert [a["model"] for a in example["answers"]] == ["C", "B"]
