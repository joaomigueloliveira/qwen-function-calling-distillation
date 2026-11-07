import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fc_distill.data import to_messages
from fc_distill.scoring import score
from fc_distill.sft import (
    ComputeLogger,
    SftRecord,
    cost_summary,
    load_teacher_run,
    sft_record,
    teacher_records,
)
from fc_distill.tracking import GpuClock
from tests.factories import make_example
from tests.fakes import FakeTracker


def test_record_prompts_like_the_eval_and_completes_with_the_gold_calls() -> None:
    ex = make_example(1, answers=[{"name": "f", "arguments": {"x": 5}}])

    record = sft_record(ex)

    assert record["prompt"] == to_messages(ex)
    [completion] = record["completion"]
    assert completion["role"] == "assistant"
    assert completion["content"] == '[{"name": "f", "arguments": {"x": 5}}]'
    assert score(completion["content"], ex["answers"])["exact"]


def test_record_turns_thinking_off() -> None:
    assert sft_record(make_example(1))["chat_template_kwargs"] == {"enable_thinking": False}


def test_cost_summary_reports_the_issue_36_numbers() -> None:
    cost = cost_summary(rows=2000, tokens=1_800_000, seconds=3600.0, peak_bytes=6 * 2**30)

    assert cost == {
        "rows": 2000,
        "tokens": 1_800_000,
        "gpu_hours": 1.0,
        "tokens_per_s": 500.0,
        "gpu_hours_per_1k_rows": 0.5,
        "peak_gpu_mem_gb": 6.0,
    }


def test_compute_logger_logs_step_time_throughput_and_memory_on_logging_steps() -> None:
    ticks = iter([0.0, 1800.0, 2000.0, 3800.0])  # two optimizer steps of 30 min, 200 s apart
    tracker = FakeTracker()
    logger = ComputeLogger(tracker, GpuClock(now=lambda: next(ticks)), peak_bytes=lambda: 3 * 2**30)

    for step, tokens, should_log in [(1, 900_000, False), (2, 1_800_000, True)]:
        state = SimpleNamespace(global_step=step, num_input_tokens_seen=tokens)
        control = SimpleNamespace(should_log=should_log)
        logger.on_step_begin(None, state, control)
        logger.on_step_end(None, state, control)

    assert tracker.history == [
        {
            "compute/student_gpu_hours": 1.0,
            "compute/teacher_gpu_hours": 0.0,
            "compute/gpu_hours": 1.0,
            "compute/tokens_per_s": 500.0,
            "compute/peak_gpu_mem_gb": 3.0,
            # HF's W&B callback steps every metric on train/global_step: without it, the row would
            # plot at the previous logging step.
            "train/global_step": 2,
        }
    ]


def test_teacher_record_prompts_like_gold_and_completes_with_the_teacher_calls_in_gold_format() -> (
    None
):
    ex = make_example(1, answers=[{"name": "f", "arguments": {"x": 5}}])
    # The teacher's own layout (a fence, newlines); the target is its calls, written as gold is.
    teacher = '```json\n[\n  {"name": "f", "arguments": {"x": 7}}\n]\n```'

    [record], _ = teacher_records([ex], {1: teacher}, lenient_only=False)

    assert record["prompt"] == sft_record(ex)["prompt"]
    assert record["chat_template_kwargs"] == {"enable_thinking": False}
    assert record["completion"] == [
        {"role": "assistant", "content": '[{"name": "f", "arguments": {"x": 7}}]'}
    ]


def test_teacher_record_keeps_only_name_and_arguments_in_gold_key_order() -> None:
    ex = make_example(1)
    teacher = '[{"arguments": {"x": 3}, "name": "f", "type": "function"}, {"name": "f"}]'

    [record], _ = teacher_records([ex], {1: teacher}, lenient_only=False)

    assert record["completion"][0]["content"] == (
        '[{"name": "f", "arguments": {"x": 3}}, {"name": "f", "arguments": {}}]'
    )


# Gold leaves out an optional argument; teacher outputs keyed by row id.
_PARAMS = {"x": {"type": "int"}, "unit": {"type": "str, optional", "default": "km"}}
_ROWS = [
    make_example(i, answers=[{"name": "f", "arguments": {"x": 1}}], params=_PARAMS)
    for i in (1, 2, 3, 4)
]
_OUTPUTS = {
    1: '[{"name": "f", "arguments": {"x": 1}}]',  # exact
    2: '[{"name": "f", "arguments": {"x": 1, "unit": "km"}}]',  # lenient only: default spelled out
    3: '[{"name": "f", "arguments": {"x": 2}}]',  # wrong, but valid
    4: "Sure! I would call f.",  # doesn't parse
}


def _answers(records: list[SftRecord]) -> list[str]:
    return [r["completion"][0]["content"] for r in records]


def test_unfiltered_teacher_records_keep_wrong_calls_and_drop_what_does_not_parse() -> None:
    records, counts = teacher_records(_ROWS, _OUTPUTS, lenient_only=False)

    assert _answers(records) == [
        '[{"name": "f", "arguments": {"x": 1}}]',
        '[{"name": "f", "arguments": {"x": 1, "unit": "km"}}]',
        '[{"name": "f", "arguments": {"x": 2}}]',
    ]
    assert counts == {"offered": 4, "kept": 3, "unparsed": 1, "not_lenient": 0}


def test_lenient_teacher_records_keep_only_calls_that_lenient_match_gold() -> None:
    records, counts = teacher_records(_ROWS, _OUTPUTS, lenient_only=True)

    assert _answers(records) == [
        '[{"name": "f", "arguments": {"x": 1}}]',
        '[{"name": "f", "arguments": {"x": 1, "unit": "km"}}]',
    ]
    assert counts == {"offered": 4, "kept": 2, "unparsed": 1, "not_lenient": 1}


def _write_teacher_run(run_dir: Path, split: str, outputs: dict[int, str]) -> None:
    """A teacher run as fc-eval saves it: summary.json and predictions.jsonl."""
    run_dir.mkdir()
    summary = {
        "model": "Qwen/Qwen3.5-2B",
        "split": split,
        "seconds": 1800.0,
        "provenance": {"model_revision": "abc123"},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    lines = [json.dumps({"id": i, "prediction": p, "parse_ok": True}) for i, p in outputs.items()]
    (run_dir / "predictions.jsonl").write_text("\n".join(lines) + "\n")


def test_teacher_run_pairs_each_output_with_its_own_row(tmp_path: Path) -> None:
    rows = [make_example(i, answers=[{"name": "f", "arguments": {"x": i}}]) for i in (1, 2, 3)]
    # fc-eval writes rows in split order; here the file order differs from `rows`.
    outputs = {i: f'[{{"name": "f", "arguments": {{"x": {i}}}}}]' for i in (3, 1, 2)}
    _write_teacher_run(tmp_path / "teacher", "train", outputs)

    run = load_teacher_run(tmp_path / "teacher")
    records, _ = teacher_records(rows, run.outputs, lenient_only=False)

    assert [r["prompt"] for r in records] == [to_messages(ex) for ex in rows]
    assert _answers(records) == [json.dumps(ex["answers"]) for ex in rows]
    assert (run.model, run.revision, run.gpu_hours) == ("Qwen/Qwen3.5-2B", "abc123", 0.5)


def test_teacher_run_must_answer_the_train_split(tmp_path: Path) -> None:
    _write_teacher_run(tmp_path / "teacher", "validation", {1: "[]"})

    with pytest.raises(ValueError, match="train"):
        load_teacher_run(tmp_path / "teacher")
