"""Measured next-token probabilities for the story site's figures (#107).

uv run --extra train fc-probe

Two files in data/probe/, both from Qwen3.5-2B (teacher) and the untrained Qwen3.5-0.8B (student),
teacher-forced on one answer each, thinking off:

- forecast.json: a made-up query, the 2B's greedy answer, and at every answer position a summary of
  both models' full next-token distributions (`summarize_position`), so the site can rebuild the
  softmax at any temperature (Figure 2).
- grading.json: a real test-clean row the 0.8B gets wrong and the 2B right (the shortest such
  prompt), the 0.8B's own greedy answer from its eval run, and each token's log-probability under
  both models: what D's loss sees at its first steps (Figure 4).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from fc_distill.methods import STUDENT, TEACHER, eval_dir


def summarize_position(
    teacher: np.ndarray, student: np.ndarray, vocab: Sequence[str], k: int = 8, bins: int = 120
) -> dict[str, Any]:
    """One position's logits, compressed: for each model, the logits (minus its max) of the union of
    both models' top-k tokens, in the teacher's order, and a histogram [mean logit, count] of every
    other token's, so softmax at any temperature is rebuilt from a few hundred numbers."""
    union = list(dict.fromkeys([*np.argsort(-teacher)[:k], *np.argsort(-student)[:k]]))
    union.sort(key=lambda i: -teacher[i])
    rest = np.ones(len(vocab), dtype=bool)
    rest[union] = False
    out: dict[str, Any] = {}
    for name, logits in (("2B", teacher), ("0.8B", student)):
        shifted = logits - logits.max()
        others = shifted[rest]
        edges = np.linspace(others.min(), others.max(), bins + 1)
        which = np.clip(np.digitize(others, edges) - 1, 0, bins - 1)
        hist = [
            [round(float(others[which == b].mean()), 4), int((which == b).sum())]
            for b in range(bins)
            if (which == b).any()
        ]
        out[name] = {
            "top": [[vocab[i], round(float(shifted[i]), 4)] for i in union],
            "rest": hist,
        }
    return out


FORECAST_QUERY = "Give me the forecast for Lisbon."
# Made up in xLAM's format: `days` is optional with a default, which is where the teacher hesitates.
FORECAST_TOOLS = [
    {
        "name": "get_forecast",
        "description": "Get the weather forecast for a city.",
        "parameters": {
            "city": {"description": "The city name.", "type": "str"},
            "days": {
                "description": "How many days to forecast.",
                "type": "int, optional",
                "default": 7,
            },
        },
    },
    {
        "name": "get_time",
        "description": "Get the current local time in a city.",
        "parameters": {"city": {"description": "The city name.", "type": "str"}},
    },
]


def _prompt_ids(tok: Any, messages: Sequence[Any]) -> list[int]:
    from fc_distill.data import CHAT_TEMPLATE_KWARGS

    text = tok.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS
    )
    return list(tok(text, add_special_tokens=False)["input_ids"])


def _answer_logits(model: Any, prompt: list[int], answer: list[int]) -> np.ndarray:
    """Next-token logits (float32) at each answer position, teacher-forced on `answer`."""
    import torch

    ids = torch.tensor([prompt + answer], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, len(prompt) - 1 : len(prompt) + len(answer) - 1]
    return np.asarray(logits.float().cpu().numpy(), dtype=np.float32)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return np.asarray(shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True)))


def forecast(tok: Any, teacher: Any, student: Any) -> dict[str, Any]:
    """The 2B's greedy answer to the made-up query, and both models' distributions along it."""
    import torch

    from fc_distill.data import prompt_messages

    prompt = _prompt_ids(tok, prompt_messages(FORECAST_QUERY, FORECAST_TOOLS))
    with torch.no_grad():
        out = teacher.generate(
            torch.tensor([prompt], device=teacher.device), max_new_tokens=96, do_sample=False
        )
    answer = [int(i) for i in out[0, len(prompt) :] if int(i) not in tok.all_special_ids]
    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    vocab = [tok.convert_tokens_to_string([v]) for v in vocab]
    t, s = _answer_logits(teacher, prompt, answer), _answer_logits(student, prompt, answer)
    n = min(t.shape[1], s.shape[1], len(vocab))
    return {
        "query": FORECAST_QUERY,
        "tools": FORECAST_TOOLS,
        "answer": tok.decode(answer),
        "answer_tokens": [vocab[i] for i in answer],
        "prompt_tokens": len(prompt),
        "positions": {
            str(j): {"tok": vocab[answer[j]], **summarize_position(t[j, :n], s[j, :n], vocab[:n])}
            for j in range(len(answer))
        },
    }


def grading_row(runs_dir: Any) -> tuple[dict[str, Any], str]:
    """The test-clean row the 0.8B gets wrong and the 2B right with the shortest prompt (ties to the
    lowest id), and the 0.8B's greedy answer to it from its eval run."""
    from fc_distill.data import load_split
    from fc_distill.evaluate import read_run
    from fc_distill.examples import shortest

    student = {r["id"]: r for r in read_run(runs_dir / eval_dir("test", STUDENT))[1]}
    teacher = {r["id"]: r for r in read_run(runs_dir / eval_dir("test", TEACHER))[1]}
    rows = [
        ex
        for ex in load_split("test-clean")
        if not student[ex["id"]]["exact"] and teacher[ex["id"]]["exact"]
    ]
    best = shortest(rows)
    return dict(best), student[best["id"]]["prediction"]


def grading(tok: Any, teacher: Any, student: Any, runs_dir: Any) -> dict[str, Any]:
    """Each token of the 0.8B's own answer, with its log-probability under both models."""
    from fc_distill.data import to_messages

    row, text = grading_row(runs_dir)
    prompt = _prompt_ids(tok, to_messages(row))  # type: ignore[arg-type]
    answer = list(tok(text, add_special_tokens=False)["input_ids"])
    lt = _log_softmax(_answer_logits(teacher, prompt, answer))
    ls = _log_softmax(_answer_logits(student, prompt, answer))
    return {
        "id": row["id"],
        "query": row["query"],
        "tools": row["tools"],
        "gold": row["answers"],
        "answer": text,
        "tokens": [
            {
                "tok": tok.decode([a]),
                "student": round(float(ls[j, a]), 4),
                "teacher": round(float(lt[j, a]), 4),
            }
            for j, a in enumerate(answer)
        ],
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("data/probe"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(TEACHER)
    teacher, student = (
        AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        for name in (TEACHER, STUDENT)
    )
    about = (
        "Measured with transformers by fc-probe (#107): Qwen3.5-2B (teacher, '2B') and the "
        "untrained Qwen3.5-0.8B (student, '0.8B'), bf16, thinking off, teacher-forced. "
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data, what in (
        (
            "forecast.json",
            forecast(tok, teacher, student),
            "A made-up query and the 2B's greedy answer. Per answer position and model: the logits "
            "(minus the max) of the union of both models' top-8 tokens, and a 120-bin histogram "
            "[mean logit, count] of every other token.",
        ),
        (
            "grading.json",
            grading(tok, teacher, student, args.runs),
            "The test-clean row the 0.8B gets wrong and the 2B right with the shortest prompt; the "
            "0.8B's greedy answer from its eval run, and each token's log-probability under both.",
        ),
    ):
        (args.out / name).write_text(
            json.dumps({"about": about + what, **data}, ensure_ascii=False) + "\n"
        )
        print(f"wrote {args.out / name}")
