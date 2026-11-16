# 0003. Thinking mode off for every model and run

Date: 2026-11-05 · Status: accepted

## Context

Qwen3.5 can write hidden reasoning before answering. With it on, a ~50-token tool call becomes hundreds
or thousands of tokens, multiplying the cost of every eval, every teacher generation and every training
step. A production tool router answers directly.

## Decision

Thinking is disabled (`enable_thinking=False` in the chat template) for student and teacher, in training
and evaluation.

## Consequences

- Every method stays cheap enough to run several times on one 12 GB GPU.
- The teacher's ceiling may be lower than with thinking on. Distilling the teacher's reasoning traces is
  a stretch goal (map #2), measured against this baseline.
