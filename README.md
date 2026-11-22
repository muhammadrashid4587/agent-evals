# agent-evals

A lightweight, deterministic eval harness for AI agents. Pure Python 3
standard library — no model calls, no API keys, no network.

An eval suite is a list of cases (prompt + expected behaviour). Each case
runs against a user-supplied agent function, and a deterministic scorer
judges the output. The runner records per-case pass/fail, latency, and token
counts (when the agent reports them), prints a summary table, and writes a
JSON report. A case may set a `timeout` in seconds, so a hung agent fails
that case instead of the run.

## What it is

- A small runner for evaluating an agent callable against a suite of cases.
- Eight deterministic scorers: exact match, substring, contains-all,
  contains-any, regex, JSON-schema shape, JSON-path value, numeric
  tolerance.
- A CLI that runs a suite JSON file against an agent named as
  `module:attribute` and exits 0 only when every case passes.

## What it is not

- Not a model-as-judge: nothing here calls an LLM, and no scorer asks a
  model for an opinion. Same output, same verdict, every run.
- Not a benchmark dataset: the two example suites are tiny on purpose.
  Bring your own cases.
- Not a tracing or observability tool. For recording what an agent did,
  see [blackbox](https://github.com/muhammadrashid4587/blackbox).

## Quickstart

```bash
python -m agent_evals.cli --suite examples/calculator.json \
    --agent examples.calculator_agent:agent
python -m agent_evals.cli --suite examples/kv_store.json \
    --agent examples.kv_agent:agent --report reports/kv.json
```

Or from Python:

```python
from agent_evals import Suite, run_suite, format_summary
from examples.kv_agent import make_agent

suite = Suite.from_dict({
    "name": "kv",
    "cases": [
        {"id": "set", "prompt": "SET a 1",
         "scorer": {"type": "exact", "expected": "OK"}},
        {"id": "get", "prompt": "GET a",
         "scorer": {"type": "exact", "expected": "1"}},
    ],
})
report = run_suite(suite, make_agent())
print(format_summary(report))
report.save("report.json")
```

## The agent under test

A callable `agent(prompt) -> ...` returning one of:

- a `str` — the agent's output (token count unknown),
- a `(output, tokens)` tuple,
- a `dict` with an `"output"` key and an optional `"tokens"` key.

An agent that raises an exception fails that case with the exception in the
detail; the run continues with the remaining cases.

## Scorers

| type | spec | passes when |
| --- | --- | --- |
| `exact` | `{"expected": "19"}` | output equals the expected string (surrounding whitespace ignored) |
| `contains` | `{"substring": "hello"}` | output contains the substring |
| `contains_all` | `{"substrings": ["hello", "world"]}` | output contains every listed substring (empty list passes) |
| `contains_any` | `{"substrings": ["hello", "hi"]}` | output contains at least one listed substring (empty list fails) |
| `regex` | `{"pattern": "^\\d+$"}` | the regex matches anywhere in the output |
| `json_schema` | `{"schema": {"name": "str", "age": "int"}}` | output is valid JSON with the given shape |
| `json_path` | `{"path": "$.result.total", "expected": 42}` | the value at the JSON path equals the expected value (optional `tolerance` for numbers) |
| `numeric` | `{"expected": 3.333, "tolerance": 0.001}` | the first number in the output is within tolerance of expected |

The JSON-schema shape language: a type name (`str`, `int`, `float`,
`bool`, `list`, `dict`, `number`), a dict of `field -> subschema` for
objects (a field name ending in `?` is optional), or a one-element list
`[subschema]` for arrays. `bool` is never accepted as `int`.

The JSON-path language: an optional leading `$`, dot-separated keys, and
`[n]` array indices, e.g. `$.result.total` or `items[0].name`. A missing
key or index fails the case; a malformed path is a scorer error. Numbers
compare with an optional `tolerance`; `bool` is never accepted as `int`.

## Suite files

A suite is JSON: `{"name": ..., "cases": [{"id", "prompt", "scorer"}]}`.
See `examples/calculator.json` and `examples/kv_store.json`.

A case may also carry an optional `timeout` in seconds, e.g.
`{"id": "slow", "prompt": "...", "scorer": {...}, "timeout": 5}`. If the
agent takes longer than that, the case fails as timed out (reported as
`"timed_out": true` in the JSON report and `TIMEOUT` in the summary) and
the run continues with the next case. The timeout is enforced on a
daemon thread, so a hung agent cannot stall the run; the thread itself
cannot be killed, so it keeps running in the background until it returns
on its own. Non-positive or non-numeric timeouts are rejected when the
suite loads.

## Report format

```json
{
  "suite": "kv-store",
  "started_at": "2026-11-21T20:52:00+00:00",
  "ended_at": "2026-11-21T20:52:00+00:00",
  "total": 8,
  "passed": 8,
  "pass_rate": 1.0,
  "total_latency_ms": 1.234,
  "total_tokens": null,
  "results": [
    {"case_id": "set-a", "passed": true, "detail": "exact match",
     "latency_ms": 0.123, "tokens": null, "output": "OK",
     "timed_out": false}
  ]
}
```

## Examples

- `examples/calculator_agent.py` — evaluates arithmetic in the prompt
  with a safe AST walk (no `eval()`), returns the result as a string.
- `examples/kv_agent.py` — a stateful in-memory key-value store speaking
  `SET` / `GET` / `DEL` / `EXISTS`, so a suite can SET in one case and
  GET in a later one.

## Tests

```bash
make test    # python3 -m unittest discover -s tests -v
make demo    # run both example suites through the CLI
```

## Roadmap

- ~~Per-case timeouts, so a hung agent fails the case instead of the run.~~ (done in v0.1.3)
- Weighted cases and partial credit.
- Parallel case execution for slow agents.
