# agent-evals

A lightweight, deterministic eval harness for AI agents. Pure Python 3
standard library — no model calls, no API keys, no network.

An eval suite is a list of cases (prompt + expected behaviour). Each case
runs against a user-supplied agent function, and a deterministic scorer
judges the output. The runner records per-case pass/fail, latency, and token
counts (when the agent reports them), prints a summary table, and writes a
JSON report. A case may set a `timeout` in seconds, so a hung agent fails
that case instead of the run. A case may also carry a `weight`, so some
cases count more than others, and some scorers give partial credit.

## What it is

- A small runner for evaluating an agent callable against a suite of cases.
- Eight deterministic scorers: exact match, substring, contains-all,
  contains-any, regex, JSON-schema shape, JSON-path value, numeric
  tolerance.
- A CLI that runs a suite JSON file against an agent named as
  `module:attribute` and exits 0 only when every case passes (a case
  that passes its checks but exceeds its latency budget does not fail
  the run unless `--fail-on-budget` is given). With `--repeat N` it
  runs the suite N times and flags flaky cases that pass sometimes
  and fail other times.

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

## Weights and partial credit

A case may also carry an optional `weight`, a positive number that
defaults to 1.0, e.g. `{"id": "core", ..., "weight": 2}`. The case is
worth that many points out of the suite's total weight. Each case earns
`score * weight` points, where `score` is in [0, 1]: 1.0 on a pass,
0.0 on a fail. The `contains_all` scorer gives partial credit: a case
that finds 3 of 4 required substrings earns a score of 0.75, so it
still earns most of its weight even though it does not count as passed.
Non-positive or non-numeric weights are rejected when the suite loads.

The run summary reports the weighted score alongside the pass count,
e.g. `Weighted score: 2.00/3 (66.7%)`, and the JSON report carries
`total_weight`, `earned_weight`, and `weighted_score`, plus a `score`
and `weight` per result. The CLI exit code is unchanged: 0 only when
every case passes, even if the weighted score is high.

## Latency budgets

A case may also carry an optional `max_latency_ms` budget, e.g.
`{"id": "fast", ..., "max_latency_ms": 200}`. If the agent's answer
passes its checks but took longer than the budget, the case is reported
as a budget violation instead of a pass: `passed` is false,
`budget_violated` is true, the score is 0, and the detail records the
latency and the budget, e.g. `exact match; exceeded latency budget
243.1 ms > 200 ms`. The JSON report carries the budget per result
(`max_latency_ms`, `budget_violated`) and a `budget_violations` count
for the run, and the summary marks such cases `BUDGET`. A case that
fails its checks is never also a budget violation, and neither is a
timed-out case. A case with no budget behaves exactly as before.

By default a budget violation does not fail the CLI run — the answer
was right, just slow — so the exit code stays 0 when every case
passed its checks. Pass `--fail-on-budget` to make budget violations
fail the run as well, e.g. to gate a CI step on latency:

```bash
python -m agent_evals.cli --suite examples/calculator.json \
    --agent examples.calculator_agent:agent --fail-on-budget
```

Non-positive or non-numeric budgets are rejected when the suite loads.

## Parallel execution

`run_suite(suite, agent, workers=N)` runs cases concurrently on a
thread pool of N workers, and the CLI takes `--workers N`. Results are
always reported in suite order, whatever order the cases finished in,
and per-case timeouts are still enforced per case — a hung case fails
on its own timeout while the other workers carry on. A suite of slow
cases finishes in roughly the time of its slowest case instead of the
sum of all of them. Parallel mode needs a thread-safe agent: a stateful
agent whose cases depend on each other's order (a SET in one case and
a GET in a later one, like the `kv_store` example) can give different
verdicts when cases run at the same time, so leave `workers` at 1 for
those suites.

## Comparing two runs

Save a baseline report, change the agent, run again, and compare:

```bash
python -m agent_evals.cli --suite examples/calculator.json \
    --agent examples.calculator_agent:agent --report reports/baseline.json
# ... change the agent ...
python -m agent_evals.cli --suite examples/calculator.json \
    --agent examples.calculator_agent:agent --report reports/current.json
python -m agent_evals.compare --baseline reports/baseline.json \
    --current reports/current.json
```

Or from Python:

```python
from agent_evals import compare_reports, format_comparison, load_report

comp = compare_reports(load_report("reports/baseline.json"),
                       load_report("reports/current.json"))
print(format_comparison(comp))
```

The comparison classifies every case by how it moved: **regression**
(passed, now failed — including a pass that became a timeout),
**fix** (failed, now passed), **score change** (failed in both runs
but earned a different partial score), **new**, **removed**, or
unchanged. It also reports the pass-rate delta, the weighted-score
delta, and the total latency delta, plus a per-case latency change,
so a run that passes everything but got much slower is visible too.
Cases are matched by `case_id`; a suite-name mismatch is noted but
does not stop the comparison. The CLI exits 1 when at least one case
regressed, 0 otherwise, so it can gate a CI step. `--json` writes the
comparison as JSON. Fixes, new cases, and score improvements never
fail the comparison — only a regression does.

## Repeated runs and flaky cases

A single run cannot tell a genuinely broken case from a flaky one:
an agent that answers nondeterministically passes a case sometimes
and fails it other times. Run the suite N times and aggregate:

```bash
python -m agent_evals.cli --suite examples/calculator.json \
    --agent examples.calculator_agent:agent --repeat 5
```

Or from Python:

```python
from agent_evals import repeat_suite, format_stability

stability = repeat_suite(suite, agent, repetitions=5)
print(format_stability(stability))
```

Each case gets a pass rate over the N runs plus its mean latency and
mean score. A case is **flaky** when its pass rate is strictly between
0 and 1: it passed at least once and failed at least once, so the
verdict is not reproducible. That is the signal to look at the agent
(nondeterminism, a race, a timeout that is too tight) or at the case
(a scorer too brittle for the agent's output), rather than treating
the run as a clean pass or a clean fail. The CLI exits 0 only when
every case passed every time; `--report` writes the stability report
as JSON. `repetitions` must be a positive integer.

## Report format

```json
{
  "suite": "kv-store",
  "started_at": "2026-11-21T20:52:00+00:00",
  "ended_at": "2026-11-21T20:52:00+00:00",
  "total": 8,
  "passed": 8,
  "pass_rate": 1.0,
  "total_weight": 9,
  "earned_weight": 9.0,
  "weighted_score": 1.0,
  "budget_violations": 0,
  "total_latency_ms": 1.234,
  "total_tokens": null,
  "results": [
    {"case_id": "set-a", "passed": true, "detail": "exact match",
     "latency_ms": 0.123, "tokens": null, "output": "OK",
     "timed_out": false, "score": 1.0, "weight": 1.0,
     "max_latency_ms": null, "budget_violated": false}
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
- ~~Weighted cases and partial credit.~~ (done in v0.1.4)
- ~~Parallel case execution for slow agents.~~ (done in v0.1.5)
- ~~Comparing two run reports to catch regressions.~~ (done in v0.2.0)
- ~~Latency budgets per case, so a case that passes but gets too slow fails.~~ (done in v0.3.0)
- ~~Repeated runs that flag flaky cases by per-case pass rate.~~ (done in v0.4.0)
