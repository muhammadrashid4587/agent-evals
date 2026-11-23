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
Cases may also carry `tags`, so `--tag smoke` runs only the smoke cases.

## What it is

- A small runner for evaluating an agent callable against a suite of cases.
- Nine scorers: exact match, substring, contains-all,
  contains-any, regex, JSON-schema shape, JSON-path value, numeric
  tolerance, and a judge scorer that scores free-form output against
  a rubric with a pluggable judge (a deterministic built-in stub
  judge by default; bring your own LLM judge as a callable or an
  HTTP endpoint).
- A CLI that runs a suite JSON file against an agent named as
  `module:attribute` and exits 0 only when every case passes (a case
  that passes its checks but exceeds its latency budget does not fail
  the run unless `--fail-on-budget` is given). With `--repeat N` it
  runs the suite N times and flags flaky cases that pass sometimes
  and fail other times. With `--junit results.xml` it also writes the
  run as JUnit XML, so CI systems can ingest the results.

## What it is not

- Not a model-as-judge by default: nothing here calls an LLM, and no
  scorer asks a model for an opinion unless you register a judge that
  does. Same output, same verdict, every run — the built-in stub
  judge is deterministic, and a custom judge is opt-in.
- Not a benchmark dataset: the three example suites are tiny on purpose.
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
| `judge` | `{"judge": "stub", "rubric": {"criteria": [...]}}` | a judge scores the output against the rubric (see below) |

The JSON-schema shape language: a type name (`str`, `int`, `float`,
`bool`, `list`, `dict`, `number`), a dict of `field -> subschema` for
objects (a field name ending in `?` is optional), or a one-element list
`[subschema]` for arrays. `bool` is never accepted as `int`.

The JSON-path language: an optional leading `$`, dot-separated keys, and
`[n]` array indices, e.g. `$.result.total` or `items[0].name`. A missing
key or index fails the case; a malformed path is a scorer error. Numbers
compare with an optional `tolerance`; `bool` is never accepted as `int`.

## LLM-as-judge scorer

Free-form output — an explanation, a summary, a support answer — cannot
be checked with substrings. The `judge` scorer sends the prompt, the
output, and a rubric to a judge and records its verdict:

```json
{"type": "judge", "judge": "stub", "rubric": {
  "criteria": [
    {"name": "names the refund window",
     "must_mention": ["30 days", "refund"]},
    {"name": "offers the next step",
     "must_mention": ["Contact support"]}
  ],
  "must_not_mention": ["as an AI"],
  "pass_threshold": 1.0
}}
```

A judge is a callable `judge(prompt, output, rubric)` returning a
verdict: `passed` (bool), `score` (0–1), and a `rationale` string. The
verdict's score earns partial credit like `contains_all` does — a case
that meets half the rubric's weighted criteria earns half its weight —
and the rationale lands in the report's detail, so a human can see why
the output scored the way it did. A judge that raises, or returns
something that is not a verdict, is a scorer error: the case fails and
the run continues.

Three ways to get a judge:

- **Built-in stub** (`"stub"`, the default when the spec omits
  `"judge"`): deterministic, offline, no model. Each criterion lists
  substrings the output must mention (a `weight` per criterion is
  optional, default 1.0); the score is the weighted fraction of
  criteria met. The case passes when the score reaches
  `pass_threshold` (default 1.0) and no `must_not_mention` substring
  appears — a forbidden substring always fails the case. Use it in
  tests and demos, and as the reference for a real judge.
- **Your own callable**: pass a registry to `run_suite` —
  `run_suite(suite, agent, judges={"my-judge": my_judge})` — and name
  it in the spec (`"judge": "my-judge"`). The CLI takes
  `--judge module:attribute` and registers it as `"custom"`.
- **An HTTP endpoint**: `make_http_judge("https://...")` returns a
  judge callable that POSTs `{"prompt", "output", "rubric"}` and
  expects `{"passed", "score", "rationale"}` back, so you can point
  the harness at any LLM-judge service you run — a wrapper around an
  LLM API, a local model server — without the harness holding API
  keys. The CLI takes `--judge-url URL` and registers it as `"http"`.

```bash
python -m agent_evals.cli --suite examples/support.json \
    --agent examples.support_agent:agent
```

Judge cases work with the rest of the harness: weights, latency
budgets, timeouts, parallel runs (the judge must be thread-safe for
`--workers` above 1), repeated runs, and run comparison all apply.
A judge that calls a remote model is nondeterministic by nature, so a
`--repeat` run is the honest way to check whether its verdicts are
stable — a case the judge passes sometimes and fails other times shows
up as flaky, same as any other.

## Suite files

A suite is JSON: `{"name": ..., "cases": [{"id", "prompt", "scorer"}]}`.
See `examples/calculator.json` and `examples/kv_store.json`. A suite
may also carry an optional `"version"` label, e.g. `"version": "1.1"`,
recorded in run reports alongside the set's content fingerprint (see
"Eval-set versioning").

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

## Case tags

A case may also carry an optional `tags` list, e.g.
`{"id": "add", ..., "tags": ["arithmetic", "smoke"]}`. Tags are labels
for selecting cases: run only the smoke cases with `--tag smoke`, or
only the cases carrying both tags with `--tag smoke --tag fast
--tag-mode all`:

```bash
# OR: any case carrying at least one of the tags (the default)
python -m agent_evals.cli --suite examples/calculator.json \
    --agent examples.calculator_agent:agent --tag smoke --tag fast
# AND: only a case carrying every tag
python -m agent_evals.cli --suite examples/calculator.json \
    --agent examples.calculator_agent:agent --tag smoke --tag fast --tag-mode all
```

Or from Python: `suite.select(["smoke"])` returns a new suite with
only the matching cases (the original is unchanged), and
`suite.select(["smoke", "fast"], match="all")` is the AND form. An
empty tag list selects the whole suite. A tag filter that matches
no cases exits 2 with a message naming the tags and the mode, so a
typo fails loudly instead of silently passing an empty run. Tags
also work with `--repeat` and `--workers`.

To skip cases instead of selecting them, `--exclude-tag slow` drops
every case carrying `slow` (repeatable), and it combines with
`--tag`: select first, then exclude, so `--tag smoke --exclude-tag
slow` runs the smoke cases that are not slow. From Python,
`suite.exclude(["slow"])` returns the suite without the slow cases.
An excluded tag that matches no case in the suite is the same clean
error as an unmatched `--tag` — a typo that would otherwise silently
exclude nothing — and excluding every case exits 2 as well.

Tags are recorded per result in the JSON report (`"tags"`) and shown
in the summary after the detail, e.g. `exact match  [arithmetic,
smoke]`. Tags are part of a case's fingerprint, so retagging a case
changes it — but only the set of tags matters, not their order, and
duplicates are dropped when the suite loads. A non-list `tags`, or a
tag that is not a non-empty string, is rejected when the suite loads.

To see which tags a suite uses before running it, `--list-tags`
prints every tag with its case count and exits without running the
agent (`--agent` is not needed for it), and it applies `--tag` and
`--exclude-tag` first, so it lists the tags of the selected set:

```bash
python -m agent_evals.cli --suite examples/calculator.json --list-tags
# Tags in examples/calculator.json:
#   arithmetic  5 cases
#   negative    1 case
#   smoke       2 cases
```

Or from Python: `suite.tag_counts()` returns the same list of
`(tag, count)` pairs, sorted by tag.

A run report also carries a per-tag pass-rate breakdown, so a tag
that keeps failing is visible without digging through the rows: the
JSON report has a `tag_breakdown` object mapping each tag to its
case count, pass count, pass rate, and weighted score, and the text
summary ends with a `Per-tag pass rates:` section. A case carrying
several tags counts under each of them; untagged cases appear under
no tag.

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

### Was it the agent or the eval set?

A score delta between two runs can come from the agent changing, or
from the eval set changing underneath it. Run reports record the set
each run was measured against — a human version label and a content
fingerprint — plus a fingerprint per case, so the diff attributes
the movement:

- **Eval set changed**: the two runs' suite fingerprints differ
  (different cases), so score deltas may come from the set, not the
  agent. The diff prints both sets, e.g. `v1.0 (sha f2a43e864e07) ->
  v1.1 (sha 5ea6c4a5e807)`.
- **Changed cases**: shared cases whose own definition changed
  (prompt, scorer, timeout, weight, or budget). A regression or fix
  on a changed case is annotated `(case changed — may be the set,
  not the agent)` and listed in its own section; a regression on an
  unchanged case is an agent regression.
- **Added / removed cases** are the set changing too: a case only in
  the current run is new, one only in the baseline is removed.

A regression still fails the diff (exit 1) even when its case
changed — a case that now fails still fails the gate — but the
annotation says where to look first. Reports written before v0.6.0
carry no fingerprints: the diff falls back to comparing version
labels for the set, and reports per-case change as unknown rather
than guessing.

## Comparing across a reports directory

A two-run diff says what changed between yesterday and today. A
directory of reports says how the agent has been trending:

```bash
python -m agent_evals.trends --reports-dir reports/
python -m agent_evals.trends --reports-dir reports/ --json trends.json
```

Or from Python:

```python
from agent_evals import compare_history, format_trend, load_reports_dir

trend = compare_history(load_reports_dir("reports/"))
print(format_trend(trend))
```

The trend report shows a per-run table (pass rate and weighted score
per run, plus the headline trend from the first run to the latest),
per-case pass rates over the whole history with first-to-last score
movement, the cases that newly failed or newly passed in the latest
run, the flaky cases across the history, and the worst regressions —
cases that passed at least once and are failing in their latest run,
ranked by how reliably they used to pass, so a case that passed 9
of 10 runs and now fails outranks one that was already failing half
the time. Reports load in `started_at` order; a case added or removed
mid-history gets no verdict for the runs it was not part of, so it
does not skew the other cases' pass rates. A file in the directory
that is not a run report is skipped with a warning, not an error.
The CLI exits 1 when at least one case newly failed in the latest
run, 0 otherwise, so it can gate a CI step the same way the two-run
comparison does. `--json` writes the trend as JSON.

### Per-case latency trends

Pass rates say whether the agent still works; latency says whether
it is quietly getting slower. The trend report carries a latency
history per case and summarises it:

- **first / last / mean / min / max** latency in ms, plus the
  first-to-last change in ms and as a percent;
- a **least-squares slope** in ms per run, fitted against run index
  (so a case absent from a run keeps its spacing), which uses every
  sample instead of just the endpoints;
- a **trend label** — `slowing`, `improving`, or `flat` — from the
  slope relative to the case's mean latency: a slope bigger than
  10% of the mean per run counts as slowing (or improving, when
  negative). Latency is noisy, so the label is a signal to look at
  the case, not a verdict.

The trend lists the **slowest growing** cases — the ones trending
slower, ranked by slope relative to their mean latency — so a case
that keeps passing but creeps from 50 ms to 200 ms shows up before
it starts tripping its latency budget. A case with fewer than two
latency samples has no slope and no trend. The per-case table shows
each case's first-to-last latency alongside its pass rate and score
movement, and the JSON trend carries the full latency history per
case.

### Latency-trend CI gate

A trend report is a warning until it can fail a build. Two flags
turn the latency slope into a gate:

```bash
# Fail when any case's latency slope exceeds +2 ms per run
python -m agent_evals.trends --reports-dir reports/ --fail-on-trend 2.0

# Fail when any case is labelled "slowing" at all
python -m agent_evals.trends --reports-dir reports/ --fail-on-slowing
```

`--fail-on-trend N` fails (exit 1) when any case's least-squares
latency slope exceeds N ms per run. The comparison is strict — a
slope exactly at the threshold passes — so the threshold is a
budget, not a tripwire, and a case with fewer than two latency
samples has no slope and never trips the gate. `--fail-on-slowing`
fails when any case carries the `slowing` label (slope bigger than
10% of its mean latency per run), which catches slow creep on fast
cases whose absolute slope is too small for `--fail-on-trend` to
notice. Either flag combines with the existing newly-failing rule:
the exit code is 1 when a case newly failed, the slope gate tripped,
or a case is slowing, and 0 otherwise. The text report names the
cases that tripped the gate with their slope and mean latency, and
the JSON output carries `latency_gate_threshold_ms_per_run`,
`latency_gate_failures`, and `latency_gate_passed` (plus
`slowing_cases` with `--fail-on-slowing`), so CI can report exactly
which cases are getting slower and by how much. From Python,
`TrendReport.latency_gate_failures(threshold)` returns the tripping
cases worst first, and `latency_gate_passed(threshold)` is the
boolean form.

## JUnit XML export

Most CI systems understand JUnit XML even though they know nothing
about eval harnesses. `--junit results.xml` writes the same run as
one `<testsuite>` (named after the suite), with one `<testcase>` per
case and the latency in seconds:

```bash
python -m agent_evals.cli --suite examples/calculator.json \
    --agent examples.calculator_agent:agent --junit results.xml
```

Or from Python:

```python
from agent_evals import to_junit_xml

xml = to_junit_xml(report)
report.save_junit("results.xml")
```

A case that failed its checks becomes a `<failure>` whose message is
the result's detail; a case that could not be scored at all — a
timeout, an agent crash, or a scorer error — becomes an `<error>`
instead, because the failure was in the harness or the agent, not in
the answer. A budget violation is a `<failure>`: the answer was
right but too slow, and the detail says by how much. The suite
version, fingerprint, and weighted score ride along as
`<properties>`, so a CI job can tell which eval set a run was
measured against. Agent output can contain control characters that
are illegal in XML 1.0, so details are sanitised before they are
written (anything below U+0020 other than tab, newline, and carriage
return becomes U+FFFD). `--junit` is not supported with `--repeat`,
which reports stability across runs rather than a single run.

## Eval-set versioning

A suite file may carry an optional `"version"` label, e.g.
`{"name": "calculator", "version": "1.1", "cases": [...]}`. The label
is a human marker only — what identifies a set is its fingerprint,
a short SHA-256 of the suite's name and cases. Relabelling a version
without touching a case keeps the same fingerprint; changing any
case changes it even if the label was not bumped. Every run report
records `suite_version`, `suite_fingerprint`, and a
`case_fingerprint` per result, so any two reports can be diffed for
both verdict movement and set movement (see above).

## Output diffs on failures

When an `exact` case fails, the report carries a unified diff of the
expected output vs the actual output, so a regression is visible at a
glance instead of requiring two blobs to be compared by eye. The diff
is in the JSON report as `"diff"` on the failing result, and printed
under the failing row in the text summary (capped at 30 lines there;
the JSON report keeps it whole):

```
bad-1                  FAIL      0.00      0.1       -  expected 'alpha\nbeta', got 'alpha\ngamma'
    --- expected
    +++ actual
    @@ -1,2 +1,2 @@
     alpha
    -beta
    +gamma
```

Only `exact` cases carry a diff: other scorers have no expected text
to diff against — `contains` checks for a substring, `numeric` and
`json_path` compare values, and the judge scorer's rationale is already
in the detail. A passing case has `"diff": null`.

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
  "suite_version": "1.1",
  "suite_fingerprint": "f2a43e864e07",
  "started_at": "2026-11-21T20:52:00+00:00",
  "ended_at": "2026-11-21T20:52:00+00:00",
  "total": 8,
  "passed": 8,
  "pass_rate": 1.0,
  "total_weight": 9,
  "earned_weight": 9.0,
  "weighted_score": 1.0,
  "budget_violations": 0,
  "tag_breakdown": {
    "smoke": {"cases": 2, "passed": 2, "pass_rate": 1.0,
              "weighted_score": 1.0}
  },
  "total_latency_ms": 1.234,
  "total_tokens": null,
  "results": [
    {"case_id": "set-a", "passed": true, "detail": "exact match",
     "latency_ms": 0.123, "tokens": null, "output": "OK",
     "timed_out": false, "score": 1.0, "weight": 1.0,
     "max_latency_ms": null, "budget_violated": false,
     "case_fingerprint": "a1b2c3d4e5f6", "diff": null,
     "tags": ["smoke"]}
  ]
}
```

## Examples

- `examples/calculator_agent.py` — evaluates arithmetic in the prompt
  with a safe AST walk (no `eval()`), returns the result as a string.
- `examples/kv_agent.py` — a stateful in-memory key-value store speaking
  `SET` / `GET` / `DEL` / `EXISTS`, so a suite can SET in one case and
  GET in a later one.
- `examples/support_agent.py` — a canned support agent answering refund
  questions, so `examples/support.json` can show the judge scorer:
  one case passes outright, one earns partial credit against its
  rubric.

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
- ~~Judge / LLM-as-judge scorer: a pluggable judge that scores free-form
  output against a rubric, with a deterministic built-in stub judge
  and an HTTP judge for real LLM endpoints.~~ (done in v0.5.0)
- ~~Eval-set versioning with run diffing: versioned eval sets (hash +
  version id in run reports) and a diff that flags cases
  added/removed/changed and attributes score deltas to set changes
  vs agent changes.~~ (done in v0.6.0)
- ~~Per-case output diffs: when an exact-match case fails, the report
  carries a unified diff of the expected output vs the actual output
  (in JSON and in the text summary), so a regression is visible at a
  glance.~~ (done in v0.7.0)
- ~~Compare across a reports directory: per-case pass rates over time,
  score trends, newly failing / newly passing cases, and the worst
  regressions in the history (not just between two runs).~~ (done in
  v0.8.0)
- ~~Per-case latency trends: first/last/mean/min/max latency, a
  least-squares slope per run, and a slowest-growing list, so a case
  that keeps passing but gets steadily slower is visible before it
  trips its latency budget.~~ (done in v0.9.0)
- ~~Latency-trend CI gate: `--fail-on-trend N` fails the trend run
  when any case's latency slope exceeds N ms per run, and
  `--fail-on-slowing` fails it when any case is labelled slowing,
  so latency creep can gate CI before a case trips its latency
  budget.~~ (done in v0.10.0)
- ~~JUnit XML export: `--junit results.xml` writes a run as JUnit
  XML — check failures as `<failure>`, timeouts, agent crashes,
  and scorer errors as `<error>`, budget violations as
  `<failure>` — so CI systems can ingest eval results.~~ (done in
  v0.11.0)
- ~~Case tags with `--tag` filtering: cases carry a `tags` list, and
  `--tag smoke` runs only the matching cases (`--tag-mode all`
  requires every tag), with tags recorded per result in the JSON
  report and shown in the summary.~~ (done in v0.12.0)
- ~~`--exclude-tag` filtering: skip the cases carrying a tag
  (`--exclude-tag slow`), repeatable, combining with `--tag`
  (select first, then exclude); an excluded tag that matches no
  case is a clean error, same as an unmatched `--tag`.~~ (done in
  v0.13.0)
- ~~`--list-tags` and per-tag pass rates: `--list-tags` lists every
  tag in the suite with its case count and exits without running
  the agent, and a run report carries a per-tag pass-rate
  breakdown (`tag_breakdown` in JSON, a `Per-tag pass rates:`
  section in the text summary), so a tag that keeps failing is
  visible at a glance.~~ (done in v0.14.0)
