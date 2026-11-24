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
- Nineteen scorers: exact match, substring, contains-all,
  contains-any, not-contains, regex, contains-regex, JSON-schema shape, JSON-path value, JSON
  subset, JSON equality, fuzzy similarity, starts with, ends with, one of,
  length range,
  numeric range, numeric tolerance, and a judge scorer that scores
  free-form output against a rubric with a pluggable judge (a
  deterministic built-in stub judge by default; bring your own LLM
  judge as a callable or an HTTP endpoint).
- A CLI that runs a suite JSON file against an agent named as
  `module:attribute` and exits 0 only when every case passes (a case
  that passes its checks but exceeds its latency budget does not fail
  the run unless `--fail-on-budget` is given, and a case that passes
  its checks but exceeds its token budget does not fail the run
  unless `--fail-on-cost` is given). With `--repeat N` it
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
| `exact` | `{"expected": "19"}` | output equals the expected string (surrounding whitespace ignored; optional `"ignore_case": true` for a case-insensitive match) |
| `contains` | `{"substring": "hello"}` | output contains the substring (optional `"ignore_case": true` for a case-insensitive match) |
| `contains_all` | `{"substrings": ["hello", "world"]}` | output contains every listed substring (empty list passes; optional `"ignore_case": true`) |
| `contains_any` | `{"substrings": ["hello", "hi"]}` | output contains at least one listed substring (empty list fails; optional `"ignore_case": true`) |
| `not_contains` | `{"substrings": ["as an AI", "sorry"]}` | output contains none of the listed substrings (empty list passes; optional `"ignore_case": true`) |
| `regex` | `{"pattern": "^\\d+$"}` | the regex matches anywhere in the output (optional `"ignore_case": true` for a case-insensitive match) |
| `contains_regex` | `{"pattern": "^\\d+$"}` | the regex matches anywhere in the output (optional `"ignore_case": true` for a case-insensitive match; a non-string `pattern` is a scorer error) |
| `json_schema` | `{"schema": {"name": "str", "age": "int"}}` | output is valid JSON with the given shape |
| `json_path` | `{"path": "$.result.total", "expected": 42}` | the value at the JSON path equals the expected value (optional `tolerance` for numbers) |
| `json_subset` | `{"expected": {"status": "ok", "total": 42}}` | the output JSON contains every expected field with a matching value; extra fields are ignored (optional `tolerance` for numbers) |
| `json_equals` | `{"expected": {"status": "ok", "total": 42}}` | the output JSON is deeply equal to the expected value (optional `ignore_order` to compare arrays as multisets, optional `tolerance` for numbers) |
| `fuzzy` | `{"expected": "hello world", "threshold": 0.8}` | the output is similar enough to the expected string (SequenceMatcher ratio at or above `threshold`, default 0.8; optional `"ignore_case": true` for a case-insensitive comparison) |
| `starts_with` | `{"prefix": "The answer is"}` | the output starts with the prefix (surrounding whitespace ignored; optional `"ignore_case": true`) |
| `ends_with` | `{"suffix": "is 42"}` | the output ends with the suffix (surrounding whitespace ignored; optional `"ignore_case": true`) |
| `one_of` | `{"values": ["yes", "no", "maybe"]}` | the output equals one of the listed values (surrounding whitespace ignored; optional `"ignore_case": true`) |
| `length` | `{"min_length": 10, "max_length": 100}` | the output's length falls in the range (at least one bound required; optional `"unit": "words"` for word count instead of characters) |
| `range` | `{"min": 0, "max": 100}` | the first number in the output falls in the range (at least one bound required; optional `"tolerance"` widens the accepted band) |
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

The JSON-subset scorer checks several fields at once where
`json_path` checks one: `expected` is an object, and every field it
names must be present in the output JSON with a matching value,
recursively, while extra fields in the output are ignored — the
check is a subset match, not a full equality. That is the shape of
an agent that returns JSON with fields you did not ask for: a
request id, a timestamp, debug metadata. A dict inside the expected
object is itself a subset match, and a list matches pairwise by
position with the same length (a dict inside a list is a subset
too, so an expected list item only pins the fields it names).
Numbers compare with an optional `tolerance`; `bool` is never
accepted as `int`. It gives partial credit like `contains_all`:
the score is the fraction of expected leaf values that matched, so
a case that gets most of the fields right still earns most of its
weight. An empty `expected` object passes vacuously; a non-object
`expected` is a scorer error.

The `json_equals` scorer is the strict counterpart of `json_subset`:
it passes when the output JSON is deeply equal to the expected value
— every key must match and extra keys fail the case, so an agent that
returns a request id or debug metadata alongside the answer does not
pass. Object key order does not matter, and lists must match pairwise
by position unless `"ignore_order": true` is set, in which case
arrays compare as multisets (duplicates still have to match, and the
option applies to nested arrays too). Numbers compare numerically, so
`1` equals `1.0`; with an optional `"tolerance"` they compare within
that tolerance instead, recursively, so a float answer that is close
but not exact still passes — `tolerance` composes with
`ignore_order`, which then matches array elements pairwise rather
than by canonical encoding. `bool` is never accepted as `int`, so
`true` does not equal `1` and `1` does not equal `"1"`. An output
that is not valid JSON scores 0.0, and an `expected` value that is
not valid strict JSON (not serialisable, or containing NaN/Infinity)
is a scorer error, as is a missing `expected`, a non-boolean
`ignore_order`, or a non-numeric or negative `tolerance`.
The score is all-or-nothing, like `exact`: 1.0 on a pass, 0.0 on a
fail.

The `fuzzy` scorer is for output that should be close to an
expected string without matching it exactly — a rephrased answer,
a typo, an extra word. Similarity is `difflib`'s SequenceMatcher
ratio on the stripped strings (deterministic, standard library
only, no model): the case passes when the ratio reaches
`threshold` (default 0.8, must be in [0, 1]). It gives partial
credit like `contains_all`: the score is the ratio itself, so a
near miss still earns most of its weight. An exact match is a
ratio of 1.0; an empty `expected` only matches an empty output.

The `starts_with` scorer is for output that must open with a fixed
lead-in — an answer that has to start with "The answer is", a JSON
blob that has to start with `{"status"`, a greeting that has to
start with "Hello". It compares the stripped output against the
stripped prefix, so leading whitespace in the output does not fail
the case. It gives partial credit like `fuzzy`: the score is the
fraction of the prefix the output matches from the start, so an
output that gets the first few words right still earns part of its
weight. An empty `prefix` passes vacuously: every string starts
with it.

The `ends_with` scorer is the mirror of `starts_with`, for output
that must close with a fixed sign-off — an answer that has to end
with "is 42", a JSON blob that has to end with `}`, a closing line
that has to end with "Thank you". It compares the stripped output
against the stripped suffix, so trailing whitespace in the output
does not fail the case. It gives partial credit like `fuzzy`: the
score is the fraction of the suffix the output matches from the
end, so an output that gets the last few words right still earns
part of its weight. An empty `suffix` passes vacuously: every
string ends with it.

The `one_of` scorer is for output that must be one of a fixed set
of allowed values — a classification answer that has to be "yes",
"no", or "maybe", a verdict from a fixed vocabulary, a choice from
a menu. It passes when the stripped output equals one of the listed
`values`, whole-output equality like `exact`, not containment: "yes
please" is not "yes". It is all-or-nothing like `exact`: 1.0 on a
pass, 0.0 on a fail. It accepts an optional `"ignore_case": true`,
which case-folds both the values and the output before comparing.
A missing `values`, a non-list `values`, or an empty list is a
scorer error: a set of allowed values with nothing in it can never
pass, so the suite is wrong, not the agent.

The `not_contains` scorer is the mirror of `contains_all`, for
output that must not contain certain phrases — an answer that must
not say "as an AI", a reply that must not apologise, a summary that
must not leak a forbidden token. It passes when none of the listed
substrings appears in the output. It gives partial credit like
`contains_all`: the score is the fraction of listed substrings that
are absent, so an output that slips in one of four forbidden phrases
still earns most of its weight. An empty `substrings` list passes
vacuously: there is nothing forbidden to find.

The `length` scorer is for output that must be a certain size — a
summary that has to fit in a tweet, an answer that has to be at least
a paragraph, a code snippet that must not ramble. It passes when the
output's length falls in `[min_length, max_length]`; at least one
bound is required, and a missing bound is unbounded on that side, so
`{"min_length": 10}` only sets a floor and `{"max_length": 100}`
only sets a ceiling. Length is measured in characters by default
(the stripped output, like the other string scorers) or in words
with `"unit": "words"` (whitespace-split word count). It gives
partial credit: the score falls linearly from 1.0 at the bound to
0.0 at a length of 0 (too short: the score is `n / min_length`) or
at twice the maximum (too long: the score is
`max(0, (2 * max_length - n) / max_length)`), so an output just
outside the range still earns most of its weight and one wildly
outside earns nothing. A bound that is not a non-negative integer,
`min_length` above `max_length`, or a `unit` other than
`"characters"` or `"words"` is a scorer error.

The `range` scorer is the numeric mirror of `length`: it passes
when the first number in the output falls in `[min, max]` — a score
that has to land between 0 and 100, a temperature that has to stay
in a safe band, a probability that has to stay in [0, 1]. At least
one bound is required, and a missing bound is unbounded on that
side, so `{"min": 0}` only sets a floor and `{"max": 100}` only
sets a ceiling; bounds may be negative, and may be integers or
floats. The number is parsed the same way as the `numeric` scorer
(the first number in the output, which may be embedded in
surrounding text), and an output with no number fails with a score
of 0.0. An optional `tolerance` widens the accepted band by that
amount on each bounded side, so `{"min": 0, "max": 100,
"tolerance": 10}` accepts any value in [-10, 110] — a float answer
that is close but just outside the range still passes. It gives
partial credit like `length`: the score falls linearly from 1.0 at
the widened bound to 0.0 one scale away, where the scale is the
widened range width when both bounds are present (`max - min`
plus twice the tolerance) or the magnitude of the single widened
bound otherwise, so a value just outside the widened range still
earns most of its weight and one a full scale outside earns
nothing. A zero scale (`min == max` with no tolerance, or a lone
bound of 0 with no tolerance) gives no partial credit: any value
outside the range scores 0.0. A non-numeric bound, `min` above
`max`, or a non-numeric, negative, NaN, or infinite `tolerance`
is a scorer error, and a tolerance of 0 keeps the old behaviour
exactly.

The `contains_regex` scorer is the strict counterpart of
`regex`: it passes when the regex matches anywhere in the output,
like `regex`, but the `pattern` must be a string — a non-string
pattern is a scorer error instead of being stringified, so a suite
that passes a number where a pattern belongs fails loudly instead
of matching the digits of that number. An invalid regex is a scorer
error too. It is all-or-nothing: 1.0 on a match, 0.0 otherwise.

The `exact` scorer and the seven string scorers — `contains`, `contains_all`,
`contains_any`, `not_contains`, `starts_with`, `ends_with`, `one_of` — and the
`fuzzy`, `regex`, and `contains_regex` scorers all accept an optional `"ignore_case": true`
in the spec, e.g.
`{"type": "exact", "expected": "Hello", "ignore_case": true}`.
When set, the expected strings and the output are case-folded
(`str.casefold`, so non-ASCII case pairs compare equal too) before
comparing — `regex` and `contains_regex` match with `re.IGNORECASE` instead — so `"Hello"`
matches `"hello"` and a forbidden phrase
fails `not_contains` in any casing. Scores and partial credit are
unchanged — a `contains_all` case that finds 3 of 4 substrings in
the wrong case still earns 0.75, and `exact` stays all-or-nothing —
and the detail still names the substrings as written in the spec. The option defaults to off, so
existing suites behave exactly as before, and a non-boolean value
is a scorer error.

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

## Per-case pass thresholds

A case may also carry an optional `threshold` in [0, 1], e.g.
`{"id": "mostly", ..., "threshold": 0.75}`. By default a case passes
only on a full score, but a case with a threshold passes when its
scorer's score reaches it — so a `contains_all` case that finds 3 of
4 substrings (score 0.75) passes at a threshold of 0.75, and a case
whose scorer passes it on a low score can still fail at a higher
threshold. The case keeps the scorer's score either way, so it earns
its partial weight; the detail records why the verdict changed, e.g.
`missing substrings: 'date'; score 0.75 meets threshold 0.75`. The
threshold is part of the case, so setting or changing it changes the
case fingerprint, and the JSON report carries the threshold per
result. A timeout, an agent error, or a scorer error still fails the
case whatever the threshold is — the threshold only applies to a
score the scorer actually produced.

A non-numeric threshold, a boolean, or a value outside [0, 1] is
rejected when the suite loads, the same way a bad weight or timeout
is.

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

## Token budgets

A case may also carry an optional `max_tokens` budget, e.g.
`{"id": "terse", ..., "max_tokens": 100}`. If the agent's answer
passes its checks but the agent reported more tokens than the budget,
the case is reported as a cost violation instead of a pass: `passed`
is false, `cost_violated` is true, the score is 0, and the detail
records the token count and the budget, e.g. `exact match; exceeded
token budget 250 tokens > 100 tokens`. The JSON report carries the
budget per result (`max_tokens`, `cost_violated`) and a
`cost_violations` count for the run, and the summary marks such cases
`COST`. A case that fails its checks is never also a cost violation,
and neither is a timed-out case; a case over both its latency and
token budgets is a latency violation only, so one case gets one
reason. The budget only applies when the agent reports a token count
(a `(output, tokens)` tuple or a dict with a `"tokens"` key): an
agent that returns a plain string reports no tokens, so its cases
can never trip the budget.

By default a cost violation does not fail the CLI run — the answer
was right, just expensive — so the exit code stays 0 when every case
passed its checks. Pass `--fail-on-cost` to make cost violations
fail the run as well, e.g. to gate a CI step on token spend:

```bash
python -m agent_evals.cli --suite examples/calculator.json \
    --agent examples.calculator_agent:agent --fail-on-cost
```

The flag only bites when the agent reports a token count: the
example agents return plain strings, so they never trip a token
budget. An agent that returns `(output, tokens)` or a dict with a
`"tokens"` key is what a token budget measures.

`max_tokens` must be a positive integer; anything else is rejected
when the suite loads. It is part of a case's fingerprint, so adding
or changing a token budget changes the case — a run diff attributes
the movement to the eval set, not the agent.

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
delta, the total latency delta, and the total token delta, names an
overall **winner** (weighted score, then pass rate, then lowest total
latency, then lowest total tokens — a tie has no winner), and prints
every case side by side in one table — baseline vs current score,
latency, and tokens with their deltas — so a run that passes
everything but got much slower or more expensive is visible too.
Cases are matched by `case_id`; a suite-name mismatch is noted but
does not stop the comparison. The CLI exits 1 when at least one case
regressed, 0 otherwise, so it can gate a CI step. `--json` writes the
comparison as JSON. Fixes, new cases, and score improvements never
fail the comparison — only a regression does.

The comparison also counts **case wins** per run: a shared case is
won by the run with the higher score, with a tie on score broken by
the lowest latency — the same rule the per-model comparison uses.
A case both runs scored 0 on has no winner, and neither does an
exact tie or a case present in only one run. The text summary prints
`Case wins: baseline N, current M` and the JSON carries a
`case_wins` count per run. The wins are also broken down per tag —
which run won how many cases under each tag, with the cases no run
won recorded as ties — so a tag one run keeps losing on is visible
at a glance, the same breakdown the per-model comparison carries.
The text summary prints a `Case wins by tag:` section and the JSON
carries `case_wins_by_tag`. A case's tags come from the current
run's report when it is in it, else the baseline's; a case carrying
several tags counts under each of them, and untagged cases appear
under no tag. `--fail-on-lose-per-case` turns the count
into a gate: it exits 1 when the baseline run won strictly more
individual cases than the current run (a tie in case wins is not a
loss), so a current run that loses case by case still fails CI even
if it wins the aggregate — for example when it wins the weighted
score on one heavy case but loses several lighter ones on partial
credit. The gate is independent of the regression rule: either one
failing exits 1.

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

## Comparing models on one suite

A two-run diff compares one agent against its own past. To compare
several agents (or several models) against each other, run the same
suite against each of them and get one comparison view:

```bash
python -m agent_evals.model_compare --suite examples/calculator.json \
    --agent fast=examples.calculator_agent:agent \
    --agent slow=my_agents:slow_calc
```

Or from Python:

```python
from agent_evals import compare_models, format_model_comparison

comp = compare_models(suite, {"fast": fast_agent, "slow": slow_agent})
print(format_model_comparison(comp))
```

The view shows an overall ranking (weighted score, then pass rate,
then lowest total latency), per-case and aggregate scores, latency,
and token totals side by side, a winner per case (highest score,
ties broken by lowest latency — a case every agent failed has no
winner), a case-wins count per agent, and a case-wins breakdown per
tag (which agent won how many cases under each tag, with the cases
no agent won recorded as ties), so a tag one agent keeps losing on
is visible without digging through the per-case table. A case
carrying several tags counts under each of them; untagged cases
appear under no tag. `--json` writes the comparison as JSON
(`case_wins_by_tag` in the JSON, with each case's `tags` in its
per-case row). At least two `--agent name=module:attribute`
specs are required. The comparison exits 0 by default: it reports
which agent won, it does not gate CI. `--fail-on-lose BASELINE`
turns it into a gate: it exits 1 when the named baseline agent
beats any other agent (higher weighted score, then higher pass
rate, then lower total latency — an exact tie on all three is not
a loss), so a candidate agent that regresses against the baseline
fails CI. `--fail-on-lose-per-case BASELINE` is the per-case
variant: it exits 1 when the baseline won strictly more
individual cases than any other agent (a tie in case wins is not
a loss), so a candidate that loses case by case still fails CI
even if it wins the aggregate. Use the two-run compare to gate on regressions between
two runs of the same agent.

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
the time. It also breaks case wins down per tag — which run won how
many cases under each tag, with the cases no run won recorded as
ties (a case wins the run with its highest score, ties broken by
lowest latency; a case every run scored 0 on, an exact tie, or a
case present in only one run has no winner) — so a tag one run
keeps losing on is visible at a glance, the same breakdown the
two-run comparison carries, keyed by run. A case carrying several
tags counts under each of them; untagged cases appear under no tag,
and a case's tags are the ones it carried in the latest run that
contained it. Reports load in `started_at` order; a case added or removed
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

### Per-case token-cost trends

Latency says whether the agent is getting slower; token counts say
whether it is getting more expensive. The same trend machinery
applies to the token counts agents report: the trend report carries
a token-cost history per case and summarises it:

- **first / last / mean / min / max** tokens per run, plus the
  first-to-last change in tokens and as a percent;
- a **least-squares slope** in tokens per run, fitted against run
  index (so a case absent from a run — or a run where the agent
  reported no tokens — keeps its spacing);
- a **trend label** — `rising`, `falling`, or `flat` — from the
  slope relative to the case's mean token count: a slope bigger than
  10% of the mean per run counts as rising (or falling, when
  negative).

The trend lists the **most expensive cases** — ranked by mean token
count, so the cases that dominate the bill are visible at a glance —
and the **costliest growing** cases — the ones trending more
expensive, ranked by slope relative to their mean — so a case that
keeps passing but creeps from 100 tokens to 300 shows up before it
starts tripping its token budget. The per-run table carries each
run's total token count, and the per-case table shows each case's
first-to-last token count alongside its pass rate and latency. The
JSON trend carries the full token history per case, plus
`most_expensive` and `costliest_growing`.

Token trends only cover cases whose agent reported a token count:
an agent that returns a plain string reports no tokens, so its
cases have no token history and never appear in the cost sections —
a run with no token counts at all shows `-` in the tokens column.
A case with fewer than two token samples has no slope and no trend.

### Token-cost trend CI gate

Two flags turn the token slope into a gate, mirroring the latency
gate:

```bash
# Fail when any case's token slope exceeds +10 tokens per run
python -m agent_evals.trends --reports-dir reports/ --fail-on-cost-trend 10

# Fail when any case is labelled "rising" at all
python -m agent_evals.trends --reports-dir reports/ --fail-on-rising-cost
```

`--fail-on-cost-trend N` fails (exit 1) when any case's
least-squares token slope exceeds N tokens per run. The comparison
is strict — a slope exactly at the threshold passes — so the
threshold is a budget, not a tripwire, and a case with fewer than
two token samples has no slope and never trips the gate.
`--fail-on-rising-cost` fails when any case carries the `rising`
label (slope bigger than 10% of its mean token count per run),
which catches cost creep on cheap cases whose absolute slope is too
small for `--fail-on-cost-trend` to notice. Either flag combines
with the existing newly-failing, latency-gate, and slowing rules:
the exit code is 1 when any of them trips, and 0 otherwise. The
text report names the cases that tripped the gate with their slope
and mean token count, and the JSON output carries
`cost_gate_threshold_tokens_per_run`, `cost_gate_failures`, and
`cost_gate_passed` (plus `rising_cost_cases` with
`--fail-on-rising-cost`). From Python,
`TrendReport.cost_gate_failures(threshold)` returns the tripping
cases worst first, and `cost_gate_passed(threshold)` is the boolean
form.

### Per-tag case-wins CI gate

A run can win the aggregate while losing a whole tag case by case —
the tag one run keeps losing on is visible in the `Case wins by
tag:` breakdown, and `--fail-on-lose-per-tag RUN` turns it into a
gate:

```bash
python -m agent_evals.trends --reports-dir reports/ \
    --fail-on-lose-per-tag baseline.json
```

It fails (exit 1) when the named baseline run won strictly more
cases under any tag than any other run in the history (a tie in wins
under a tag is not a loss), mirroring the `--fail-on-lose-per-case`
gate on the two-run comparison but per tag, using the
`case_wins_by_tag` breakdown. The run is named by its path, the
same identifier the per-run table uses (usually the report
filename); naming a run that is not in the history is a clean
error. It combines with the existing newly-failing, latency-gate,
slowing, cost-gate, and rising-cost rules: the exit code is 1 when
any of them trips, and 0 otherwise. The text report names the tags
and runs that tripped the gate with their win counts, and the JSON
output carries `per_tag_gate_baseline`, `per_tag_gate_failures`,
and `per_tag_gate_passed`. From Python,
`TrendReport.per_tag_losers_to(baseline)` returns the tripping
(tag, run) pairs, and `per_tag_gate_passed(baseline)` is the
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
as JSON. In the JSON report each case carries its run count, passes,
fails, and flakiness flag (`"runs"`, `"passes"`, `"fails"`,
`"flaky"`), and the text summary lists the flaky cases by name with
their pass counts. Passes are threshold-aware: a case's pass
threshold decides the verdict in every run, so a case that passes
its checks on a low score still fails when the score is below its
threshold, and a partial-credit case passes when its score reaches
it. The report carries the threshold per case — `"threshold"` in the
JSON (null when the case has none), a `threshold` column in the text
table (`>=0.75`, or `-` with no threshold) — plus a
`"below_threshold"` flag per case, and the text summary lists the
cases whose mean score does not reach their threshold, so a case
that fails on its threshold instead of its scorer says so. A
threshold change is already part of the case fingerprint, so
changing a threshold changes the case a stability report is
measuring, not just its verdict. `repetitions` must be a positive integer. Tag filters apply
to repeated runs too: `--tag smoke --repeat 5` repeats only the
smoke cases, and the stability report records the filter that was
applied — `tag_filter` in the JSON report (null when no filter was
given) and a `Tags: ...` line in the text summary — so a report
says which subset of the suite it covers. From Python,
`repeat_suite(suite, agent, 5, tags=["smoke"])` is the same run,
with `tag_mode="all"` and `exclude_tags=["slow"]` available.

The stability report also carries a per-tag pass-rate breakdown,
the same shape as a single run's `tag_breakdown` but aggregated
over repetitions: `tag_breakdown` in JSON maps each tag to its case
count, total passes, total case-runs, pass rate, mean score, and
how many cases under the tag were flaky, and the text summary ends
with a `Per-tag pass rates:` section. A case carrying several tags
counts under each of them; untagged cases appear under no tag.

### Flaky-case history across stability reports

One stability report says which cases flipped in a single burst of
repeated runs. A directory of stability reports says which cases
*keep* flipping — a case that is flaky in every report is a
different problem from one that flipped once and has been stable
since:

```bash
python -m agent_evals.flaky_history --reports-dir stability/
python -m agent_evals.flaky_history --reports-dir stability/ \
    --json flaky-history.json
```

Or from Python:

```python
from agent_evals import compare_flaky_history, format_flaky_history
from agent_evals import load_stability_reports_dir

history = compare_flaky_history(
    load_stability_reports_dir("stability/"))
print(format_flaky_history(history))
```

The history shows, per case, its pass-rate history and flaky
history across the reports, and groups cases into **persistently
flaky** (flaky in at least two reports), **flaky in the latest
report** (the live signal), **newly flaky** (stable in every earlier
report, flaky in the latest one), and **stopped flaky** (flaky
before, stable now). A case that first appears flaky in the latest
report was never stable before, so it counts as flaky in the latest
report but not as newly flaky; a case added or removed mid-history
gets no verdict for the reports it was not part of, so it does not
skew the other cases' histories. The history is threshold-aware:
each case carries the pass threshold in effect per report
(`threshold_history` in the JSON, null when the case has none or
was absent) alongside its mean score, and when a case moves
between the flaky and stable groups the history says why — a
transition is flagged threshold-explained when the threshold
changed between the two reports bracketing the move while the mean
score stayed the same, so a case that only started flipping
because its bar moved is not mistaken for an agent regression; a
move with a genuine score change reads as score movement instead.
The text summary annotates both kinds in the newly and stopped
sections and lists the threshold-explained transitions as their
own section. The JSON history carries the same
groups plus per-case totals (`total_runs`, `total_passes`,
`total_fails`, `flaky_reports`) and the pass-rate history, so CI
can track a case's reliability over time. The CLI exits 1 while any
case is flaky in the latest stability report, 0 otherwise, so it
can gate a CI step on current flakiness — a case that stopped
flipping passes the gate, whatever its history says. Files in the
directory that are not stability reports (single-run reports, bad
JSON) are skipped with a warning, not an error.

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
  "cost_violations": 0,
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
- ~~Per-tag filtering for `--repeat` stability reports: the `--tag`
  / `--exclude-tag` filters apply to repeated runs too, and the
  stability report records which tag filter was applied
  (`tag_filter` in JSON, a `Tags: ...` line in the text summary),
  so a stability report says which subset of the suite it covers.~~
  (done in v0.15.0)
- ~~Per-tag pass-rate breakdowns in `--repeat` stability reports:
  the stability report carries a `tag_breakdown` (JSON) and a
  `Per-tag pass rates:` section (text summary), aggregated over
  repetitions with a flaky count per tag, so a tag that keeps
  failing or flipping is visible at a glance.~~ (done in v0.16.0)
- ~~Flaky-case history across stability reports: per-case
  pass/fail history across a directory of `--repeat` reports —
  persistently flaky, newly flaky, and stopped-flaky cases, with a
  `fails` count and a flaky-case list in each stability report —
  so a case that keeps flipping is visible as a history, not a
  single burst, and the CLI gates on flakiness in the latest
  report.~~ (done in v0.17.0)
- ~~Per-case token budgets: a case may carry a `max_tokens`
  budget, and a case that passes its checks but whose agent
  reported more tokens than the budget is a cost violation
  instead of a pass (reported as `COST`, with a
  `cost_violations` count in the JSON report), and
  `--fail-on-cost` makes cost violations fail the CLI run, so
  token spend can gate CI the same way latency budgets do. The
  budget only applies when the agent reports a token count.~~
  (done in v0.18.0)
- ~~Cost-aware trend reports: per-case token-cost trends across a
  reports directory — first/last/mean/min/max tokens, a
  least-squares token slope per run, and a rising/falling/flat
  label — plus a most-expensive ranking and a costliest-growing
  list, with `--fail-on-cost-trend N` and `--fail-on-rising-cost`
  CI gates, so a case that keeps passing but gets steadily more
  expensive is visible before it trips its token budget. Token
  trends only cover cases whose agent reports a token count.~~
  (done in v0.19.0)
- ~~Per-model comparison view: run the same suite against several
  agents (or models) and get one comparison view — per-case and
  aggregate scores, latency, and token/cost totals side by side,
  with a winner per case and an overall ranking — as a CLI entry
  (`python -m agent_evals.model_compare --agent name=module:attr
  ...`) and from Python (`compare_models` /
  `format_model_comparison`).~~ (done in v0.20.0)
- ~~Compare saved reports side by side: a two-run diff shows
  per-case score, latency, and token/cost deltas in one table,
  names an overall winner (weighted score, then pass rate, then
  lowest latency, then lowest tokens), and carries the total token
  delta in the headline and in the JSON output, so a run that
  passes everything but got slower or more expensive is visible
  at a glance.~~ (done in v0.21.0)
- ~~`--fail-on-lose` CI gate for the per-model comparison:
  `python -m agent_evals.model_compare ... --fail-on-lose
  BASELINE` exits 1 when the named baseline agent beats any
  other agent in the comparison (weighted score, then pass rate,
  then lowest total latency — an exact tie is not a loss), so a
  candidate agent that regresses against the baseline fails CI
  instead of just losing a row in the table.~~ (done in v0.22.0)
- ~~Per-case variant of the `--fail-on-lose` gate:
  `--fail-on-lose-per-case BASELINE` exits 1 when the named
  baseline agent won strictly more individual cases than any
  other agent (a tie in case wins is not a loss), so a candidate
  that loses case by case fails CI even if it wins the aggregate;
  the comparison JSON also carries a `case_wins` count per
  agent.~~ (done in v0.23.0)
- ~~Per-case-wins gate for the two-run comparison: the diff counts
  case wins per run (higher score wins a shared case, ties broken
  by lowest latency; a case both runs scored 0 on, an exact tie,
  or a new/removed case counts for nobody), prints them and
  carries them in the JSON, and `--fail-on-lose-per-case` exits 1
  when the baseline run won strictly more individual cases than
  the current run (a tie is not a loss), so a run that loses case
  by case fails CI even if it wins the aggregate.~~ (done in
  v0.24.0)
- ~~Per-tag case wins in the per-model comparison: the multi-agent
  comparison view breaks case wins down per tag — which agent won
  how many cases under each tag, with the cases no agent won
  recorded as ties and not counted as wins — in the text summary
  (`Case wins by tag:`) and in the JSON (`case_wins_by_tag`), with
  each case's tags in its per-case row, so a tag one agent keeps
  losing on is visible at a glance.~~ (done in v0.25.0)
- ~~Per-tag case wins in the two-run comparison: the two-run diff
  breaks case wins down per tag the same way — which run won how
  many cases under each tag, with the cases no run won recorded
  as ties — in the text summary (`Case wins by tag:`) and in the
  JSON (`case_wins_by_tag`), with each case's tags in its per-case
  row, so a tag one run keeps losing on is visible at a glance.~~
  (done in v0.26.0)
- ~~Per-tag case wins in the trend report: the trend across a
  reports directory breaks case wins down per tag the same way —
  which run won how many cases under each tag, with the cases no
  run won recorded as ties — in the text summary (`Case wins by
  tag:`) and in the JSON (`case_wins_by_tag`), with each case's
  tags in its per-case row, so a tag one run keeps losing on is
  visible at a glance. A case wins the run with its highest score
  over the history, ties broken by lowest latency; a case every
  run scored 0 on, an exact tie, or a case present in only one run
  has no winner.~~ (done in v0.27.0)
- ~~Per-tag case-wins CI gate for the trend report:
  `--fail-on-lose-per-tag RUN` exits 1 when the named baseline run
  won strictly more cases under any tag than any other run in the
  history (a tie in wins under a tag is not a loss), so a run that
  loses a whole tag case by case fails CI even if it wins the
  aggregate — the per-tag variant of the `--fail-on-lose-per-case`
  gate, using the `case_wins_by_tag` breakdown.~~ (done in v0.28.0)
- ~~JSON-subset scorer: `json_subset` checks that the output JSON
  contains every expected field with a matching value, recursively,
  while extra fields are ignored — a subset match, not a full
  equality — with partial credit per matched leaf value, so an agent
  that returns JSON with fields you did not ask for can still be
  scored on the fields you did.~~ (done in v0.29.0)
- ~~Fuzzy similarity scorer: `fuzzy` passes when the output is
  close enough to an expected string — a `difflib` SequenceMatcher
  ratio at or above a `threshold` (default 0.8) — with partial
  credit equal to the ratio, so a rephrased answer, a typo, or an
  extra word can still pass without an exact match.~~ (done in
  v0.30.0)
- ~~`starts_with` scorer: `starts_with` passes when the output
  starts with a fixed prefix, with partial credit equal to the
  fraction of the prefix matched from the start, so an output that
  gets the first few words right still earns part of its weight.~~
  (done in v0.31.0)
- ~~`ends_with` scorer: `ends_with` passes when the output
  ends with a fixed suffix, with partial credit equal to the
  fraction of the suffix matched from the end, so an output that
  gets the last few words right still earns part of its weight.~~
  (done in v0.32.0)
- ~~`not_contains` scorer: `not_contains` passes when the output
  contains none of the listed substrings — the mirror of
  `contains_all` — with partial credit equal to the fraction of
  listed substrings absent, so an output that slips in one
  forbidden phrase still earns most of its weight.~~
  (done in v0.33.0)
- ~~Case-insensitive string scorers: the six string scorers
  (`contains`, `contains_all`, `contains_any`, `not_contains`,
  `starts_with`, `ends_with`) accept an optional
  `"ignore_case": true`, which case-folds both the expected strings
  and the output before comparing, so `"Hello"` matches `"hello"`
  and a forbidden phrase fails `not_contains` in any casing —
  with scores and partial credit unchanged, details still naming
  the substrings as written, and a non-boolean value rejected as a
  scorer error.~~ (done in v0.34.0)
- ~~`ignore_case` for the `exact` scorer: `exact` accepts an optional
  `"ignore_case": true`, which case-folds both the expected string
  and the output before comparing, so `"Hello"` matches `"hello"` —
  with the score still all-or-nothing, the option defaulting to off,
  and a non-boolean value rejected as a scorer error.~~ (done in v0.35.0)
- ~~`ignore_case` for the `fuzzy` and `regex` scorers: `fuzzy` accepts
  an optional `"ignore_case": true`, which case-folds both strings
  before the SequenceMatcher ratio, and `regex` accepts an optional
  `"ignore_case": true`, which matches with `re.IGNORECASE` — with
  scores and partial credit unchanged, the option defaulting to off,
  and a non-boolean value rejected as a scorer error.~~ (done in v0.36.0)
- ~~`length` scorer: `length` passes when the output's length falls in
  `[min_length, max_length]` — at least one bound required, a missing
  bound unbounded on that side — measured in characters by default or
  in words with `"unit": "words"`, with linear partial credit falling
  to 0 at a length of 0 (too short) or twice the maximum (too long),
  and non-integer, negative, or inverted bounds rejected as scorer
  errors.~~ (done in v0.37.0)
- ~~`range` scorer: `range` passes when the first number in the output
  falls in `[min, max]` — at least one bound required, a missing bound
  unbounded on that side — with linear partial credit falling to 0 one
  range width outside the bound (or one bound magnitude with a single
  bound), an output with no number scoring 0, and non-numeric or
  inverted bounds rejected as scorer errors.~~ (done in v0.38.0)
- ~~`json_equals` scorer: `json_equals` passes when the output JSON is
  deeply equal to the expected value — full equality, not a subset,
  so extra keys fail — with numbers compared numerically (`1` equals
  `1.0`), `bool` never accepted as `int`, an optional `ignore_order`
  that compares arrays as multisets, an invalid-JSON output scoring
  0.0, and a non-serialisable `expected` rejected as a scorer
  error.~~ (done in v0.39.0)
- ~~`tolerance` for the `json_equals` scorer: an optional `tolerance`
  compares numbers within that tolerance instead of requiring exact
  numeric equality, recursively — deep equality otherwise unchanged,
  `bool` still never accepted as `int`, invalid-JSON output still
  scoring 0.0, `ignore_order` still comparing arrays as multisets
  (matched pairwise under tolerance), and a non-numeric or negative
  `tolerance` rejected as a scorer error.~~ (done in v0.40.0)
- ~~`tolerance` for the `range` scorer: an optional `tolerance`
  widens the accepted band by that amount on each bounded side —
  the value passes when it falls in `[min - tolerance,
  max + tolerance]` — with the partial-credit falloff computed
  from the widened band, so a float answer that is close but just
  outside the range still passes; validated the same way as the
  `json_equals` tolerance (non-numeric, negative, NaN, or infinite
  rejected as a scorer error), and a tolerance of 0 keeps the old
  behaviour exactly.~~ (done in v0.41.0)
- ~~`contains_regex` scorer: `contains_regex` passes when the regex
  matches anywhere in the output — the strict counterpart of
  `regex`, so a non-string `pattern` is a scorer error instead of
  being stringified — with an invalid regex also a scorer error,
  `ignore_case` composing through `re.IGNORECASE`, and the scorer
  flowing through suite JSON like the other scorers.~~ (done in v0.42.0)
- ~~Per-case pass thresholds: a case may carry a `threshold` in [0, 1]
  and passes when its scorer's score reaches it — so a partial-credit
  case can pass without a full score, and a case its scorer passes on
  a low score can still fail — with the threshold in the case
  fingerprint and per result in the JSON report, the verdict reason
  in the detail, and timeouts, agent errors, and scorer errors still
  failing whatever the threshold is; a non-numeric or out-of-range
  threshold is rejected when the suite loads.~~ (done in v0.43.0)
- ~~Threshold-aware stability reports: a `--repeat` report carries
  each case's pass threshold (`"threshold"` in the JSON, null when
  the case has none; a `threshold` column in the text summary) and
  a `"below_threshold"` flag, and the text summary lists the cases
  whose mean score does not reach their threshold — so a case that
  passes on raw score but fails its threshold reads as failing
  with the reason visible, and a partial-credit case that only
  passes because of its threshold says so; the pass counts were
  already threshold-aware, and changing a threshold already
  changes the case fingerprint.~~ (done in v0.44.0)
- ~~Threshold-aware flaky history: a flaky-history report records
  each case's pass threshold in effect per report
  (`"threshold_history"` in the JSON, null when the case has none
  or was absent) alongside its mean score, and when a case moves
  between the flaky and stable groups the history says why — a
  transition is flagged threshold-explained when the threshold
  changed between the two reports bracketing the move while the
  mean score stayed the same, so a case that only started flipping
  because its bar moved is not mistaken for an agent regression,
  and a move with a genuine score change reads as score movement
  instead; the text summary annotates both kinds in the newly and
  stopped sections and lists the threshold-explained transitions
  as their own section.~~ (done in v0.45.0)
- ~~`one_of` scorer: `one_of` passes when the stripped output
  equals one of a listed set of allowed values — the scorer for
  classification-style evals, an answer that has to be "yes", "no",
  or "maybe" — with whole-output equality like `exact` (not
  containment), all-or-nothing scoring, an optional
  `"ignore_case": true`, and a missing, non-list, or empty
  `values` rejected as a scorer error.~~ (done in v0.46.0)
