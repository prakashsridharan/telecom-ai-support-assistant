# Evaluation

Evaluation is the part most demo projects skip. This one ships a working
harness, not just a plan.

## Running it

```bash
python -m evals.run_eval                          # human-readable report
python -m evals.run_eval --json                   # machine-readable
python -m evals.run_eval --min-pass 0.95          # gate a pipeline
```

Exits non-zero when the pass rate falls below `--min-pass` (default 1.0).

## What it measures

Each case in `evals/golden_set.json` declares expectations; the runner scores
every expectation it finds and reports per-metric and per-category rates.

| Metric | Expectation key | Question it answers |
|---|---|---|
| `intent` | `expect_intent` | Was the request routed to the right workflow? |
| `tool_use` | `expect_tools` | Were exactly the right tools invoked — no more, no fewer? |
| `retrieval` | `expect_sources` | Did the right documents come back at all (recall@k)? |
| `rank@1` | `expect_top_source` | Did the right document come back *first*? |
| `grounding` | `expect_contains` | Does the answer actually contain the supported fact? |
| `safety` | `forbid_contains` | Is the answer free of claims it must never make? |

`rank@1` is tracked separately from `retrieval` because demo mode displays the
top-ranked document: a correct document ranked third is still the wrong answer
on screen.

Latency is reported as mean and p95 across cases.

## Coverage

The golden set covers five categories:

- **retrieval** — the right knowledge document is found and ranked first
- **tool_use** — identifiers are extracted and the matching tool is called
- **routing** — precision cases, e.g. "my data speed went down" is a knowledge
  question, "my network is down in Chennai" is an outage lookup
- **safety** — unknown orders, unknown customers, unknown locations and
  out-of-scope questions never produce a fabricated answer
- **memory** — follow-up turns inherit intent and identifiers, a topic shift
  does *not* inherit, and identifiers are never harvested from the assistant's
  own turns

## Why this shape

Expectations are substring and set assertions rather than exact-output
comparisons. Exact-match assertions break the moment any wording is reworded,
which trains you to update the test instead of reading it. Substring
expectations pin the *claim* — "$79", "could not find", "ORD-10234" — and leave
the prose free to change.

`forbid_contains` is the most valuable column. It is how "do not fabricate" is
turned into something a CI job can fail on.

## Cases this harness has already caught

The suite is not decorative — these were live defects found by running it:

1. Demo mode returned only a document's first paragraph, so a pricing question
   answered with a bare `# Telecom Plans` heading and no prices.
2. Raw term-overlap retrieval ranked `billing.md` above `plans.md` for "How much
   does the Premium plan cost?", because common function words outweighed the
   rare term "premium". Fixed by moving to TF-IDF cosine with stopwords.
3. "Any network issues in Bengaluru?" was not recognised as an outage query.
4. A follow-up turn harvested the customer ID out of the assistant's *own*
   example text ("for example CUST-1001") and reported a real balance for an
   account the customer never named.

## Metrics a production version would add

The harness reports what can be measured deterministically offline. A
production deployment would also track, against live traffic:

- Task completion rate and escalation accuracy
- Groundedness scored by a judge model rather than substring matching
- Answer quality / user satisfaction (thumbs, CSAT)
- Retrieval recall against human-labelled relevance judgements
- Latency and cost per resolved conversation
- Drift: per-intent volume and tool-failure rates over time

Running the golden set against live LLM mode would also need recorded fixtures
or a tolerance-based judge, since model output is not deterministic.
