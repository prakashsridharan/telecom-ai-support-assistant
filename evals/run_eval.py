"""Golden-set evaluation harness.

Scores the orchestrator against `golden_set.json` on the dimensions listed in
docs/evaluation.md: intent routing, tool-call accuracy, retrieval recall,
grounding (required substrings present) and safety (forbidden substrings
absent).

Usage:
    python -m evals.run_eval                 # human-readable report
    python -m evals.run_eval --json          # machine-readable, for CI
    python -m evals.run_eval --min-pass 0.95 # fail below a pass-rate threshold

Exits non-zero when the pass rate falls below the threshold, so it can gate a
pipeline. Run it from the repository root.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.models import ChatMessage
from app.services.orchestrator import Orchestrator

GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.json"


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    latency_ms: float
    failures: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)


def _check(case: dict, result: dict) -> tuple[dict[str, bool], list[str]]:
    """Compare one orchestrator response against one expectation block."""
    checks: dict[str, bool] = {}
    failures: list[str] = []
    answer = result["answer"].lower()

    if "expect_intent" in case:
        ok = result["intent"] == case["expect_intent"]
        checks["intent"] = ok
        if not ok:
            failures.append(
                f"intent: expected {case['expect_intent']!r}, "
                f"got {result['intent']!r}"
            )

    if "expect_tools" in case:
        ok = sorted(result["tool_calls"]) == sorted(case["expect_tools"])
        checks["tool_use"] = ok
        if not ok:
            failures.append(
                f"tools: expected {case['expect_tools']}, "
                f"got {result['tool_calls']}"
            )

    if "expect_sources" in case:
        missing = [s for s in case["expect_sources"] if s not in result["sources"]]
        checks["retrieval"] = not missing
        if missing:
            failures.append(
                f"retrieval: {missing} absent from {result['sources']}"
            )

    if "expect_top_source" in case:
        # Recall@1. Stricter than membership and it matters: demo mode shows
        # the top-ranked document, so a correct document ranked third is still
        # the wrong answer on screen.
        top = result["sources"][0] if result["sources"] else None
        ok = top == case["expect_top_source"]
        checks["rank@1"] = ok
        if not ok:
            failures.append(
                f"rank@1: expected {case['expect_top_source']!r} first, got {top!r}"
            )

    if "expect_contains" in case:
        missing = [s for s in case["expect_contains"] if s.lower() not in answer]
        checks["grounding"] = not missing
        if missing:
            failures.append(f"grounding: answer missing {missing}")

    if "forbid_contains" in case:
        present = [s for s in case["forbid_contains"] if s.lower() in answer]
        checks["safety"] = not present
        if present:
            failures.append(f"safety: answer contains forbidden {present}")

    return checks, failures


def run(cases: list[dict]) -> list[CaseResult]:
    orchestrator = Orchestrator()
    results = []

    for case in cases:
        history = [ChatMessage(**t) for t in case.get("history", [])]
        started = time.perf_counter()
        response = orchestrator.respond(case["message"], history)
        latency_ms = (time.perf_counter() - started) * 1000

        checks, failures = _check(case, response)
        results.append(
            CaseResult(
                case_id=case["id"],
                category=case.get("category", "uncategorised"),
                passed=not failures,
                latency_ms=latency_ms,
                failures=failures,
                checks=checks,
            )
        )

    return results


def summarise(results: list[CaseResult]) -> dict:
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    by_check: dict[str, list[bool]] = defaultdict(list)

    for r in results:
        by_category[r.category].append(r)
        for name, ok in r.checks.items():
            by_check[name].append(ok)

    passed = sum(1 for r in results if r.passed)
    latencies = sorted(r.latency_ms for r in results)

    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "by_category": {
            name: {
                "total": len(rs),
                "passed": sum(1 for r in rs if r.passed),
            }
            for name, rs in sorted(by_category.items())
        },
        "by_metric": {
            name: {
                "total": len(values),
                "passed": sum(values),
                "rate": sum(values) / len(values),
            }
            for name, values in sorted(by_check.items())
        },
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p95": round(latencies[int(len(latencies) * 0.95) - 1], 2)
            if latencies
            else 0.0,
        },
        "failures": [
            {"id": r.case_id, "category": r.category, "reasons": r.failures}
            for r in results
            if not r.passed
        ],
    }


def print_report(summary: dict) -> None:
    print("\nTelecom AI Support Assistant — golden-set evaluation")
    print("=" * 60)

    print("\nBy metric")
    for name, stats in summary["by_metric"].items():
        print(
            f"  {name:<12} {stats['passed']:>3}/{stats['total']:<3} "
            f"{stats['rate']:>7.1%}"
        )

    print("\nBy category")
    for name, stats in summary["by_category"].items():
        print(f"  {name:<12} {stats['passed']:>3}/{stats['total']:<3}")

    lat = summary["latency_ms"]
    print(f"\nLatency  mean {lat['mean']} ms   p95 {lat['p95']} ms")

    if summary["failures"]:
        print("\nFailures")
        for failure in summary["failures"]:
            print(f"  [{failure['category']}] {failure['id']}")
            for reason in failure["reasons"]:
                print(f"      - {reason}")

    print(
        f"\nOverall  {summary['passed']}/{summary['total']} "
        f"({summary['pass_rate']:.1%})\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument(
        "--min-pass",
        type=float,
        default=1.0,
        help="minimum pass rate before exiting non-zero (default: 1.0)",
    )
    args = parser.parse_args()

    cases = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    summary = summarise(run(cases))

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_report(summary)

    return 0 if summary["pass_rate"] >= args.min_pass else 1


if __name__ == "__main__":
    sys.exit(main())
