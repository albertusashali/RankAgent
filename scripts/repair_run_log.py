"""Repair token accounting in an existing run log.

WHY THIS EXISTS
---------------
Runs produced before the token-accounting fix wrote the *running totals* into each
iteration's ``prompt_tokens`` / ``completion_tokens`` fields instead of what that
iteration actually spent. Reading such a log makes a 1,516-token iteration look
like a 5,670-token one, and Feasibility is scored on those numbers.

The per-iteration values are exactly recoverable, because the logged column is a
strictly increasing cumulative series: iteration N spent ``cum[N] - cum[N-1]``.
This script does that differencing, moves the original figures into explicit
``cumulative_*`` fields, and rewrites the log. Nothing about the run changes — the
same tokens were spent, they are simply attributed to the right iteration. The
run-level totals in the summary are untouched and are re-verified afterwards.

It is idempotent: a log already carrying ``cumulative_prompt_tokens`` is left alone.

Usage:
    python scripts/repair_run_log.py logs/run_summary.json          # repair
    python scripts/repair_run_log.py logs/run_summary.json --dry-run
"""
import argparse
import json
import os
import shutil
import sys


def looks_cumulative(iterations) -> bool:
    """True if the token column is non-decreasing across iterations with LLM calls."""
    seen = [it.get("prompt_tokens", 0) for it in iterations]
    nonzero = [v for v in seen if v]
    if len(nonzero) < 2:
        return False
    return all(b >= a for a, b in zip(nonzero, nonzero[1:])) and nonzero[-1] > nonzero[0]


def repair(payload: dict) -> tuple:
    iterations = payload.get("iterations", [])
    if any("cumulative_prompt_tokens" in it for it in iterations):
        return payload, ["already repaired — cumulative_* fields present; nothing to do"]

    if not looks_cumulative(iterations):
        return payload, ["token column is not a cumulative series; nothing to repair"]

    notes = []
    prev_p = prev_c = 0
    total_calls = payload.get("llm_calls", 0)
    llm_iters = [it for it in iterations if it.get("proposal_source") == "llm"]

    for it in iterations:
        cum_p = int(it.get("prompt_tokens", 0) or 0)
        cum_c = int(it.get("completion_tokens", 0) or 0)
        per_p, per_c = cum_p - prev_p, cum_c - prev_c

        it["prompt_tokens"] = per_p
        it["completion_tokens"] = per_c
        it["cumulative_prompt_tokens"] = cum_p
        it["cumulative_completion_tokens"] = cum_c

        # Call attribution: one proposal call per LLM-sourced iteration, plus any
        # repair calls. This run recorded no recoveries, so the mapping is 1:1.
        if it.get("proposal_source") == "llm":
            calls = 1 + len((it.get("error_recovery") or {}).get("attempts", []) or [])
        else:
            calls = 0
        it["llm_calls"] = calls

        notes.append(f"  iteration {it['iteration_id']}: "
                     f"{cum_p} cumulative -> {per_p} this iteration "
                     f"({per_c} completion)")
        prev_p, prev_c = cum_p, cum_c

    # The per-iteration figures must still sum to the run totals.
    sum_p = sum(int(it.get("prompt_tokens", 0)) for it in iterations)
    sum_c = sum(int(it.get("completion_tokens", 0)) for it in iterations)
    if sum_p != payload.get("total_prompt_tokens") or sum_c != payload.get("total_completion_tokens"):
        raise SystemExit(
            f"REFUSING TO WRITE: repaired per-iteration tokens sum to "
            f"({sum_p}, {sum_c}) but the run totals are "
            f"({payload.get('total_prompt_tokens')}, {payload.get('total_completion_tokens')}). "
            f"The log is not a simple cumulative series; inspect it by hand.")
    notes.append(f"  verified: per-iteration sums {sum_p} in / {sum_c} out "
                 f"match the run totals exactly")

    inferred_calls = sum(int(it.get("llm_calls", 0)) for it in iterations)
    if total_calls and inferred_calls != total_calls:
        notes.append(f"  NOTE: inferred {inferred_calls} LLM calls but the summary "
                     f"records {total_calls}; left the summary total authoritative")
    return payload, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="logs/run_summary.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    with open(a.path, encoding="utf-8") as fh:
        payload = json.load(fh)

    repaired, notes = repair(payload)
    print(f"repairing {a.path}")
    for n in notes:
        print(n)

    if a.dry_run:
        print("\n--dry-run: no files written")
        return

    backup = a.path + ".pre-repair.bak"
    if not os.path.exists(backup):
        shutil.copy2(a.path, backup)
        print(f"  original preserved at {backup}")

    with open(a.path, "w", encoding="utf-8") as fh:
        json.dump(repaired, fh, indent=2)
    print(f"  wrote {a.path}")


if __name__ == "__main__":
    main()
