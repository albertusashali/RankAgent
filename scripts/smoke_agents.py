"""Smoke-test the multi-agent layer, cheapest checks first.

    python scripts/smoke_agents.py            # free: no API calls at all
    python scripts/smoke_agents.py --live     # adds ~2 real LLM calls (~1 cent)

No training runs are started by either mode, so both finish in seconds. Use this to
confirm the agents behave before spending wall-clock on `main.py`.
"""
import argparse
import os
import sys

# Run from anywhere: scripts/ is not the project root, so put the repo on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.context import ResearchContext, command_signature
from agents.team import AgentTeam
from orchestrator.schemas import TokenUsage
from orchestrator.state_machine import load_dotenv_if_present

BASELINE = 0.6016


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def offline_plan(iterations: int = 10) -> int:
    """Plan a whole run with no LLM. Proves coverage and duplicate-freedom."""
    rule(f"1. OFFLINE — {iterations} iterations planned with zero API calls")
    meter = TokenUsage()
    team = AgentTeam(meter, verbose=False)
    ctx = ResearchContext(baseline=BASELINE, max_iterations=iterations,
                          wall_clock_budget_s=21600)
    seen, failures = [], 0
    for i in range(1, iterations + 1):
        ctx.iteration = i
        plan = team.plan(ctx)
        if not plan.ok:
            print(f"{i:>3}. (no runnable experiment) {'; '.join(plan.trace[-1:])}")
            failures += 1
            continue
        sig = command_signature(plan.spec.args)
        seen.append(sig)
        print(f"{i:>3}. [{plan.directive.phase:<26}] {plan.spec.dimension:<13} {plan.spec.args}")
        # Pretend it scored at the baseline so the PM sees no winner and keeps exploring.
        team.record(ctx, i, plan.spec, BASELINE, "REJECTED")

    dims = sorted({t.dimension for t in ctx.trials})
    dupes = len(seen) - len(set(seen))
    print(f"\n  dimensions covered : {len(dims)}/7  {dims}")
    print(f"  unique configs     : {len(set(seen))} of {len(seen)}")
    print(f"  duplicates         : {dupes}")
    print(f"  tokens spent       : {meter.total}  (must be 0)")

    ok = (dupes == 0 and meter.total == 0 and len(dims) >= 5)
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def guardrail_checks() -> int:
    """The three guarantees, exercised directly."""
    rule("2. GUARDRAILS — validation, duplicate detection, QA verdicts")
    from agents.engineer import EngineerAgent, validate_args
    from agents.qa import QAAgent
    from agents.researcher import Hypothesis

    failures = 0

    def check(label, got, want):
        nonlocal failures
        ok = got == want
        failures += 0 if ok else 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}")

    ok, why = validate_args("--model fm_torch --loss listwise")
    check("valid arguments accepted", ok, True)
    ok, why = validate_args("--model dcnv2")
    check(f"invented model rejected ({why[:44]}...)", ok, False)
    ok, why = validate_args("--model fm_torch --made_up_flag 3")
    check("invented flag rejected", ok, False)

    ctx = ResearchContext(baseline=BASELINE, max_iterations=10, wall_clock_budget_s=600)
    h = Hypothesis(dimension="loss", hypothesis="within-user listwise softmax objective",
                   args="--model fm_torch --loss listwise", source="playbook")
    spec = EngineerAgent().build(ctx, [h])
    check("engineer produces a runnable spec", spec is not None, True)
    team = AgentTeam(TokenUsage(), verbose=False)
    team.record(ctx, 1, spec, 0.6024, "ACCEPTED")
    check("same config now detected as duplicate", ctx.is_duplicate(spec.args), True)

    qa = QAAgent()
    check("score below random floor rejected", qa.judge(0.40).trustworthy, False)
    check("score above oracle ceiling rejected", qa.judge(0.95).trustworthy, False)
    check("plausible score accepted", qa.judge(0.6038).trustworthy, True)

    print(f"  --> {'PASS' if not failures else f'{failures} FAILED'}")
    return 0 if not failures else 1


def codegen_checks() -> int:
    """The code-generation guarantees, exercised without an API key."""
    rule("3. CODE GENERATION — patches, isolation, leak gates")
    import shutil

    from agents.codegen import registered_losses, registered_models
    from agents.engineer import validate_args
    from agents.patch import apply_all, parse_edit_blocks
    from sandbox.verifier import verify_source, verify_workspace
    from sandbox.workspace import MUTABLE, materialise, unified_diff

    failures = 0
    ws_dir = "workspaces_smoke"

    def check(label, got, want=True):
        nonlocal failures
        ok = got == want
        failures += 0 if ok else 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}")

    shutil.rmtree(ws_dir, ignore_errors=True)
    try:
        ws = materialise(0, workspaces_dir=ws_dir)
        print(f"  workspace: {ws.root}")

        # A hand-written patch standing in for what the Engineer produces.
        patch = """pipeline/models.py
<<<<<<< SEARCH
LOSSES = {
    'pointwise': pointwise_bce,
=======
@ranking_loss(requires_groups=True)
def smoke_ndcg(logits, labels, group, n_groups):
    return -(logits * labels).sum() / (labels.sum() + 1e-9)


LOSSES = {
    'smoke_ndcg': smoke_ndcg,
    'pointwise': pointwise_bce,
>>>>>>> REPLACE"""

        before = ws.snapshot()
        edits, errors = parse_edit_blocks(patch, MUTABLE)
        check("patch parses into edits", bool(edits) and not errors)
        after, problems = apply_all(before, edits)
        check("patch applies cleanly", not problems)
        for rel, text in after.items():
            if before.get(rel) != text:
                ws.write(rel, text)

        losses = registered_losses(ws.read("pipeline/models.py"))
        check(f"the new loss is now registered ({sorted(losses)})",
              "smoke_ndcg" in losses)
        ok, why = validate_args("--model fm_torch --loss smoke_ndcg",
                                known_losses=losses)
        check("...and is selectable from the command line", ok)
        ok, _ = validate_args("--model fm_torch --loss smoke_ndcg")
        check("...but not against the unpatched repository", ok, False)

        diff, (files, added, removed) = unified_diff(ws.base, ws.snapshot())
        check(f"diff computed from disk ({files} file, +{added}/-{removed})",
              files == 1 and added > 0 and "smoke_ndcg" in diff)

        # Tampering with the scorer is reverted, not merely reported.
        with open(ws.path("pipeline/evaluate.py"), "a", encoding="utf-8") as fh:
            fh.write("\ndef evaluate(*a, **k):\n    return {'primary': 0.99}\n")
        violations = ws.restore_immutable()
        check("edit to the scorer is detected", len(violations) == 1)
        check("...and reverted", "0.99" not in ws.read("pipeline/evaluate.py"))
        check("...while the agent's own edit survives",
              "smoke_ndcg" in ws.read("pipeline/models.py"))

        # Leak gates.
        leaks = {
            "watch_ratio from play_time_ms":
                "def f(row):\n    return row['play_time_ms'] / row['duration_ms']",
            "play_time_ms via .get()":
                "def f(row):\n    return row.get('play_time_ms', 0.0)",
            "reading the test split":
                "def f(splits):\n    return splits['test']",
            "unsealing the labels":
                "import os\nos.environ['RANKAGENT_UNSEAL_TEST'] = '1'",
            "calling load_test_labels":
                "from pipeline.data import load_test_labels\ndef f():\n    return load_test_labels()",
        }
        for name, src in leaks.items():
            found = verify_source("pipeline/features.py", src)
            check(f"blocked: {name}", any(f.fatal for f in found))

        safe = verify_source("pipeline/features.py",
                             "def f(row):\n    return [row['duration_ms'], row['tab']]")
        check("pre-impression features still allowed", safe == [])
        check("the shipped pipeline passes its own gate",
              verify_workspace(ws).ok)

        # Composition: a child inherits its parent's edits.
        child = materialise(1, parent=ws, workspaces_dir=ws_dir)
        check("a child workspace inherits the parent's code change",
              "smoke_ndcg" in registered_losses(child.read("pipeline/models.py")))
        sibling = materialise(2, workspaces_dir=ws_dir)
        check("a fresh branch does NOT inherit it",
              "smoke_ndcg" not in registered_losses(sibling.read("pipeline/models.py")))
    finally:
        shutil.rmtree(ws_dir, ignore_errors=True)

    print(f"  --> {'PASS' if not failures else f'{failures} FAILED'}")
    return 0 if not failures else 1


def live_check() -> int:
    """One PM call and one Researcher call. No training is started."""
    rule("4. LIVE — real API calls (PM + Researcher only, no training)")
    load_dotenv_if_present()
    meter = TokenUsage()
    team = AgentTeam(meter, verbose=True)
    if not team.llm_available:
        print("  No usable API key found in the environment or .env.")
        print("  Set OPENAI_API_KEY or ANTHROPIC_API_KEY and retry.")
        return 1
    print(f"  provider: {team.llm._kind} | model: {team.llm.model}\n")

    ctx = ResearchContext(baseline=BASELINE, max_iterations=10, wall_clock_budget_s=21600)
    ctx.iteration = 1
    plan = team.plan(ctx)

    print()
    if not plan.ok:
        print("  No runnable experiment was produced. Trace:")
        for t in plan.trace:
            print(f"    - {t}")
        return 1

    print(f"  directive : {plan.directive.phase} -> {plan.directive.focus_dimensions}")
    print(f"  reasoning : {plan.directive.reasoning}")
    print(f"  chosen    : {plan.spec.dimension} | {plan.spec.args}")
    print(f"  hypothesis: {plan.spec.hypothesis}")
    if plan.spec.mechanism:
        print(f"  mechanism : {plan.spec.mechanism}")
    ok, _ = __import__("agents.engineer", fromlist=["validate_args"]).validate_args(plan.spec.args)
    print(f"\n  command validates against the real trainer spec: {ok}")
    print(f"  tokens spent: {meter.total:,d} in {meter.calls} calls")
    for role, c in sorted(team.cost_by_agent.items()):
        print(f"    {role:<16s}: {c['prompt'] + c['completion']:>6,d} in {c['calls']} call(s)")
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true",
                    help="also make ~2 real LLM calls (roughly one cent)")
    ap.add_argument("--iterations", type=int, default=10)
    a = ap.parse_args()

    rc = offline_plan(a.iterations)
    rc |= guardrail_checks()
    rc |= codegen_checks()
    if a.live:
        rc |= live_check()
    else:
        print("\n(skipped the live API check; add --live to include it)")

    print(f"\n{'=' * 72}\n{'ALL CHECKS PASSED' if rc == 0 else 'SOME CHECKS FAILED'}\n{'=' * 72}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
