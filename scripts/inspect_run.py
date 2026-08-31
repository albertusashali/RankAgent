"""Summarise the last run: what was tried, what was written, what stacked.

    python scripts/inspect_run.py

Reads logs/run_summary.json. Answers the three questions a reader of the run
actually has — did the agent write code, did the code compose, and is the
reported result the one that got submitted.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(path="logs/run_summary.json"):
    if not os.path.exists(path):
        print(f"{path} not found — run main.py first.")
        return 1
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)

    print("=" * 78)
    print(f"run {d.get('run_id')} — {d.get('halt_reason')}")
    print("=" * 78)
    base, drift = d.get("baseline_measured"), d.get("baseline_drift")
    print(f"baseline reproduced : {d.get('baseline_reproduced')}"
          + (f"  measured {base:.4f} (drift {drift:+.5f})" if base else ""))
    best = d.get("best_valid_primary")
    print(f"best valid primary  : {best:.4f} (delta {d.get('best_delta', 0):+.4f})"
          if best else "best valid primary  : none")

    a = d.get("autonomy", {})
    print(f"llm-authored        : {a.get('iterations_llm_authored')}"
          f"/{a.get('iterations_total')} iterations")
    print(f"code patches        : {a.get('nodes_with_generated_code')} "
          f"(+{a.get('generated_lines_added')} lines)")
    print(f"best node source    : {a.get('best_node_source')}, "
          f"code change: {a.get('best_node_had_code_change')}")
    print(f"interventions       : {len(d.get('interventions', []))} "
          f"{[i['kind'] for i in d.get('interventions', [])]}")

    print("\n" + "-" * 78)
    print(f"{'it':>3} {'status':<10} {'primary':>8} {'diff':>7}  target / hypothesis")
    print("-" * 78)
    for e in d.get("iterations", []):
        m = e.get("metrics") or {}
        p = f"{m.get('primary_score'):.4f}" if m.get("primary_score") else "—"
        diff = (e.get("code_diff") or "")
        lines = sum(1 for l in diff.splitlines()
                    if l.startswith("+") and not l.startswith("+++"))
        print(f"{e.get('iteration_id'):>3} {e.get('status',''):<10} {p:>8} "
              f"{('+' + str(lines)) if lines else '—':>7}  "
              f"{(e.get('target_file') or '')[:22]:<22} "
              f"{(e.get('hypothesis') or '')[:60]}")

    sd = d.get("submission_decision", {})
    if sd.get("rationale"):
        print("\n" + "-" * 78)
        print("SUBMISSION")
        print("-" * 78)
        print(f"  iteration {sd.get('chosen_iteration')} at "
              f"{sd.get('valid_primary', 0):.4f}")
        print(f"  {sd['rationale']}")

    # Composition: what each surviving workspace has registered.
    try:
        from agents.codegen import registered_losses, registered_models
        rows = []
        for name in sorted(os.listdir("workspaces")):
            root = os.path.join("workspaces", name)
            try:
                t = open(os.path.join(root, "pipeline/train.py"), encoding="utf-8").read()
                m = open(os.path.join(root, "pipeline/models.py"), encoding="utf-8").read()
            except OSError:
                continue
            rows.append((name, registered_models(t, m), registered_losses(m)))
        if rows:
            base_models = rows[0][1]
            base_losses = rows[0][2]
            print("\n" + "-" * 78)
            print("COMPOSITION — what each node's code had that the baseline did not")
            print("-" * 78)
            for name, models, losses in rows:
                added = sorted((models - base_models) | (losses - base_losses))
                print(f"  {name}: {', '.join(added) if added else '(baseline only)'}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
