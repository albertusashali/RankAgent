# Multi-Agent Architecture

RankAgent runs its research loop as four specialised agents coordinating over a
shared blackboard, rather than as one prompt trying to do domain reasoning, code
generation and debugging at once.

This document explains what each role owns, why the roles are split the way they
are, and — most importantly — **which measured failures each role exists to
prevent**. Every design choice below is traceable to something that went wrong in
a logged run, not to an org chart.

---

## 1. The shape

```
                    ┌──────────────────────────────────────────────┐
                    │            Product Manager Agent             │
                    │   Owns COVERAGE: which axis to work on next  │
                    │   Runs every N iterations, or when stalled   │
                    └───────────────────────┬──────────────────────┘
                                            │ Directive
                    ┌───────────────────────▼──────────────────────┐
                    │            ML Research Agent                 │
                    │   Owns HYPOTHESES: what to try, and why      │
                    │   Returns k candidates per call              │
                    └───────────────────────┬──────────────────────┘
                                            │ Hypothesis[]
                    ┌───────────────────────▼──────────────────────┐
                    │            Engineer (SWE) Agent              │
                    │   Owns RUNNABILITY: validated, non-duplicate │
                    │   Parses against the REAL argparse spec      │
                    └───────────────────────┬──────────────────────┘
                                            │ TrialSpec
                    ┌───────────────────────▼──────────────────────┐
                    │            QA & Debugger Agent               │
                    │   Owns TRUST: pre-flight, verdicts, healing   │
                    └──────────────────────────────────────────────┘

        All four read and write one ResearchContext (the blackboard).
```

Code: [`agents/`](../agents/) — `product_manager.py`, `researcher.py`,
`engineer.py`, `qa.py`, coordinated by `team.py` over `context.py`.

---

## 2. Blackboard, not a pipeline

The obvious wiring is a chain: PM feeds Research, which feeds SWE, which feeds QA.
We deliberately did not do that.

In a chain, every handoff is a **lossy paraphrase**. The PM says "focus on
cold-start", Research invents a cold-start hypothesis, SWE emits
`--model din --embed_dim 64`, and the link to cold-start is fiction. This project
has already been bitten by exactly that failure: an earlier single-agent run
logged a hypothesis about DCN-v2 while executing `--model mmoe`. A four-stage
chain would have given that drift three more places to happen.

Instead, every agent reads the same structured record and writes structured
objects back to it. Nothing is retold in prose between roles. Two consequences:

- **No telephone game.** The Engineer reads the Researcher's typed `Hypothesis`
  object, not a summary of it.
- **Each role is independently testable.** Every agent can be driven from a
  hand-built `ResearchContext`, which is why the agent layer has 28 tests that run
  in 0.1s with no API key.

---

## 3. What each role owns, and the failure it prevents

### Product Manager — coverage and stopping

It would be easy to dismiss this role as decoration. The logs say otherwise. In
the LLM-driven run of 2026-08-29 the single agent:

- proposed MMoE at iteration 1, tuned it at iteration 3, and at iteration 4
  **re-ran an identical configuration** while claiming it had changed a parameter;
- **never once varied the loss function** across four iterations — despite
  listwise being the strongest single lever measured on this benchmark
  (0.6024 vs 0.6011 pointwise, on identical FM capacity).

Neither is a failure of hypothesis quality. Both are failures of *portfolio
management*. A single agent optimising locally will exploit the first thing that
works and never notice an untouched axis.

The PM tracks seven dimensions — `loss`, `architecture`, `multi_task`, `sequence`,
`features`, `capacity`, `optimisation` — and directs effort at untried ones first,
in measured-priority order. It also rotates away when three consecutive
experiments fail to beat the best, which matters because of the halt rule:
convergence fires when the best-so-far curve gains under 0.002 over 3 iterations,
so a run that front-loads its best idea halts early with most of its budget
unspent.

**Measured effect.** Over a simulated 10-iteration run with no LLM, the team
covers **all 7 dimensions with 10 unique configurations and zero duplicates**.
The previous single-agent run covered 2 dimensions in 4 iterations, one of which
was a duplicate.

### ML Researcher — hypotheses with mechanisms

Proposes `k` experiments per call rather than one. Same round trip, but the
Engineer gets alternatives to fall through to when the first choice turns out to
be invalid or already run — which is how the earlier run wasted an iteration.

Every hypothesis must state a **mechanism**: the reason the change should move a
*within-user* ranking metric. "It is a stronger model" is rejected as a mechanism.
The agent is also fed the organizers' measured dead ends (static side features,
larger embeddings) so it does not re-derive known-null results.

Proposals outside the PM's directive are dropped rather than silently followed.

### Engineer — the anti-drift guarantee

This role exists to make one promise: **the experiment that runs is the experiment
that was proposed.** Splitting roles does not achieve that by itself. Validation at
the boundary does:

1. **Every command is parsed against the real `pipeline.train` argparse spec**
   before it may run. An invented flag or out-of-range choice is caught here
   instead of becoming a failed training run.
2. **Every command is checked against run history.** A configuration already
   executed is rejected rather than burning an iteration.
3. **The spec carries its hypothesis**, so the logger records a matched pair
   rather than two independently-authored strings.

Ordering matters here and is deliberate: the Engineer tries *every* alternative
hypothesis before falling back to re-running an old configuration under a fresh
seed. Exploring a new config always beats replicating an old one. (A first
implementation had this backwards and collapsed coverage from 7 dimensions to 2 —
caught by `test_team_covers_multiple_dimensions_over_a_run`.)

Seed replication is not waste when it is *chosen*: repeating a config under a new
seed measures the noise floor, which this benchmark badly needs given the whole
improvement so far is roughly 3σ. Repeating it under the *same* seed is pure waste.

### QA — pre-flight, verdicts, and self-healing

This is the only role that was already mostly built. `SelfHealingDebugger` already
classified failures, applied heuristic repairs, re-ran, and logged every attempt;
rebuilding it as a fresh agent would have been duplication for the sake of the
diagram. The QA agent **wraps** it and adds two things a reviewer should own:

- **Pre-flight.** Cheap assertions before a trial runs, so a doomed experiment
  costs nothing rather than a full training run. Includes a shallow consistency
  check that the hypothesis prose mentions the model the command actually runs —
  the direct guard against the DCN-v2/MMoE drift mode.
- **Result verdicts.** A trial that "succeeds" can still be junk. A score below
  the random-scoring floor (0.4834) means the pipeline broke, not that the model
  is weak. A score above the oracle ceiling (0.8484) is only reachable with label
  leakage. QA rejects both, so a broken run cannot become the new best.

---

## 4. Cost control

Naively this is four LLM calls per iteration, which would move the run from the
cheapest Feasibility tier to the most expensive one for no gain in score. So:

| Role | When it calls out |
| :--- | :--- |
| Product Manager | every `pm_refresh` iterations (default 3), or when progress stalls |
| Researcher | once per iteration, returning several hypotheses |
| Engineer | only when argument validation fails |
| QA | only when a trial actually breaks |

In the steady state that is roughly **one call per iteration, not four**.

Token spend is attributed **per role** (`RunSummary.cost_by_agent`) so the cost of
the multi-agent design is visible rather than buried in a single total.

---

## 5. Running without an API key

Every agent has a deterministic fallback, and a failed LLM call is never partial:
the agent returns either a fully validated object from the model, or a fully formed
fallback. It never mixes the two.

That rule is the direct fix for the earlier bug where a half-usable LLM response
had its command silently swapped while its prose was kept, producing a run log
that claimed one experiment and ran another.

The fallbacks are not stubs — the PM's coverage-first planner and the Researcher's
`agents/playbook.py` produce a full 7-dimension exploration on their own, and the
whole team runs at **zero tokens** in that mode.

---

## 6. Known gap

The Engineer currently emits **validated configurations**, not code patches. That
is a real limitation against the brief, which says writing the code for each stage
is the agent's job. The architecture is shaped to accept it — `TrialSpec` is the
seam — but a patch-writing Engineer needs a plugin contract, git-backed
apply/rollback, and QA validation of generated modules before it is honest to
claim it.

Until then this is a disciplined search over a config space with a strong audit
trail, and the documentation says so rather than overstating it.
