# Overhaul Plan — from config search to code generation

**Branch:** `plan` (planning only; implementation lands on feature branches off `agent_creation`)
**Status:** proposal, not yet implemented
**Date:** 2026-08-30

---

## 0. Verdict on the critique

The central charge is correct and I am not going to soften it. `pipeline/train.py`
exposes 16 argparse flags over 5 pre-written architectures. `agents/engineer.py`
validates a candidate string against `build_parser()` and returns it. The agent's
entire action space is the Cartesian product of flags a human already wrote. That
is Optuna with a language model attached, and the run log proves it: every
`code_diff` field in `logs/runs/*.json` is empty, because nothing on disk ever
changed.

Two corrections, both of which shrink the work:

**Point 3 is already half-solved.** `sandbox/logger.py:49` `capture_diff()` shells
out to `git diff --unified=3`, and `IterationLogEntry.code_diff`
(`orchestrator/schemas.py:71`) is written into both the JSON and the Markdown log
inside a fenced diff block. The deliverable field exists and is wired end to end. It
is empty because there is nothing to capture, not because it is missing. Give the
agent a file to edit and the log requirement is satisfied by code that already
runs.

**Point 1 overstates the scope.** "Complete overhaul" is the wrong frame and would
cost you the hackathon. The action space is wrong; almost nothing else is. These
are all orthogonal to how a trial is specified, all already tested, and all things
a code-generating agent needs anyway:

| Keep as-is | Why it survives the overhaul |
| :--- | :--- |
| Hidden-test seal (`pipeline/data.py`) | A code-writing agent makes this *more* necessary, not less |
| Baseline reproduction (0.6015 vs 0.6016) | Still the anchor every delta is measured against |
| `TreeManager` convergence (eps=0.002, N=3) | Organizer-defined; unchanged |
| `ExecutionRunner` subprocess isolation | Now load-bearing: generated code segfaults |
| `RunLogger` JSON+MD, per-run archive | Deliverable format; only the payload changes |
| PM coverage / QA verdicts / token attribution | Portfolio management is orthogonal to action space |
| The four-role split itself | Only the Engineer's job description changes |

What actually gets replaced: **the Engineer's output type**, and **what a tree node
holds**. Everything else is refactored around those two.

---

## 1. Core design

### 1.1 Mutable surface / immutable surface

The moment an LLM can write files, "can it rewrite the scorer to return 0.99?"
stops being paranoia. So the pipeline splits in two, enforced by the filesystem
rather than by a prompt:

```
MUTABLE  — the agent may rewrite these, freely
  pipeline/models.py        architectures, loss functions
  pipeline/features.py      feature construction, encoders, causal stats
  pipeline/train.py         training loop, optimiser, schedule, early stop

IMMUTABLE — restored from canonical repo before every single run
  pipeline/data.py          the hidden-test seal
  pipeline/evaluate.py      the official scorer
  pipeline/models_np.py     the reproducible FM baseline
  sandbox/, orchestrator/   the harness itself
```

This is the single most defensible design decision in the overhaul and it should
be stated prominently in the writeup: **the agent cannot reach the scorer, the
data loader, or the test seal.** QA's oracle-ceiling check (0.8484) becomes a
second line of defence rather than the only one.

### 1.2 Per-node workspaces

A tree node is no longer `(hypothesis, flags, score)`. It is a **code state**.

```
workspaces/
  node_000/pipeline/{models,features,train}.py   <- root, = canonical repo
  node_004/pipeline/{models,features,train}.py   <- root + a NeuralNDCG loss
  node_009/pipeline/{models,features,train}.py   <- node_004 + user x item crosses
```

Materialising a child: copy the parent workspace, let the Engineer rewrite one
file, then **hard-restore every immutable file from the canonical repo and verify
its SHA-256** before handing the workspace to the runner. A generated edit to
`evaluate.py` is silently reverted and logged as a contract violation.

Execution: `ExecutionRunner.run_command(cmd, cwd=workspace, env={'PYTHONPATH':
workspace})`, with `--data_dir` and `CHECKPOINTS_DIR` pointed at absolute repo
paths so artefacts still land in one place.

Rollback is free: a bad node is simply never made a parent. Nothing in the
canonical repo is ever mutated, so `git status` stays clean during a run and the
existing `capture_diff()` is replaced by an explicit parent→child
`difflib.unified_diff`, which is more honest anyway.

**This is what makes changes compose.** Today the best result is a single MMoE
config at 0.6038. With `improve`-from-parent, listwise loss + new crosses +
sequence modelling stack on one another instead of each restarting from the
baseline. That compounding is the actual score story, not any individual patch.

### 1.3 Patch format

Do **not** ask the LLM for unified diffs. Context lines must match byte-for-byte
and models are unreliable at it; you will spend the hackathon debugging patch
application instead of the agent.

Primary format: **anchored search/replace blocks** (Aider-style):

```
<<<<<<< SEARCH
LOSSES = {
    'pointwise': pointwise_bce,
=======
LOSSES = {
    'neural_ndcg': neural_ndcg,
    'pointwise': pointwise_bce,
>>>>>>> REPLACE
```

Applied by exact string match; unique-match required, otherwise rejected. After
two consecutive application failures on the same target, fall back to **whole-file
rewrite** (`features.py` is 320 lines ~ 4k tokens; the reliability is worth the
tokens). The logged `code_diff` is computed by *us* with `difflib` from the before
and after file contents, so the log records what actually happened rather than
what the model claimed.

---

## 2. The four roles, restated

| Role | Today | After |
| :--- | :--- | :--- |
| **PM** | picks among 7 flag dimensions | picks among code dimensions: `loss`, `features`, `architecture`, `training`, `ensemble`; and picks the **node mode** (draft / improve / debug) |
| **Researcher** | prose + a flag string | prose + mechanism + **target file** + a concrete implementation sketch, grounded in a paper KB |
| **Engineer** | validates flags against argparse | **reads the target file, writes a patch, applies it, makes it import-clean** |
| **QA** | argparse re-check, score sanity | **AST parse, contract check, leakage scan, smoke run**, then score sanity |

### 2.1 Researcher — where the "academic papers" rubric item lands

`prompts/recsys_kb.py` already holds a 36-line playbook. Extend it into a small
citation-bearing KB — 8-12 entries, each with the paper, the one-paragraph idea,
and the *shape* of the implementation. Candidates that are genuinely well-matched
to within-user ranking on this data:

- **NeuralNDCG** (Pobrotyn & Bialobrzeski 2021) — differentiable nDCG via NeuralSort
- **LambdaLoss** (Wang et al. CIKM 2018) — lambdarank as a probabilistic loss
- **ApproxNDCG** (Qin et al. 2010) — sigmoid-smoothed rank positions
- **PLE** (Tang et al. RecSys 2020) — beats MMoE where tasks conflict
- **DCN-v2** (Wang et al. WWW 2021) — explicit bounded-degree crosses
- **ESMM** (Ma et al. SIGIR 2018) — entire-space multi-task, sample-selection bias
- **Focal loss** (Lin et al. 2017) — `long_view` positives are sparse
- **Popularity debiasing / PDA** (Zhang et al. SIGIR 2021)

The Researcher cites the entry it is drawing on; the citation goes into the log
entry. That is the artefact a judge looks for under "Innovation & Problem Insight".

### 2.2 Engineer — the new component

```
build(ctx, hypothesis, workspace) -> Patch | None
  1. read workspace/pipeline/<target_file>
  2. prompt: file contents + contract + hypothesis + KB entry + last traceback (if debugging)
  3. parse SEARCH/REPLACE blocks (or whole file on the 3rd attempt)
  4. apply; on ambiguity or no-match, one reprompt with the failure reason
  5. return Patch(target_file, before, after, unified_diff, blocks_applied)
```

The anti-drift guarantee it currently makes ("the experiment that runs is the
experiment that was proposed") is *preserved and strengthened*: the diff is
computed from the file that ran, so the log physically cannot claim a change it
did not make.

### 2.3 QA — verification gate

Runs in order, cheapest first. Anything that fails returns to the Engineer with
the reason, max 3 repair attempts, then the node is pruned.

1. **Restore + hash immutable files.** Any modification = contract violation, logged.
2. **`ast.parse` + `compile`.** Syntax errors cost milliseconds, not 20 minutes.
3. **Contract check.** A `pipeline/contracts.py` declares required public symbols
   per mutable file — `LOSSES` still contains `pointwise`/`listwise`/`bpr`;
   `TorchFM`/`DeepFM`/`MMoE`/`DIN` still defined; `encode_features`,
   `extract_dense_tabular_features`, `extract_sequential_features` still present
   with compatible `inspect.signature`. Additions welcome, removals rejected.
4. **Leakage scan.** Reject any generated code referencing `load_test_labels`,
   `RANKAGENT_UNSEAL_TEST`, or a `'test'` split label. This is a hard rule and a
   strong thing to be able to show a judge.
5. **Import allowlist.** numpy / torch / lightgbm / stdlib. No `os.system`,
   `subprocess`, `requests`, `eval`, `exec`, or writes outside `checkpoints/`.
6. **Smoke run.** `python -m pipeline.train --smoke ...` — subsampled to ~60k train
   rows, 1 epoch, ~20-30s. Requires exit 0 and one `[EVAL]` line. Score discarded.
   *This is the single highest-leverage addition in the plan:* generated code fails
   often, and without it every failure costs a full training run.
7. **Full run**, then the existing floor/ceiling verdicts.

### 2.4 PM — node mode selection

`TreeManager.select_parent()` becomes real. Policy:

- last node failed and repair attempts < 3 → **debug** (parent = failed node)
- best node beats baseline and depth < max → **improve** (parent = best node)
- PM directs an untouched dimension, or the improve branch has stalled 3x → **draft** (parent = root)

Plus a fifth dimension, `ensemble`, which is not a code patch at all: it calls the
existing `pipeline/submit.py` `export_predictions` / `blend_weight_on_valid` to
blend two prior nodes' cached validation predictions. Cheap, decorrelated, and
historically the largest single gain on this kind of benchmark.

---

## 3. File-by-file change list

**New**
```
sandbox/workspace.py      materialise/copy/restore/hash, unified_diff
sandbox/patcher.py        SEARCH/REPLACE parsing and application
sandbox/verifier.py       AST, contract, leakage, import allowlist
pipeline/contracts.py     required public symbols per mutable file
prompts/paper_kb.py       citation-bearing KB (replaces recsys_kb.py)
tests/test_workspace.py
tests/test_patcher.py
tests/test_verifier.py
```

**Rewritten**
```
agents/engineer.py        flag validator -> code writer
agents/qa.py              keep judge(); add verify(workspace, patch)
agents/researcher.py      emit target_file + implementation sketch + citation
```

**Modified**
```
pipeline/train.py         + --smoke, + --max_rows; CHECKPOINTS_DIR from env
sandbox/runner.py         + cwd/env override per node; STRIP API KEYS (see 5.3)
orchestrator/state_machine.py  iteration = materialise -> patch -> verify -> smoke -> run
orchestrator/tree_manager.py   nodes carry workspace_path + node_mode
orchestrator/schemas.py        + Patch, + node_mode, + verification_report
sandbox/logger.py         code_diff from the Patch, not from git diff
agents/product_manager.py new dimension set + node-mode policy
```

**Untouched (deliberately)**
```
pipeline/data.py  pipeline/evaluate.py  pipeline/models_np.py
kuairand-starter-kit/*
```

---

## 4. Staging — a working system at every step

Each stage is independently shippable. If you run out of time at the end of any
stage, you still have a demo and a log.

| Stage | Content | Output that proves it |
| :--- | :--- | :--- |
| **0** | Tag current `agent_creation` as `v1-config-search`. Freeze the 0.6038 run log and submission. | A fallback submission that cannot be lost |
| **1** | `workspace.py` + immutable restore + runner `cwd`. No LLM change — run the *existing* flag experiments through workspaces. | Baseline still reproduces 0.6015 through the workspace path |
| **2** | Engineer writes code, **`models.py` loss functions only**. Narrow, verifiable, high yield. + `--smoke`, verifier, patcher. | A log entry with a real diff adding `neural_ndcg` to `LOSSES` |
| **3** | Extend the mutable surface to `features.py`. Highest actual score upside on this benchmark (user x item crosses). | A diff in `features.py` that moves the primary metric |
| **4** | `improve`-from-parent composition + `ensemble` node type. | A node at depth 3 whose diff stacks on two ancestors |
| **5** | Paper KB with citations, PM dimension rework, docs + `AGENT_ARCHITECTURE.md` section 6 rewritten (the "known gap" section becomes the headline feature). | Devpost-ready writeup |

**Minimum viable overhaul = stages 0-2.** That alone answers all four of your
objections: real diffs, real code generation, real AIDE-shaped action space, and a
log a judge can read.

---

## 5. Risks, honestly

### 5.1 Score regression — the biggest one
Generated code will score worse than the hand-tuned MMoE (0.6038) for the first
several iterations. Mitigations, in order of importance:

1. The root workspace **is** the current pipeline, so the agent starts from a
   strong state and improves from it rather than re-deriving it.
2. Final submission selects best-on-validation **across all nodes**, including the
   seeded flag-space ones. You cannot end below today's number.
3. Keep the flag-space action as a fallback the PM can select when code-gen has
   failed N times in a row. It is a legitimate research action, not a cop-out.

### 5.2 Wall clock and tokens
Full trial ~5 min (measured: 27.4 min for 5 iterations incl. baseline). Add ~30s
LLM + ~30s smoke, and assume a 30-50% early failure rate. A 6h budget lands around
35-45 iterations, which is comfortably inside the 50 cap.

Tokens: whole-file context is the cost driver. Roughly 15-25k tokens per iteration
(researcher ~1.5k, engineer 5k in / 3k out, plus repairs) x ~40 iterations
= **0.6-1.0M tokens**, single-digit dollars on gpt-4o. Compare 8,774 tokens for the
whole current 5-iteration run. Budget for it and say so in the writeup — a
code-generating agent costing 100x a flag-toggler is the correct trade, but it
should be a stated choice, not a surprise on the invoice.

### 5.3 Security — one real bug to fix first
`sandbox/runner.py:26` does `env = os.environ.copy()`, which hands
`OPENAI_API_KEY` (and everything else in `.env`) to every trial subprocess. Today
that is harmless because we wrote the code. The moment an LLM writes the code that
runs in that subprocess, it is a live credential-exfiltration path. **Strip
`*_API_KEY` from the trial environment in Stage 1.** Also: pin `cwd` to the
workspace and keep the immutable-restore step, but be honest in the writeup that
this is defence-in-depth, not a sandbox — real isolation needs a container and is
out of scope for a hackathon.

### 5.4 Contract rot
An agent that keeps adding to `models.py` will eventually produce a 900-line file
that no longer fits in a prompt cheaply. Cap it: if a mutable file exceeds ~600
lines, the Engineer is instructed to extract into a new module rather than append.

---

## 6. Team split — four parallel workstreams

Dependencies are shallow: A must land first, then B/C/D proceed concurrently
against A's interface.

- **A — Workspace & executor** (`sandbox/workspace.py`, `runner.py`, tree node
  state). Unblocks everyone; do this first, ideally in a day.
  *Done when:* baseline reproduces 0.6015 through a materialised workspace.
- **B — Code generation** (`agents/engineer.py`, `sandbox/patcher.py`). The
  hardest single piece; give it your strongest Python person.
  *Done when:* a SEARCH/REPLACE patch adding a new loss applies cleanly and the
  diff appears in the log.
- **C — Verification** (`sandbox/verifier.py`, `pipeline/contracts.py`,
  `--smoke`). Fully testable offline with hand-written bad patches; needs no API
  key and no GPU.
  *Done when:* the eight checks in section 2.3 each have a test that fails a
  deliberately broken patch.
- **D — Research quality & deliverables** (`prompts/paper_kb.py`, PM dimensions,
  logger payload, `AGENT_ARCHITECTURE.md`, Devpost). Underrated: the rubric
  weights this at 20% and it is the one workstream with no code dependency on A.

---

## 7. Definition of done

The overhaul is complete when a single run produces a log in which:

1. at least one iteration shows a non-empty `code_diff` against
   `pipeline/models.py` or `pipeline/features.py`;
2. that diff implements a named technique with a citation in the hypothesis;
3. a later iteration's diff is applied **on top of** an earlier accepted one;
4. at least one iteration shows a failed patch, a traceback, and a successful
   Engineer repair;
5. the baseline reproduction (0.6015 +/- 0.0005) still appears at iteration 0;
6. the final validation-best score is >= 0.6038 (the v1 result);
7. no iteration references the hidden test set, and the immutable-file hash check
   passed on every node.
