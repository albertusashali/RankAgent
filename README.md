# RankAgent

An autonomous ML research agent for the **KuaiRand-Pure** within-user ranking
benchmark. It reproduces the organizers' baseline, then improves on it by
**writing and running its own code** — new loss functions, new architectures, new
feature logic — verifying each change before spending a training run on it.

```
python main.py --data_dir data/KuaiRand-Pure/data --max_iterations 30
```

---

## 1. What it does

The benchmark ranks a user's own impressions. Label `long_view`; metrics GAUC and
nDCG@5; the primary score is their mean. The official Factorization Machine
baseline scores **0.6016** on validation.

Each iteration, five agents run one experiment:

| Role | Owns | Output |
| :--- | :--- | :--- |
| **Product Manager** | coverage across seven research dimensions | a directive |
| **ML Researcher** | hypotheses, grounded in a citation-bearing knowledge base | k hypotheses |
| **Engineer** | writing the code | a source patch |
| **Feature Steward** | the feature space | a validated feature recipe |
| **QA** | trust | gate verdicts, and repairs |

The agent's action space is **file I/O**, not a flag menu. A typical iteration
reads `pipeline/models.py`, writes a patch implementing (say) ApproxNDCG,
registers it so `--loss approx_ndcg` becomes selectable, verifies it, trains it,
and records the diff. Runs have produced working implementations of ApproxNDCG,
ListMLE, focal loss, DCN-v2, PLE and a multi-head cross-attention ranker — none
of which existed in the codebase.

### Measured results

| | validation primary | vs baseline |
| :--- | ---: | ---: |
| Random scoring floor | 0.4834 | — |
| **Official FM baseline (published)** | **0.6016** | — |
| Our reproduction of it | 0.6015 | −0.0001 |
| Best agent-written model (`cross_attention`) | **0.6041** | **+0.0026 (3.2σ)** |
| `dense_deepfm` (causal features in a neural model) | 0.6024 | +0.0009 |
| Oracle ceiling | 0.8484 | — |

Seed noise is σ = 0.0008, so a gain under 0.0024 is not evidence. The agent
designates its own final submission and records the margin in units of σ.

---

## 2. Setup

**Requirements:** Python 3.11+ (developed on 3.14), ~4 GB RAM, no GPU needed.
A full run takes 10–40 minutes on a laptop CPU.

```bash
git clone https://github.com/albertusashali/RankAgent.git
cd RankAgent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Data.** Download KuaiRand-Pure and unpack it so the logs sit at
`data/KuaiRand-Pure/data/`:

```bash
mkdir -p data && curl -L https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz \
  | tar -xz -C data
ls data/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv
```

**API key** (optional — the system runs fully without one, see §3.4):

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY or ANTHROPIC_API_KEY
```

---

## 3. Reproducing our results

### 3.1 Verify the harness — free, ~20 seconds

```bash
.venv/bin/python -m pytest tests/ -q
```
Expect `86 passed`. No API key, no data, no training.

```bash
.venv/bin/python scripts/smoke_agents.py
```
Expect `ALL CHECKS PASSED`. This plans ten iterations with zero API calls,
applies a real patch, proves a newly registered loss becomes CLI-selectable,
tampers with the scorer and shows it reverted, and attempts five known leaks and
shows each blocked.

### 3.2 Reproduce the official baseline — ~90 seconds

```bash
.venv/bin/python -m pipeline.train --model fm --data_dir "$(pwd)/data/KuaiRand-Pure/data"
```
Expect `Primary: 0.6015` against the published 0.6016. The agent runs this itself
at iteration 0 of every run and **halts** if the drift exceeds 0.005 — every
later delta is measured against a reference that was verified, not typed in.

### 3.3 A full autonomous run — 10–40 min, roughly $1–3 of tokens

```bash
.venv/bin/python main.py --data_dir "$(pwd)/data/KuaiRand-Pure/data" --max_iterations 30
```

Then read what it did:

```bash
.venv/bin/python scripts/inspect_run.py
```

Artefacts: `logs/run_log.md` (human-readable, with every diff),
`logs/run_summary.json` (machine-readable), `logs/runs/<run_id>.*` (archived),
`workspaces/node_NNN/` (each iteration's code), and
`submissions/kuairand_pure_final.csv`.

### 3.4 Deterministic mode — no API key, no cost

```bash
OPENAI_API_KEY=none ANTHROPIC_API_KEY=none \
  .venv/bin/python main.py --data_dir "$(pwd)/data/KuaiRand-Pure/data" --max_iterations 5
```
Every agent has a hand-written fallback, so the loop runs end to end with zero
tokens. The log says so plainly — `llm_available: false`, and a banner stating
the run demonstrates the harness rather than autonomy. A run with no key and a
run driven by a model must never be confusable in the artefact a judge reads.

### 3.5 Feature governance

```bash
.venv/bin/python -m pipeline.feature_agent --dynamic \
  --data_dir "$(pwd)/data/KuaiRand-Pure/data"
```
Expect `[FEATURE AUDIT] PASS`.

---

## 4. Limitations, honestly

**The gain is small and near the noise.** +0.0026 at 3.2σ clears the
significance bar, but only just, and run-to-run variance (0.6038 vs 0.6041 on
identical setups) is comparable to the effect. Two runs are not evidence of a
reliable improvement, and the final ranking is on a hidden test set where a
validation gain this size may not transfer.

**Improvements compound only weakly.** Each node inherits its parent's code, and
a node within 3σ of the best stays eligible as a parent — so neutral changes are
not discarded. But acceptance is still driven by a single validation score, and
in practice chains stay short. A 30-iteration run has not yet produced a node
whose result depends on three stacked edits.

**Generated losses are worse than generated architectures.** ApproxNDCG, ListMLE
and focal loss all ran and all scored well below baseline (0.48–0.55), while
DCN-v2 and PLE landed near it on a first attempt. The nDCG-family objectives
need per-group sorting over a ragged batch, which is genuinely hard to get right;
we added a `group_padded` helper to make it writable, and they still come out
poor. The PM keeps directing effort at `loss` first because it is the
highest-priority dimension, which spends budget on the agent's weakest area.

**A patch can be a silent no-op.** One run produced a `CyclicLR` scheduler that
was constructed and never stepped — a one-line diff claiming a training-schedule
change that did nothing. Nothing detects a patch that cannot affect the result.

**Converges early.** Halting needs no 0.002 gain over three iterations, which is
the *normal* state of this benchmark, so runs stop around iteration 11 of 30.

**Scope we did not cover.** Only KuaiRand-Pure; the loaders hard-code the Pure
filenames and split dates, so the 1K/27K variants would silently produce empty
splits. Preprocessing and post-processing (calibration, N-way blending, per-user
re-ranking) remain immutable, so *"every upstream and downstream module is fair
game"* is only partly satisfied. There is no run resumption: a process killed at
iteration 25 loses the tree, though the log and submission remain readable.

### Given more time

1. **Make composition the point of the search.** Explicit `draft` / `improve` /
   `debug` node modes with UCB selection over a frontier, and lineage in the
   Researcher's prompt, so the agent reasons about *what it has already built*.
   This is the single change most likely to convert more iterations into a
   better score.
2. **Ensembling.** `blend_weight_on_valid` exists and is capped at two models.
   Rank-normalised blending of decorrelated members (a GBDT and a neural ranker)
   is historically the largest single gain on this kind of benchmark and we never
   spent an iteration on it.
3. **No-op detection.** AST dataflow to reject a patch whose new symbols are
   never read.
4. **Make preprocessing agent-editable**, behind the differential seal test that
   already exists — eleven raw columns (`hourmin`, `time_ms`, `is_hate`, …) are
   currently discarded before any agent sees them.
5. **Multi-seed confirmation** before designating a submission, so the final
   choice rests on n=3 means rather than a single noisy run.

---

## 5. Team

Four contributors, by the branch each authored:

| Contributor | Contribution |
| :--- | :--- |
| **albertus.ashali** | Initial RankAgent framework, orchestrator state machine, tree manager and run logging (`main`); PLE / DCN-v2 / BST models with dense feature support (`feature/domain_prompt`) |
| **gohpk** | Multi-agent framework — PM / Researcher / Engineer / QA roles and the blackboard (`agent_creation`); the code-generation overhaul — per-node workspaces, patch engine, verification gates, submission designation (`codegen`) |
| **brianyeo02** | Feature engineering and governance — the feature manifest, mutation-based leakage audit, and the validated recipe search space (`feature-engineering`) |
| **kz** | Duplicate-experiment guard and failure containment (`feat/dupe_guard`) |

Integration of `feature-engineering` into the code-generation branch, and the
documentation, were done on `codegen`.

> Roles above are inferred from commit authorship and branch contents — please
> correct them before submission.

---

## 6. Where things are

```
main.py                  entry point
agents/                  the five roles, the patch engine, the paper KB
orchestrator/            the research loop, tree search, intervention ledger
sandbox/                 workspaces, execution, verification, logging
pipeline/                the ML pipeline the agent edits (and the parts it cannot)
tests/                   86 tests, none needing an API key
docs/ARCHITECTURE.md     system architecture and diagrams
scripts/inspect_run.py   summarise the last run
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the component map, data
flow, and the mutable/immutable safety boundary.
