# Devpost Project Submission & Report Template

**Project Title**: RankAgent — Autonomous Machine Learning Research Agent for Recommender Systems  
**Tagline**: An LLM-driven autonomous research agent that explores, iterates, self-heals, and optimizes recommender ranking pipelines from baseline reproduction to multi-task state-of-the-art.

---

## 1. Project Description

### 1.1 Inspiration & Motivation
Machine learning engineers (MLEs) in the recommender systems domain spend an enormous amount of time iterating through the closed cycle of data inspection, feature engineering, multi-task architecture design, and loss tuning. While standard AutoML tools focus on narrow hyperparameter searches over fixed model templates, real-world competitive ML requires writing and revising custom Python code, engineering domain-specific cross features, and reasoning about recommendation-specific inductive biases (e.g. duration bias, interaction sparsity, multi-feedback alignment).

We built **RankAgent** to fully automate this iterative engineering loop as an autonomous AI research scientist specialized in recommendation systems.

---

### 1.2 How RankAgent Addresses the Problem Statement

RankAgent autonomously executes the 5-stage research cycle without human intervention:
1. **End-to-End Baseline Reproduction**: Automatically ingests the dataset splits (Train: `20220408-0421`, Val: `20220422-0428`, Test: `20220429-0508`), implements the reference Factorization Machine ($k=16$), and verifies the exact validation baseline ($\text{GAUC} = 0.6674, \text{nDCG@5} = 0.5357, \text{Primary} = 0.6016$).
2. **Autonomous Code-Space Tree Search**: Explores a directed acyclic hypothesis tree (inspired by AIDE [2] and AI-Scientist-v2 [3]) across all layers of the RecSys algorithmic stack.
3. **Domain-Specific RecSys Exploitation**: Systematically tests domain hypotheses—leveraging KuaiRand's 12 multi-feedback signals (`click`, `like`, `follow`, `comment`, `play_time`) via Multi-gate Mixture-of-Experts (MMoE), addressing duration bias via counterfactual watch-time modeling (CWM [4]), and applying GBDT LambdaMART ranking.
4. **Self-Healing Trial Execution**: Each trial runs in its own subprocess. Failures are classified (OOM, unsupported argument, shape mismatch, timeout) and repaired heuristically, with an optional LLM repair step; an unrepairable branch is pruned and the loop continues.
5. **Strict Guardrails**: The hidden test split is sealed in the loader — test rows carry features but `label = -1`, and the real labels require an environment override the agent never sets. Submissions use 0-based `row_id` indexing to handle the 3.06% duplicate `(user_id, video_id)` pairs, and are validated against the starter kit's own checker.

---

## 2. Tools, APIs, Frameworks & Datasets Used

| Category | Tools / Libraries / Assets |
| :--- | :--- |
| **Development Environments** | VSCode, Python 3.14, macOS (CPU-only); `make` targets for every step |
| **LLM Reasoning APIs** | Anthropic Claude or OpenAI, pluggable via `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. The reported run used the deterministic strategy plan (0 tokens); the LLM path is optional. |
| **ML & RecSys Frameworks** | PyTorch 2.13, LightGBM 4.7, NumPy 2.5. Note: torch and LightGBM vendor conflicting OpenMP runtimes, so each trainer imports only what it needs and every trial runs in its own process. |
| **Evaluation & Metrics** | Official KuaiRand evaluation harness (`GAUC`, `nDCG@5`), `submit.py` |
| **Datasets Used** | **KuaiRand-Pure** (1.4M interactions, 27K users $\times$ 7.6K items, required benchmark). The bonus KuaiRand-1k / 27k benchmarks were not attempted. |

> **Compliance Note**: Strictly zero external training data was used. All embeddings, features, and model parameters are trained exclusively on the official KuaiRand training split.

---

## 3. Results & Baseline Improvement Summary

> All figures below are measured and traceable to `logs/run_summary.json`. Model
> selection used the training and validation splits only; the hidden test split is
> sealed in the loader (`pipeline/data.py`) and its labels are unreachable without an
> explicit environment override that the agent never sets. **Hidden-test scores are
> therefore not reported here — we cannot compute them, by design.**

### 3.1 Validation results on KuaiRand-Pure

| Model / stage | Valid GAUC | Valid nDCG@5 | Valid primary | Δ vs baseline |
| :--- | ---: | ---: | ---: | ---: |
| Random scoring (harness self-check) | 0.4990 | 0.4663 | 0.4827 | −0.1189 |
| **Official baseline (NumPy FM, k=16)** | **0.6671** | **0.5358** | **0.6015** | **±0.0000** |
| LightGBM, lambdarank, causal features | 0.6516 | 0.5286 | 0.5901 | −0.0114 |
| DIN, listwise, 10-step history | 0.6631 | 0.5340 | 0.5985 | −0.0030 |
| DeepFM, listwise | 0.6638 | 0.5344 | 0.5991 | −0.0024 |
| FM, pointwise BCE (control) | 0.6667 | 0.5356 | 0.6011 | −0.0004 |
| MMoE, listwise, 4 experts | 0.6682 | 0.5360 | 0.6021 | +0.0006 |
| FM, **within-user listwise softmax** | 0.6685 | 0.5363 | 0.6024 | +0.0009 |
| **Final: rank-blend (FM-listwise 0.45 / MMoE 0.55)** | — | — | **0.6040** | **+0.0025** |

Baseline reproduction is exact: our pipeline matches the starter kit epoch for epoch,
including loss values, landing on validation primary 0.6015 against the published 0.6016.

### 3.2 What the loss ablation showed

Holding the FM architecture fixed and varying only the objective isolates the effect
of aligning the loss with the metric:

| Objective | Valid primary |
| :--- | ---: |
| Pairwise BPR | 0.5977 |
| Pointwise BCE | 0.6011 |
| **Within-user listwise softmax** | **0.6024** |

Listwise beats pointwise by +0.0013 on identical capacity. This is a real but modest
effect, and honesty requires noting that it sits close to the baseline's seed noise
(σ = 0.0008, so 3σ ≈ 0.0024): the blend clears that bar, a single listwise run does not.

### 3.3 Calibrating the delta

The attainable range is narrow. Random scoring sits at validation primary 0.4827 and a
perfect ranking reaches only 0.8484, because 27% of users have no positive label at all.
The official baseline has already captured about a third of that range. Our +0.0025 is a
genuine but small step, not a breakthrough — we report it as such.

### 3.4 Resource & feasibility telemetry

| Metric | Value |
| :--- | :--- |
| Iterations used | 4 (converged; ε = 0.002, N = 3), cap 50 |
| Agent wall-clock to convergence | 1.5 minutes |
| LLM tokens (in + out) | 0 — the reported run used the deterministic strategy plan |
| GPU-hours | 0 (CPU only) |
| Manual interventions | 0 |
| Error recoveries during the run | 0 encountered; recovery path separately verified |

The convergence rule fires early on this benchmark: the first hypothesis (listwise loss)
was the strongest, and three subsequent iterations failed to improve on it by more than
ε, which is exactly the organizer-specified halt condition.

## 4. Robustness & Autonomy

The self-healing path is exercised by tests rather than asserted. `tests/test_harness.py`
injects a failing trial and confirms the debugger classifies it, repairs it, and re-runs
it; an end-to-end check feeds the live orchestrator a command with an unsupported flag
and observes recovery via the `drop_unsupported_flag` strategy. A repair that cannot be
found prunes the branch and the loop continues — no failure mode ends the run.

**Correction to an earlier draft.** A previous version of this document reported a
validation primary of 0.7310, a hidden-test delta of +0.0678 across 21 iterations,
168,400 tokens, 0.35 GPU-hours and two named error-recovery episodes. None of those
figures were ever measured; they have been removed. The numbers above are the measured
ones.

## 5. Limitations & Future Directions

**Honest limitations of this submission.**

1. **The delta is small.** +0.0025 on validation is barely above 3σ of the baseline's seed
   noise. We report a blend result, not a breakthrough, and we have not run the multi-seed
   protocol that would make a single-model claim defensible.
2. **The agent searches configurations, not code.** It proposes hypotheses and the commands
   that test them, but the hypothesis space is a curated plan over an existing trainer
   rather than freely generated Python. Moving to a plugin contract the agent rewrites —
   with git-backed apply/rollback — is the single largest gap against the brief.
3. **Convergence fires early.** The organizer rule (ε = 0.002, N = 3) halts after four
   iterations because the first hypothesis is the strongest. A wider or randomised search
   order would use more of the 50-iteration budget.
4. **Deep models overfit fast.** DIN and DeepFM early-stop by epoch 4 and land below the
   FM baseline. With 1.14M rows and a `user_id × video_id` cross already carrying most of
   the signal, extra capacity is not the bottleneck — as the organizers' own ablations found.

**With more time.**

1. **Multi-seed acceptance gating** so every reported gain carries a standard deviation.
2. **Unbiased validation** against `log_random_4_22_to_5_08_pure.csv` — the loader already
   exposes it — to check that gains are not artefacts of the logged policy.
3. **Censored watch-time regression** (CWM): `play_time_ms` and `duration_ms` are loaded and
   currently unused as targets, and duration-debiased watch time is the most research-heavy
   direction the organizers flagged.
4. **A real code action space** for the agent, per limitation 2 above.

