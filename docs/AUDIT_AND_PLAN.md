# RankAgent — Code Audit & Hackathon Work Breakdown

**Scope**: audit of the repository at commit `1091e05`, against `kuairand-starter-kit/` and the challenge brief.
**Sources for every number below**: `logs/run_summary.json`, `kuairand-starter-kit/baseline_scores.json`.

---

## 0. Where we actually are

The validation primary score does **not** span `[0, 1]`. Positioning on the real attainable range:

| Reference | Valid primary | Position in attainable range |
| :--- | ---: | ---: |
| Random scoring | 0.4834 | 0% |
| Item popularity | 0.5807 | 26.7% |
| **Official FM baseline** | **0.6016** | **32.4%** |
| **Our best measured run** | **0.6038** | **33.0%** |
| Oracle ceiling (true labels as scores) | 0.8484 | 100% |

Our best measured run sits **0.6 percentage points of the range** past the baseline it is meant to beat, which is within 3σ of the baseline's own seed noise (σ = 0.0008). Every remaining point has to come out of the 0.2446 gap on the right.

### Verdict

| | Value | Note |
| :--- | :--- | :--- |
| Measured best (valid) | **0.6038** | Iteration 1, MMoE, single seed |
| Delta over baseline | **+0.0022** | 2.75σ on one seed — not yet defensible |
| Iterations completed | **4 / 50** | Loop has never reached the convergence rule |
| Report vs. reality | **0.7310** | The number the Devpost draft claims. Never measured. |

The repository has the right shape: an FSM loop, a subprocess sandbox, a Pydantic contract layer, a pipeline with FM / DeepFM / MMoE / DIN / LightGBM, and a domain knowledge base. That is genuinely most of a submission.

But three things are true at once. The agent has not beaten the baseline in any measured run. It cannot survive a failed trial, because the error branch calls three things that do not exist. And `docs/DEVPOST_SUBMISSION.md` reports results that did not happen.

The good news: the two highest-value modelling levers are untouched, and the organizers have already told us they are untested. That is where the score is.

---

## 1. Red flags — fix these before anything else ships

### 01 — The results table in the Devpost draft is fabricated · **BLOCKING**

`docs/DEVPOST_SUBMISSION.md` reports a validation primary of 0.7310 and a hidden-test delta of +0.0678 across 21 iterations, plus an "Iteration 8 DCN-v2 shape-mismatch recovery" and an "Iteration 14 CUDA OOM mitigation", 168,400 tokens and 0.35 GPU-hours.

`logs/run_summary.json` contains four iterations, a best of 0.6038, zero recorded errors, and zero tokens.

Submitting that table is not a presentation problem, it is a fabricated-results problem, and it puts the whole entry at risk regardless of how the code scores.

> **Fix**: strip every unmeasured number out now. Keep the table structure, fill it from `logs/run_summary.json`, and mark unfilled cells `TBD` until a real run produces them.

### 02 — Any failed trial crashes the whole run · **BLOCKING**

In `orchestrator/state_machine.py`, the error branch of `start_loop`:

- reads `res.stderr`, but `ExecutionResult` defines `error_traceback`;
- calls `self.debugger.attempt_repair(...)`, which is not defined on `SelfHealingDebugger`;
- constructs `SelfHealingDebugger(self.runner)` against a signature of `__init__(self, max_retries=3)`, binding the runner to the retry count.

Robustness is graded on how the agent handles a failure, not on whether it hits one. Right now the first failure raises `AttributeError` and the run dies — the worst possible outcome on that criterion.

> **Fix**: make the debugger real — `attempt_repair(cmd, traceback, target_file) -> Optional[str]`, backed by an LLM patch plus the existing heuristics, wrapped in `try/except` so a repair failure prunes the branch instead of killing the loop.

### 03 — DIN's sequence features leak evaluation labels · **BLOCKING**

In `pipeline/features.py`, `extract_sequential_features`, history is appended with:

```python
if r['label'] == 1 or split_name == 'train':
    user_hist[u].append(v_idx)
```

On valid and test that means a row enters the user's history **only when its ground-truth label is positive** — so the labels of earlier evaluation rows are feeding the features of later ones for the same user. Every DIN number measured on valid or test is contaminated. It happens to have scored badly anyway, which is why nobody noticed.

> **Fix**: append every impression to history regardless of label, on all three splits, strictly in log order. If you want a positives-only history, build it from the **train** window only and freeze it.

### 04 — Our evaluator is not the official evaluator · **BLOCKING**

`pipeline/evaluate.py` is a reimplementation, and it diverges in three places:

1. The official `auc` averages ranks across ties (Mann-Whitney with tie correction); ours does not.
2. The official nDCG sorts stably (`lst.sort(key=lambda x: -x[0])`), so tied scores keep log order; our `np.argsort(-preds)` is not stable.
3. We cast predictions to `float32`, which manufactures ties that would not otherwise exist.

Every selection decision, every logged score, and the final submission are all keyed off a scorer that can disagree with the one that ranks us.

> **Fix**: delete the reimplementation. Import `kuairand-starter-kit/evaluate.py` verbatim as the single source of truth. If you need a fast path for the inner loop, keep it — but add a parity test asserting agreement to 1e-9 against the official one, including on tied inputs.

---

## 2. Full audit — everything else, by severity

| Sev | Location | What is wrong | Fix |
| :--- | :--- | :--- | :--- |
| **High** | `orchestrator/tree_manager.py` · `add_node` | A single `improvement > epsilon` test gates both the best-score update **and** the stagnation reset, so any gain under 0.002 is never recorded as the new best. The convergence rule is also misread: it compares the current score against the best of the last N iterations, not against a running best that only moves on large jumps. | Split the two. Update `best` whenever the score improves at all; reset stagnation only when it beats the best as of N iterations ago. Keep the official 0.6016 as a fixed reference, separate from our own running best. |
| **High** | `orchestrator/state_machine.py` · `query_llm_hypothesis` | The logged hypothesis does not match the command executed. Iter 1 proposes DCN-v2 and runs `--model mmoe`; iter 2 proposes MMoE and runs `--model din`; iter 3 proposes DIN and runs `--model lgb`. The strategy-bank fallback silently overwrites the LLM's command while keeping its prose. | Carry hypothesis and command from the same response, atomically. If a fallback fires, replace the hypothesis text too and mark the entry `source: fallback`. The run-log is a graded deliverable; incoherence reads as fabrication. |
| **High** | `orchestrator/state_machine.py` · `sandbox/logger.py` | Every iteration logs `prompt_tokens: 0` and `completion_tokens: 0` despite the hypotheses being visibly LLM-written. No wall-clock total, no intervention counter, no GPU accounting. | Feasibility & Practicality is 15% and is scored on tokens plus agent wall-clock. Accumulate both per call, persist a run-level total, and write an explicit `manual_interventions` tally. |
| **High** | `orchestrator/state_machine.py` · `query_llm_hypothesis` | The agent never writes code. It selects among five hard-coded CLI strings, and `code_diff` is literally `"Executed: <cmd>"`. The brief says writing the code for each stage is the agent's job, and deliverable 3 requires the code diff per iteration. | Give the LLM a code action space, not an argument space. See workstream **W2**. |
| **High** | `pipeline/train.py` · all trainers | Every model optimizes pointwise `BCELoss` while the metrics are within-user ranking metrics. LightGBM uses `objective: binary`, not `lambdarank`, even though the docs claim LambdaMART. | The starter kit names loss alignment as the single most promising untested direction. See **W3**. |
| **High** | `pipeline/submit.py` · `generate_from_best_checkpoint` | Hard-codes `MMoE(embed_dim=16, num_experts=4)` and prefers `best_mmoe.pt` regardless of which configuration actually won — a winning run at `--embed_dim 32 --experts 6` fails to load. Separately, nothing in the main loop generates a submission at all; only `train_ensemble` does. | Persist the winning config alongside the weights and reconstruct from it. Make submission generation a terminal state of the FSM, driven by the validation-best node. |
| **High** | `pipeline/features.py` · `extract_dense_tabular_features` | Target encodings are fit on train and applied to train, so each training row's `video_hist_long_view_rate` contains its own label. Low-count videos leak hard. It also creates a train/serve mismatch: sharp in-sample statistics at fit time, noisy out-of-sample ones at evaluation. | Time-shifted encoding — compute statistics from an earlier window, apply to a later one — or out-of-fold. Cheap, and it is probably suppressing LightGBM right now. |
| **Med** | `pipeline/models.py` · `DIN.forward` | The candidate embedding comes from `factor_embeddings` (the offset categorical space) while history comes from a separate `item_embeddings` table. The attention unit compares vectors from two unrelated spaces, so `cand - hist` and `cand * hist` are meaningless. | One item embedding table for both sides, with history indexed in the same offset space as the `video_id` field. |
| **Med** | `pipeline/features.py` · `feature_names` | Four of twelve LightGBM features are pure user-side (`user_hist_count`, `user_hist_long_view_rate`, `user_hist_click_rate`, `user_hist_like_rate`). Ranking happens within a user, so anything constant within a user cannot change the ordering except through an interaction — the starter kit warns about this explicitly. | Replace them with user × item crosses: the user's historical long-view rate *for this author*, *for this duration bucket*, *for this tab*. Those vary within a user and carry real signal. |
| **Med** | `pipeline/train.py` · all trainers | Test metrics are computed on every single training run. Nothing currently selects on them, but there is no guard, and KuaiRand's test labels are locally available. | Make honesty mechanical: compute test predictions only in the terminal submission state, behind an explicit flag, and never print or log test metrics during iteration. A guard you can point at is worth real credit here. |
| **Med** | `pipeline/submit.py` · `check_submission` | Validates header, row count, `row_id` continuity and NaN/Inf, but never checks that `user_id` / `video_id` align with the evaluation split row-for-row. The official checker does. | Load the split and compare pairwise. Then run the starter kit's own `submit.py --check` as the final gate. |
| **Med** | `pipeline/data.py` · `find_data_dir` · `main.py` | Nothing runs out of the box: the dataset is not present, `find_data_dir` does not include the starter-kit path, and there is no download step. `main.py --data_dir` is accepted and then silently ignored, because the generated trial commands never pass it through. | Add a `make data` target, extend the candidate list, and thread `--data_dir` into every generated command. |
| **Med** | `configs/*.yaml` | Neither YAML file is read by any code path. Budgets, timeouts, retry counts and the convergence parameters are all hard-coded in Python, so the config directory is decorative — and the README's launch command passes flags the orchestrator does not accept. | Load both configs at startup and make the README's documented command line actually work. Judges will run it. |
| **Med** | *methodology* | No multi-seed protocol. The baseline's seed σ is 0.0008; a single-seed +0.0022 is inside the range where seed choice alone could explain it. | Accept a change only on the mean of ≥3 seeds, and report the standard deviation next to every headline number. |
| **Low** | `.gitignore` | `*.csv` and `submissions/*.csv` are both ignored, so the required final submission file cannot be committed. | Add an explicit negation, e.g. `!submissions/kuairand_pure_final.csv`. |
| **Low** | *unused data* | `log_random_4_22_to_5_08_pure.csv`, the 1.18M-row randomized-exposure log, is never touched. | Use it as an unbiased second validation set. Cheap, catches overfitting to logged-policy bias, and a genuinely distinctive thing to show a judge. |

### Two dead ends the organizers have already measured — do not spend iterations here

- **Static side features.** Wiring in all 13 CWM domains scored 0.5940 against 0.5950 for the base five — no better, marginally worse.
- **Model capacity.** Embedding dimension 8 / 16 / 32 gave 0.5895 / 0.5902 / 0.5887.

The `user_id × video_id` cross has already absorbed most of the learnable signal, and 1.14M rows will not support more capacity. **The bottleneck is neither features nor size.**

---

## 3. Workstreams — seven parcels, one blocking, six parallel

Split along interface boundaries so that after **W0** lands, nobody waits on anybody. Each parcel owns its files, ships behind a stated contract, and has a definition of done you can check without reading the diff.

### W0 — Ground truth
**Owner**: first person free · **Blocks**: everything · **Size**: ~2 hours

Nothing else is trustworthy until the scorer, the data and the baseline are pinned.

- Download KuaiRand-Pure; add a `make data` target and fix `find_data_dir`.
- Delete `pipeline/evaluate.py`; import the starter kit's scorer verbatim.
- Parity test: random scoring must give primary ≈ 0.4834 on valid, and the fast path must match the official one to 1e-9 on tied inputs.
- Reproduce FM at 0.6016 val and pin it in a regression test.
- Add the split guard so test labels are unreachable outside the submission state.

**Done when** `make baseline` prints 0.6016 ± 0.001 and `pytest` is green on a clean clone.

---

### W1 — Loop & telemetry
**Owner**: M1 (orchestrator) · **Depends on**: W0 · **Scores**: robustness, autonomy, feasibility

Make the loop survive 50 iterations untouched, and make it prove it did.

- Fix the convergence tracker (best-update and stagnation-reset are separate conditions).
- Repair the crash path: real `attempt_repair`, correct field names, correct constructor, `try/except` around the whole iteration.
- Bind hypothesis and command atomically; label fallbacks as fallbacks.
- Real token, wall-clock, and intervention accounting, persisted per iteration and per run.
- Load `configs/*.yaml`; make the README's launch command work verbatim.
- Resume-from-state so a crash costs one iteration, not the run.

**Done when** a run with three deliberately broken trials injected still reaches iteration 50 or convergence, with every failure and recovery in the log.

---

### W2 — Code-writing sandbox
**Owner**: M2 (sandbox) + M3 (prompts) · **Depends on**: W0 · **Scores**: autonomy, innovation

The differentiator, and the largest gap against the brief. Move the agent from picking flags to writing code.

- Define a plugin contract — one file the agent owns, e.g. `pipeline/candidates/cand_<n>.py` exposing `build_features(splits)` and `fit_predict(...)`.
- Agent emits the whole file; the sandbox writes it, runs it, and captures the real `git diff` into the run-log.
- Apply / roll back through git so a bad candidate never corrupts the tree.
- Traceback-driven repair, three attempts, each recorded as an error-recovery event.
- Feed the starter kit's dead-end list into the prompt so the agent does not re-test what the organizers already killed.

**Done when** the run-log contains genuine unified diffs the agent authored, and at least one of them was repaired from a traceback without a human.

---

### W3 — Ranking loss
**Owner**: M4 (pipeline) · **Depends on**: W0 · **Scores**: primary metric — *highest expected value*

The organizers' own number-one untested direction. Objective and metric are currently misaligned: we optimize per-impression probability, we are scored on within-user order.

- Within-user listwise softmax (ListNet-style) over each user's impressions, on top of the existing FM and DeepFM scorers.
- BPR / pairwise logistic on within-user positive–negative pairs as the comparison arm.
- Top-heavy weighting to match nDCG@5's position discount.
- LightGBM switched to `objective: lambdarank`, `metric: ndcg`, `eval_at: [5]`, `lambdarank_truncation_level: 5`, rows grouped by user — this needs the data sorted by user and the group array built correctly.

**Done when** a listwise variant of the plain FM is measured against pointwise FM over three seeds, with the delta and its σ reported either way.

---

### W4 — Leak-free features & sequence
**Owner**: M4 (pipeline), second seat · **Depends on**: W0 · **Scores**: primary metric, innovation

Directions two and three on the organizers' list, plus the leak repairs. Static side features are a known dead end — build crosses and history instead.

- Fix the DIN history leak; rebuild history causally over all splits.
- Share one item embedding table between candidate and history so target attention is meaningful.
- Time-shifted target encoding to remove the in-sample leak and the train/serve mismatch.
- Drop the four inert user-side features; add user × author, user × duration-bucket, user × tab affinity crosses.
- Multi-task auxiliaries done properly: `click`, `like`, `forward` as tasks with tuned weights, plus censored watch-time regression (CWM) as the research-depth arm.

**Done when** the leak fixes are proven by a test (shuffling eval labels must not move eval-side features) and each cross-feature group is ablated over three seeds.

---

### W5 — Validation & submission
**Owner**: M1 or M2, whoever lands first · **Depends on**: W0 · **Size**: small, but non-negotiable

Small surface, total downside if it is wrong on the last night.

- Three-seed protocol with mean ± σ as the acceptance gate for any change.
- Second unbiased validation set from the randomized-exposure log; report both numbers side by side.
- Submission generated from the validation-best node, with its config persisted next to the weights.
- Alignment check against the split, then the starter kit's own `--check` as the final gate.
- `.gitignore` exception so the final CSV can actually be committed.

**Done when** `python3 submit.py --check --split test submission.csv` passes inside the starter-kit directory, on a file produced end-to-end by the agent.

---

### W6 — Report & run-log
**Owner**: M3 (prompts/docs) · **Depends on**: W1 telemetry · **Scores**: presentation, feasibility evidence

Start by removing risk, then build the deliverable back up from measured numbers only.

- Purge the fabricated results table and the two invented recovery episodes **today**.
- Run-log per iteration: hypothesis, real diff, metrics, error and recovery events — the format the brief specifies.
- Intervention summary with an honest count; a small number, stated plainly, beats an unbelievable zero.
- Resource summary: tokens in + out, agent wall-clock, iterations used of 50.
- README reproduction steps that a judge can run on a clean clone; limitations section written honestly.
- Optional 3-minute video — worth it if the code-writing loop is demoable by then.

**Done when** every number in every document traces to a line in `logs/run_summary.json`.

---

## 4. Sequencing — one serial hour, then everything at once

| Block | Work |
| :--- | :--- |
| **Block 1 · serial** | **W0** alone, plus the Devpost purge from **W6**. One person on ground truth while another deletes the fabricated table. Nothing else starts until the scorer is the official scorer, because every measurement taken before that has to be retaken. |
| **Block 2 · parallel** | **W1**, **W2**, **W3**, **W4** run concurrently on disjoint files. W3 and W4 land measured deltas into a shared results sheet; W1 and W2 rebuild the loop underneath them. **W3 is the one to staff first if you are short-handed** — cheapest change, highest expected gain. |
| **Block 3 · convergent** | **W5** gates the winners through the three-seed protocol; the full 50-iteration autonomous run goes off with the repaired loop; **W6** writes the report from whatever that run actually produced. Reserve the final hours for the run, not for the writing. |

### If you only do three things

1. Purge the fabricated report.
2. Repair the error path so a run can finish.
3. Put a listwise loss on the plain FM and measure it over three seeds.

Those cover the disqualification risk, the robustness score, and the most likely source of a real delta — and none of them takes a full day.

---

## 5. Scoring map — where each parcel actually pays

| Criterion | Weight | Carried by | What the judge is looking at |
| :--- | ---: | :--- | :--- |
| Technical Execution | 35% | W0 · W3 · W4 · W1 · W2 | The converged hidden-test delta over 0.5946 — scored continuously, so falling short is not fatal, but the trajectory has to be real. Robustness is judged on recovery from failure, which is entirely W1 and W2. |
| Innovation & Problem Insight | 20% | W3 · W4 · W2 | What the agent chose to target and why — judged on the reasoning, not the implementation. Loss/metric alignment, causal history modelling and unbiased validation are the arguments that land here. |
| Impact & Relevance (autonomy) | 20% | W1 · W2 | How much of the loop the agent drives itself, measured mainly by intervention count. A CLI-flag bandit reads as tuning; an agent that writes and repairs code reads as autonomy. |
| Feasibility & Practicality | 15% | W1 · W6 | Tokens and agent wall-clock, in coarse tiers — but only scored if the hidden-test score clears the baseline. Zeros in the token field score nothing; there is no credit for cheap failure. |
| Presentation & Communication | 10% | W6 | Final event only. Reproducible README, coherent run-log, honest limitations. Numbers that trace back to logs. |

### Calibrating the target

The baseline at 0.5946 has already taken **30.7%** of the attainable range on hidden test, and the ceiling is **0.8645**, not 1.0. A delta of **+0.01 is a real result** here; **+0.03 would be strong**. Treat any single-seed claim below **+0.0024** as noise, since that is 3σ of the baseline's own seed variance.

Compute is deliberately not the constraint — 100 iterations of the reference pipeline is about 28 minutes on one CPU core. The binding constraints are the 50-iteration cap, the 6-hour ceiling, and the convergence rule at ε = 0.002 over N = 3, which normally fires first. **Spend the budget on ideas, not on epochs.**

### One inconsistency in the brief itself

The constraints table lists `NDCG@10 / Recall@50` with `click` as the positive label, while every other section — and the starter kit, which is pinned and authoritative — specifies `GAUC / nDCG@5` on `long_view`. Build against the starter kit, and raise the discrepancy with the organizers rather than hedging across both.
