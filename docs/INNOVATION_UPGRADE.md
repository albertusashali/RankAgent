# RankAgent Innovation Upgrade

This document describes the metric-aligned, causal feature-engineering and
debiasing upgrade added to RankAgent. It covers the motivation, implementation,
commands, safety properties, experiment protocol, and current limitations.

## 1. Executive summary

The upgrade changes RankAgent from a model/configuration search loop into a more
complete recommender-system research workflow. It adds:

1. Power-user-preserving grouped batches.
2. A hybrid GAUC and nDCG@5 training objective.
3. Causal lifetime, 3-day, and 7-day feature families.
4. Historical watch-completion and recency signals without candidate-row leakage.
5. Rank-aware feature selection based on within-user variance.
6. A manifest-driven Feature Engineering Agent and preflight audit.
7. Randomized-exposure IPS and SNIPS training ablations.
8. Validation-only cold/warm, head/tail, and duration diagnostics.
9. Configuration-specific checkpoints so experiments cannot overwrite one another.

The hidden-test contract is unchanged: training and selection use train and public
validation labels only. Hidden-test rows are read only when producing the final
submission, and their labels remain replaced by `-1`.

## 2. Why the previous implementation needed these changes

### 2.1 Power-user truncation

The old grouped batcher sampled at most 64 rows from each user. This reduced
memory, but it was not aligned with GAUC. The official GAUC weights eligible users
by their number of positives; a heavily active user therefore contributes more to
the metric. Silently forcing every user toward the same group size discarded much
of the signal from power users.

The new default is `--max_group_rows 0`, meaning no truncation. A positive value
is still available as an explicit compute ablation. Sampling is repeated without
replacement per epoch when a cap is deliberately selected.

### 2.2 Objective mismatch

The primary score is:

```text
0.5 * GAUC + 0.5 * nDCG@5
```

BPR emphasizes pairwise ordering and therefore resembles AUC optimization.
Listwise softmax emphasizes within-user positive probability, but it does not
explicitly prioritize mistakes at rank 5. The new `hybrid` objective combines:

```text
L_hybrid = alpha * L_BPR + (1 - alpha) * L_delta_nDCG
```

`L_delta_nDCG` samples a negative for each positive within the same user and
weights the pair by the absolute nDCG change produced by swapping their current
predicted ranks. Items outside the top-5 receive zero discount unless their swap
would affect the top-5.

`--auc_weight` controls `alpha`. Start with `0.5`, then test `0.25` and `0.75`.

### 2.3 Candidate-row outcome leakage

`play_time_ms`, click, like, follow, comment, and forward are observed after an
item is exposed. They must not be candidate inputs when predicting `long_view`.

The safe implementation follows this order for training date `d`:

```text
feature rows from dates < d
then observe outcomes from date d
```

Consequently, historical watch completion is available, while the current row's
watch time cannot affect its own feature vector. Validation and test use a frozen
state built from training only.

## 3. Feature profiles

The LightGBM ranker exposes three profiles.

### `core`

Contains item/author support and smoothed rates, user history used as a conditioning
signal, duration transforms, and historical duration-bucket completion priors.

### `affinity`

Adds candidate-varying personalized crosses:

- user-video affinity;
- user-author affinity;
- user-duration-bucket affinity;
- user-tab affinity.

Unseen crosses back off to the user's historical rate instead of the global rate.

### `full`

Adds:

- historical video, author, and user-author watch-completion log-ratios;
- days since user, video, author, and user-author activity;
- exact trailing 3-day and 7-day video rates;
- exact trailing 3-day and 7-day user-author rates;
- short-term versus lifetime momentum features.

All rates use additive smoothing. The recent rolling stores are advanced only
after a full day's rows have been featurized, preventing same-day target leakage.
The smoothing prior is also expanding-window. An early dynamic audit discovered
that using the full-training label mean as a prior leaked future dates even though
the entity counters themselves were causal; that path has been removed.

## 4. Rank-aware feature selection

`--select_features` calculates the mean within-user variance of each training
column. A feature that is constant inside every user's candidate set cannot alter
that user's order when used additively, so columns below the variance threshold
are removed.

Selection uses no labels and no validation values. The resulting training mask is
applied unchanged to validation and test.

This is intentionally different from ordinary global variance selection. A field
can vary greatly across users while remaining constant within each user, making it
irrelevant to direct within-user ranking.

Static user features are not universally forbidden. They can help through explicit
interactions with candidate fields in FM/deep models. They should nevertheless be
tested with an ablation rather than assumed useful.

## 5. Feature Engineering Agent

`pipeline/feature_agent.py` defines a manifest for every dense feature. Each
manifest records:

- feature name and family;
- entity keys;
- source columns;
- information availability;
- historical window;
- smoothing and fallback;
- expected ranking mechanism.

The static audit checks:

- every produced feature has exactly one manifest;
- no stale manifest remains after code changes;
- no duplicate feature name exists;
- outcome columns are marked historical-only when referenced.

The optional dynamic audit mutates the current row's outcomes and confirms that
its feature vector is unchanged. It also rejects NaN and infinity.

The orchestrator runs the static audit before baseline/trial execution and halts
if it fails. This prevents an expensive autonomous run from continuing with an
unreviewed or inconsistent feature definition.

Commands:

```powershell
python -m pipeline.feature_agent
python -m pipeline.feature_agent --dynamic
```

Reports are written to `logs/feature_audit.json` by default.

### AI-controlled feature recipes

Feature search is now recipe-driven. The AI does not patch `features.py` during a
run. Instead it returns a bounded `feature_recipe` in the same JSON response as
its hypothesis. The orchestrator validates the recipe, rejects duplicates, assigns
a content hash, saves it under `experiments/recipes`, and replaces the proposed
command with the audited LightGBM recipe command.

A recipe controls:

- `base_profile`: core, affinity, or full;
- explicit feature inclusion and exclusion;
- item and interaction-cross smoothing;
- expanding global-prior strength;
- historical completion-ratio clipping;
- recency clipping horizon;
- within-user feature selection and its threshold.

Recipe identity excludes its display name. Two differently named recipes with
identical behavior therefore receive the same hash and cannot waste two iterations.
Every checkpoint metadata file embeds the complete recipe rather than relying on
the original JSON path. Final submission reconstruction is consequently stable
even if experiment files are moved later.

After a recipe ranker trains, it writes normalized LightGBM gain importance to:

```text
checkpoints/<checkpoint>.importance.json
```

The next AI prompt includes the five most important fields and the prior recipe
ID alongside validation history. This gives the model evidence for an attributable
mutation—for example increasing cross smoothing when sparse affinities are noisy,
or excluding completion fields when they contribute negligible gain.

The end-to-end recipe integration check produced a deliberately weak standalone
recipe ranker (`0.5894` primary), correctly rejected it, then found that a 15%
recipe contribution raised the listwise FM from about `0.6032` to `0.6036`.
This illustrates the intended safety boundary: feature search may discover useful
decorrelated information even when its standalone model is worse, but it cannot
replace the stronger model unless validation improves.

If neither `OPENAI_API_KEY` nor `ANTHROPIC_API_KEY` is configured in `.env`, the
same `python main.py` command uses five deterministic, deduplicated recipes. Thus
the pipeline remains fully executable offline; API-backed operation adds adaptive
reasoning rather than being required for correctness.

At startup, `main.py` prints either `[AI] openai connected`, `[AI] anthropic
connected`, or `[AI] no API key detected`. A run summary showing zero LLM calls
and zero tokens means deterministic recipe search was used, not AI communication.

## 6. Randomized-exposure debiasing

KuaiRand provides a randomized-exposure log. `pipeline/propensity.py` uses it to
fit a density-ratio classifier that distinguishes randomized from standard
exposures using pre-exposure covariates only:

- video identity;
- author identity;
- tab;
- log-duration bucket.

For a standard training row, the raw ratio is:

```text
w(x) = P(random | x) / P(standard | x)
```

Two modes are available:

- `ips`: clip the density ratio into `[1/clip, clip]`;
- `snips`: apply the same clipping and normalize weights to mean one.

The weights are applied to the main pointwise, BPR, listwise, or hybrid loss.
Auxiliary MMoE heads remain unweighted because this change is intended as an
isolated ablation on the scored `long_view` objective.

Example:

```powershell
python -m pipeline.train --model fm_torch --loss hybrid `
  --propensity snips --propensity_clip 10
```

Important limitation: density-ratio weighting does not prove all confounding has
been removed. Random and standard logs may differ for reasons not captured by the
selected covariates. IPS/SNIPS must therefore compete against an unweighted control
on the official validation metric. SNIPS is the recommended first experiment due
to its lower scale instability.

## 7. Validation segment diagnostics

The diagnostic command evaluates a selected checkpoint on public validation only:

```powershell
python -m pipeline.diagnostics --checkpoint fm_torch_hybrid_aw50
```

It reports:

- all rows;
- cold and warm users;
- head and tail videos;
- short and long videos.

These reports explain where an aggregate score changed. They must not replace the
official primary metric or create a hidden-test selection path.

## 8. Checkpoint reproducibility

Neural checkpoint names now include behavior-changing options. Examples:

```text
fm_torch_hybrid_aw50
fm_torch_hybrid_aw50_snips10
fm_torch_hybrid_aw50_cap64
mmoe_hybrid_aw50
```

LightGBM checkpoints include their feature profile:

```text
lgb_core_selected
lgb_affinity
lgb_full_selected
```

Metadata stores loss weights, nDCG cutoff, group cap, propensity mode and clipping,
feature profile, feature-selection state, and the exact model artifact. The
submission loader reconstructs the matching feature matrix before prediction.

This fixes a serious prior failure mode where all LightGBM experiments overwrote
`checkpoints/lgb.txt`, allowing the final export to use a model different from the
validation-best trial.

## 9. Recommended experiment sequence

Change one major factor at a time:

```powershell
# Control: metric-aligned full user groups
python -m pipeline.train --model fm_torch --loss bpr --max_group_rows 0

# Hybrid objective
python -m pipeline.train --model fm_torch --loss hybrid --auc_weight 0.5

# Objective balance
python -m pipeline.train --model fm_torch --loss hybrid --auc_weight 0.25
python -m pipeline.train --model fm_torch --loss hybrid --auc_weight 0.75

# Feature-family ablations
python -m pipeline.train --model lgb --feature_profile core --select_features
python -m pipeline.train --model lgb --feature_profile affinity --select_features
python -m pipeline.train --model lgb --feature_profile full --select_features

# Debiasing ablations
python -m pipeline.train --model fm_torch --loss hybrid --propensity ips --propensity_clip 10
python -m pipeline.train --model fm_torch --loss hybrid --propensity snips --propensity_clip 10
python -m pipeline.train --model fm_torch --loss hybrid --propensity snips --propensity_clip 5
```

For gains smaller than roughly three baseline standard deviations, repeat at least
one additional seed before treating the change as real.

### Single-command operation

The normal workflow requires only:

```powershell
python main.py
```

That command now performs the real-data dynamic feature audit, reproduces the
official baseline, runs ordered autonomous experiments, retains only validation
improvements, searches safe two-model rank blends, writes validation diagnostics,
exports test predictions in isolated processes, validates alignment, and writes
the final run summary and submission. The module-level commands above remain
developer/debugging interfaces; they are not required for a normal agent run.

The baseline is always a candidate, so a rejected feature or debiasing experiment
cannot lower the validation-selected final output. This does not guarantee a hidden
test improvement, which no validation-only system can promise, but it guarantees
that the agent does not knowingly replace a stronger validation model with a weaker
one.

### Why the first feature run appeared worse

The observed results had three different meanings:

- `0.6032`: the actual selected listwise FM, above the `0.6016` baseline;
- `0.5903`: the full causal feature-only LightGBM, showing that aggregates do not
  replace memorized user/item interactions;
- `0.5811`: a five-tree integration smoke test, not a converged experiment.

The corrected workflow treats the feature ranker as a decorrelated ensemble
component. On the recorded validation predictions, blending 81% listwise FM with
19% full-feature LightGBM increased primary from approximately `0.6032` to `0.6040`.
The blend weight is retuned on validation during every complete run.

### Windows Unicode handling

All child processes declare UTF-8, replace malformed output bytes instead of
raising `UnicodeDecodeError`, and clear the hidden-test unseal flag. The remaining
locale-dependent path was the logger's `git diff` capture; it now also specifies
`encoding="utf-8", errors="replace"`. `main.py` reconfigures its IDE/terminal
streams similarly. Consequently, an unusual filename or subprocess byte cannot
terminate the autonomous loop.

## 10. Files changed

| File | Responsibility |
|---|---|
| `pipeline/models.py` | Hybrid BPR/delta-nDCG loss and weighted losses |
| `pipeline/train.py` | Full-user batches, new CLI, propensity threading, unique checkpoints |
| `pipeline/features.py` | Feature profiles, affinities, recency, rolling windows, selector |
| `pipeline/feature_agent.py` | Feature manifests and leakage governance |
| `pipeline/propensity.py` | Random-exposure density-ratio estimation |
| `pipeline/diagnostics.py` | Validation-only segment reports |
| `pipeline/submit.py` | Exact feature/checkpoint reconstruction |
| `sandbox/logger.py` | Configuration-aware checkpoint inference |
| `orchestrator/state_machine.py` | Audit preflight and new research strategies |
| `prompts/templates.py` | New action space and leakage rules for the LLM |
| `tests/test_harness.py` | Leakage, manifests, group coverage, and hybrid-loss tests |

## 11. What was deliberately not claimed

The implementation includes historical duration-debiased watch-completion signals,
but it is not a full reproduction of the KDD 2024 Counterfactual Watch Model. A
faithful CWM requires a separately validated censored/counterfactual likelihood and
should be introduced as its own model family, not mislabeled as a log-ratio feature.

Likewise, IPS/SNIPS are experiments rather than guaranteed improvements. The final
model must be chosen by public validation and reproducibility checks.

## 12. Verification

Run:

```powershell
python -m compileall -q pipeline orchestrator prompts sandbox tests
python -m pipeline.feature_agent
python -m pytest tests/test_harness.py -q
```

The project declares `pytest` in `requirements.txt`. If it is absent from the
active environment, install project dependencies before running the complete suite.

## 13. Research references

- Wang et al., *The LambdaLoss Framework for Ranking Metric Optimization*, CIKM 2018:
  https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/
- Joachims et al., *Unbiased Learning-to-Rank with Biased Feedback*, IJCAI 2018:
  https://www.ijcai.org/proceedings/2018/738
- Zhao et al., *Counteracting Duration Bias in Video Recommendation via
  Counterfactual Watch Time*, KDD 2024: https://doi.org/10.1145/3637528.3671817
# Latest ranking upgrade: fused causal-feature DeepFM

Feature recipes now feed `deepfm_dense`, a hybrid ranker that combines user/video
categorical embeddings with strictly historical numerical features. The dense
branch receives affinity, popularity, duration preference, historical completion,
recency, and momentum signals; training-only mean and standard deviation values
are saved in checkpoint metadata and reused unchanged for validation/test scoring.

AI-generated Feature Engineering proposals are compiled into this fused model
instead of being restricted to LightGBM. Each recipe receives a unique checkpoint
name, and submission reconstruction checks the exact feature-name schema before
loading weights. This makes a feature mutation both measurable in the strong
model family and safe from train/serve skew. Model selection remains validation
only and uses the challenge primary score (mean of GAUC and nDCG@5).

The first full-data validation trial selected the pointwise fused model at epoch
3: GAUC `0.6724`, nDCG@5 `0.5380`, primary `0.6052`. This is higher than the
previous best validation primary (`0.6047`) and the official baseline (`0.6016`).
The listwise fused ablation reached only `0.6029`, so the agent's default recipe
runner uses the empirically validated pointwise objective while retaining other
losses as explicit ablations.

## GAUC noise-reduction ablation

Three training-only denoising approaches were validated without changing the
evaluation labels or inference features:

- `label_smoothing=0.03`: GAUC `0.6720`; rejected.
- `label_smoothing=0.03` plus a 14-day recency half-life: GAUC `0.6721`; rejected.
- a second random seed followed by validation-selected rank averaging: GAUC
  `0.67286`, nDCG@5 `0.53845`, primary `0.60565`; accepted.

The result shows that older interactions still contain useful preference signal,
whereas averaging optimization noise is beneficial. The autonomous strategy now
runs seed-safe checkpoints (`_s<seed>`) and allows the terminal validation-only
blend search to select their weight. Unsuccessful smoothing and temporal-decay
settings remain available as explicit CLI ablations but are not default choices.
