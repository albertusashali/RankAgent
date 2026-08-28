# Run-Log Specification & Protocol

**Project**: RankAgent — Autonomous Machine Learning Research Agent for Recommender Systems  
**Purpose**: Standardized reporting of autonomous iterations, hypotheses, code diffs, metrics, and error recovery events for hackathon judging (Autonomy, Robustness, Feasibility).

---

## 1. Run-Log Requirements

As specified in the hackathon guidelines, every autonomous iteration must record:
1. **Hypothesis**: What the agent intended to try and the scientific/engineering rationale.
2. **Code Diff**: Unified diff of changes applied to the baseline / previous best code.
3. **Evaluation Metrics**: Exact validation $\text{GAUC}$, $\text{nDCG@5}$, and $\text{Primary Score} = \frac{\text{GAUC} + \text{nDCG@5}}{2}$.
4. **Error & Recovery Events**: Any compiler/runtime exceptions, timeout handling, or numerical divergence encountered and the self-healing resolution.
5. **Autonomy & Resource Telemetry**: Manual intervention count, LLM token consumption (prompt + completion), iteration wall-clock duration, and peak memory.

---

## 2. JSON Schema Definition

All iteration logs are serialized into `logs/run_summary.json` following the schema below:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "RankAgentRunLog",
  "type": "object",
  "required": [
    "run_id",
    "benchmark",
    "official_baseline_validation",
    "total_iterations",
    "total_wall_clock_seconds",
    "total_tokens",
    "manual_interventions",
    "converged",
    "best_iteration_id",
    "best_validation_metrics",
    "iterations"
  ],
  "properties": {
    "run_id": { "type": "string" },
    "benchmark": { "type": "string", "enum": ["KuaiRand-Pure", "KuaiRand-1k", "KuaiRand-27k"] },
    "official_baseline_validation": {
      "type": "object",
      "properties": {
        "gauc": { "type": "number" },
        "ndcg_at_5": { "type": "number" },
        "primary_score": { "type": "number" }
      }
    },
    "total_iterations": { "type": "integer", "maximum": 50 },
    "total_wall_clock_seconds": { "type": "number" },
    "total_tokens": {
      "type": "object",
      "properties": {
        "prompt_tokens": { "type": "integer" },
        "completion_tokens": { "type": "integer" },
        "total": { "type": "integer" }
      }
    },
    "manual_interventions": { "type": "integer" },
    "converged": { "type": "boolean" },
    "convergence_reason": { "type": "string" },
    "best_iteration_id": { "type": "integer" },
    "best_validation_metrics": {
      "type": "object",
      "properties": {
        "gauc": { "type": "number" },
        "ndcg_at_5": { "type": "number" },
        "primary_score": { "type": "number" },
        "delta_over_baseline": { "type": "number" }
      }
    },
    "iterations": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "iteration_id",
          "parent_iteration_id",
          "stage",
          "hypothesis",
          "code_diff",
          "metrics",
          "status",
          "wall_clock_seconds",
          "tokens_used"
        ],
        "properties": {
          "iteration_id": { "type": "integer" },
          "parent_iteration_id": { "type": ["integer", "null"] },
          "stage": {
            "type": "string",
            "enum": ["Feature Engineering", "Architecture", "Multi-Task", "Loss Function", "Ensembling", "Hyperparameter Tuning"]
          },
          "hypothesis": { "type": "string" },
          "code_diff": { "type": "string" },
          "metrics": {
            "type": "object",
            "properties": {
              "gauc": { "type": "number" },
              "ndcg_at_5": { "type": "number" },
              "primary_score": { "type": "number" },
              "delta_from_parent": { "type": "number" },
              "delta_from_baseline": { "type": "number" }
            }
          },
          "status": { "type": "string", "enum": ["SUCCESS", "ACCEPTED", "REJECTED", "ERROR_RECOVERED", "FAILED"] },
          "error_event": {
            "type": ["object", "null"],
            "properties": {
              "error_type": { "type": "string" },
              "traceback": { "type": "string" },
              "recovery_action": { "type": "string" },
              "recovery_success": { "type": "boolean" }
            }
          },
          "wall_clock_seconds": { "type": "number" },
          "tokens_used": { "type": "integer" }
        }
      }
    }
  }
}
```

---

## 3. Human-Readable Log Format (`logs/run_log.md`)

Each run also generates a human-readable iteration journal formatted as follows:

```markdown
# RankAgent Experiment Run Log
- **Run ID**: `run-kuairand-pure-20260828-01`
- **Target Benchmark**: KuaiRand-Pure
- **Official Baseline Val Primary**: 0.6016 (GAUC: 0.6674, nDCG@5: 0.5357)
- **Manual Interventions**: 0 (100% Autonomous)

---

### Iteration 0: Baseline Reproduction
* **Status**: ACCEPTED (Reproduced)
* **Hypothesis**: Stand up NumPy Factorization Machine baseline on KuaiRand-Pure train/val date splits.
* **Validation Metrics**: GAUC: 0.6674 | nDCG@5: 0.5357 | Primary: 0.6016 (Δ Baseline: +0.0000)
* **Resource Usage**: 42s CPU | 1,420 tokens

---

### Iteration 1: Historical Engagement Target Aggregations
* **Status**: ACCEPTED
* **Hypothesis**: Computing cumulative out-of-fold user/item long_view interaction frequencies and smooth target rates will provide informative dense priors.
* **Validation Metrics**: GAUC: 0.6812 | nDCG@5: 0.5510 | Primary: 0.6161 (Δ Baseline: +0.0145)
* **Code Diff**:
```diff
+ def compute_historical_stats(df_train, df_val):
+     user_stats = df_train.groupby('user_id')['long_view'].agg(['count', 'mean']).reset_index()
+     item_stats = df_train.groupby('video_id')['long_view'].agg(['count', 'mean']).reset_index()
+     return merge_stats(df_train, df_val, user_stats, item_stats)
```
* **Resource Usage**: 68s | 3,850 tokens | 0 errors

---

### Iteration 2: DeepFM Architecture with Categorical Embeddings
* **Status**: ERROR_RECOVERED -> ACCEPTED
* **Hypothesis**: Replace linear FM with DeepFM (FM + 3-layer MLP) to capture high-order feature interactions.
* **Error Event**:
  * `RuntimeError: CUDA out of memory. Tried to allocate 2.40 GiB`
  * **Recovery**: Debugger automatically reduced batch size from 4096 to 2048 with gradient accumulation (2 steps).
* **Validation Metrics**: GAUC: 0.7045 | nDCG@5: 0.5780 | Primary: 0.6412 (Δ Baseline: +0.0396)
* **Resource Usage**: 145s | 7,120 tokens | Self-healed in 1 try

---

### Iteration 3: MMoE Multi-Task Learning on 12 KuaiRand Feedback Signals
* **Status**: ACCEPTED (Best Checkpoint)
* **Hypothesis**: Auxiliary prediction heads for `click`, `like`, `follow`, and `comment` using MMoE gating network to resolve sparsity of `long_view`.
* **Validation Metrics**: GAUC: 0.7230 | nDCG@5: 0.5985 | Primary: 0.6608 (Δ Baseline: +0.0592)
* **Resource Usage**: 210s | 9,450 tokens

---

## 4. Run Telemetry & Convergence Summary

| Benchmark | Final Val GAUC | Final Val nDCG@5 | Final Val Primary | Δ Baseline | Iterations | Total Wall-Clock | Total LLM Tokens | Interventions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **KuaiRand-Pure** | **0.7230** | **0.5985** | **0.6608** | **+0.0592** | 18 / 50 | 48 min | 142,500 | **0** |

