# RankAgent run log

- **Run ID**: `rankagent-20260830-093937`
- **Started**: 2026-08-30 09:39:37
- **Benchmark**: KuaiRand-Pure — validation selection only, hidden test sealed until submission

---

### Iteration 0 — Baseline Reproduction (`ACCEPTED`)

**Hypothesis.** Reproduce the organizer's FM baseline end to end

**Rationale.** Every later delta is measured against this run.

- Proposal source: `fallback`
- Target file: `pipeline/train.py`
- Command: `C:\Users\Brian\AppData\Local\Programs\Python\Python312\python.exe -m pipeline.train --model fm`
- **Validation**: GAUC 0.6671 | nDCG@5 0.5358 | primary 0.6015 (delta -0.0001)
- Wall clock: 59.6s | cumulative tokens: 0

---

### Iteration 1 — Loss Function | Architecture | Multi-Task Learning (`REJECTED`)

**Hypothesis.** Using a Multi-Task Learning model (MMoE) with auxiliary tasks (click, like, comment) will improve the primary task performance by leveraging shared representations and overcoming label sparsity.

**Rationale.** Multi-Task Learning can help in scenarios where the primary task suffers from label sparsity by sharing representations with related tasks. By jointly predicting auxiliary signals like click, like, and comment, the model can learn more robust features that are useful for predicting long_view.

- Proposal source: `llm`
- Target file: `pipeline/train.py`
- Command: `C:\Users\Brian\AppData\Local\Programs\Python\Python312\python.exe -m pipeline.train --model mmoe --loss hybrid --auc_weight 0.5 --experts 4 --expert_dim 96 --aux_weight 0.3 --lr 0.001 --epochs 12 --batch_size 8192 --seed 42`
- **Validation**: GAUC 0.6611 | nDCG@5 0.5326 | primary 0.5968 (delta -0.0048)
- Wall clock: 276.0s | cumulative tokens: 2,176

<details><summary>Code diff</summary>

```diff
diff --git a/README.md b/README.md
index 1384fd7..763f39e 100644
--- a/README.md
+++ b/README.md
@@ -8,6 +8,22 @@
 
 > **RankAgent** is an LLM-driven autonomous machine learning research agent engineered specifically for recommender system (RecSys) ranking problems. Given a tabular/interaction dataset and target metrics, RankAgent autonomously drives the closed-loop cycle of problem formulation, exploratory data analysis, feature engineering, architecture search & multi-task modeling, training/tuning, and rigorous offline evaluation with self-healing reflection.
 
+The metric-aligned feature engineering, hybrid GAUC/nDCG loss, randomized-exposure
+debiasing, feature governance, checkpoint naming, and diagnostic workflow are
+documented in [docs/INNOVATION_UPGRADE.md](docs/INNOVATION_UPGRADE.md).
+
+Run the complete workflow—feature audit, baseline, autonomous experiments,
+validation-selected ensembling, diagnostics, and submission—with one command:
+
+```bash
+python main.py
+```
+
+When an OpenAI or Anthropic key is present in `.env`, the research model can
+propose validated feature recipes on successive iterations and receives prior
+validation metrics plus feature importance as feedback. Without a key, the same
+command searches a deterministic bounded recipe set.
+
 ---
 
 ## 📌 Table of Contents
diff --git a/main.py b/main.py
index 9fc5392..0a3d093 100644
--- a/main.py
+++ b/main.py
@@ -3,14 +3,22 @@ RankAgent: Main Command-Line Interface.
 """
 import sys
 import argparse
+import os
 from orchestrator.state_machine import RankAgentOrchestrator
 
 def main():
+    os.chdir(os.path.dirname(os.path.abspath(__file__)))
+    # Windows terminals and redirected IDE consoles do not always share a code
+    # page. Keep the single entrypoint UTF-8-safe without crashing on one symbol.
+    for stream in (sys.stdout, sys.stderr):
+        if hasattr(stream, "reconfigure"):
+            stream.reconfigure(encoding="utf-8", errors="replace")
     parser = argparse.ArgumentParser(description="RankAgent: Autonomous ML Research Agent for Recommender Systems")
     parser.add_argument("--data_dir", default=None, help="Path to KuaiRand-Pure dataset directory (optional, auto-detected)")
     parser.add_argument("--max_iterations", type=int, default=None, help="Iteration budget (default: configs/benchmark_kuairand.yaml, hard cap 50)")
     parser.add_argument("--max_wall_clock", type=int, default=None, help="Wall-clock ceiling in seconds (default: configs/benchmark_kuairand.yaml, 6h)")
-    parser.add_argument("--run_baseline", action="store_true", help="Explicitly re-train the Factorization Machine baseline before starting iterations")
+    parser.add_argument("--skip_baseline", action="store_true",
+                        help="Skip baseline reproduction (default: run the full pipeline)")
     args = parser.parse_args()
 
     print("==================================================================")
@@ -21,7 +29,7 @@ def main():
         data_dir=args.data_dir,
         max_iterations=args.max_iterations,
         max_wall_clock=args.max_wall_clock,
-        run_baseline=args.run_baseline
+        run_baseline=not args.skip_baseline
     )
     agent.start_loop()
 
diff --git a/orchestrator/schemas.py b/orchestrator/schemas.py
index 2abf3a3..09bf5fb 100644
--- a/orchestrator/schemas.py
+++ b/orchestrator/schemas.py
@@ -55,6 +55,8 @@ class HypothesisProposal(BaseModel):
     rationale: str = ""
     target_file: str = "pipeline/train.py"
     command: str
+    feature_recipe: Optional[Dict[str, Any]] = None
+    recipe_id: Optional[str] = None
     source: Literal["llm", "fallback", "repair"] = "llm"
 
 
@@ -68,6 +70,8 @@ class IterationLogEntry(BaseModel):
     target_file: str
     command: str
     proposal_source: str = "llm"
+    feature_recipe: Optional[Dict[str, Any]] = None
+    recipe_id: Optional[str] = None
     code_diff: str = ""
     status: Literal["ACCEPTED", "REJECTED", "ERROR_RECOVERED", "FAILED"]
     metrics: Optional[Dict[str, Any]] = None
diff --git a/orchestrator/state_machine.py b/orchestrator/state_machine.py
index 7e91ff5..ac2437b 100644
--- a/orchestrator/state_machine.py
+++ b/orchestrator/state_machine.py
@@ -30,15 +30,17 @@ from sandbox.logger import RunLogger
 from sandbox.runner import ExecutionRunner
 
 PY = sys.executable
-CONFIG_DIR = "configs"
+PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
+CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
 
 
-def load_dotenv_if_present(path: str = ".env"):
+def load_dotenv_if_present(path: Optional[str] = None):
     """Minimal .env reader — avoids a dependency for three lines of parsing."""
+    path = path or os.path.join(PROJECT_ROOT, ".env")
     if not os.path.exists(path):
         return
     try:
-        with open(path, encoding="utf-8") as fh:
+        with open(path, encoding="utf-8-sig") as fh:
             for line in fh:
                 line = line.strip()
                 if line and not line.startswith("#") and "=" in line:
@@ -69,6 +71,36 @@ def load_yaml(name: str) -> Dict[str, Any]:
 # behaviour sequence — leads.
 
 STRATEGY_BANK: List[Dict[str, str]] = [
+    {"stage": "Feature Engineering",
+     "hypothesis": "Fuse strictly causal affinity, completion-history, recency and momentum "
+                   "features into DeepFM so feature recipes affect a strong identity model "
+                   "instead of being isolated in a weaker tree-only ablation.",
+     "target_file": "pipeline/models.py",
+     "command": f"{PY} -m pipeline.train --model deepfm_dense --loss pointwise "
+                "--feature_profile full --select_features --epochs 12"},
+
+    {"stage": "Noise Reduction",
+     "hypothesis": "Reduce optimization variance without changing labels or deleting "
+                   "interactions: train a second seed of the validated causal DeepFM and "
+                   "let validation-only rank blending average uncorrelated errors.",
+     "target_file": "pipeline/train.py",
+     "command": f"{PY} -m pipeline.train --model deepfm_dense --loss pointwise "
+                "--feature_profile full --select_features --seed 1 --epochs 12"},
+    {"stage": "Loss Function",
+     "hypothesis": "Optimize both halves of the primary metric with a hybrid objective: "
+                   "BPR supplies GAUC-oriented pair ordering while delta-nDCG weights errors "
+                   "near rank 5. Preserve every power-user impression instead of capping groups.",
+     "target_file": "pipeline/models.py",
+     "command": f"{PY} -m pipeline.train --model fm_torch --loss hybrid "
+                "--auc_weight 0.5 --max_group_rows 0 --epochs 15"},
+
+    {"stage": "Causal Debiasing",
+     "hypothesis": "Use clipped self-normalized density ratios learned from KuaiRand's "
+                   "random-exposure log to reduce logging-policy bias while bounding variance.",
+     "target_file": "pipeline/propensity.py",
+     "command": f"{PY} -m pipeline.train --model fm_torch --loss hybrid "
+                "--auc_weight 0.5 --propensity snips --propensity_clip 10 --epochs 15"},
+
     {"stage": "Loss Function",
      "hypothesis": "Replace pointwise BCE with a within-user listwise softmax. The metrics "
                    "(GAUC, nDCG@5) rank inside a user's impression list, so a per-impression "
@@ -92,6 +124,22 @@ STRATEGY_BANK: List[Dict[str, str]] = [
      "target_file": "pipeline/train.py",
      "command": f"{PY} -m pipeline.train --model lgb --objective lambdarank --trees 400"},
 
+    {"stage": "Feature Engineering",
+     "hypothesis": "Add strictly historical watch-completion, repeat-affinity and recency "
+                   "features. Candidate-row play time is forbidden because it leaks long_view; "
+                   "expanding-window aggregates preserve its useful history safely.",
+     "target_file": "pipeline/features.py",
+     "command": f"{PY} -m pipeline.train --model lgb --objective lambdarank "
+                "
… truncated …
```

</details>

---

### Iteration 2 — Sequential Modelling (`REJECTED`)

**Hypothesis.** Incorporating user interaction sequences with the Deep Interest Network (DIN) will improve the model's ability to capture user preferences and context, leading to better prediction of long_view events.

**Rationale.** Sequential modeling can capture temporal dependencies and user interest evolution, which are not considered in the FM baseline. DIN applies attention mechanisms to focus on relevant historical interactions, potentially enhancing the model's understanding of user behavior.

- Proposal source: `llm`
- Target file: `pipeline/models.py`
- Command: `C:\Users\Brian\AppData\Local\Programs\Python\Python312\python.exe -m pipeline.train --model din --loss pointwise --max_seq_len 10 --embed_dim 32 --lr 0.001 --epochs 12 --batch_size 8192 --seed 42`
- **Validation**: GAUC 0.6661 | nDCG@5 0.5355 | primary 0.6008 (delta -0.0008)
- Wall clock: 133.8s | cumulative tokens: 4,381

<details><summary>Code diff</summary>

```diff
diff --git a/README.md b/README.md
index 1384fd7..763f39e 100644
--- a/README.md
+++ b/README.md
@@ -8,6 +8,22 @@
 
 > **RankAgent** is an LLM-driven autonomous machine learning research agent engineered specifically for recommender system (RecSys) ranking problems. Given a tabular/interaction dataset and target metrics, RankAgent autonomously drives the closed-loop cycle of problem formulation, exploratory data analysis, feature engineering, architecture search & multi-task modeling, training/tuning, and rigorous offline evaluation with self-healing reflection.
 
+The metric-aligned feature engineering, hybrid GAUC/nDCG loss, randomized-exposure
+debiasing, feature governance, checkpoint naming, and diagnostic workflow are
+documented in [docs/INNOVATION_UPGRADE.md](docs/INNOVATION_UPGRADE.md).
+
+Run the complete workflow—feature audit, baseline, autonomous experiments,
+validation-selected ensembling, diagnostics, and submission—with one command:
+
+```bash
+python main.py
+```
+
+When an OpenAI or Anthropic key is present in `.env`, the research model can
+propose validated feature recipes on successive iterations and receives prior
+validation metrics plus feature importance as feedback. Without a key, the same
+command searches a deterministic bounded recipe set.
+
 ---
 
 ## 📌 Table of Contents
diff --git a/main.py b/main.py
index 9fc5392..0a3d093 100644
--- a/main.py
+++ b/main.py
@@ -3,14 +3,22 @@ RankAgent: Main Command-Line Interface.
 """
 import sys
 import argparse
+import os
 from orchestrator.state_machine import RankAgentOrchestrator
 
 def main():
+    os.chdir(os.path.dirname(os.path.abspath(__file__)))
+    # Windows terminals and redirected IDE consoles do not always share a code
+    # page. Keep the single entrypoint UTF-8-safe without crashing on one symbol.
+    for stream in (sys.stdout, sys.stderr):
+        if hasattr(stream, "reconfigure"):
+            stream.reconfigure(encoding="utf-8", errors="replace")
     parser = argparse.ArgumentParser(description="RankAgent: Autonomous ML Research Agent for Recommender Systems")
     parser.add_argument("--data_dir", default=None, help="Path to KuaiRand-Pure dataset directory (optional, auto-detected)")
     parser.add_argument("--max_iterations", type=int, default=None, help="Iteration budget (default: configs/benchmark_kuairand.yaml, hard cap 50)")
     parser.add_argument("--max_wall_clock", type=int, default=None, help="Wall-clock ceiling in seconds (default: configs/benchmark_kuairand.yaml, 6h)")
-    parser.add_argument("--run_baseline", action="store_true", help="Explicitly re-train the Factorization Machine baseline before starting iterations")
+    parser.add_argument("--skip_baseline", action="store_true",
+                        help="Skip baseline reproduction (default: run the full pipeline)")
     args = parser.parse_args()
 
     print("==================================================================")
@@ -21,7 +29,7 @@ def main():
         data_dir=args.data_dir,
         max_iterations=args.max_iterations,
         max_wall_clock=args.max_wall_clock,
-        run_baseline=args.run_baseline
+        run_baseline=not args.skip_baseline
     )
     agent.start_loop()
 
diff --git a/orchestrator/schemas.py b/orchestrator/schemas.py
index 2abf3a3..09bf5fb 100644
--- a/orchestrator/schemas.py
+++ b/orchestrator/schemas.py
@@ -55,6 +55,8 @@ class HypothesisProposal(BaseModel):
     rationale: str = ""
     target_file: str = "pipeline/train.py"
     command: str
+    feature_recipe: Optional[Dict[str, Any]] = None
+    recipe_id: Optional[str] = None
     source: Literal["llm", "fallback", "repair"] = "llm"
 
 
@@ -68,6 +70,8 @@ class IterationLogEntry(BaseModel):
     target_file: str
     command: str
     proposal_source: str = "llm"
+    feature_recipe: Optional[Dict[str, Any]] = None
+    recipe_id: Optional[str] = None
     code_diff: str = ""
     status: Literal["ACCEPTED", "REJECTED", "ERROR_RECOVERED", "FAILED"]
     metrics: Optional[Dict[str, Any]] = None
diff --git a/orchestrator/state_machine.py b/orchestrator/state_machine.py
index 7e91ff5..ac2437b 100644
--- a/orchestrator/state_machine.py
+++ b/orchestrator/state_machine.py
@@ -30,15 +30,17 @@ from sandbox.logger import RunLogger
 from sandbox.runner import ExecutionRunner
 
 PY = sys.executable
-CONFIG_DIR = "configs"
+PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
+CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
 
 
-def load_dotenv_if_present(path: str = ".env"):
+def load_dotenv_if_present(path: Optional[str] = None):
     """Minimal .env reader — avoids a dependency for three lines of parsing."""
+    path = path or os.path.join(PROJECT_ROOT, ".env")
     if not os.path.exists(path):
         return
     try:
-        with open(path, encoding="utf-8") as fh:
+        with open(path, encoding="utf-8-sig") as fh:
             for line in fh:
                 line = line.strip()
                 if line and not line.startswith("#") and "=" in line:
@@ -69,6 +71,36 @@ def load_yaml(name: str) -> Dict[str, Any]:
 # behaviour sequence — leads.
 
 STRATEGY_BANK: List[Dict[str, str]] = [
+    {"stage": "Feature Engineering",
+     "hypothesis": "Fuse strictly causal affinity, completion-history, recency and momentum "
+                   "features into DeepFM so feature recipes affect a strong identity model "
+                   "instead of being isolated in a weaker tree-only ablation.",
+     "target_file": "pipeline/models.py",
+     "command": f"{PY} -m pipeline.train --model deepfm_dense --loss pointwise "
+                "--feature_profile full --select_features --epochs 12"},
+
+    {"stage": "Noise Reduction",
+     "hypothesis": "Reduce optimization variance without changing labels or deleting "
+                   "interactions: train a second seed of the validated causal DeepFM and "
+                   "let validation-only rank blending average uncorrelated errors.",
+     "target_file": "pipeline/train.py",
+     "command": f"{PY} -m pipeline.train --model deepfm_dense --loss pointwise "
+                "--feature_profile full --select_features --seed 1 --epochs 12"},
+    {"stage": "Loss Function",
+     "hypothesis": "Optimize both halves of the primary metric with a hybrid objective: "
+                   "BPR supplies GAUC-oriented pair ordering while delta-nDCG weights errors "
+                   "near rank 5. Preserve every power-user impression instead of capping groups.",
+     "target_file": "pipeline/models.py",
+     "command": f"{PY} -m pipeline.train --model fm_torch --loss hybrid "
+                "--auc_weight 0.5 --max_group_rows 0 --epochs 15"},
+
+    {"stage": "Causal Debiasing",
+     "hypothesis": "Use clipped self-normalized density ratios learned from KuaiRand's "
+                   "random-exposure log to reduce logging-policy bias while bounding variance.",
+     "target_file": "pipeline/propensity.py",
+     "command": f"{PY} -m pipeline.train --model fm_torch --loss hybrid "
+                "--auc_weight 0.5 --propensity snips --propensity_clip 10 --epochs 15"},
+
     {"stage": "Loss Function",
      "hypothesis": "Replace pointwise BCE with a within-user listwise softmax. The metrics "
                    "(GAUC, nDCG@5) rank inside a user's impression list, so a per-impression "
@@ -92,6 +124,22 @@ STRATEGY_BANK: List[Dict[str, str]] = [
      "target_file": "pipeline/train.py",
      "command": f"{PY} -m pipeline.train --model lgb --objective lambdarank --trees 400"},
 
+    {"stage": "Feature Engineering",
+     "hypothesis": "Add strictly historical watch-completion, repeat-affinity and recency "
+                   "features. Candidate-row play time is forbidden because it leaks long_view; "
+                   "expanding-window aggregates preserve its useful history safely.",
+     "target_file": "pipeline/features.py",
+     "command": f"{PY} -m pipeline.train --model lgb --objective lambdarank "
+                "
… truncated …
```

</details>

---

### Iteration 3 — Tree-based Ranker (`REJECTED`)

**Hypothesis.** Using a tree-based ranker like LightGBM with the LambdaMART objective will better capture complex interactions between features and improve ranking performance compared to the FM baseline.

**Rationale.** Tree-based models like LightGBM are known for their ability to capture non-linear interactions between features and handle categorical variables effectively. By using the LambdaMART objective, we can directly optimize for ranking metrics, which might lead to better performance in terms of GAUC and nDCG@5.

- Proposal source: `llm`
- Target file: `pipeline/train.py`
- Command: `C:\Users\Brian\AppData\Local\Programs\Python\Python312\python.exe -m pipeline.train --model lgb --objective lambdarank --trees 400 --num_leaves 63 --feature_profile full --select_features --seed 42`
- **Validation**: GAUC 0.6503 | nDCG@5 0.5280 | primary 0.5892 (delta -0.0124)
- Wall clock: 94.6s | cumulative tokens: 6,654

<details><summary>Code diff</summary>

```diff
diff --git a/README.md b/README.md
index 1384fd7..763f39e 100644
--- a/README.md
+++ b/README.md
@@ -8,6 +8,22 @@
 
 > **RankAgent** is an LLM-driven autonomous machine learning research agent engineered specifically for recommender system (RecSys) ranking problems. Given a tabular/interaction dataset and target metrics, RankAgent autonomously drives the closed-loop cycle of problem formulation, exploratory data analysis, feature engineering, architecture search & multi-task modeling, training/tuning, and rigorous offline evaluation with self-healing reflection.
 
+The metric-aligned feature engineering, hybrid GAUC/nDCG loss, randomized-exposure
+debiasing, feature governance, checkpoint naming, and diagnostic workflow are
+documented in [docs/INNOVATION_UPGRADE.md](docs/INNOVATION_UPGRADE.md).
+
+Run the complete workflow—feature audit, baseline, autonomous experiments,
+validation-selected ensembling, diagnostics, and submission—with one command:
+
+```bash
+python main.py
+```
+
+When an OpenAI or Anthropic key is present in `.env`, the research model can
+propose validated feature recipes on successive iterations and receives prior
+validation metrics plus feature importance as feedback. Without a key, the same
+command searches a deterministic bounded recipe set.
+
 ---
 
 ## 📌 Table of Contents
diff --git a/main.py b/main.py
index 9fc5392..0a3d093 100644
--- a/main.py
+++ b/main.py
@@ -3,14 +3,22 @@ RankAgent: Main Command-Line Interface.
 """
 import sys
 import argparse
+import os
 from orchestrator.state_machine import RankAgentOrchestrator
 
 def main():
+    os.chdir(os.path.dirname(os.path.abspath(__file__)))
+    # Windows terminals and redirected IDE consoles do not always share a code
+    # page. Keep the single entrypoint UTF-8-safe without crashing on one symbol.
+    for stream in (sys.stdout, sys.stderr):
+        if hasattr(stream, "reconfigure"):
+            stream.reconfigure(encoding="utf-8", errors="replace")
     parser = argparse.ArgumentParser(description="RankAgent: Autonomous ML Research Agent for Recommender Systems")
     parser.add_argument("--data_dir", default=None, help="Path to KuaiRand-Pure dataset directory (optional, auto-detected)")
     parser.add_argument("--max_iterations", type=int, default=None, help="Iteration budget (default: configs/benchmark_kuairand.yaml, hard cap 50)")
     parser.add_argument("--max_wall_clock", type=int, default=None, help="Wall-clock ceiling in seconds (default: configs/benchmark_kuairand.yaml, 6h)")
-    parser.add_argument("--run_baseline", action="store_true", help="Explicitly re-train the Factorization Machine baseline before starting iterations")
+    parser.add_argument("--skip_baseline", action="store_true",
+                        help="Skip baseline reproduction (default: run the full pipeline)")
     args = parser.parse_args()
 
     print("==================================================================")
@@ -21,7 +29,7 @@ def main():
         data_dir=args.data_dir,
         max_iterations=args.max_iterations,
         max_wall_clock=args.max_wall_clock,
-        run_baseline=args.run_baseline
+        run_baseline=not args.skip_baseline
     )
     agent.start_loop()
 
diff --git a/orchestrator/schemas.py b/orchestrator/schemas.py
index 2abf3a3..09bf5fb 100644
--- a/orchestrator/schemas.py
+++ b/orchestrator/schemas.py
@@ -55,6 +55,8 @@ class HypothesisProposal(BaseModel):
     rationale: str = ""
     target_file: str = "pipeline/train.py"
     command: str
+    feature_recipe: Optional[Dict[str, Any]] = None
+    recipe_id: Optional[str] = None
     source: Literal["llm", "fallback", "repair"] = "llm"
 
 
@@ -68,6 +70,8 @@ class IterationLogEntry(BaseModel):
     target_file: str
     command: str
     proposal_source: str = "llm"
+    feature_recipe: Optional[Dict[str, Any]] = None
+    recipe_id: Optional[str] = None
     code_diff: str = ""
     status: Literal["ACCEPTED", "REJECTED", "ERROR_RECOVERED", "FAILED"]
     metrics: Optional[Dict[str, Any]] = None
diff --git a/orchestrator/state_machine.py b/orchestrator/state_machine.py
index 7e91ff5..ac2437b 100644
--- a/orchestrator/state_machine.py
+++ b/orchestrator/state_machine.py
@@ -30,15 +30,17 @@ from sandbox.logger import RunLogger
 from sandbox.runner import ExecutionRunner
 
 PY = sys.executable
-CONFIG_DIR = "configs"
+PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
+CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
 
 
-def load_dotenv_if_present(path: str = ".env"):
+def load_dotenv_if_present(path: Optional[str] = None):
     """Minimal .env reader — avoids a dependency for three lines of parsing."""
+    path = path or os.path.join(PROJECT_ROOT, ".env")
     if not os.path.exists(path):
         return
     try:
-        with open(path, encoding="utf-8") as fh:
+        with open(path, encoding="utf-8-sig") as fh:
             for line in fh:
                 line = line.strip()
                 if line and not line.startswith("#") and "=" in line:
@@ -69,6 +71,36 @@ def load_yaml(name: str) -> Dict[str, Any]:
 # behaviour sequence — leads.
 
 STRATEGY_BANK: List[Dict[str, str]] = [
+    {"stage": "Feature Engineering",
+     "hypothesis": "Fuse strictly causal affinity, completion-history, recency and momentum "
+                   "features into DeepFM so feature recipes affect a strong identity model "
+                   "instead of being isolated in a weaker tree-only ablation.",
+     "target_file": "pipeline/models.py",
+     "command": f"{PY} -m pipeline.train --model deepfm_dense --loss pointwise "
+                "--feature_profile full --select_features --epochs 12"},
+
+    {"stage": "Noise Reduction",
+     "hypothesis": "Reduce optimization variance without changing labels or deleting "
+                   "interactions: train a second seed of the validated causal DeepFM and "
+                   "let validation-only rank blending average uncorrelated errors.",
+     "target_file": "pipeline/train.py",
+     "command": f"{PY} -m pipeline.train --model deepfm_dense --loss pointwise "
+                "--feature_profile full --select_features --seed 1 --epochs 12"},
+    {"stage": "Loss Function",
+     "hypothesis": "Optimize both halves of the primary metric with a hybrid objective: "
+                   "BPR supplies GAUC-oriented pair ordering while delta-nDCG weights errors "
+                   "near rank 5. Preserve every power-user impression instead of capping groups.",
+     "target_file": "pipeline/models.py",
+     "command": f"{PY} -m pipeline.train --model fm_torch --loss hybrid "
+                "--auc_weight 0.5 --max_group_rows 0 --epochs 15"},
+
+    {"stage": "Causal Debiasing",
+     "hypothesis": "Use clipped self-normalized density ratios learned from KuaiRand's "
+                   "random-exposure log to reduce logging-policy bias while bounding variance.",
+     "target_file": "pipeline/propensity.py",
+     "command": f"{PY} -m pipeline.train --model fm_torch --loss hybrid "
+                "--auc_weight 0.5 --propensity snips --propensity_clip 10 --epochs 15"},
+
     {"stage": "Loss Function",
      "hypothesis": "Replace pointwise BCE with a within-user listwise softmax. The metrics "
                    "(GAUC, nDCG@5) rank inside a user's impression list, so a per-impression "
@@ -92,6 +124,22 @@ STRATEGY_BANK: List[Dict[str, str]] = [
      "target_file": "pipeline/train.py",
      "command": f"{PY} -m pipeline.train --model lgb --objective lambdarank --trees 400"},
 
+    {"stage": "Feature Engineering",
+     "hypothesis": "Add strictly historical watch-completion, repeat-affinity and recency "
+                   "features. Candidate-row play time is forbidden because it leaks long_view; "
+                   "expanding-window aggregates preserve its useful history safely.",
+     "target_file": "pipeline/features.py",
+     "command": f"{PY} -m pipeline.train --model lgb --objective lambdarank "
+                "
… truncated …
```

</details>

---

### Iteration 4 — Loss Function | Architecture | Multi-Task Learning (`REJECTED`)

**Hypothesis.** Switching to a hybrid loss function with a mix of pointwise and listwise approaches will improve the model's ability to rank items by balancing individual prediction accuracy and overall ranking quality.

**Rationale.** The previous attempts with different architectures did not surpass the FM baseline. A hybrid loss function can leverage both pointwise and listwise objectives, potentially capturing a better balance between predicting individual scores and optimizing the overall ranking order. This approach may address the limitations seen in previous iterations by improving the model's ranking capability.

- Proposal source: `llm`
- Target file: `pipeline/train.py`
- Command: `C:\Users\Brian\AppData\Local\Programs\Python\Python312\python.exe -m pipeline.train --model fm_torch --loss hybrid --auc_weight 0.5 --embed_dim 32 --lr 0.001 --epochs 12 --batch_size 8192 --seed 42`
- **Validation**: GAUC 0.6611 | nDCG@5 0.5329 | primary 0.5970 (delta -0.0046)
- Wall clock: 159.0s | cumulative tokens: 9,068

<details><summary>Code diff</summary>

```diff
diff --git a/README.md b/README.md
index 1384fd7..763f39e 100644
--- a/README.md
+++ b/README.md
@@ -8,6 +8,22 @@
 
 > **RankAgent** is an LLM-driven autonomous machine learning research agent engineered specifically for recommender system (RecSys) ranking problems. Given a tabular/interaction dataset and target metrics, RankAgent autonomously drives the closed-loop cycle of problem formulation, exploratory data analysis, feature engineering, architecture search & multi-task modeling, training/tuning, and rigorous offline evaluation with self-healing reflection.
 
+The metric-aligned feature engineering, hybrid GAUC/nDCG loss, randomized-exposure
+debiasing, feature governance, checkpoint naming, and diagnostic workflow are
+documented in [docs/INNOVATION_UPGRADE.md](docs/INNOVATION_UPGRADE.md).
+
+Run the complete workflow—feature audit, baseline, autonomous experiments,
+validation-selected ensembling, diagnostics, and submission—with one command:
+
+```bash
+python main.py
+```
+
+When an OpenAI or Anthropic key is present in `.env`, the research model can
+propose validated feature recipes on successive iterations and receives prior
+validation metrics plus feature importance as feedback. Without a key, the same
+command searches a deterministic bounded recipe set.
+
 ---
 
 ## 📌 Table of Contents
diff --git a/main.py b/main.py
index 9fc5392..0a3d093 100644
--- a/main.py
+++ b/main.py
@@ -3,14 +3,22 @@ RankAgent: Main Command-Line Interface.
 """
 import sys
 import argparse
+import os
 from orchestrator.state_machine import RankAgentOrchestrator
 
 def main():
+    os.chdir(os.path.dirname(os.path.abspath(__file__)))
+    # Windows terminals and redirected IDE consoles do not always share a code
+    # page. Keep the single entrypoint UTF-8-safe without crashing on one symbol.
+    for stream in (sys.stdout, sys.stderr):
+        if hasattr(stream, "reconfigure"):
+            stream.reconfigure(encoding="utf-8", errors="replace")
     parser = argparse.ArgumentParser(description="RankAgent: Autonomous ML Research Agent for Recommender Systems")
     parser.add_argument("--data_dir", default=None, help="Path to KuaiRand-Pure dataset directory (optional, auto-detected)")
     parser.add_argument("--max_iterations", type=int, default=None, help="Iteration budget (default: configs/benchmark_kuairand.yaml, hard cap 50)")
     parser.add_argument("--max_wall_clock", type=int, default=None, help="Wall-clock ceiling in seconds (default: configs/benchmark_kuairand.yaml, 6h)")
-    parser.add_argument("--run_baseline", action="store_true", help="Explicitly re-train the Factorization Machine baseline before starting iterations")
+    parser.add_argument("--skip_baseline", action="store_true",
+                        help="Skip baseline reproduction (default: run the full pipeline)")
     args = parser.parse_args()
 
     print("==================================================================")
@@ -21,7 +29,7 @@ def main():
         data_dir=args.data_dir,
         max_iterations=args.max_iterations,
         max_wall_clock=args.max_wall_clock,
-        run_baseline=args.run_baseline
+        run_baseline=not args.skip_baseline
     )
     agent.start_loop()
 
diff --git a/orchestrator/schemas.py b/orchestrator/schemas.py
index 2abf3a3..09bf5fb 100644
--- a/orchestrator/schemas.py
+++ b/orchestrator/schemas.py
@@ -55,6 +55,8 @@ class HypothesisProposal(BaseModel):
     rationale: str = ""
     target_file: str = "pipeline/train.py"
     command: str
+    feature_recipe: Optional[Dict[str, Any]] = None
+    recipe_id: Optional[str] = None
     source: Literal["llm", "fallback", "repair"] = "llm"
 
 
@@ -68,6 +70,8 @@ class IterationLogEntry(BaseModel):
     target_file: str
     command: str
     proposal_source: str = "llm"
+    feature_recipe: Optional[Dict[str, Any]] = None
+    recipe_id: Optional[str] = None
     code_diff: str = ""
     status: Literal["ACCEPTED", "REJECTED", "ERROR_RECOVERED", "FAILED"]
     metrics: Optional[Dict[str, Any]] = None
diff --git a/orchestrator/state_machine.py b/orchestrator/state_machine.py
index 7e91ff5..ac2437b 100644
--- a/orchestrator/state_machine.py
+++ b/orchestrator/state_machine.py
@@ -30,15 +30,17 @@ from sandbox.logger import RunLogger
 from sandbox.runner import ExecutionRunner
 
 PY = sys.executable
-CONFIG_DIR = "configs"
+PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
+CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
 
 
-def load_dotenv_if_present(path: str = ".env"):
+def load_dotenv_if_present(path: Optional[str] = None):
     """Minimal .env reader — avoids a dependency for three lines of parsing."""
+    path = path or os.path.join(PROJECT_ROOT, ".env")
     if not os.path.exists(path):
         return
     try:
-        with open(path, encoding="utf-8") as fh:
+        with open(path, encoding="utf-8-sig") as fh:
             for line in fh:
                 line = line.strip()
                 if line and not line.startswith("#") and "=" in line:
@@ -69,6 +71,36 @@ def load_yaml(name: str) -> Dict[str, Any]:
 # behaviour sequence — leads.
 
 STRATEGY_BANK: List[Dict[str, str]] = [
+    {"stage": "Feature Engineering",
+     "hypothesis": "Fuse strictly causal affinity, completion-history, recency and momentum "
+                   "features into DeepFM so feature recipes affect a strong identity model "
+                   "instead of being isolated in a weaker tree-only ablation.",
+     "target_file": "pipeline/models.py",
+     "command": f"{PY} -m pipeline.train --model deepfm_dense --loss pointwise "
+                "--feature_profile full --select_features --epochs 12"},
+
+    {"stage": "Noise Reduction",
+     "hypothesis": "Reduce optimization variance without changing labels or deleting "
+                   "interactions: train a second seed of the validated causal DeepFM and "
+                   "let validation-only rank blending average uncorrelated errors.",
+     "target_file": "pipeline/train.py",
+     "command": f"{PY} -m pipeline.train --model deepfm_dense --loss pointwise "
+                "--feature_profile full --select_features --seed 1 --epochs 12"},
+    {"stage": "Loss Function",
+     "hypothesis": "Optimize both halves of the primary metric with a hybrid objective: "
+                   "BPR supplies GAUC-oriented pair ordering while delta-nDCG weights errors "
+                   "near rank 5. Preserve every power-user impression instead of capping groups.",
+     "target_file": "pipeline/models.py",
+     "command": f"{PY} -m pipeline.train --model fm_torch --loss hybrid "
+                "--auc_weight 0.5 --max_group_rows 0 --epochs 15"},
+
+    {"stage": "Causal Debiasing",
+     "hypothesis": "Use clipped self-normalized density ratios learned from KuaiRand's "
+                   "random-exposure log to reduce logging-policy bias while bounding variance.",
+     "target_file": "pipeline/propensity.py",
+     "command": f"{PY} -m pipeline.train --model fm_torch --loss hybrid "
+                "--auc_weight 0.5 --propensity snips --propensity_clip 10 --epochs 15"},
+
     {"stage": "Loss Function",
      "hypothesis": "Replace pointwise BCE with a within-user listwise softmax. The metrics "
                    "(GAUC, nDCG@5) rank inside a user's impression list, so a per-impression "
@@ -92,6 +124,22 @@ STRATEGY_BANK: List[Dict[str, str]] = [
      "target_file": "pipeline/train.py",
      "command": f"{PY} -m pipeline.train --model lgb --objective lambdarank --trees 400"},
 
+    {"stage": "Feature Engineering",
+     "hypothesis": "Add strictly historical watch-completion, repeat-affinity and recency "
+                   "features. Candidate-row play time is forbidden because it leaks long_view; "
+                   "expanding-window aggregates preserve its useful history safely.",
+     "target_file": "pipeline/features.py",
+     "command": f"{PY} -m pipeline.train --model lgb --objective lambdarank "
+                "
… truncated …
```

</details>

---

## Run summary

| | |
|---|---|
| Halt reason | validation primary improved by <= 0.002 over the last 3 iterations |
| Best validation primary | 0.6034 (iteration 0) |
| Delta over official baseline | +0.0018 |
| Iterations used | 4 / 50 |
| Agent wall clock | 17.9 min |
| LLM tokens (in + out) | 9,068 in 4 calls |
| Error recoveries | 0 |
| Failed iterations | 0 |
| Manual interventions | 0 |
| Submission | `submissions\kuairand_pure_final.csv` |
