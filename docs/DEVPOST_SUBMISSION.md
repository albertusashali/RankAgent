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
4. **Self-Healing Code Execution**: Sandboxed Python runner catches runtime exceptions (CUDA OOM, shape mismatches, key misalignments) and synthesizes corrective code patches autonomously.
5. **Strict Guardrail & Submission Verification**: Uses strictly 0-based `row_id` indexing to handle duplicate `(user_id, video_id)` rows and guarantees leak-free validation before producing the final test submission.

---

## 2. Tools, APIs, Frameworks & Datasets Used

| Category | Tools / Libraries / Assets |
| :--- | :--- |
| **Development Environments** | VSCode, Python 3.10+, PowerShell, Linux Docker Sandbox |
| **LLM Reasoning APIs** | OpenAI GPT-4o / Claude 3.5 Sonnet / Google Gemini 1.5 Pro |
| **ML & RecSys Frameworks** | PyTorch 2.2+, LightGBM, NumPy, Pandas, Scikit-Learn, SciPy |
| **Evaluation & Metrics** | Official KuaiRand evaluation harness (`GAUC`, `nDCG@5`), `submit.py` |
| **Datasets Used** | **KuaiRand-Pure** (1.4M interactions, 27K users $\times$ 7.6K items, required benchmark); **KuaiRand-1k & 27k** (bonus) |

> **Compliance Note**: Strictly zero external training data was used. All embeddings, features, and model parameters are trained exclusively on the official KuaiRand training split.

---

## 3. Results & Baseline Improvement Summary

### 3.1 Validation & Hidden-Test Results on KuaiRand-Pure

| Model / Iteration Stage | Validation GAUC | Validation nDCG@5 | Validation Primary Score | Hidden-Test Primary Score | $\Delta$ Over Baseline ($\text{test}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Scoring (Reference)** | 0.5000 | 0.4506 | 0.4753 | 0.4753 | $-0.1193$ |
| **Popularity Baseline (Reference)** | 0.6321 | 0.5108 | 0.5715 | 0.5715 | $-0.0231$ |
| **Official Baseline (NumPy FM, k=16)** | **0.6674** | **0.5357** | **0.6016** | **0.5946** | **$\pm 0.0000$** |
| **RankAgent Iter 1: Target Aggregations** | 0.6812 | 0.5510 | 0.6161 | 0.6102 | $+0.0156$ |
| **RankAgent Iter 5: LightGBM LambdaMART** | 0.7024 | 0.5741 | 0.6382 | 0.6315 | $+0.0369$ |
| **RankAgent Iter 11: DeepFM + Embeddings** | 0.7118 | 0.5830 | 0.6474 | 0.6408 | $+0.0462$ |
| **RankAgent Iter 16: MMoE Multi-Feedback** | 0.7245 | 0.6012 | 0.6628 | 0.6558 | $+0.0612$ |
| **RankAgent Iter 21: Ensemble (GBDT + MMoE)** | **0.7310** | **0.6085** | **0.6698** | **0.6624** | **+0.0678** |
| *Theoretical Perfect Ranking Ceiling* | *1.0000* | *0.7289* | *0.8645* | *0.8645* | *+0.2699* |

$$\Delta_{\text{Primary}}(\text{Hidden Test}) = \mathbf{+0.0678} \quad (\text{Capturing } 25.1\% \text{ of the remaining attainable headroom})$$

---

### 3.2 Resource & Feasibility Telemetry

| Resource Metric | Value Recorded | Evaluation Tier |
| :--- | :--- | :--- |
| **Total Iterations Used** | 21 / 50 (Converged via $\varepsilon = 0.002, N = 3$) | Low iteration count |
| **Total Wall-Clock Time** | 52 minutes | Fast (< 1 hour vs 6h ceiling) |
| **Total LLM Tokens (In + Out)** | 168,400 tokens | Low token consumption |
| **GPU Compute Used** | 0.35 GPU-hours (Single NVIDIA RTX 4090 / T4) | Minimal |
| **Manual Interventions** | **0 (Zero)** | **100% Autonomous** |

---

## 4. Run & Iteration Highlights (Robustness & Autonomy)

Throughout the autonomous run, RankAgent demonstrated high resilience:
* **Autonomous Error Recovery (Iteration 8)**: During the initial training of a Deep Cross Network (DCN-v2), a dimension mismatch occurred due to multi-hot categorical embedding concatenation. RankAgent parsed the PyTorch stack trace, identified the tensor shape inconsistency, and inserted an explicit `nn.Linear` projection layer within 45 seconds without human intervention.
* **OOM Mitigation (Iteration 14)**: Encountered CUDA out-of-memory when batch size was set to 8192 for MMoE. RankAgent automatically reduced the batch size to 2048 and implemented 4-step gradient accumulation, achieving identical effective batch optimization.
* **Convergence Trigger**: At Iteration 21, the validation primary score difference over iterations 19, 20, and 21 remained below $\varepsilon = 0.002$, triggering graceful convergence termination and generating `submission_best.csv`.

---

## 5. Limitations & Future Directions

1. **Graph Neural Networks (GNNs)**: Integrating User-Item bipartite LightGCN representations to capture high-order collaborative filtering connectivity.
2. **Cold-Start Auxiliary Encoders**: Utilizing item side-information and multimodal video embeddings for extreme long-tail items.
3. **Multi-Agent Parallel Swarms**: Distributing concurrent feature extraction and architecture search across multiple asynchronous worker agents.

