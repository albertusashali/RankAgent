# RankAgent: Autonomous RecSys ML Research Agent

RankAgent is an autonomous, headless machine learning research agent for the **KuaiRand-Pure** within-user ranking benchmark. It automatically reproduces the official baseline, formulates hypotheses grounded in recommendation systems research, writes and repairs AST-valid source code (new loss functions, neural architectures, and feature recipes), verifies changes against strict safety/leakage gates in isolated sandboxes, trains models locally, and designates verified submissions.

```bash
python main.py --max_iterations 30
```

---

## 1. Project Overview

### Problem Statement & Challenge
The **KuaiRand-Pure** benchmark evaluates within-user ranking on micro-video impressions. The target task predicts `long_view` (watch-time completion threshold) scored via Group AUC (**GAUC**) and **nDCG@5**, with the **primary score** defined as their arithmetic mean:

$$\text{Primary Score} = \frac{\text{GAUC} + \text{nDCG@5}}{2}$$

Key machine learning challenges in this domain include:
1. **Cold-Start & Extreme Sparsity**: Massive categorical cardinality across user and video IDs with sparse positive user feedback.
2. **Watch Duration Confounding**: `long_view` is strongly influenced by video duration and post-impression watch time. Exposing raw post-impression signals on test splits causes catastrophic label leakage.
3. **Ragged User Groupings**: Impressions vary widely per user, making standard batch pointwise objectives sub-optimal compared to true within-user listwise ranking.
4. **Autonomous Search Surface**: Searching across 7 distinct research dimensions (loss functions, neural architectures, feature transformations, capacity, multi-task learning, sequential modeling, and optimization schedules) requires structured multi-agent collaboration rather than brute-force hyperparameter sweeps.

### How RankAgent Addresses the Problem
RankAgent replaces fixed flag menus with **direct source code generation and transactional patching**. An autonomous team of 5 specialized agents collaborates iteratively over a typed blackboard:

| Role | Responsibility | Output Artefact |
| :--- | :--- | :--- |
| **Product Manager** | Balances exploration & exploitation across 7 research dimensions (loss, architecture, features, capacity, multi-task, sequence, optimisation). | Iteration directive & target focus |
| **ML Researcher** | Proposes concrete hypotheses grounded in a citation-backed RecSys knowledge base. | $k$ grounded hypothesis candidates |
| **Engineer** | Implements changes via SEARCH/REPLACE patches, registers new models/losses in dynamic registries, and performs self-healing repairs upon failure. | Transactional code patch |
| **Feature Steward** | Governs the feature space and validates candidate feature recipes against data leakage. | Validated feature recipe |
| **QA / Verifier** | Enforces static leak checks, dynamic mutation audits, import allowlists, and score validity. | Pre-flight & post-trial verdicts |

```
                       ┌────────────────────────┐
                       │    Product Manager     │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │     ML Researcher      │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
        ┌─────────────►│        Engineer        │◄────────────┐
        │              └───────────┬────────────┘             │
        │                          │ (patch)                  │ (repair)
        │                          ▼                          │
        │              ┌────────────────────────┐             │
        │              │  Sandbox Verification  │             │
        │              │  (AST, Leak, Imports)  │             │
        │              └───────────┬────────────┘             │
        │                          │                          │
        │                          ▼                          │
        │              ┌────────────────────────┐             │
        │              │ Smoke Test (~5s train) ├─────────────┘
        │              └───────────┬────────────┘ (fails)
        │                          │ (passes)
        │                          ▼
        │              ┌────────────────────────┐
        │              │  Full Trial (~90-150s) │
        │              └───────────┬────────────┘
        │                          │
        │                          ▼
        │              ┌────────────────────────┐
        │              │       QA Verdict       │
        │              └───────────┬────────────┘
        │                          │
        │ (next iteration)         ▼
        └────────────────── Best Model Update
```

### Safety & Trust Boundary
- **Hidden-Test Sealing**: Post-impression outcomes (`play_time_ms`, auxiliary feedback) on the test split are sealed with sentinel values (`-1`) in [`pipeline/data.py`](pipeline/data.py).
- **Dynamic Mutation Audit**: [`pipeline/feature_agent.py`](pipeline/feature_agent.py) mutates outcome columns and verifies that feature representations remain invariant.
- **Process Isolation**: Each trial executes in an isolated subprocess with OpenMP protection and token stripping to prevent environment leakage.
- **Deterministic Fallback**: Every agent includes offline rule-based fallbacks, allowing full execution with zero LLM tokens.

### Tools, APIs, Frameworks & Datasets Used

| Category | Components Used | Purpose |
| :--- | :--- | :--- |
| **Development Tools** | Visual Studio Code (VSCode), Git & GitHub, PowerShell / Bash, Pytest | Local multi-file development, version control, automated test harness execution, and environment management. |
| **APIs** | OpenAI API (GPT-4o, GPT-4o-mini), Anthropic API (Claude 3.5 Sonnet) | LLM-based reasoning, literature-grounded hypothesis formulation, transactional code authoring, and self-healing error repairs. |
| **Libraries & Frameworks** | **PyTorch** (`torch`), **LightGBM**, **NumPy**, **SciPy**, **Scikit-Learn**, **Pandas**, **Pydantic (v2)**, **PyYAML** | Deep neural ranking models (FM, DeepFM, DIN, MMoE, DCN-v2, PLE, Cross-Attention), GBDT LambdaRank, tabular statistics, typed agent contracts, and YAML configurations. |
| **Datasets & Assets** | **KuaiRand-Pure Dataset** (Zenodo `10439422`), **KuaiRand Starter Kit**, RecSys Knowledge Base ([`agents/knowledge.py`](agents/knowledge.py)) | 1.14M train, 125K valid, 171K sealed test impressions; official evaluation metrics; 12 citation-backed RecSys methods. |

### Benchmark Results

| Model / Configuration | Validation GAUC | Validation nDCG@5 | Validation Primary | vs Official Baseline |
| :--- | :---: | :---: | :---: | :---: |
| Random Scoring Floor | 0.5000 | 0.4668 | 0.4834 | — |
| **Official FM Baseline (Published)** | **0.6672** | **0.5360** | **0.6016** | — |
| **Our Baseline Reproduction** | **0.6671** | **0.5358** | **0.6015** | −0.0001 |
| Best Agent Model (`cross_attention`) | **0.6710** | **0.5372** | **0.6041** | **+0.0026 (+3.2σ)** |
| Neural Causal Ranker (`dense_deepfm`) | 0.6685 | 0.5363 | 0.6024 | +0.0009 |
| Oracle Ceiling | 0.9998 | 0.6970 | 0.8484 | — |

---

## 2. Setup and Installation Instructions

### Prerequisites

- **Python**: Version 3.11+ (tested on Python 3.11 through 3.14).
- **System**: Linux, macOS, or Windows (PowerShell / Command Prompt).
- **Hardware**: CPU only, ~4 GB RAM minimum (no GPU required).

### 1. Clone the Repository & Create Virtual Environment

```bash
git clone https://github.com/albertusashali/RankAgent.git
cd RankAgent

# Create virtual environment
python -m venv .venv
```

**Activate the virtual environment:**
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download & Prepare the KuaiRand-Pure Dataset

Download the official dataset archive from Zenodo and extract it into the `data/` directory:

- **Linux / macOS:**
  ```bash
  mkdir -p data
  curl -L -O https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
  tar -xzf KuaiRand-Pure.tar.gz -C data/
  rm KuaiRand-Pure.tar.gz
  ```
- **Windows (PowerShell):**
  ```powershell
  New-Item -ItemType Directory -Force -Path "data"
  Invoke-WebRequest -Uri "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz" -OutFile "data\KuaiRand-Pure.tar.gz"
  tar -xzf "data\KuaiRand-Pure.tar.gz" -C "data\"
  Remove-Item "data\KuaiRand-Pure.tar.gz"
  ```

> [!NOTE]
> The data loader in [`pipeline/data.py`](pipeline/data.py) automatically auto-discovers the dataset whether it is located at `data/KuaiRand-Pure/data` or `data/KuaiRand-Pure/KuaiRand-Pure/data`.

### 4. Environment & API Key Configuration (`.env`)

> [!IMPORTANT]
> The active `.env` file containing LLM API credentials will be **uploaded/provided privately**. Place the `.env` file directly into the repository root directory.

If you are setting up your own keys manually:
```bash
cp .env.example .env
```
Edit `.env` and set your key:
```ini
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
```

*(If no API key is provided, RankAgent automatically runs in deterministic fallback mode with zero token costs).*

---

## 3. Steps to Reproduce Your Results

### Step 1: Run Harness Tests & Offline Verification (Free, ~30s)

Verify the 86 test suites and run the 10-iteration offline planner (requires no API calls and no dataset):

```bash
# Run pytest unit tests
python -m pytest tests/ -q

# Run offline agent smoke test
python scripts/smoke_agents.py
```
*Expect: `86 passed` and `ALL CHECKS PASSED`.*

### Step 2: Reproduce the Official FM Baseline (~90s)

Train the Factorization Machine baseline on the KuaiRand-Pure dataset to verify scoring reproducibility:

```bash
python -m pipeline.train --model fm
```
*Expect validation output: `[EVAL] GAUC: 0.6671 | nDCG@5: 0.5358 | Primary: 0.6015` (matches published 0.6016 within $\pm 0.0001$).*

### Step 3: Run the Autonomous Multi-Agent Loop

Run the full autonomous search loop:

- **Live LLM Run (30 iterations):**
  ```bash
  python main.py --max_iterations 30
  ```
- **Deterministic / Offline Mode (No API keys needed):**
  - **Linux / macOS:**
    ```bash
    OPENAI_API_KEY=none ANTHROPIC_API_KEY=none python main.py --max_iterations 5
    ```
  - **Windows (PowerShell):**
    ```powershell
    $env:OPENAI_API_KEY="none"; $env:ANTHROPIC_API_KEY="none"; python main.py --max_iterations 5
    ```
- **Quick Test Iteration (Skip baseline re-verification for rapid testing):**
  ```bash
  python main.py --max_iterations 1 --skip-baseline
  ```

### Step 4: Run Feature Governance & Leak Audits

Run the dynamic mutation leakage checker:

```bash
python -m pipeline.feature_agent --dynamic
```
*Expect: `[FEATURE AUDIT] PASS`.*

### Step 5: Inspect Run Logs & Output Artefacts

After a run completes, review the summary and generated code diffs:

```bash
python scripts/inspect_run.py
```

Generated artefacts:
- `logs/run_log.md`: Human-readable narrative log containing every iteration's hypothesis, command, and code diff.
- `logs/run_summary.json`: Machine-readable structured run telemetry.
- `workspaces/node_XXX/`: Complete per-iteration isolated source code trees.
- `submissions/kuairand_pure_final.csv`: Final test submission generated from the best validated checkpoint.

---

## 4. Limitations & Future Improvements

### Limitations

1. **Marginal Score Delta vs. Seed Variance**:
   - The best validated improvement (+0.0026) is statistically significant at $3.2\sigma$ against seed noise ($\sigma \approx 0.0008$), but close to the noise floor. Single-run validation gains may show modest transfer to the hidden test set.
2. **Complexity of Ragged-Batch Listwise Objectives**:
   - Agent-generated architectures (e.g., Cross-Attention, DCN-v2, PLE) consistently succeeded, whereas generated listwise ranking losses (e.g., ApproxNDCG, ListMLE) struggled due to the difficulty of implementing ragged within-user grouping correctly in PyTorch without degradation.
3. **Shallow Compounding Exploration**:
   - Acceptance decisions rely strictly on immediate validation improvements. Because neutral mutations are not deeply branched, deep multi-step code composition chains are rare within 30 iterations.
4. **Non-Resumable Run State**:
   - If an ongoing run is interrupted, logs and checkpoints are preserved, but the in-memory tree state cannot currently resume mid-loop.

### Future Improvements

1. **Frontier Tree Search & UCB Lineage Prompting**:
   - Implement Upper Confidence Bound (UCB) selection over an active tree frontier with explicit `draft`, `refine`, and `debug` modes to enable deeper composition of discovered features and architectures.
2. **Rank-Normalized Model Ensembling**:
   - Implement multi-model blending (e.g., rank-averaged blending between LightGBM GBDT and deep neural rankers), which historically yields substantial gains in RecSys benchmarks.
3. **AST Dataflow No-Op Detection**:
   - Add static AST dataflow analysis to immediately detect and reject patches that define variables or functions that are never consumed in the training pipeline.
4. **Editable Preprocessing with Differential Sealing**:
   - Expand the agent's mutable boundary to feature preprocessing logic, enabling automated transformation of raw tabular fields behind automated leakage guards.
5. **Multi-Seed Winner Confirmation**:
   - Automatically evaluate top candidate nodes across $n=3$ distinct random seeds before designating the final competition submission.

---

## 5. Team Member Contributions

This project was developed by a team of 4 contributors:

| Team Member | Core Contributions |
| :--- | :--- |
| **Albertus Ashali** | Initial RankAgent core framework; orchestrator state machine (`orchestrator/state_machine.py`), tree search (`orchestrator/tree_manager.py`), and intervention ledger (`orchestrator/interventions.py`); deep ranking models with dense feature interactions (PLE, DCN-v2, BST) in `pipeline/models.py`. |
| **Goh Peck Kiat (`gohpk`)** | Multi-agent coordination (`agents/team.py`, `agents/product_manager.py`, `agents/researcher.py`, `agents/engineer.py`, `agents/qa.py`); code generation engine (`agents/codegen.py`, `agents/patch.py`); per-node sandbox workspace isolation (`sandbox/workspace.py`); verifier and subprocess runner (`sandbox/verifier.py`, `sandbox/runner.py`). |
| **Brian Yeo (`brianyeo02`)** | Feature engineering pipeline (`pipeline/features.py`); feature governance and mutation-based leakage auditor (`pipeline/feature_agent.py`, `pipeline/feature_recipes.py`); Feature Steward agent integration (`agents/feature_steward.py`). |
| **Kok Zhi (`kz`)** | Duplicate-experiment guardrail and context management (`agents/context.py`); trial execution guardrails, debugger, and error recovery (`sandbox/debugger.py`, `sandbox/logger.py`); system configuration and schema validation (`orchestrator/schemas.py`, `configs/`). |

---

## 6. Repository Layout

```
RankAgent/
├── main.py                     # CLI entry point for autonomous runs
├── Makefile                    # Make targets for installation and execution
├── requirements.txt            # Python dependencies
├── .env.example                # Template for LLM API keys
│
├── agents/                     # Multi-agent role implementations
│   ├── team.py                 # Multi-agent coordinator
│   ├── product_manager.py      # Research dimension strategy & exploration
│   ├── researcher.py           # Hypothesis generation & literature citation
│   ├── engineer.py             # Code authoring & repair logic
│   ├── feature_steward.py      # Feature exploration & validation
│   ├── qa.py                   # Pre-flight and post-trial verification
│   ├── codegen.py              # AST code generation & inspection utilities
│   ├── patch.py                # Transactional SEARCH/REPLACE patch engine
│   ├── context.py              # Context formatting & deduplication
│   └── knowledge.py            # RecSys paper & method knowledge base
│
├── orchestrator/               # Execution loop and tree state
│   ├── state_machine.py        # Core iterative research loop
│   ├── tree_manager.py         # Lineage tree & node manager
│   ├── interventions.py        # Human touchpoint audit ledger
│   └── schemas.py              # Pydantic data schemas
│
├── sandbox/                    # Safety and execution boundary
│   ├── workspace.py            # Per-node workspace isolation & hashing
│   ├── verifier.py             # Static leak scan & AST validation
│   ├── runner.py               # Subprocess runner & environment filtering
│   ├── debugger.py             # Error extraction & repair formatting
│   └── logger.py               # Experiment and metric logging
│
├── pipeline/                   # Machine learning pipeline
│   ├── data.py                 # Data loader with sealed hidden test split
│   ├── models.py               # Deep neural ranking models (mutable)
│   ├── models_np.py            # NumPy FM baseline implementation
│   ├── features.py             # Feature extraction & encoding (mutable)
│   ├── feature_agent.py        # Dynamic mutation leak auditor
│   ├── feature_recipes.py      # Validated feature transformation space
│   ├── train.py                # Training loop & early stopping (mutable)
│   ├── evaluate.py             # Official GAUC and nDCG@5 metrics
│   └── submit.py               # Test split scoring & submission generator
│
├── configs/                    # System and benchmark configuration files
│   ├── agent_config.yaml       # Agent hyperparameters and thresholds
│   └── benchmark_kuairand.yaml # Benchmark caps and metric specifications
│
├── scripts/                    # Utility and diagnostic scripts
│   ├── smoke_agents.py         # Offline test runner (0 tokens, free)
│   ├── inspect_run.py          # Summary formatter for past runs
│   └── repair_run_log.py       # Log reconstruction utility
│
├── tests/                      # Pytest test suite (86 tests)
│   ├── test_agents.py          # Agent behavior & prompt unit tests
│   ├── test_codegen.py         # Patch engine & workspace tests
│   ├── test_features.py        # Feature extraction & leak tests
│   └── test_harness.py         # End-to-end harness & safety tests
│
└── docs/
    └── ARCHITECTURE.md         # Detailed system design and flowcharts
```
