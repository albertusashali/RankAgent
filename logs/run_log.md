# RankAgent Experiment Run Log
- **Run ID**: `rankagent-1787915608`

---

### Iteration 0: Baseline Reproduction
* **Status**: `ACCEPTED`
* **Target File**: `pipeline/models.py`
* **Hypothesis**: Stand up official Factorization Machine baseline on KuaiRand-Pure.
* **Metrics**: GAUC: 0.6664 | nDCG@5: 0.5360 | Primary: 0.6012
* **Telemetry**: 86.0s | Tokens: 0

---

### Iteration 1: Feature Engineering
* **Status**: `REJECTED`
* **Target File**: `pipeline/train.py`
* **Hypothesis**: Expand 5 fields to CWM 13 user/video domains.
* **Metrics**: GAUC: 0.6649 | nDCG@5: 0.5347 | Primary: 0.5998
* **Telemetry**: 135.2s | Tokens: 0

---

### Iteration 0: Baseline Reproduction
* **Status**: `ACCEPTED`
* **Target File**: `pipeline/models.py`
* **Hypothesis**: Stand up official Factorization Machine baseline on KuaiRand-Pure.
* **Metrics**: GAUC: 0.6664 | nDCG@5: 0.5360 | Primary: 0.6012
* **Telemetry**: 88.4s | Tokens: 0

---

### Iteration 1: Feature Engineering
* **Status**: `REJECTED`
* **Target File**: `pipeline/train.py`
* **Hypothesis**: Expand 5 fields to CWM 13 user/video domains.
* **Metrics**: GAUC: 0.6649 | nDCG@5: 0.5347 | Primary: 0.5998
* **Telemetry**: 133.3s | Tokens: 0

---

### Iteration 2: Architecture
* **Status**: `REJECTED`
* **Target File**: `pipeline/models.py`
* **Hypothesis**: Train DeepFM with 2nd order feature factor embeddings & Deep MLP.
* **Metrics**: GAUC: 0.6665 | nDCG@5: 0.5344 | Primary: 0.6004
* **Telemetry**: 197.4s | Tokens: 0

---

### Iteration 3: Hyperparameter Tuning
* **Status**: `REJECTED`
* **Target File**: `pipeline/train.py`
* **Hypothesis**: Tune learning rate to 0.0005 with weight decay on DeepFM.
* **Metrics**: GAUC: 0.6652 | nDCG@5: 0.5349 | Primary: 0.6001
* **Telemetry**: 169.4s | Tokens: 0

---

