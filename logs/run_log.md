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

### Iteration 1: Multi-Task Learning
* **Status**: `ACCEPTED`
* **Target File**: `pipeline/models.py`
* **Hypothesis**: Train Multi-Task MMoE on long_view + click + like to leverage shared representations.
* **Metrics**: GAUC: 0.6706 | nDCG@5: 0.5370 | Primary: 0.6038
* **Telemetry**: 570.0s | Tokens: 0

---

### Iteration 2: Sequential Attention
* **Status**: `REJECTED`
* **Target File**: `pipeline/models.py`
* **Hypothesis**: Train Deep Interest Network (DIN) with Target-Attention pooling over user past watch history.
* **Metrics**: GAUC: 0.6652 | nDCG@5: 0.5350 | Primary: 0.6001
* **Telemetry**: 185.3s | Tokens: 0

---

### Iteration 3: Tree Ranker
* **Status**: `REJECTED`
* **Target File**: `pipeline/train.py`
* **Hypothesis**: Train LightGBM GBDT Ranker with dense historical engagement aggregations.
* **Metrics**: GAUC: 0.6396 | nDCG@5: 0.5227 | Primary: 0.5811
* **Telemetry**: 51.0s | Tokens: 0

---

### Iteration 1: Architecture
* **Status**: `ACCEPTED`
* **Target File**: `pipeline/models.py`
* **Hypothesis**: Incorporating a Deep & Cross Network (DCN-v2) can capture both low-order and high-order feature interactions more effectively than a standard Factorization Machine, potentially improving the model's ability to predict the 'long_view' target.
* **Metrics**: GAUC: 0.6706 | nDCG@5: 0.5370 | Primary: 0.6038
* **Telemetry**: 580.6s | Tokens: 0

---

### Iteration 2: Multi-Task
* **Status**: `REJECTED`
* **Target File**: `pipeline/models.py`
* **Hypothesis**: Incorporating Multi-Task Learning (MTL) using a Multi-gate Mixture-of-Experts (MMoE) architecture can leverage auxiliary signals such as 'click', 'like', and 'comment' to improve the prediction of the primary target 'long_view'. This approach can help overcome label sparsity and capture shared representations across tasks, potentially improving the model's performance.
* **Metrics**: GAUC: 0.6652 | nDCG@5: 0.5350 | Primary: 0.6001
* **Telemetry**: 193.5s | Tokens: 0

---

### Iteration 3: Architecture
* **Status**: `REJECTED`
* **Target File**: `pipeline/models.py`
* **Hypothesis**: Incorporating a Deep Interest Network (DIN) can enhance the model's ability to capture user-specific interests by applying attention mechanisms to the sequence of previously interacted videos. This can improve the prediction of 'long_view' by focusing on the most relevant historical interactions.
* **Metrics**: GAUC: 0.6396 | nDCG@5: 0.5227 | Primary: 0.5811
* **Telemetry**: 47.2s | Tokens: 0

---

### Iteration 4: Feature Engineering
* **Status**: `REJECTED`
* **Target File**: `pipeline/features.py`
* **Hypothesis**: Incorporating historical user behavior features such as historical click-through rate (CTR) and long_view rate can provide additional context about user preferences and engagement patterns. These features can be computed as smooth Bayesian target encodings to avoid overfitting and can enhance the model's ability to predict the 'long_view' target.
* **Metrics**: GAUC: 0.6664 | nDCG@5: 0.5344 | Primary: 0.6004
* **Telemetry**: 210.4s | Tokens: 0

---

