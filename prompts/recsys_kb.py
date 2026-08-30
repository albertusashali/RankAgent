"""
Domain Knowledge Base and Playbook for Short-Video Recommender Systems (KuaiRand Focus).
"""

RECSYS_KB = """
### SHORT-VIDEO RECOMMENDATION PLAYBOOK & DOMAIN DYNAMICS

1. TARGET VARIABLE & RANKING METRIC REALITIES:
   - Primary Label: `long_view` (Binary classification: 1 if user watched the video for a long duration or completed, 0 otherwise).
   - Metrics: GAUC (User-weighted AUC excluding 0-pos and all-pos users) & nDCG@5 (Discounted gain: 2^rel - 1).
   - Official FM Baseline: GAUC 0.6674, nDCG@5 0.5357 -> Primary Score: 0.6016 (val) / 0.5946 (test).
   - Theoretical Ceiling: 0.8484 (val) / 0.8645 (test) — 27% of users have no positive labels.
   - Ranking Alignment: Within-user ranking means all pure user-side signals (constant for a user's candidate list) have zero ranking gradient unless interacted with item attributes (cross-features). Pointwise BCE optimizes whole-corpus calibration; within-user listwise softmax or LambdaMART directly optimizes relative within-user ordering.

2. SHORT-VIDEO PSYCHOLOGICAL REALITIES (Crucial for Feature Engineering & Prompting):
   - Topic Fatigue & Negative Diversity:
     * Short-video consumption is rapid-swipe. Users experience sharp boredom when exposed to consecutive videos from the same author or narrow topic.
     * Users rarely watch 3 consecutive comedy sketches or dance clips from the exact same author.
     * Sequential models without negative penalty signals or fatigue tracking suffer from topic clustering.
     * Key Levers: Track consecutive author impressions (`user_author_recent_streak`), recency gap since author was last seen (`author_recency_gap`), and apply non-linear negative penalties.
   - Watch Time vs. Instant Click (The Clickbait Duality):
     * A `click` is an impulsive reaction triggered by an intriguing cover/thumbnail, title, or the first 1-2 seconds.
     * `long_view` (>5s or full completion) represents genuine user satisfaction and content quality.
     * Deceptive clickbait achieves high click rates but near-zero completion. Recommending clickbait destroys user retention.
     * Key Levers: Compute Bayesian-smoothed clickbait discrepancy signals (`click_long_view_gap` = `click_rate` - `long_view_rate`, `click_to_long_view_ratio`). Use multi-task regularizers that penalize high click probability when `long_view` is absent.
   - Duration Bias & Natural Completion Decay:
     * Video duration fundamentally biases completion rates: a 10-second clip naturally achieves ~40% completion rate, whereas a 60-second clip achieves ~5% completion rate.
     * Unnormalized models over-recommend trivial micro-clips because completion rate acts as an easy shortcut for short duration.
     * Key Levers: Duration bucket conditioning (`dur_bucket`), duration-normalized completion expectations (observed completion progress relative to the bucket's expected prior), and user-by-duration affinity crosses (`user_durbucket_affinity`).
   - Author Loyalty vs. Session Depth Dynamics:
     * User loyalty to favorite creators creates high tolerance for longer content and repeated exposure, whereas casual users require broad diversity.
     * Intra-session fatigue accumulates: as a user's session depth (daily impression count) grows, attention span decays and tolerance for sub-par content drops.
     * Key Levers: Track intra-day session depth index (`user_daily_session_depth`) and author loyalty indices (`user_author_loyalty` = user author long_view rate / user base rate).

3. CAUSAL FEATURE ENGINEERING CONTRACT:
   - Causal Expanding Window: Training rows on date d must only see statistics computed from dates < d. Valid and test rows must see frozen full-training statistics.
   - Cross-Features: User x Author affinity, User x Duration bucket affinity, User x Tab affinity back off smoothly to the user's prior rate.
   - Discrepancy & Fatigue Signals: Tabular GBDTs (LightGBM LambdaMART) excel at learning non-linear threshold cuts on clickbait gaps, author streaks, and normalized completion expectations.

4. MULTI-TASK & AUXILIARY SIGNALS (MMoE vs PLE):
   - KuaiRand logs auxiliary signals: `click`, `like`, `follow`, `comment`, `forward`, `play_time_ms`, `duration_ms`.
   - Multi-Task Learning (MMoE): Jointly train auxiliary heads (`click`, `like`, `forward`) to regularize shared embeddings without diluting task 0 (`long_view`).
   - Progressive Layered Extraction (PLE / CGC): Unlike MMoE where all experts are shared, PLE equips each task with private experts alongside shared experts. This eliminates the "seesaw effect" (negative transfer) where noisy `click` gradients corrupt representations for `long_view`.

5. SEQUENTIAL & ATTENTION MODELING (DIN vs BST):
   - Causal impression sequence captures evolving user taste over the last 5-20 impressions.
   - Deep Interest Network (DIN): Target attention queries history using candidate item.
   - Behavior Sequence Transformer (BST): Self-attention Transformer encoder models item-to-item sequential transitions and positional dynamics across the user history before scoring the candidate.

6. EXPLICIT CROSSING (DCN-v2):
   - Deep & Cross Network v2 (DCN-v2): Applies explicit polynomial feature crossing to break the static user feature cancellation trap, learning multiplicative user x item interactions automatically without manual feature engineering.

7. CRITICAL RECSYS TRAPS & HIGH-VALUE LEVERS:
   - THE TRAP: Never Drop/Clip Power Users (Outliers):
     * In standard tabular classification, users with 5,000 views look like outliers. In RecSys, clipping or dropping them is FATAL.
     * The number of informative ranking pairs for GAUC scales as O(N_pos * N_neg). The top 1% power users contribute the vast majority of all pairwise ranking pairs in the dataset. Dropping them ruins validation scores.
   - THE TRAP: Static User Features Have Zero Direct Ranking Gradient:
     * Static user features (e.g., age bracket, registration days) add a constant +C_u to every video candidate in user u's impression list.
     * In within-user ranking (GAUC / nDCG@5), (Score(u, i) - Score(u, j)) = (g(i) + C_u) - (g(j) + C_u) = g(i) - g(j). The user constant cancels out completely!
     * Static user features are completely useless unless explicitly crossed/interacted with video or author attributes (use DCN-v2 or causal aggregations).
   - HIGH-VALUE LEVER: Causal Interaction Aggregations:
     * Compute smoothed expanding-window historical interaction statistics, e.g., Affinity(u, author) = (clicks(u, author) + 1) / (impressions(u, author) + 10).
   - HIGH-VALUE LEVER: Duration Log-Ratio Normalization:
     * Use log(1 + play_time_ms) - log(1 + duration_ms) to capture proportional completion without raw duration scale bias.
   - HIGH-VALUE LEVER: Feature Selection via Dropping Redundant Columns:
     * Dropping redundant static demographic columns reduces embedding table size and memory overhead, accelerating training epochs without hurting within-user ranking.
"""

