"""
Domain Knowledge Base and Playbook for Recommender Systems on KuaiRand-Pure.
"""

RECSYS_KB = """
### SHORT VIDEO RECOMMENDATION PLAYBOOK (KuaiRand Focus)

1. TARGET VARIABLE & METRICS:
   - Primary Label: `long_view` (Binary classification: 1 if user watched the video for long duration / completed, 0 otherwise).
   - Metrics: GAUC (User-weighted AUC excluding 0-pos and all-pos users) & nDCG@5 (Discounted gain: 2^rel - 1).
   - Official FM Baseline: GAUC 0.6674, nDCG@5 0.5357 -> Primary Score: 0.6016 (val) / 0.5946 (test).
   - Theoretical Ceiling: 0.8645 (Not 1.0, because 27.1% of test users have no positive labels).

2. KUAIRAND DATASETS & FEATURES:
   - Base features: user_id, video_id, author_id, tab, dur_bucket (5 fields).
   - Expanded features: user_features_pure.csv (5 user fields: follow_user_num_range, register_days_range, fans_user_num_range, friend_user_num_range, user_active_degree) + video_features_basic_pure.csv (music_id, video_type, upload_type) = 13 feature domains!
   - Target aggregations: Historical user click/long_view rate, item long_view count, smooth Bayesian target encoding.

3. MULTI-TASK & AUXILIARY SIGNALS:
   - KuaiRand logs 12 feedback signals: `click`, `like`, `follow`, `comment`, `forward`, `play_time`, `is_profile_enter`, `is_rand`, etc.
   - Use Multi-Task Learning (MMoE, PLE, Shared-Bottom) to jointly predict auxiliary signals (`click`, `like`, `comment`) and `long_view` to overcome label sparsity.

4. DURATION BIAS & WATCH TIME (CWM):
   - Raw `play_time` is heavily biased by video duration.
   - Implement counterfactual watch-time modeling or censored regression (Zhao et al., KDD 2024 CWM) to de-bias continuous watch time before using it as an auxiliary feature/target.

5. SEQUENTIAL & ATTENTION MODELING (DIN / SASRec):
   - Incorporate the sequence of the user's last 5-10 watched video_ids.
   - Apply target attention (Deep Interest Network) between the candidate video and user historical sequence.

6. MODEL ARCHITECTURES & ENSEMBLING:
   - Factorization Machines (FM), DeepFM, DCN-v2 (Deep & Cross Network).
   - GBDT Ranker (LightGBM LambdaMART) for tabular historical features.
   - Final Ensemble: Rank-normalized weighted blend of GBDT + Deep Multi-Task MMoE.
"""

