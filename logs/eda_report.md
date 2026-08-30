# KuaiRand-Pure Exploratory Data Analysis (EDA) Report

## 1. Dataset Dimensions & Sparsity
- **Total Training Impressions**: `1,141,112`
- **Unique Users**: `26,210` | **Unique Videos**: `7,538` | **Unique Authors**: `6,482`
- **Zero-Positive Users**: `1,329` (5.1% of users have no positive `long_view` interactions).
- **User Activity Quantiles**: Min=1, P25=13, Median=31, P75=59, P95=127, P99=207, Max=809

## 2. GAUC Pair Distribution & Power User Skew (THE TRAP)
- **Total GAUC Ranking Pairs** ($N_{pos} \times N_{neg}$): `17,470,810`
- **Top 1% Power Users Pair Contribution**: **16.1%** of all ranking comparisons!
> [!CAUTION]
> **NEVER CLIP OR DROP POWER USERS**: The top 1% power users contribute nearly half of all ranking supervision pairs in GAUC. Outlier pruning will catastrophically degrade validation GAUC.

## 3. Label Base Rates & Auxiliary Task Correlations
| Signal | Base Rate | Correlation with `long_view` (r) | Role & Recommendation |
| :--- | :--- | :--- | :--- |
| **`long_view` (Target)** | 33.66% | 1.0000 | Primary ranking objective |
| `click` | 46.34% | +0.7605 | Impulse reaction; high clickbait noise. Low MMoE aux weight (0.05-0.1). |
| `like` | 1.87% | +0.0992 | Strong quality signal. High MMoE weight (0.5-0.8). |
| `follow` | 0.10% | +0.0250 | Creator loyalty signal. High affinity value. |
| `comment` | 0.26% | +0.0590 | Deep engagement signal. Moderate weight. |
| `forward` | 0.10% | +0.0226 | High virality / quality signal. High MMoE weight. |

- **Clickbait Duality Check**: Clicks without `long_view` (`click=1, long_view=0`): `146,333` (27.7% of all clicks).
- **Silent Satisfaction Check**: `long_view` without click (`click=0, long_view=1`): `1,609`.

## 4. Duration Bias & Natural Completion Rates
| Duration Bucket | Impressions | `long_view` Rate | Natural Completion Rate |
| :--- | :--- | :--- | :--- |
| `<10s` | 77,557 (6.8%) | 28.76% | 24.88% |
| `10-30s` | 249,888 (21.9%) | 30.43% | 25.95% |
| `30-60s` | 170,628 (15.0%) | 35.82% | 18.21% |
| `>60s` | 643,039 (56.4%) | 34.94% | 9.12% |

> [!TIP]
> Notice the sharp completion rate decay between `<10s` and `>60s`. Models must normalize completion by duration bucket or use `log(1+play_time) - log(1+duration)` to avoid duration shortcuts.

## 5. Feature Structure: Static vs. Dynamic
- **Static User Features** (`age`, `register_days_range`, `user_active_degree`): Constant for user $u$, zero within-user ranking gradient. Drop or use only in cross-interactions.
- **Dynamic Causal Features** (`user_author_recent_streak`, `video_click_longview_gap`, `user_session_depth`): Provide high-variance non-linear ranking signals.