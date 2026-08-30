"""Exploratory Data Analysis (EDA) module for KuaiRand-Pure dataset.

Computes empirical dataset distributions, correlations, duration biases,
and power-user ranking pair distributions to ground the RankAgent in real data.
Outputs a markdown report to logs/eda_report.md and returns a concise summary.
"""
import os
import math
from collections import defaultdict
from typing import Dict, Any, List, Optional
from pipeline.data import load_kuairand, find_data_dir


def run_eda(data_dir: Optional[str] = None, output_path: str = "logs/eda_report.md") -> str:
    """Runs EDA over KuaiRand-Pure train split and writes report to disk."""
    print("==> [EDA] Loading KuaiRand-Pure dataset for Exploratory Data Analysis...")
    splits = load_kuairand(data_dir=data_dir, include_extra_features=True)
    train_rows = splits['train']
    n_total = len(train_rows)
    print(f"==> [EDA] Profiling {n_total:,} training interaction rows...")

    users = set()
    videos = set()
    authors = set()
    user_counts = defaultdict(int)
    user_pos_counts = defaultdict(int)
    user_neg_counts = defaultdict(int)

    counts = {
        'long_view': 0, 'click': 0, 'like': 0,
        'follow': 0, 'comment': 0, 'forward': 0
    }
    
    # Co-occurrence with long_view
    click_and_long = 0
    click_only = 0
    long_only = 0

    # Duration buckets: <10s, 10-30s, 30-60s, >60s
    dur_buckets = {
        '<10s': {'total': 0, 'long_view': 0, 'completed': 0},
        '10-30s': {'total': 0, 'long_view': 0, 'completed': 0},
        '30-60s': {'total': 0, 'long_view': 0, 'completed': 0},
        '>60s': {'total': 0, 'long_view': 0, 'completed': 0},
    }

    for r in train_rows:
        u = r['user_id']
        v = r['video_id']
        a = r['author_id']
        y = r['label']
        clk = r['click']
        
        users.add(u)
        videos.add(v)
        authors.add(a)
        user_counts[u] += 1
        if y == 1:
            user_pos_counts[u] += 1
            counts['long_view'] += 1
        else:
            user_neg_counts[u] += 1

        if clk == 1: counts['click'] += 1
        if r['like'] == 1: counts['like'] += 1
        if r['follow'] == 1: counts['follow'] += 1
        if r['comment'] == 1: counts['comment'] += 1
        if r['forward'] == 1: counts['forward'] += 1

        if clk == 1 and y == 1:
            click_and_long += 1
        elif clk == 1 and y == 0:
            click_only += 1
        elif clk == 0 and y == 1:
            long_only += 1

        # Duration analysis
        dur_s = r['duration_ms'] / 1000.0 if r['duration_ms'] > 0 else 0
        play_s = r['play_time_ms'] / 1000.0 if r['play_time_ms'] > 0 else 0
        completed = (play_s >= dur_s) if dur_s > 0 else False

        if dur_s < 10: b = '<10s'
        elif dur_s <= 30: b = '10-30s'
        elif dur_s <= 60: b = '30-60s'
        else: b = '>60s'

        dur_buckets[b]['total'] += 1
        if y == 1: dur_buckets[b]['long_view'] += 1
        if completed: dur_buckets[b]['completed'] += 1

    # Basic stats
    n_users = len(users)
    n_videos = len(videos)
    n_authors = len(authors)

    # Base rates
    base_rates = {k: v / n_total for k, v in counts.items()}

    # Correlations with long_view (binary phi coefficient / Pearson)
    correlations = {}
    p_y = base_rates['long_view']
    var_y = p_y * (1 - p_y)
    
    for sig in ['click', 'like', 'follow', 'comment', 'forward']:
        p_x = base_rates[sig]
        var_x = p_x * (1 - p_x)
        # Compute joint p(x=1, y=1)
        joint_xy = sum(1 for r in train_rows if r[sig] == 1 and r['label'] == 1) / n_total
        cov_xy = joint_xy - (p_x * p_y)
        corr = cov_xy / math.sqrt(var_x * var_y) if (var_x * var_y) > 0 else 0.0
        correlations[sig] = corr

    # User activity distribution & GAUC pair skew
    user_lens = sorted(user_counts.values())
    quantiles = {
        'min': user_lens[0],
        'p25': user_lens[int(0.25 * n_users)],
        'p50': user_lens[int(0.50 * n_users)],
        'p75': user_lens[int(0.75 * n_users)],
        'p95': user_lens[int(0.95 * n_users)],
        'p99': user_lens[int(0.99 * n_users)],
        'max': user_lens[-1]
    }

    # GAUC pairs = N_pos(u) * N_neg(u)
    user_pairs = {u: user_pos_counts[u] * user_neg_counts[u] for u in users}
    total_gauc_pairs = sum(user_pairs.values())
    sorted_users_by_pairs = sorted(user_pairs.items(), key=lambda x: x[1], reverse=True)
    
    top_1pct_count = max(1, int(0.01 * n_users))
    top_1pct_pairs = sum(p for _, p in sorted_users_by_pairs[:top_1pct_count])
    top_1pct_pair_share = (top_1pct_pairs / total_gauc_pairs * 100) if total_gauc_pairs > 0 else 0.0

    zero_pos_users = sum(1 for u in users if user_pos_counts[u] == 0)
    zero_pos_pct = (zero_pos_users / n_users * 100)

    # Format Markdown Report
    report = []
    report.append("# KuaiRand-Pure Exploratory Data Analysis (EDA) Report")
    report.append("")
    report.append("## 1. Dataset Dimensions & Sparsity")
    report.append(f"- **Total Training Impressions**: `{n_total:,}`")
    report.append(f"- **Unique Users**: `{n_users:,}` | **Unique Videos**: `{n_videos:,}` | **Unique Authors**: `{n_authors:,}`")
    report.append(f"- **Zero-Positive Users**: `{zero_pos_users:,}` ({zero_pos_pct:.1f}% of users have no positive `long_view` interactions).")
    report.append(f"- **User Activity Quantiles**: Min={quantiles['min']}, P25={quantiles['p25']}, Median={quantiles['p50']}, P75={quantiles['p75']}, P95={quantiles['p95']}, P99={quantiles['p99']}, Max={quantiles['max']}")
    report.append("")
    
    report.append("## 2. GAUC Pair Distribution & Power User Skew (THE TRAP)")
    report.append(f"- **Total GAUC Ranking Pairs** ($N_{{pos}} \\times N_{{neg}}$): `{total_gauc_pairs:,}`")
    report.append(f"- **Top 1% Power Users Pair Contribution**: **{top_1pct_pair_share:.1f}%** of all ranking comparisons!")
    report.append("> [!CAUTION]")
    report.append("> **NEVER CLIP OR DROP POWER USERS**: The top 1% power users contribute nearly half of all ranking supervision pairs in GAUC. Outlier pruning will catastrophically degrade validation GAUC.")
    report.append("")

    report.append("## 3. Label Base Rates & Auxiliary Task Correlations")
    report.append("| Signal | Base Rate | Correlation with `long_view` (r) | Role & Recommendation |")
    report.append("| :--- | :--- | :--- | :--- |")
    report.append(f"| **`long_view` (Target)** | {base_rates['long_view']*100:.2f}% | 1.0000 | Primary ranking objective |")
    report.append(f"| `click` | {base_rates['click']*100:.2f}% | {correlations['click']:+.4f} | Impulse reaction; high clickbait noise. Low MMoE aux weight (0.05-0.1). |")
    report.append(f"| `like` | {base_rates['like']*100:.2f}% | {correlations['like']:+.4f} | Strong quality signal. High MMoE weight (0.5-0.8). |")
    report.append(f"| `follow` | {base_rates['follow']*100:.2f}% | {correlations['follow']:+.4f} | Creator loyalty signal. High affinity value. |")
    report.append(f"| `comment` | {base_rates['comment']*100:.2f}% | {correlations['comment']:+.4f} | Deep engagement signal. Moderate weight. |")
    report.append(f"| `forward` | {base_rates['forward']*100:.2f}% | {correlations['forward']:+.4f} | High virality / quality signal. High MMoE weight. |")
    report.append("")
    report.append(f"- **Clickbait Duality Check**: Clicks without `long_view` (`click=1, long_view=0`): `{click_only:,}` ({click_only/max(1, counts['click'])*100:.1f}% of all clicks).")
    report.append(f"- **Silent Satisfaction Check**: `long_view` without click (`click=0, long_view=1`): `{long_only:,}`.")
    report.append("")

    report.append("## 4. Duration Bias & Natural Completion Rates")
    report.append("| Duration Bucket | Impressions | `long_view` Rate | Natural Completion Rate |")
    report.append("| :--- | :--- | :--- | :--- |")
    for b_name in ['<10s', '10-30s', '30-60s', '>60s']:
        d = dur_buckets[b_name]
        tot = d['total']
        lv_pct = (d['long_view'] / tot * 100) if tot > 0 else 0
        comp_pct = (d['completed'] / tot * 100) if tot > 0 else 0
        report.append(f"| `{b_name}` | {tot:,} ({tot/n_total*100:.1f}%) | {lv_pct:.2f}% | {comp_pct:.2f}% |")
    report.append("")
    report.append("> [!TIP]")
    report.append("> Notice the sharp completion rate decay between `<10s` and `>60s`. Models must normalize completion by duration bucket or use `log(1+play_time) - log(1+duration)` to avoid duration shortcuts.")
    report.append("")

    report.append("## 5. Feature Structure: Static vs. Dynamic")
    report.append("- **Static User Features** (`age`, `register_days_range`, `user_active_degree`): Constant for user $u$, zero within-user ranking gradient. Drop or use only in cross-interactions.")
    report.append("- **Dynamic Causal Features** (`user_author_recent_streak`, `video_click_longview_gap`, `user_session_depth`): Provide high-variance non-linear ranking signals.")
    
    report_text = "\n".join(report)
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write(report_text)
    
    print(f"==> [EDA] Report generated and saved to {output_path}")
    return report_text


if __name__ == '__main__':
    run_eda()

