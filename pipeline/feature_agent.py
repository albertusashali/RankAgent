"""Feature governance and leakage auditing for RankAgent.

This agent does not invent arbitrary Python and execute it. It governs a typed
feature manifest, runs deterministic safety checks, and emits a machine-readable
report that the research orchestrator and human reviewers can audit.
"""
import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Sequence

import numpy as np

from pipeline.features import CausalStats, extract_dense_tabular_features
from pipeline.feature_recipes import FeatureRecipe, load_recipe

FORBIDDEN_CURRENT_ROW_SOURCES = {
    'label', 'long_view', 'play_time_ms', 'click', 'like', 'follow',
    'comment', 'forward', 'is_profile_enter',
}


@dataclass(frozen=True)
class FeatureManifest:
    name: str
    family: str
    entity_keys: Sequence[str]
    source_columns: Sequence[str]
    availability: str
    window_days: int = 0
    smoothing: float = 0.0
    fallback: str = 'zero'
    expected_mechanism: str = ''


def _manifest(name, family, keys, sources, mechanism, window=0, smoothing=0.0):
    return FeatureManifest(name, family, keys, sources, 'historical_only', window,
                           smoothing, 'smoothed prior', mechanism)


FEATURE_MANIFESTS: List[FeatureManifest] = [
    _manifest(n, 'core', ['video_id'], ['label', 'click', 'date'],
              'historical item quality or exposure support', smoothing=15.0)
    for n in CausalStats.CORE_FEATURES[:5]
] + [
    _manifest(n, 'core', ['user_id'], ['label', 'duration_ms', 'date'],
              'conditions heterogeneous user behaviour', smoothing=15.0)
    for n in CausalStats.CORE_FEATURES[5:]
] + [
    _manifest(n, 'affinity', ['user_id', 'candidate entity'], ['label', 'date'],
              'candidate-varying personalized preference', smoothing=8.0)
    for n in CausalStats.AFFINITY_FEATURES
] + [
    _manifest(n, 'temporal', ['user/item/author'],
              ['play_time_ms', 'duration_ms', 'label', 'date'],
              'recent interest, duration-debiased history, or preference momentum',
              window=(3 if '_3d' in n else 7 if '_7d' in n else 0), smoothing=8.0)
    for n in CausalStats.TEMPORAL_FEATURES
]


class FeatureEngineeringAgent:
    """Audits feature definitions before an expensive training iteration."""

    def static_audit(self) -> dict:
        declared = [m.name for m in FEATURE_MANIFESTS]
        produced = list(CausalStats.FEATURE_NAMES)
        duplicate = sorted({n for n in declared if declared.count(n) > 1})
        missing = sorted(set(produced) - set(declared))
        stale = sorted(set(declared) - set(produced))
        unsafe = sorted(m.name for m in FEATURE_MANIFESTS
                        if FORBIDDEN_CURRENT_ROW_SOURCES.intersection(m.source_columns)
                        and m.availability != 'historical_only')
        return {
            'status': 'PASS' if not (duplicate or missing or stale or unsafe) else 'FAIL',
            'feature_count': len(produced), 'duplicates': duplicate,
            'missing_manifests': missing, 'stale_manifests': stale,
            'unsafe_current_row_features': unsafe,
            'forbidden_current_row_sources': sorted(FORBIDDEN_CURRENT_ROW_SOURCES),
        }

    def recipe_audit(self, recipe: FeatureRecipe) -> dict:
        available = set(CausalStats.FEATURE_NAMES)
        requested = set(recipe.include_features or []) | set(recipe.exclude_features)
        unknown = sorted(requested - available)
        selected = list(recipe.include_features or {
            'core': CausalStats.CORE_FEATURES,
            'affinity': CausalStats.CORE_FEATURES + CausalStats.AFFINITY_FEATURES,
            'full': CausalStats.FEATURE_NAMES,
        }[recipe.base_profile])
        selected = [n for n in selected if n not in recipe.exclude_features]
        return {'status': 'PASS' if not unknown and selected else 'FAIL',
                'recipe_id': recipe.recipe_id, 'recipe_name': recipe.name,
                'unknown_features': unknown, 'selected_features': selected}

    def dynamic_audit(self, splits: dict, recipe: FeatureRecipe = None) -> dict:
        """Verify current-row outcomes cannot affect that row's feature vector."""
        recipe = recipe or FeatureRecipe(base_profile='full')
        original, names = extract_dense_tabular_features(splits, recipe=recipe)
        changed = {k: [dict(r) for r in rows] for k, rows in splits.items()}
        probe = next((i for i, r in enumerate(changed['train'])
                      if r['date'] != changed['train'][0]['date']), 0)
        for source in FORBIDDEN_CURRENT_ROW_SOURCES:
            if source in changed['train'][probe]:
                changed['train'][probe][source] = 1e9 if source == 'play_time_ms' else 1
        mutated, _ = extract_dense_tabular_features(changed, recipe=recipe)
        same = np.array_equal(original['train'][0][probe], mutated['train'][0][probe])
        finite = all(np.isfinite(parts[0]).all() for parts in original.values())
        return {'status': 'PASS' if same and finite else 'FAIL',
                'current_row_outcome_invariant': bool(same),
                'all_values_finite': bool(finite), 'features_checked': names}

    def report(self, splits=None, recipe: FeatureRecipe = None) -> dict:
        recipe = recipe or FeatureRecipe(base_profile='full')
        report = {'static_audit': self.static_audit(),
                  'recipe_audit': self.recipe_audit(recipe),
                  'manifests': [asdict(m) for m in FEATURE_MANIFESTS]}
        if splits is not None:
            report['dynamic_audit'] = self.dynamic_audit(splits, recipe)
        statuses = [v['status'] for k, v in report.items() if k.endswith('_audit')]
        report['status'] = 'PASS' if all(s == 'PASS' for s in statuses) else 'FAIL'
        return report


def main(argv=None):
    p = argparse.ArgumentParser(description='Audit RankAgent feature manifests and leakage rules')
    p.add_argument('--data_dir', default=None)
    p.add_argument('--dynamic', action='store_true', help='also run outcome-mutation checks')
    p.add_argument('--recipe', default=None, help='optional FeatureRecipe JSON')
    p.add_argument('--output', default=os.path.join('logs', 'feature_audit.json'))
    args = p.parse_args(argv)
    splits = None
    if args.dynamic:
        from pipeline.data import load_kuairand
        full = load_kuairand(args.data_dir)
        # Dynamic invariance needs at least two dates, not the full million rows.
        dates = sorted({r['date'] for r in full['train']})[:3]
        train = []
        per_date = {d: 0 for d in dates}
        for row in full['train']:
            if row['date'] in per_date and per_date[row['date']] < 2500:
                train.append(row)
                per_date[row['date']] += 1
            if all(n >= 2500 for n in per_date.values()):
                break
        splits = {'train': train, 'valid': full['valid'][:5000]}
    recipe = load_recipe(args.recipe, base_profile='full')
    report = FeatureEngineeringAgent().report(splits, recipe)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print(f"[FEATURE AUDIT] {report['status']} -> {args.output}")
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
