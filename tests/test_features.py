"""Feature governance: manifests, recipes, and leakage.

Ported from the `feature-engineering` branch, plus the regression that motivated
integrating it. Everything here runs without an API key or a training run.
"""
import numpy as np
import pytest

from pipeline.feature_agent import FEATURE_MANIFESTS, FeatureEngineeringAgent
from pipeline.feature_recipes import FeatureRecipe, deterministic_recipes
from pipeline.features import (CausalStats, extract_dense_tabular_features,
                               select_rank_features)


def _synthetic():
    rows = []
    for d, date in enumerate(('20220408', '20220409', '20220410')):
        for i in range(6):
            rows.append({
                'date': date, 'user_id': f'u{i % 3}', 'video_id': f'v{i % 4}',
                'author_id': f'a{i % 2}', 'tab': '1',
                'duration_ms': 10000.0 + 1000 * i, 'play_time_ms': 5000.0 + 100 * i,
                'label': (i + d) % 2, 'click': i % 2, 'like': 0, 'follow': 0,
                'comment': 0, 'forward': 0, 'is_rand': 0, 'v_extra': [], 'u_extra': [],
            })
    return {'train': rows, 'valid': [dict(r) for r in rows[:6]]}


def test_full_features_are_strictly_historical():
    splits = _synthetic()
    before, names = extract_dense_tabular_features(splits, profile='full')
    changed = {'train': [dict(r) for r in splits['train']], 'valid': splits['valid']}
    changed['train'][4]['play_time_ms'] = 999999.0
    after, _ = extract_dense_tabular_features(changed, profile='full')
    assert np.array_equal(before['train'][0][4], after['train'][0][4])
    assert 'video_completion_logratio' in names


def test_rank_feature_selector_drops_user_constants():
    X = np.asarray([[1, 1], [1, 2], [3, 4], [3, 5]], dtype=np.float32)
    dense = {'train': (X, np.zeros(4), ['a', 'a', 'b', 'b']),
             'valid': (X.copy(), np.zeros(4), ['a', 'a', 'b', 'b'])}
    selected, names = select_rank_features(dense, ['user_constant', 'candidate_signal'])
    assert names == ['candidate_signal']
    assert selected['train'][0].shape == (4, 1)


def test_feature_manifest_covers_every_dense_feature():
    from pipeline.feature_agent import FeatureEngineeringAgent
    report = FeatureEngineeringAgent().static_audit()
    assert report['status'] == 'PASS'
    assert report['missing_manifests'] == []


def test_feature_recipe_hash_ignores_display_name_and_detects_behavior():
    from pipeline.feature_recipes import FeatureRecipe
    a = FeatureRecipe(name='first')
    b = FeatureRecipe(name='renamed')
    c = FeatureRecipe(name='first', cross_smoothing=9)
    assert a.recipe_id == b.recipe_id
    assert a.recipe_id != c.recipe_id


def test_feature_recipe_rejects_unknown_features_at_compile_time():
    from pipeline.feature_recipes import FeatureRecipe
    recipe = FeatureRecipe(include_features=['not_a_real_feature'])
    with pytest.raises(ValueError, match='unknown feature'):
        extract_dense_tabular_features(_synthetic(), recipe=recipe)


def test_the_dynamic_audit_catches_a_leak_the_static_scanner_cannot():
    """The reason this agent was integrated at all.

    `sandbox/verifier.py` matches tokens, so it catches `row['play_time_ms']`
    but not the same value reached through a helper with a computed key name.
    The audit tests information flow instead: mutate a row's own outcome
    columns, recompute, and require that row's feature vector not to move.

    This matters because `long_view` is close to a deterministic function of
    play_time/duration, and `features.py` is on the agent's mutable surface — so
    a leak here scores brilliantly on validation and is worthless on the hidden
    test split, where watch time is withheld.
    """
    from sandbox.verifier import verify_source

    src = open("pipeline/features.py", encoding="utf-8").read()
    leaky = src.replace("BASE_FIELDS = [", "_K = 'play' + '_time_ms'\nBASE_FIELDS = [", 1)
    leaky = leaky.replace(
        "    def featurise(self, row: dict) -> List[float]:",
        "    def featurise(self, row: dict) -> List[float]:\n"
        "        out = self._inner(row)\n"
        "        r = float(row.get(_K, 0.0)) / max(float(row.get('duration_ms', 1.0)), 1.0)\n"
        "        return [v + r for v in out]\n\n"
        "    def _inner(self, row: dict) -> List[float]:", 1)
    assert "_inner" in leaky, "the fixture failed to patch featurise"

    # The static scanner sees nothing: the literal key never appears.
    assert verify_source("pipeline/features.py", leaky) == []

    # The mutation test sees it, by running the code.
    ns: dict = {}
    exec(compile(leaky, "features_leaky.py", "exec"), ns)
    splits = _synthetic()
    before, _ = ns["extract_dense_tabular_features"](splits, profile="full")
    changed = {"train": [dict(r) for r in splits["train"]], "valid": splits["valid"]}
    changed["train"][4]["play_time_ms"] = 999999.0
    after, _ = ns["extract_dense_tabular_features"](changed, profile="full")
    assert not np.array_equal(before["train"][0][4], after["train"][0][4]), (
        "a feature derived from the row's own watch time must be detectable")


def test_the_shipped_pipeline_passes_its_own_audits():
    """A gate that flags the baseline would block every legitimate change."""
    report = FeatureEngineeringAgent().static_audit()
    assert report["status"] == "PASS", report
    assert report["feature_count"] == len(CausalStats.FEATURE_NAMES)

    for recipe in deterministic_recipes():
        audit = FeatureEngineeringAgent().recipe_audit(recipe)
        assert audit["status"] == "PASS", (recipe.name, audit)
        assert audit["selected_features"], recipe.name


def test_the_auditor_and_the_recipe_schema_are_immutable():
    """A checker the agent can edit is not a check.

    If generated code could weaken FORBIDDEN_CURRENT_ROW_SOURCES or relax the
    recipe's validated ranges, the leak gate would pass whatever it was asked to.
    """
    from sandbox.workspace import IMMUTABLE, MUTABLE

    for path in ("pipeline/feature_agent.py", "pipeline/feature_recipes.py"):
        assert path in IMMUTABLE, f"{path} must not be agent-editable"
        assert path not in MUTABLE


def test_a_recipe_is_reproducible_and_bounded():
    """Recipes are a third action class: validated ranges, hashed identity."""
    a = FeatureRecipe(name="alpha", base_profile="core")
    assert a.recipe_id == FeatureRecipe(name="beta", base_profile="core").recipe_id, (
        "renaming a recipe must not change what it does")
    assert a.recipe_id != FeatureRecipe(name="alpha", base_profile="full").recipe_id

    with pytest.raises(Exception):
        FeatureRecipe(item_smoothing=9999)      # out of range
    with pytest.raises(Exception):
        FeatureRecipe(exclude_features=["nope", "nope"])   # duplicates

    ids = [r.recipe_id for r in deterministic_recipes()]
    assert len(ids) == len(set(ids)), "the fallback recipes must all differ"


def test_the_steward_never_repeats_a_recipe_and_costs_nothing():
    from agents.feature_steward import FeatureStewardAgent
    from agents.context import ResearchContext

    steward = FeatureStewardAgent(verbose=False)
    ctx = ResearchContext(baseline=0.6015, max_iterations=10, wall_clock_budget_s=600)
    seen = []
    for i in range(1, 7):
        ctx.iteration = i
        r = steward.fallback(ctx, tried_ids=seen)
        assert r.recipe_id not in seen, f"recipe {r.recipe_id} proposed twice"
        ok, why = steward.audit_recipe(r)
        assert ok, why
        seen.append(r.recipe_id)


def test_a_dense_model_reaches_the_causal_features():
    """`dense_deepfm` is what makes recipes matter beyond LightGBM.

    The causal dense features were previously reachable only through
    `--model lgb`, which has never been the best model — so the Feature
    Steward's whole search space was confined to a branch that could not win.
    """
    from pipeline.models import MODELS, resolve_model

    assert "dense_deepfm" in MODELS
    builder = resolve_model("dense_deepfm")
    assert builder.needs_dense is True
    assert builder.needs_history is False

    # Every other architecture must be unaffected.
    for name in ("fm_torch", "deepfm", "din"):
        assert resolve_model(name).needs_dense is False
    assert resolve_model("din").needs_history is True

    # The builder refuses to be constructed without features rather than
    # silently building a zero-width dense tower.
    with pytest.raises(ValueError):
        builder(1000, 5, 16, 999, num_dense=0)

    torch = pytest.importorskip("torch")
    model = builder(1000, 5, 16, 999, num_dense=15)
    logits = model(torch.zeros(4, 5, dtype=torch.long), torch.zeros(4, 15))
    assert logits.shape == (4,), "a ranking model must emit one logit per row"


def test_a_dense_checkpoint_records_what_it_needs_to_be_rebuilt():
    """Inference must reconstruct the exact feature set training used.

    A dense model restored against a different recipe has the wrong first-layer
    width — and if the widths happen to agree, it silently scores the wrong
    columns, which is worse than failing.
    """
    import inspect

    import pipeline.submit as submit
    import pipeline.train as train

    meta_src = inspect.getsource(train.train_torch)
    assert '"num_dense": num_dense' in meta_src
    assert '"feature_recipe"' in meta_src

    rebuild = inspect.getsource(submit.predict_split)
    assert 'num_dense=meta.get("num_dense", 0)' in rebuild
    assert "trained_dim" in rebuild, "a width mismatch must be detected, not ignored"

    # train and submit must agree on where checkpoints live.
    assert train.CHECKPOINTS_DIR == submit.CHECKPOINTS_DIR
