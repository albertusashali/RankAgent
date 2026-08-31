"""Validated, reproducible search space for causal feature engineering."""
import hashlib
import json
import os
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class FeatureRecipe(BaseModel):
    """A bounded feature mutation the AI may propose without editing Python."""

    name: str = "affinity_default"
    base_profile: Literal['core', 'affinity', 'full'] = 'affinity'
    include_features: Optional[List[str]] = None
    exclude_features: List[str] = Field(default_factory=list)
    item_smoothing: float = Field(15.0, ge=1.0, le=100.0)
    cross_smoothing: float = Field(8.0, ge=1.0, le=50.0)
    global_prior_strength: float = Field(50.0, ge=5.0, le=500.0)
    completion_ratio_clip: float = Field(5.0, ge=1.0, le=10.0)
    recency_cap_days: int = Field(30, ge=7, le=90)
    use_rank_selection: bool = True
    min_within_user_variance: float = Field(1e-10, ge=0.0, le=1e-3)

    @field_validator('name')
    @classmethod
    def safe_name(cls, value: str) -> str:
        cleaned = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in value)
        if not cleaned:
            raise ValueError('recipe name cannot be empty')
        return cleaned[:64]

    @field_validator('include_features', 'exclude_features')
    @classmethod
    def unique_features(cls, value):
        if value is None:
            return value
        if len(value) != len(set(value)):
            raise ValueError('feature names must be unique')
        return value

    def canonical(self) -> str:
        payload = self.model_dump(exclude={'name'})
        return json.dumps(payload, sort_keys=True, separators=(',', ':'))

    @property
    def recipe_id(self) -> str:
        return hashlib.sha256(self.canonical().encode('utf-8')).hexdigest()[:10]

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = self.model_dump()
        payload['recipe_id'] = self.recipe_id
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2)
        return path


def load_recipe(path: Optional[str] = None, base_profile: str = 'affinity',
                select_features: bool = False) -> FeatureRecipe:
    if path is None:
        return FeatureRecipe(base_profile=base_profile,
                             use_rank_selection=select_features)
    with open(path, encoding='utf-8') as fh:
        payload = json.load(fh)
    payload.pop('recipe_id', None)
    return FeatureRecipe.model_validate(payload)


def deterministic_recipes() -> List[FeatureRecipe]:
    """Diverse bounded mutations used when no external LLM is configured."""
    return [
        FeatureRecipe(name='affinity_balanced', base_profile='affinity'),
        FeatureRecipe(name='full_conservative', base_profile='full',
                      item_smoothing=30, cross_smoothing=16,
                      completion_ratio_clip=3, recency_cap_days=30),
        FeatureRecipe(name='full_responsive', base_profile='full',
                      item_smoothing=8, cross_smoothing=4,
                      global_prior_strength=25, recency_cap_days=14),
        FeatureRecipe(name='affinity_no_user_constants', base_profile='affinity',
                      exclude_features=['user_hist_count', 'user_hist_long_view_rate'],
                      cross_smoothing=12),
        FeatureRecipe(name='temporal_robust', base_profile='full',
                      exclude_features=['video_completion_logratio',
                                        'user_author_completion_logratio'],
                      item_smoothing=25, cross_smoothing=20,
                      recency_cap_days=14),
    ]
