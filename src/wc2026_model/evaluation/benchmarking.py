from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from wc2026_model.features import (
    CONFEDERATIONS,
    RecentFormConfig,
    WorldCupXGConfig,
    attach_confederation_features,
    augment_with_pre_match_form_features,
    augment_with_pre_match_h2h_features,
    augment_with_pre_match_xg_features,
)
from wc2026_model.types import (
    OUTCOME_AWAY,
    OUTCOME_DRAW,
    OUTCOME_HOME,
    ThreeWayProbabilities,
)

# All selectable xG context feature groups (used to validate explicit selections).
XG_CONTEXT_FEATURE_GROUPS = (
    "shot_accuracy",
    "confederation",
    "h2h",
    "h2h_decayed",
    "pass_completion",
    "pressures",
)

# Legacy `include_context_features=True` bundle. Pinned to the raw groups so the
# historical benchmark stays reproducible; h2h_decayed is opt-in only.
XG_CONTEXT_LEGACY_BUNDLE = ("shot_accuracy", "confederation", "h2h")

# Recency-weighted h2h shares the raw h2h augmentation pass, which emits both the
# raw `h2h_*` counts and the decayed `h2h_decayed_*` columns in a single sweep.
_H2H_AUGMENTED_GROUPS = ("h2h", "h2h_decayed")


@dataclass
class EloMultinomialBenchmark:
    classifier: LogisticRegression

    @classmethod
    def fit(cls, matches: pd.DataFrame) -> "EloMultinomialBenchmark":
        feature_frame = _benchmark_features(matches)
        classifier = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        )
        classifier.fit(
            feature_frame,
            matches["home_result"],
            sample_weight=matches["sample_weight"],
        )
        return cls(classifier=classifier)

    def predict_proba(self, row: pd.Series | pd.NamedAgg | object) -> ThreeWayProbabilities:
        if isinstance(row, pd.Series):
            feature_frame = _benchmark_features(pd.DataFrame([row]))
        else:
            feature_frame = pd.DataFrame(
                [
                    {
                        "elo_diff_scaled": float(row.elo_diff_pre) / 400.0,
                        "is_non_neutral": 0.0 if bool(row.neutral) else 1.0,
                        "abs_elo_diff_scaled": abs(float(row.elo_diff_pre)) / 400.0,
                    }
                ]
            )
        predicted = self.classifier.predict_proba(feature_frame)[0]
        probability_map = dict(zip(self.classifier.classes_, predicted, strict=True))
        return ThreeWayProbabilities(
            home=float(probability_map.get(OUTCOME_HOME, 0.0)),
            draw=float(probability_map.get(OUTCOME_DRAW, 0.0)),
            away=float(probability_map.get(OUTCOME_AWAY, 0.0)),
        )


@dataclass
class FormEloMultinomialBenchmark:
    classifier: LogisticRegression
    feature_fill_values: dict[str, float]
    form_config: RecentFormConfig

    @classmethod
    def fit(
        cls,
        matches: pd.DataFrame,
        *,
        form_config: RecentFormConfig | None = None,
    ) -> "FormEloMultinomialBenchmark":
        form_config = form_config or RecentFormConfig()
        augmented = matches
        required_columns = {
            "home_form_points_per_match",
            "away_form_points_per_match",
            "home_form_goal_diff_per_match",
            "away_form_goal_diff_per_match",
            "home_days_since_last_match",
            "away_days_since_last_match",
        }
        if not required_columns.issubset(matches.columns):
            augmented = augment_with_pre_match_form_features(matches, config=form_config)

        feature_frame = _form_benchmark_features(augmented, form_config=form_config)
        fill_values = {
            column: float(value)
            for column, value in feature_frame.mean(numeric_only=True).fillna(0.0).items()
        }
        classifier = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        )
        classifier.fit(
            feature_frame.fillna(fill_values),
            augmented["home_result"],
            sample_weight=augmented["sample_weight"],
        )
        return cls(
            classifier=classifier,
            feature_fill_values=fill_values,
            form_config=form_config,
        )

    def predict_proba(self, row: pd.Series | pd.NamedAgg | object) -> ThreeWayProbabilities:
        if isinstance(row, pd.Series):
            feature_frame = _form_benchmark_features(
                pd.DataFrame([row]),
                form_config=self.form_config,
            )
        else:
            feature_frame = pd.DataFrame(
                [
                    {
                        "elo_diff_scaled": float(getattr(row, "elo_diff_pre")) / 400.0,
                        "is_non_neutral": 0.0 if bool(getattr(row, "neutral")) else 1.0,
                        "abs_elo_diff_scaled": abs(float(getattr(row, "elo_diff_pre"))) / 400.0,
                        "form_points_per_match_diff": _float_or_nan(
                            getattr(row, "home_form_points_per_match", np.nan)
                        )
                        - _float_or_nan(getattr(row, "away_form_points_per_match", np.nan)),
                        "form_goal_diff_per_match_diff": _float_or_nan(
                            getattr(row, "home_form_goal_diff_per_match", np.nan)
                        )
                        - _float_or_nan(
                            getattr(row, "away_form_goal_diff_per_match", np.nan)
                        ),
                        "form_goals_for_per_match_diff": _float_or_nan(
                            getattr(row, "home_form_goals_for_per_match", np.nan)
                        )
                        - _float_or_nan(
                            getattr(row, "away_form_goals_for_per_match", np.nan)
                        ),
                        "form_goals_against_per_match_diff": _float_or_nan(
                            getattr(row, "home_form_goals_against_per_match", np.nan)
                        )
                        - _float_or_nan(
                            getattr(row, "away_form_goals_against_per_match", np.nan)
                        ),
                        "form_win_rate_diff": _float_or_nan(
                            getattr(row, "home_form_win_rate", np.nan)
                        )
                        - _float_or_nan(getattr(row, "away_form_win_rate", np.nan)),
                        "form_match_count_diff_scaled": (
                            _float_or_nan(getattr(row, "home_form_match_count", np.nan))
                            - _float_or_nan(getattr(row, "away_form_match_count", np.nan))
                        )
                        / float(self.form_config.window_size),
                        "form_match_count_min_scaled": min(
                            _float_or_nan(getattr(row, "home_form_match_count", np.nan)),
                            _float_or_nan(getattr(row, "away_form_match_count", np.nan)),
                        )
                        / float(self.form_config.window_size),
                        "rest_days_diff_scaled": (
                            _float_or_nan(getattr(row, "home_days_since_last_match", np.nan))
                            - _float_or_nan(getattr(row, "away_days_since_last_match", np.nan))
                        )
                        / 30.0,
                    }
                ]
            )

        predicted = self.classifier.predict_proba(
            feature_frame.fillna(self.feature_fill_values)
        )[0]
        probability_map = dict(zip(self.classifier.classes_, predicted, strict=True))
        return ThreeWayProbabilities(
            home=float(probability_map.get(OUTCOME_HOME, 0.0)),
            draw=float(probability_map.get(OUTCOME_DRAW, 0.0)),
            away=float(probability_map.get(OUTCOME_AWAY, 0.0)),
        )


@dataclass
class XGEloMultinomialBenchmark:
    classifier: LogisticRegression
    feature_fill_values: dict[str, float]
    xg_config: WorldCupXGConfig
    include_context_features: bool
    context_feature_groups: tuple[str, ...]
    feature_columns: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        matches: pd.DataFrame,
        *,
        xg_config: WorldCupXGConfig | None = None,
        include_context_features: bool = False,
        context_feature_groups: tuple[str, ...] | list[str] | set[str] | None = None,
    ) -> "XGEloMultinomialBenchmark":
        xg_config = xg_config or WorldCupXGConfig()
        resolved_context_groups = _resolve_xg_context_feature_groups(
            include_context_features=include_context_features,
            context_feature_groups=context_feature_groups,
        )
        augmented = matches
        required_columns = {
            "home_xg_for_per_match",
            "away_xg_for_per_match",
            "home_xg_against_per_match",
            "away_xg_against_per_match",
            "home_xg_per_shot",
            "away_xg_per_shot",
        }
        if not required_columns.issubset(matches.columns):
            # Raw frame: run the full xG pass, which also emits the optional
            # pass_completion / pressures pre-match columns when source data exists.
            augmented = augment_with_pre_match_xg_features(matches, config=xg_config)
        if "confederation" in resolved_context_groups and not (
            _required_xg_confederation_columns().issubset(augmented.columns)
        ):
            augmented = attach_confederation_features(augmented)
        if any(group in resolved_context_groups for group in _H2H_AUGMENTED_GROUPS) and not (
            _required_xg_context_columns(
                tuple(
                    group
                    for group in resolved_context_groups
                    if group in _H2H_AUGMENTED_GROUPS
                )
            ).issubset(augmented.columns)
        ):
            augmented = augment_with_pre_match_h2h_features(augmented)
        missing_context_columns = _required_xg_context_columns(
            resolved_context_groups
        ).difference(augmented.columns)
        if missing_context_columns:
            missing = ", ".join(sorted(missing_context_columns))
            raise ValueError(f"Missing required xG context columns: {missing}")

        feature_frame = _xg_benchmark_features(
            augmented,
            xg_config=xg_config,
            context_feature_groups=resolved_context_groups,
        )
        fill_values = {
            column: float(value)
            for column, value in feature_frame.mean(numeric_only=True).fillna(0.0).items()
        }
        classifier = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        )
        classifier.fit(
            feature_frame.fillna(fill_values),
            augmented["home_result"],
            sample_weight=augmented["sample_weight"],
        )
        return cls(
            classifier=classifier,
            feature_fill_values=fill_values,
            xg_config=xg_config,
            include_context_features=bool(resolved_context_groups),
            context_feature_groups=resolved_context_groups,
            feature_columns=tuple(feature_frame.columns),
        )

    def predict_proba(self, row: pd.Series | pd.NamedAgg | object) -> ThreeWayProbabilities:
        if isinstance(row, pd.Series):
            feature_frame = _xg_benchmark_features(
                pd.DataFrame([row]),
                xg_config=self.xg_config,
                context_feature_groups=self.context_feature_groups,
            )
        else:
            feature_frame = pd.DataFrame(
                [
                    _xg_feature_dict_from_row(
                        row,
                        xg_config=self.xg_config,
                        context_feature_groups=self.context_feature_groups,
                    )
                ]
            )
        feature_frame = feature_frame.reindex(columns=self.feature_columns)

        predicted = self.classifier.predict_proba(
            feature_frame.fillna(self.feature_fill_values)
        )[0]
        probability_map = dict(zip(self.classifier.classes_, predicted, strict=True))
        return ThreeWayProbabilities(
            home=float(probability_map.get(OUTCOME_HOME, 0.0)),
            draw=float(probability_map.get(OUTCOME_DRAW, 0.0)),
            away=float(probability_map.get(OUTCOME_AWAY, 0.0)),
        )


def uniform_three_way_probabilities() -> ThreeWayProbabilities:
    return ThreeWayProbabilities(home=1.0 / 3.0, draw=1.0 / 3.0, away=1.0 / 3.0)


def weighted_outcome_frequencies(matches: pd.DataFrame) -> ThreeWayProbabilities:
    if matches.empty:
        return uniform_three_way_probabilities()

    weighted_counts = (
        matches.groupby("home_result", observed=False)["sample_weight"].sum().to_dict()
    )
    home = float(weighted_counts.get(OUTCOME_HOME, 0.0))
    draw = float(weighted_counts.get(OUTCOME_DRAW, 0.0))
    away = float(weighted_counts.get(OUTCOME_AWAY, 0.0))
    total = home + draw + away
    if total <= 0.0:
        return uniform_three_way_probabilities()
    return ThreeWayProbabilities(home=home / total, draw=draw / total, away=away / total)


def _benchmark_features(matches: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "elo_diff_scaled": matches["elo_diff_pre"].astype(float) / 400.0,
            "is_non_neutral": (~matches["neutral"].astype(bool)).astype(float),
            "abs_elo_diff_scaled": matches["elo_diff_pre"].astype(float).abs() / 400.0,
        }
    )


def _form_benchmark_features(
    matches: pd.DataFrame,
    *,
    form_config: RecentFormConfig | None = None,
) -> pd.DataFrame:
    form_config = form_config or RecentFormConfig()
    return pd.DataFrame(
        {
            "elo_diff_scaled": matches["elo_diff_pre"].astype(float) / 400.0,
            "is_non_neutral": (~matches["neutral"].astype(bool)).astype(float),
            "abs_elo_diff_scaled": matches["elo_diff_pre"].astype(float).abs() / 400.0,
            "form_points_per_match_diff": (
                matches["home_form_points_per_match"].astype(float)
                - matches["away_form_points_per_match"].astype(float)
            ),
            "form_goal_diff_per_match_diff": (
                matches["home_form_goal_diff_per_match"].astype(float)
                - matches["away_form_goal_diff_per_match"].astype(float)
            ),
            "form_goals_for_per_match_diff": (
                matches["home_form_goals_for_per_match"].astype(float)
                - matches["away_form_goals_for_per_match"].astype(float)
            ),
            "form_goals_against_per_match_diff": (
                matches["home_form_goals_against_per_match"].astype(float)
                - matches["away_form_goals_against_per_match"].astype(float)
            ),
            "form_win_rate_diff": (
                matches["home_form_win_rate"].astype(float)
                - matches["away_form_win_rate"].astype(float)
            ),
            "form_match_count_diff_scaled": (
                matches["home_form_match_count"].astype(float)
                - matches["away_form_match_count"].astype(float)
            )
            / float(form_config.window_size),
            "form_match_count_min_scaled": np.minimum(
                matches["home_form_match_count"].astype(float),
                matches["away_form_match_count"].astype(float),
            )
            / float(form_config.window_size),
            "rest_days_diff_scaled": (
                matches["home_days_since_last_match"].astype(float)
                - matches["away_days_since_last_match"].astype(float)
            )
            / 30.0,
        }
    )


def _xg_benchmark_features(
    matches: pd.DataFrame,
    *,
    xg_config: WorldCupXGConfig | None = None,
    include_context_features: bool = False,
    context_feature_groups: tuple[str, ...] | list[str] | set[str] | None = None,
) -> pd.DataFrame:
    xg_config = xg_config or WorldCupXGConfig()
    resolved_context_groups = _resolve_xg_context_feature_groups(
        include_context_features=include_context_features,
        context_feature_groups=context_feature_groups,
    )
    features = {
        "elo_diff_scaled": matches["elo_diff_pre"].astype(float) / 400.0,
        "is_non_neutral": (~matches["neutral"].astype(bool)).astype(float),
        "abs_elo_diff_scaled": matches["elo_diff_pre"].astype(float).abs() / 400.0,
        "xg_for_per_match_diff": (
            matches["home_xg_for_per_match"].astype(float)
            - matches["away_xg_for_per_match"].astype(float)
        ),
        "xg_against_per_match_diff": (
            matches["home_xg_against_per_match"].astype(float)
            - matches["away_xg_against_per_match"].astype(float)
        ),
        "xg_diff_per_match_diff": (
            matches["home_xg_diff_per_match"].astype(float)
            - matches["away_xg_diff_per_match"].astype(float)
        ),
        "shots_for_per_match_diff": (
            matches["home_shots_for_per_match"].astype(float)
            - matches["away_shots_for_per_match"].astype(float)
        ),
        "shots_against_per_match_diff": (
            matches["home_shots_against_per_match"].astype(float)
            - matches["away_shots_against_per_match"].astype(float)
        ),
        "xg_per_shot_diff": (
            matches["home_xg_per_shot"].astype(float) - matches["away_xg_per_shot"].astype(float)
        ),
        "xg_match_count_diff_scaled": (
            matches["home_xg_match_count"].astype(float)
            - matches["away_xg_match_count"].astype(float)
        )
        / float(xg_config.window_size),
        "xg_match_count_min_scaled": np.minimum(
            matches["home_xg_match_count"].astype(float),
            matches["away_xg_match_count"].astype(float),
        )
        / float(xg_config.window_size),
    }

    if "shot_accuracy" in resolved_context_groups:
        features["shot_accuracy_for_diff"] = (
            _column_as_float(matches, "home_shot_accuracy_for")
            - _column_as_float(matches, "away_shot_accuracy_for")
        )
        features["shot_accuracy_against_diff"] = (
            _column_as_float(matches, "home_shot_accuracy_against")
            - _column_as_float(matches, "away_shot_accuracy_against")
        )
    if "pass_completion" in resolved_context_groups:
        features["pass_completion_for_diff"] = (
            _column_as_float(matches, "home_pass_completion_for")
            - _column_as_float(matches, "away_pass_completion_for")
        )
        features["pass_completion_against_diff"] = (
            _column_as_float(matches, "home_pass_completion_against")
            - _column_as_float(matches, "away_pass_completion_against")
        )
    if "pressures" in resolved_context_groups:
        features["pressures_for_diff_scaled"] = (
            _column_as_float(matches, "home_pressures_for_per_match")
            - _column_as_float(matches, "away_pressures_for_per_match")
        ) / 100.0
        features["pressures_against_diff_scaled"] = (
            _column_as_float(matches, "home_pressures_against_per_match")
            - _column_as_float(matches, "away_pressures_against_per_match")
        ) / 100.0
    if "confederation" in resolved_context_groups:
        features["same_confederation"] = _column_as_float(
            matches,
            "same_confederation",
            default=0.0,
        )
    if "h2h" in resolved_context_groups:
        features["h2h_match_count_scaled"] = _scale_h2h_match_count(
            _column_as_float(matches, "h2h_match_count", default=0.0)
        )
        features["h2h_home_win_rate"] = _column_as_float(matches, "h2h_home_win_rate")
        features["h2h_draw_rate"] = _column_as_float(matches, "h2h_draw_rate")
        features["h2h_away_win_rate"] = _column_as_float(matches, "h2h_away_win_rate")
    if "h2h_decayed" in resolved_context_groups:
        features["h2h_decayed_weight_scaled"] = _scale_h2h_match_count(
            _column_as_float(matches, "h2h_decayed_match_weight", default=0.0)
        )
        features["h2h_decayed_home_win_rate"] = _column_as_float(
            matches, "h2h_decayed_home_win_rate"
        )
        features["h2h_decayed_draw_rate"] = _column_as_float(
            matches, "h2h_decayed_draw_rate"
        )
        features["h2h_decayed_away_win_rate"] = _column_as_float(
            matches, "h2h_decayed_away_win_rate"
        )
    if "confederation" in resolved_context_groups:
        for confederation in CONFEDERATIONS:
            slug = confederation.lower()
            features[f"home_is_{slug}"] = _column_as_float(
                matches,
                f"home_is_{slug}",
                default=0.0,
            )
            features[f"away_is_{slug}"] = _column_as_float(
                matches,
                f"away_is_{slug}",
                default=0.0,
            )

    return pd.DataFrame(features)


def _xg_feature_dict_from_row(
    row: object,
    *,
    xg_config: WorldCupXGConfig,
    include_context_features: bool = False,
    context_feature_groups: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict[str, float]:
    resolved_context_groups = _resolve_xg_context_feature_groups(
        include_context_features=include_context_features,
        context_feature_groups=context_feature_groups,
    )
    home_xg_match_count = _float_or_nan(getattr(row, "home_xg_match_count", np.nan))
    away_xg_match_count = _float_or_nan(getattr(row, "away_xg_match_count", np.nan))
    feature_map = {
        "elo_diff_scaled": float(getattr(row, "elo_diff_pre")) / 400.0,
        "is_non_neutral": 0.0 if bool(getattr(row, "neutral")) else 1.0,
        "abs_elo_diff_scaled": abs(float(getattr(row, "elo_diff_pre"))) / 400.0,
        "xg_for_per_match_diff": _float_or_nan(getattr(row, "home_xg_for_per_match", np.nan))
        - _float_or_nan(getattr(row, "away_xg_for_per_match", np.nan)),
        "xg_against_per_match_diff": _float_or_nan(
            getattr(row, "home_xg_against_per_match", np.nan)
        )
        - _float_or_nan(getattr(row, "away_xg_against_per_match", np.nan)),
        "xg_diff_per_match_diff": _float_or_nan(
            getattr(row, "home_xg_diff_per_match", np.nan)
        )
        - _float_or_nan(getattr(row, "away_xg_diff_per_match", np.nan)),
        "shots_for_per_match_diff": _float_or_nan(
            getattr(row, "home_shots_for_per_match", np.nan)
        )
        - _float_or_nan(getattr(row, "away_shots_for_per_match", np.nan)),
        "shots_against_per_match_diff": _float_or_nan(
            getattr(row, "home_shots_against_per_match", np.nan)
        )
        - _float_or_nan(getattr(row, "away_shots_against_per_match", np.nan)),
        "xg_per_shot_diff": _float_or_nan(getattr(row, "home_xg_per_shot", np.nan))
        - _float_or_nan(getattr(row, "away_xg_per_shot", np.nan)),
        "xg_match_count_diff_scaled": (home_xg_match_count - away_xg_match_count)
        / float(xg_config.window_size),
        "xg_match_count_min_scaled": min(home_xg_match_count, away_xg_match_count)
        / float(xg_config.window_size),
    }
    if "shot_accuracy" in resolved_context_groups:
        feature_map["shot_accuracy_for_diff"] = _float_or_nan(
            getattr(row, "home_shot_accuracy_for", np.nan)
        ) - _float_or_nan(getattr(row, "away_shot_accuracy_for", np.nan))
        feature_map["shot_accuracy_against_diff"] = _float_or_nan(
            getattr(row, "home_shot_accuracy_against", np.nan)
        ) - _float_or_nan(getattr(row, "away_shot_accuracy_against", np.nan))
    if "pass_completion" in resolved_context_groups:
        feature_map["pass_completion_for_diff"] = _float_or_nan(
            getattr(row, "home_pass_completion_for", np.nan)
        ) - _float_or_nan(getattr(row, "away_pass_completion_for", np.nan))
        feature_map["pass_completion_against_diff"] = _float_or_nan(
            getattr(row, "home_pass_completion_against", np.nan)
        ) - _float_or_nan(getattr(row, "away_pass_completion_against", np.nan))
    if "pressures" in resolved_context_groups:
        feature_map["pressures_for_diff_scaled"] = (
            _float_or_nan(getattr(row, "home_pressures_for_per_match", np.nan))
            - _float_or_nan(getattr(row, "away_pressures_for_per_match", np.nan))
        ) / 100.0
        feature_map["pressures_against_diff_scaled"] = (
            _float_or_nan(getattr(row, "home_pressures_against_per_match", np.nan))
            - _float_or_nan(getattr(row, "away_pressures_against_per_match", np.nan))
        ) / 100.0
    if "confederation" in resolved_context_groups:
        feature_map["same_confederation"] = _float_or_nan(
            getattr(row, "same_confederation", 0.0)
        )
    if "h2h" in resolved_context_groups:
        feature_map["h2h_match_count_scaled"] = _scale_h2h_match_count(
            _float_or_nan(getattr(row, "h2h_match_count", 0.0))
        )
        feature_map["h2h_home_win_rate"] = _float_or_nan(
            getattr(row, "h2h_home_win_rate", np.nan)
        )
        feature_map["h2h_draw_rate"] = _float_or_nan(getattr(row, "h2h_draw_rate", np.nan))
        feature_map["h2h_away_win_rate"] = _float_or_nan(
            getattr(row, "h2h_away_win_rate", np.nan)
        )
    if "h2h_decayed" in resolved_context_groups:
        feature_map["h2h_decayed_weight_scaled"] = _scale_h2h_match_count(
            _float_or_nan(getattr(row, "h2h_decayed_match_weight", 0.0))
        )
        feature_map["h2h_decayed_home_win_rate"] = _float_or_nan(
            getattr(row, "h2h_decayed_home_win_rate", np.nan)
        )
        feature_map["h2h_decayed_draw_rate"] = _float_or_nan(
            getattr(row, "h2h_decayed_draw_rate", np.nan)
        )
        feature_map["h2h_decayed_away_win_rate"] = _float_or_nan(
            getattr(row, "h2h_decayed_away_win_rate", np.nan)
        )
    if "confederation" in resolved_context_groups:
        for confederation in CONFEDERATIONS:
            slug = confederation.lower()
            feature_map[f"home_is_{slug}"] = _float_or_nan(
                getattr(row, f"home_is_{slug}", 0.0)
            )
            feature_map[f"away_is_{slug}"] = _float_or_nan(
                getattr(row, f"away_is_{slug}", 0.0)
            )
    return feature_map


def _resolve_xg_context_feature_groups(
    *,
    include_context_features: bool,
    context_feature_groups: tuple[str, ...] | list[str] | set[str] | None,
) -> tuple[str, ...]:
    if context_feature_groups is None:
        return XG_CONTEXT_LEGACY_BUNDLE if include_context_features else ()

    unknown_groups = sorted(
        set(context_feature_groups).difference(XG_CONTEXT_FEATURE_GROUPS)
    )
    if unknown_groups:
        unknown = ", ".join(unknown_groups)
        supported = ", ".join(XG_CONTEXT_FEATURE_GROUPS)
        raise ValueError(
            f"Unknown xG context feature group(s): {unknown}. Supported: {supported}"
        )

    selected_groups = set(context_feature_groups)
    return tuple(group for group in XG_CONTEXT_FEATURE_GROUPS if group in selected_groups)


def _required_xg_context_columns(context_feature_groups: tuple[str, ...]) -> set[str]:
    columns: set[str] = set()
    if "shot_accuracy" in context_feature_groups:
        columns.update(
            {
                "home_shot_accuracy_for",
                "away_shot_accuracy_for",
                "home_shot_accuracy_against",
                "away_shot_accuracy_against",
            }
        )
    if "confederation" in context_feature_groups:
        columns.update(_required_xg_confederation_columns())
    if "h2h" in context_feature_groups:
        columns.update(_required_xg_h2h_columns())
    if "h2h_decayed" in context_feature_groups:
        columns.update(_required_xg_h2h_decayed_columns())
    if "pass_completion" in context_feature_groups:
        columns.update(
            {
                "home_pass_completion_for",
                "away_pass_completion_for",
                "home_pass_completion_against",
                "away_pass_completion_against",
            }
        )
    if "pressures" in context_feature_groups:
        columns.update(
            {
                "home_pressures_for_per_match",
                "away_pressures_for_per_match",
                "home_pressures_against_per_match",
                "away_pressures_against_per_match",
            }
        )
    return columns


def _required_xg_confederation_columns() -> set[str]:
    columns = {"same_confederation"}
    for confederation in CONFEDERATIONS:
        slug = confederation.lower()
        columns.add(f"home_is_{slug}")
        columns.add(f"away_is_{slug}")
    return columns


def _required_xg_h2h_columns() -> set[str]:
    return {
        "h2h_match_count",
        "h2h_home_win_rate",
        "h2h_draw_rate",
        "h2h_away_win_rate",
    }


def _required_xg_h2h_decayed_columns() -> set[str]:
    return {
        "h2h_decayed_match_weight",
        "h2h_decayed_home_win_rate",
        "h2h_decayed_draw_rate",
        "h2h_decayed_away_win_rate",
    }


def _column_as_float(
    frame: pd.DataFrame,
    column: str,
    *,
    default: float = np.nan,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return frame[column].astype(float)


def _scale_h2h_match_count(value: pd.Series | float) -> pd.Series | float:
    if isinstance(value, pd.Series):
        return value.clip(lower=0.0, upper=10.0) / 10.0
    if not np.isfinite(value):
        return float("nan")
    return min(max(value, 0.0), 10.0) / 10.0


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
