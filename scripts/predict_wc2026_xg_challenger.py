from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from wc2026_model.data import canonicalize_team_name, load_international_results
from wc2026_model.evaluation import XGEloMultinomialBenchmark
from wc2026_model.features import (
    WorldCupXGConfig,
    attach_latest_team_xg_features,
    augment_with_pre_match_xg_features,
    build_latest_team_xg_snapshot,
)
from wc2026_model.markets import compare_match_probabilities
from wc2026_model.pipeline import BaselineTrainingConfig, build_training_frame


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an Elo+xG challenger forecast for WC2026 fixtures and compare it to the current recipe."
    )
    parser.add_argument(
        "--statsbomb-input",
        default="data/interim/statsbomb_men_major_tournaments_match_features.csv",
        help="StatsBomb major-tournaments match feature dataset.",
    )
    parser.add_argument(
        "--fixture-base-input",
        default="reports/wc2026_fixture_predictions_expected_xi_plus_goalkeeper_recipe_2026-06-13.csv",
        help="Existing WC2026 fixture prediction table used as the base fixture list and live Elo adjustment source.",
    )
    parser.add_argument(
        "--fixture-elo-column",
        default="adjusted_elo_diff_pre",
        help="Column in --fixture-base-input used as the Elo differential passed to the challenger model.",
    )
    parser.add_argument(
        "--min-match-date",
        default="2018-01-01",
        help="Lower bound for the StatsBomb training sample.",
    )
    parser.add_argument(
        "--min-team-matches",
        type=int,
        default=1,
        help="Minimum team match count for the training sample.",
    )
    parser.add_argument(
        "--xg-window-size",
        type=int,
        default=3,
        help="Rolling xG window used both in training and when creating the latest team snapshot.",
    )
    parser.add_argument(
        "--output",
        default="reports/wc2026_xg_challenger_predictions.csv",
        help="Where to save challenger fixture probabilities.",
    )
    parser.add_argument(
        "--comparison-output",
        default="reports/wc2026_xg_challenger_vs_current_recipe.csv",
        help="Where to save the side-by-side comparison versus the current recipe.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/wc2026_xg_challenger_summary.json",
        help="Where to save a compact challenger summary.",
    )
    parser.add_argument(
        "--bookmaker-snapshot-input",
        default="data/interim/bookmaker_match_odds_live_sample_2026-06-13.csv",
        help="Optional bookmaker 1X2 snapshot for direct market comparison.",
    )
    parser.add_argument(
        "--bookmaker-comparison-output",
        default="reports/bookmaker_match_odds_xg_challenger_live_sample_comparison.csv",
        help="Where to save the xG challenger versus bookmaker comparison.",
    )
    parser.add_argument(
        "--bookmaker-summary-output",
        default="reports/bookmaker_match_odds_xg_challenger_live_sample_comparison_summary.json",
        help="Where to save the xG challenger bookmaker summary.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    statsbomb_results = load_international_results(args.statsbomb_input)
    fixture_base = pd.read_csv(args.fixture_base_input)
    predictions, comparison, summary = build_wc2026_xg_challenger_outputs(
        statsbomb_results=statsbomb_results,
        fixture_base=fixture_base,
        training_config=BaselineTrainingConfig(
            min_match_date=args.min_match_date,
            min_team_matches=args.min_team_matches,
        ),
        xg_config=WorldCupXGConfig(window_size=args.xg_window_size),
        fixture_elo_column=args.fixture_elo_column,
    )

    output_path = Path(args.output)
    comparison_output_path = Path(args.comparison_output)
    summary_output_path = Path(args.summary_output)
    for path in (output_path, comparison_output_path, summary_output_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    comparison.to_csv(comparison_output_path, index=False)
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bookmaker_path = Path(args.bookmaker_snapshot_input)
    if bookmaker_path.exists():
        bookmaker_snapshot = pd.read_csv(bookmaker_path)
        bookmaker_comparison = compare_match_probabilities(predictions, bookmaker_snapshot)
        bookmaker_output_path = Path(args.bookmaker_comparison_output)
        bookmaker_summary_output_path = Path(args.bookmaker_summary_output)
        bookmaker_output_path.parent.mkdir(parents=True, exist_ok=True)
        bookmaker_summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        bookmaker_comparison.to_csv(bookmaker_output_path, index=False)
        bookmaker_summary_output_path.write_text(
            json.dumps(_build_bookmaker_summary(bookmaker_comparison), indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2))
    print(f"Saved xG challenger predictions to {output_path}")
    print(f"Saved xG challenger comparison to {comparison_output_path}")
    print(f"Saved xG challenger summary to {summary_output_path}")


def build_wc2026_xg_challenger_outputs(
    *,
    statsbomb_results: pd.DataFrame,
    fixture_base: pd.DataFrame,
    training_config: BaselineTrainingConfig,
    xg_config: WorldCupXGConfig,
    fixture_elo_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    training_frame = build_training_frame(statsbomb_results, config=training_config)
    training_frame_with_xg = augment_with_pre_match_xg_features(training_frame, config=xg_config)
    challenger = XGEloMultinomialBenchmark.fit(training_frame_with_xg, xg_config=xg_config)

    team_snapshot = build_latest_team_xg_snapshot(
        statsbomb_results,
        config=xg_config,
    )
    predictions = build_fixture_challenger_predictions(
        fixture_base=fixture_base,
        team_snapshot=team_snapshot,
        challenger=challenger,
        fixture_elo_column=fixture_elo_column,
    )
    comparison = build_fixture_model_delta_frame(
        fixture_base=fixture_base,
        challenger_predictions=predictions,
    )
    summary = {
        "fixture_count": int(len(predictions)),
        "training_match_count": int(len(training_frame)),
        "training_team_count": int(
            len(set(training_frame["home_team"]).union(set(training_frame["away_team"])))
        ),
        "xg_window_size": xg_config.window_size,
        "fixtures_with_both_team_xg_histories": int(predictions["both_teams_have_xg_history"].sum()),
        "fixtures_with_any_team_xg_history": int(predictions["any_team_has_xg_history"].sum()),
        "largest_probability_deltas_vs_current_recipe": _top_probability_deltas(comparison),
    }
    return predictions, comparison, summary


def build_fixture_challenger_predictions(
    *,
    fixture_base: pd.DataFrame,
    team_snapshot: pd.DataFrame,
    challenger: XGEloMultinomialBenchmark,
    fixture_elo_column: str,
) -> pd.DataFrame:
    required_columns = {"match_id", "match_date", "home_team", "away_team"}
    missing_columns = required_columns.difference(fixture_base.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Fixture base is missing columns: {missing}")
    if fixture_elo_column not in fixture_base.columns:
        raise ValueError(f"Fixture base is missing Elo column '{fixture_elo_column}'.")

    base = fixture_base.copy()
    base["home_team"] = base["home_team"].astype(str).map(canonicalize_team_name)
    base["away_team"] = base["away_team"].astype(str).map(canonicalize_team_name)
    if "neutral" not in base.columns:
        base["neutral"] = True
    base = attach_latest_team_xg_features(base, team_snapshot)
    base["challenger_elo_diff_pre"] = pd.to_numeric(base[fixture_elo_column], errors="coerce")
    if "elo_diff_pre" in base.columns:
        base["challenger_elo_diff_pre"] = base["challenger_elo_diff_pre"].fillna(
            pd.to_numeric(base["elo_diff_pre"], errors="coerce")
        )

    rows: list[dict[str, object]] = []
    for row in base.itertuples(index=False):
        probabilities = challenger.predict_proba(
            SimpleNamespace(
                neutral=bool(getattr(row, "neutral", True)),
                elo_diff_pre=float(getattr(row, "challenger_elo_diff_pre")),
                home_xg_for_per_match=getattr(row, "home_xg_for_per_match", float("nan")),
                away_xg_for_per_match=getattr(row, "away_xg_for_per_match", float("nan")),
                home_xg_against_per_match=getattr(row, "home_xg_against_per_match", float("nan")),
                away_xg_against_per_match=getattr(row, "away_xg_against_per_match", float("nan")),
                home_xg_diff_per_match=getattr(row, "home_xg_diff_per_match", float("nan")),
                away_xg_diff_per_match=getattr(row, "away_xg_diff_per_match", float("nan")),
                home_shots_for_per_match=getattr(row, "home_shots_for_per_match", float("nan")),
                away_shots_for_per_match=getattr(row, "away_shots_for_per_match", float("nan")),
                home_shots_against_per_match=getattr(
                    row, "home_shots_against_per_match", float("nan")
                ),
                away_shots_against_per_match=getattr(
                    row, "away_shots_against_per_match", float("nan")
                ),
                home_xg_per_shot=getattr(row, "home_xg_per_shot", float("nan")),
                away_xg_per_shot=getattr(row, "away_xg_per_shot", float("nan")),
                home_xg_match_count=getattr(row, "home_xg_match_count", float("nan")),
                away_xg_match_count=getattr(row, "away_xg_match_count", float("nan")),
            )
        )

        home_xg_match_count = _coerce_float(getattr(row, "home_xg_match_count", None))
        away_xg_match_count = _coerce_float(getattr(row, "away_xg_match_count", None))
        rows.append(
            {
                "match_id": row.match_id,
                "match_date": str(row.match_date),
                "home_team": row.home_team,
                "away_team": row.away_team,
                "neutral": bool(getattr(row, "neutral", True)),
                "elo_diff_pre": float(getattr(row, "challenger_elo_diff_pre")),
                "home_xg_match_count": home_xg_match_count,
                "away_xg_match_count": away_xg_match_count,
                "home_xg_for_per_match": _coerce_float(
                    getattr(row, "home_xg_for_per_match", None)
                ),
                "away_xg_for_per_match": _coerce_float(
                    getattr(row, "away_xg_for_per_match", None)
                ),
                "home_xg_diff_per_match": _coerce_float(
                    getattr(row, "home_xg_diff_per_match", None)
                ),
                "away_xg_diff_per_match": _coerce_float(
                    getattr(row, "away_xg_diff_per_match", None)
                ),
                "both_teams_have_xg_history": bool(
                    (home_xg_match_count is not None and home_xg_match_count > 0.0)
                    and (away_xg_match_count is not None and away_xg_match_count > 0.0)
                ),
                "any_team_has_xg_history": bool(
                    (home_xg_match_count is not None and home_xg_match_count > 0.0)
                    or (away_xg_match_count is not None and away_xg_match_count > 0.0)
                ),
                "home_win_probability": probabilities.home,
                "draw_probability": probabilities.draw,
                "away_win_probability": probabilities.away,
                "home_fair_odds": _probability_to_decimal_odds(probabilities.home),
                "draw_fair_odds": _probability_to_decimal_odds(probabilities.draw),
                "away_fair_odds": _probability_to_decimal_odds(probabilities.away),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["match_date", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)


def build_fixture_model_delta_frame(
    *,
    fixture_base: pd.DataFrame,
    challenger_predictions: pd.DataFrame,
) -> pd.DataFrame:
    required_current_columns = {
        "match_id",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    }
    missing_columns = required_current_columns.difference(fixture_base.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Fixture base is missing current-model columns: {missing}")

    current_columns = [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]
    if "adjusted_elo_diff_pre" in fixture_base.columns:
        current_columns.append("adjusted_elo_diff_pre")
    comparison = challenger_predictions.merge(
        fixture_base.loc[:, current_columns].rename(
            columns={
                "home_win_probability": "current_home_win_probability",
                "draw_probability": "current_draw_probability",
                "away_win_probability": "current_away_win_probability",
            }
        ),
        on=["match_id", "match_date", "home_team", "away_team"],
        how="left",
    )
    comparison["delta_home_win_probability"] = (
        comparison["home_win_probability"] - comparison["current_home_win_probability"]
    )
    comparison["delta_draw_probability"] = (
        comparison["draw_probability"] - comparison["current_draw_probability"]
    )
    comparison["delta_away_win_probability"] = (
        comparison["away_win_probability"] - comparison["current_away_win_probability"]
    )
    comparison["max_absolute_probability_delta"] = comparison[
        [
            "delta_home_win_probability",
            "delta_draw_probability",
            "delta_away_win_probability",
        ]
    ].abs().max(axis=1)
    return comparison.sort_values(
        ["max_absolute_probability_delta", "match_date", "home_team", "away_team"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _build_bookmaker_summary(comparison: pd.DataFrame) -> dict[str, object]:
    edge_columns = [
        ("home", "home_edge_vs_no_vig", "home_no_vig_probability"),
        ("draw", "draw_edge_vs_no_vig", "draw_no_vig_probability"),
        ("away", "away_edge_vs_no_vig", "away_no_vig_probability"),
    ]
    return {
        "match_count": int(len(comparison)),
        "top_positive_edges_vs_no_vig": _top_edge_records(
            comparison,
            edge_columns=edge_columns,
            largest=True,
        ),
        "top_negative_edges_vs_no_vig": _top_edge_records(
            comparison,
            edge_columns=edge_columns,
            largest=False,
        ),
    }


def _top_edge_records(
    comparison: pd.DataFrame,
    *,
    edge_columns: list[tuple[str, str, str]],
    largest: bool,
    limit: int = 10,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for outcome_label, edge_column, market_probability_column in edge_columns:
        for row in comparison.itertuples(index=False):
            records.append(
                {
                    "match_date": getattr(row, "match_date", None),
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "outcome": outcome_label,
                    "edge_vs_no_vig": float(getattr(row, edge_column)),
                    "model_probability": float(
                        getattr(
                            row,
                            {
                                "home": "home_win_probability",
                                "draw": "draw_probability",
                                "away": "away_win_probability",
                            }[outcome_label],
                        )
                    ),
                    "market_probability": float(getattr(row, market_probability_column)),
                    "expected_value": float(
                        getattr(
                            row,
                            {
                                "home": "home_expected_value",
                                "draw": "draw_expected_value",
                                "away": "away_expected_value",
                            }[outcome_label],
                        )
                    ),
                }
            )
    return sorted(
        records,
        key=lambda record: (
            record["edge_vs_no_vig"],
            record["expected_value"],
            record["home_team"],
            record["away_team"],
            record["outcome"],
        ),
        reverse=largest,
    )[:limit]


def _top_probability_deltas(comparison: pd.DataFrame, limit: int = 10) -> list[dict[str, object]]:
    rows = []
    for row in comparison.head(limit).itertuples(index=False):
        rows.append(
            {
                "match_date": row.match_date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "max_absolute_probability_delta": float(row.max_absolute_probability_delta),
                "delta_home_win_probability": float(row.delta_home_win_probability),
                "delta_draw_probability": float(row.delta_draw_probability),
                "delta_away_win_probability": float(row.delta_away_win_probability),
                "both_teams_have_xg_history": bool(row.both_teams_have_xg_history),
            }
        )
    return rows


def _coerce_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _probability_to_decimal_odds(probability: float) -> float | None:
    if probability <= 0.0:
        return None
    return 1.0 / probability


if __name__ == "__main__":
    main()
