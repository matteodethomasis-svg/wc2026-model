from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from wc2026_model.data import canonicalize_team_name, load_scheduled_matches
from wc2026_model.evaluation import EloMultinomialBenchmark
from wc2026_model.models import BlendedMatchModel, CalibratedMatchModel

_PREDICTION_COLUMNS = [
    "match_id",
    "match_date",
    "tournament",
    "city",
    "country",
    "neutral",
    "home_team",
    "away_team",
    "home_elo",
    "away_elo",
    "elo_diff_pre",
    "home_team_squad_strength",
    "away_team_squad_strength",
    "squad_strength_diff_pre",
    "home_team_secondary_squad_strength",
    "away_team_secondary_squad_strength",
    "secondary_squad_strength_diff_pre",
    "home_expected_starter_count",
    "away_expected_starter_count",
    "home_unavailable_expected_starter_count",
    "away_unavailable_expected_starter_count",
    "home_doubtful_expected_starter_count",
    "away_doubtful_expected_starter_count",
    "home_goalkeeper_starter_available",
    "away_goalkeeper_starter_available",
    "home_lineup_confidence",
    "away_lineup_confidence",
    "home_availability_elo_adjustment",
    "away_availability_elo_adjustment",
    "availability_elo_diff_pre",
    "adjusted_elo_diff_pre",
    "home_expected_goals",
    "away_expected_goals",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "home_fair_odds",
    "draw_fair_odds",
    "away_fair_odds",
]


def save_world_cup_fixture_predictions(
    *,
    model_input: str | Path,
    fixtures_input: str | Path,
    elo_ratings_input: str | Path,
    training_frame_input: str | Path,
    output: str | Path,
    summary_output: str | Path,
    elo_blend_alpha: float = 1.0,
    blend_method: str = "linear",
    blend_temperature: float = 1.0,
    calibration_gamma_home: float = 1.0,
    calibration_gamma_draw: float = 1.0,
    calibration_gamma_away: float = 1.0,
    squad_strength_input: str | Path | None = None,
    squad_strength_column: str = "squad_club_elo_rating",
    secondary_squad_strength_column: str | None = None,
    squad_elo_scale: float = 0.0,
    secondary_squad_elo_scale: float = 0.0,
    availability_input: str | Path | None = None,
    availability_starter_absence_elo: float = 18.0,
    availability_goalkeeper_absence_elo: float = 24.0,
    tournament: str = "FIFA World Cup",
    start_date: str = "2026-06-12",
    max_goals: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    predictions, summary = predict_world_cup_fixtures(
        model_input=model_input,
        fixtures_input=fixtures_input,
        elo_ratings_input=elo_ratings_input,
        training_frame_input=training_frame_input,
        elo_blend_alpha=elo_blend_alpha,
        blend_method=blend_method,
        blend_temperature=blend_temperature,
        calibration_gamma_home=calibration_gamma_home,
        calibration_gamma_draw=calibration_gamma_draw,
        calibration_gamma_away=calibration_gamma_away,
        squad_strength_input=squad_strength_input,
        squad_strength_column=squad_strength_column,
        secondary_squad_strength_column=secondary_squad_strength_column,
        squad_elo_scale=squad_elo_scale,
        secondary_squad_elo_scale=secondary_squad_elo_scale,
        availability_input=availability_input,
        availability_starter_absence_elo=availability_starter_absence_elo,
        availability_goalkeeper_absence_elo=availability_goalkeeper_absence_elo,
        tournament=tournament,
        start_date=start_date,
        max_goals=max_goals,
    )
    output_path = Path(output)
    summary_output_path = Path(summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return predictions, summary


def predict_world_cup_fixtures(
    *,
    model_input: str | Path,
    fixtures_input: str | Path,
    elo_ratings_input: str | Path,
    training_frame_input: str | Path,
    elo_blend_alpha: float = 1.0,
    blend_method: str = "linear",
    blend_temperature: float = 1.0,
    calibration_gamma_home: float = 1.0,
    calibration_gamma_draw: float = 1.0,
    calibration_gamma_away: float = 1.0,
    squad_strength_input: str | Path | None = None,
    squad_strength_column: str = "squad_club_elo_rating",
    secondary_squad_strength_column: str | None = None,
    squad_elo_scale: float = 0.0,
    secondary_squad_elo_scale: float = 0.0,
    availability_input: str | Path | None = None,
    availability_starter_absence_elo: float = 18.0,
    availability_goalkeeper_absence_elo: float = 24.0,
    tournament: str = "FIFA World Cup",
    start_date: str = "2026-06-12",
    max_goals: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    model_path = Path(model_input)
    fixtures_path = Path(fixtures_input)
    elo_ratings_path = Path(elo_ratings_input)
    training_frame_path = Path(training_frame_input)
    squad_strength_path = Path(squad_strength_input) if squad_strength_input else None
    availability_path = Path(availability_input) if availability_input else None

    with model_path.open("rb") as file_handle:
        model = pickle.load(file_handle)
    model = _maybe_build_blended_model(
        model,
        training_frame_path=training_frame_path,
        alpha_on_base=elo_blend_alpha,
        blend_method=blend_method,
        blend_temperature=blend_temperature,
    )
    model = _maybe_build_calibrated_model(
        model,
        gamma_home=calibration_gamma_home,
        gamma_draw=calibration_gamma_draw,
        gamma_away=calibration_gamma_away,
    )

    fixtures = load_scheduled_matches(
        fixtures_path,
        tournament=tournament,
        start_date=start_date,
    )
    elo_ratings = _load_elo_ratings(elo_ratings_path)
    squad_strength_ratings = _load_team_strength_ratings(
        squad_strength_path,
        rating_column=squad_strength_column,
    )
    secondary_squad_strength_ratings = (
        _load_team_strength_ratings(
            squad_strength_path,
            rating_column=secondary_squad_strength_column,
        )
        if secondary_squad_strength_column
        else {}
    )
    availability_lookup = _load_team_availability_lookup(availability_path)

    prediction_rows = []
    for row in fixtures.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        home_elo = float(elo_ratings.get(home_team, 1500.0))
        away_elo = float(elo_ratings.get(away_team, 1500.0))
        elo_diff = home_elo - away_elo
        home_squad_strength = squad_strength_ratings.get(home_team)
        away_squad_strength = squad_strength_ratings.get(away_team)
        squad_strength_diff = _strength_diff(home_squad_strength, away_squad_strength)
        home_secondary_squad_strength = secondary_squad_strength_ratings.get(home_team)
        away_secondary_squad_strength = secondary_squad_strength_ratings.get(away_team)
        secondary_squad_strength_diff = _strength_diff(
            home_secondary_squad_strength,
            away_secondary_squad_strength,
        )
        home_availability = _get_team_availability_record(
            availability_lookup,
            match_date=row.match_date,
            team=home_team,
        )
        away_availability = _get_team_availability_record(
            availability_lookup,
            match_date=row.match_date,
            team=away_team,
        )
        home_availability_adjustment = _availability_elo_adjustment(
            home_availability,
            starter_absence_elo=availability_starter_absence_elo,
            goalkeeper_absence_elo=availability_goalkeeper_absence_elo,
        )
        away_availability_adjustment = _availability_elo_adjustment(
            away_availability,
            starter_absence_elo=availability_starter_absence_elo,
            goalkeeper_absence_elo=availability_goalkeeper_absence_elo,
        )
        availability_elo_diff = home_availability_adjustment - away_availability_adjustment
        adjusted_elo_diff = elo_diff + (squad_elo_scale * squad_strength_diff) + (
            secondary_squad_elo_scale * secondary_squad_strength_diff
        ) + availability_elo_diff

        home_xg, away_xg = model.predict_expected_goals(
            home_team,
            away_team,
            neutral_site=bool(row.neutral),
            elo_diff_pre=adjusted_elo_diff,
        )
        probabilities = model.predict_outcome_probabilities(
            home_team,
            away_team,
            neutral_site=bool(row.neutral),
            elo_diff_pre=adjusted_elo_diff,
            max_goals=max_goals,
        )
        prediction_rows.append(
            {
                "match_id": row.match_id,
                "match_date": pd.Timestamp(row.match_date).strftime("%Y-%m-%d"),
                "tournament": row.tournament,
                "city": row.city,
                "country": row.country,
                "neutral": bool(row.neutral),
                "home_team": home_team,
                "away_team": away_team,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff_pre": elo_diff,
                "home_team_squad_strength": home_squad_strength,
                "away_team_squad_strength": away_squad_strength,
                "squad_strength_diff_pre": squad_strength_diff,
                "home_team_secondary_squad_strength": home_secondary_squad_strength,
                "away_team_secondary_squad_strength": away_secondary_squad_strength,
                "secondary_squad_strength_diff_pre": secondary_squad_strength_diff,
                "home_expected_starter_count": _availability_metric(
                    home_availability,
                    "expected_starter_count",
                ),
                "away_expected_starter_count": _availability_metric(
                    away_availability,
                    "expected_starter_count",
                ),
                "home_unavailable_expected_starter_count": _availability_metric(
                    home_availability,
                    "unavailable_expected_starter_count",
                ),
                "away_unavailable_expected_starter_count": _availability_metric(
                    away_availability,
                    "unavailable_expected_starter_count",
                ),
                "home_doubtful_expected_starter_count": _availability_metric(
                    home_availability,
                    "doubtful_expected_starter_count",
                ),
                "away_doubtful_expected_starter_count": _availability_metric(
                    away_availability,
                    "doubtful_expected_starter_count",
                ),
                "home_goalkeeper_starter_available": _availability_metric(
                    home_availability,
                    "goalkeeper_starter_available",
                ),
                "away_goalkeeper_starter_available": _availability_metric(
                    away_availability,
                    "goalkeeper_starter_available",
                ),
                "home_lineup_confidence": _availability_metric(
                    home_availability,
                    "lineup_confidence",
                ),
                "away_lineup_confidence": _availability_metric(
                    away_availability,
                    "lineup_confidence",
                ),
                "home_availability_elo_adjustment": home_availability_adjustment,
                "away_availability_elo_adjustment": away_availability_adjustment,
                "availability_elo_diff_pre": availability_elo_diff,
                "adjusted_elo_diff_pre": adjusted_elo_diff,
                "home_expected_goals": home_xg,
                "away_expected_goals": away_xg,
                "home_win_probability": probabilities.home,
                "draw_probability": probabilities.draw,
                "away_win_probability": probabilities.away,
                "home_fair_odds": _fair_odds(probabilities.home),
                "draw_fair_odds": _fair_odds(probabilities.draw),
                "away_fair_odds": _fair_odds(probabilities.away),
            }
        )

    predictions = pd.DataFrame.from_records(
        prediction_rows,
        columns=_PREDICTION_COLUMNS,
    )
    summary = {
        "fixture_count": int(len(predictions)),
        "start_date": start_date,
        "tournament": tournament,
        "elo_blend_alpha": elo_blend_alpha,
        "blend_method": blend_method,
        "blend_temperature": blend_temperature,
        "calibration_gamma_home": calibration_gamma_home,
        "calibration_gamma_draw": calibration_gamma_draw,
        "calibration_gamma_away": calibration_gamma_away,
        "squad_strength_column": squad_strength_column,
        "squad_elo_scale": squad_elo_scale,
        "secondary_squad_strength_column": secondary_squad_strength_column,
        "secondary_squad_elo_scale": secondary_squad_elo_scale,
        "availability_starter_absence_elo": availability_starter_absence_elo,
        "availability_goalkeeper_absence_elo": availability_goalkeeper_absence_elo,
        "fixtures_with_squad_strength_count": int(
            (
                predictions["home_team_squad_strength"].notna()
                & predictions["away_team_squad_strength"].notna()
            ).sum()
        ),
        "fixtures_with_secondary_squad_strength_count": int(
            (
                predictions["home_team_secondary_squad_strength"].notna()
                & predictions["away_team_secondary_squad_strength"].notna()
            ).sum()
        )
        if secondary_squad_strength_column
        else 0,
        "fixtures_with_availability_count": int(
            (
                (predictions["home_expected_starter_count"].fillna(0.0) > 0.0)
                | (predictions["away_expected_starter_count"].fillna(0.0) > 0.0)
            ).sum()
        ),
        "highest_home_win_probability": _top_fixture(
            predictions,
            "home_win_probability",
        ),
        "highest_away_win_probability": _top_fixture(
            predictions,
            "away_win_probability",
        ),
        "highest_draw_probability": _top_fixture(
            predictions,
            "draw_probability",
        ),
    }
    return predictions, summary


def _load_elo_ratings(path: Path) -> dict[str, float]:
    dataframe = pd.read_csv(path)
    return {
        str(row.team): float(row.elo_rating)
        for row in dataframe.loc[:, ["team", "elo_rating"]].itertuples(index=False)
    }


def _load_team_strength_ratings(
    path: Path | None,
    *,
    rating_column: str | None,
) -> dict[str, float]:
    if path is None or rating_column is None:
        return {}
    dataframe = pd.read_csv(path)
    if "team" not in dataframe.columns:
        raise ValueError("Squad strength CSV must contain a 'team' column.")
    if rating_column not in dataframe.columns:
        raise ValueError(f"Squad strength CSV is missing column '{rating_column}'.")
    return {
        canonicalize_team_name(str(row.team)): float(row[rating_column])
        for _, row in dataframe.loc[:, ["team", rating_column]].iterrows()
        if pd.notna(row[rating_column])
    }


def _load_team_availability_lookup(
    path: Path | None,
) -> dict[tuple[str, str], dict[str, object]]:
    if path is None:
        return {}

    dataframe = pd.read_csv(path)
    return _load_team_availability_lookup_from_frame(dataframe)


def _load_team_availability_lookup_from_frame(
    dataframe: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, object]]:
    if "team" not in dataframe.columns:
        raise ValueError("Availability CSV must contain a 'team' column.")

    frame = dataframe.copy()
    frame["team"] = frame["team"].astype(str).map(canonicalize_team_name)
    if "match_date" in frame.columns:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
        frame["match_date_key"] = frame["match_date"].dt.strftime("%Y-%m-%d").fillna("")
    else:
        frame["match_date_key"] = ""

    sort_columns = [
        column for column in ("expected_starter_count", "lineup_confidence") if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(
            sort_columns,
            ascending=[False] * len(sort_columns),
            kind="stable",
        )

    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        key = _availability_lookup_key(
            match_date=row.get("match_date_key"),
            team=row.get("team"),
        )
        lookup.setdefault(key, row)
    return lookup


def _maybe_build_blended_model(
    model: object,
    *,
    training_frame_path: Path,
    alpha_on_base: float,
    blend_method: str = "linear",
    blend_temperature: float = 1.0,
) -> object:
    if alpha_on_base >= 1.0:
        return model
    training_frame = pd.read_csv(training_frame_path)
    elo_benchmark = EloMultinomialBenchmark.fit(training_frame)
    return BlendedMatchModel(
        base_model=model,
        overlay_model=elo_benchmark,
        alpha_on_base=alpha_on_base,
        blend_method=blend_method,
        blend_temperature=blend_temperature,
    )


def _maybe_build_calibrated_model(
    model: object,
    *,
    gamma_home: float,
    gamma_draw: float,
    gamma_away: float,
) -> object:
    if gamma_home == 1.0 and gamma_draw == 1.0 and gamma_away == 1.0:
        return model
    return CalibratedMatchModel(
        base_model=model,
        gamma_home=gamma_home,
        gamma_draw=gamma_draw,
        gamma_away=gamma_away,
    )


def _fair_odds(probability: float) -> float | None:
    if probability <= 0.0:
        return None
    return 1.0 / probability


def _strength_diff(
    home_strength: float | None,
    away_strength: float | None,
) -> float:
    if home_strength is None or away_strength is None:
        return 0.0
    return float(home_strength) - float(away_strength)


def _get_team_availability_record(
    lookup: dict[tuple[str, str], dict[str, object]],
    *,
    match_date: object,
    team: str,
) -> dict[str, object] | None:
    dated_key = _availability_lookup_key(match_date=match_date, team=team)
    if dated_key in lookup:
        return lookup[dated_key]
    undated_key = _availability_lookup_key(match_date="", team=team)
    return lookup.get(undated_key)


def _availability_lookup_key(
    *,
    match_date: object,
    team: object,
) -> tuple[str, str]:
    if pd.isna(match_date):
        date_key = ""
    elif isinstance(match_date, pd.Timestamp):
        date_key = match_date.strftime("%Y-%m-%d")
    else:
        date_text = str(match_date).strip()
        date_key = date_text[:10] if date_text else ""
    return (date_key, canonicalize_team_name(str(team)))


def _availability_elo_adjustment(
    availability_record: dict[str, object] | None,
    *,
    starter_absence_elo: float,
    goalkeeper_absence_elo: float,
) -> float:
    if not availability_record:
        return 0.0

    expected_starter_count = _coerce_float(availability_record.get("expected_starter_count"))
    if expected_starter_count <= 0.0:
        return 0.0

    expected_starter_weight_sum = _coerce_float(
        availability_record.get("expected_starter_availability_weight_sum")
    )
    missing_starter_equivalent = max(0.0, expected_starter_count - expected_starter_weight_sum)
    lineup_confidence = _coerce_float(availability_record.get("lineup_confidence"))
    confidence_weight = min(max(lineup_confidence, 0.25), 1.0)
    penalty = -starter_absence_elo * missing_starter_equivalent * confidence_weight

    expected_goalkeeper_count = _coerce_float(availability_record.get("expected_goalkeeper_count"))
    goalkeeper_available = _coerce_bool(
        availability_record.get("goalkeeper_starter_available"),
        default=True,
    )
    if expected_goalkeeper_count > 0.0 and not goalkeeper_available:
        penalty -= goalkeeper_absence_elo * confidence_weight
    return float(penalty)


def _availability_metric(
    availability_record: dict[str, object] | None,
    column: str,
) -> object:
    if not availability_record:
        return None
    return availability_record.get(column)


def _coerce_float(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _coerce_bool(value: object, *, default: bool) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _top_fixture(dataframe: pd.DataFrame, column: str) -> dict[str, object] | None:
    if dataframe.empty:
        return None
    row = dataframe.sort_values(column, ascending=False, kind="stable").iloc[0]
    return {
        "match_date": row["match_date"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        column: float(row[column]),
    }
