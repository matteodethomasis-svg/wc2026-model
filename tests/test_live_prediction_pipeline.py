from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from wc2026_model.pipeline import (
    predict_world_cup_fixtures,
    save_world_cup_fixture_predictions,
)
from wc2026_model.types import ThreeWayProbabilities


@dataclass
class _StubModel:
    probabilities: ThreeWayProbabilities

    def predict_expected_goals(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
    ) -> tuple[float, float]:
        elo_shift = float(elo_diff_pre) / 1000.0
        return (1.5 + elo_shift, 1.1 - elo_shift)

    def predict_outcome_probabilities(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
        max_goals: int = 10,
    ) -> ThreeWayProbabilities:
        return self.probabilities


def test_save_world_cup_fixture_predictions_applies_live_availability_overlay(
    tmp_path: Path,
) -> None:
    model_path = _write_model_pickle(tmp_path / "model.pkl")
    fixtures_path = _write_fixture_input(
        tmp_path / "fixtures.csv",
        [
            {
                "date": "2026-06-15",
                "home_team": "France",
                "away_team": "Senegal",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
                "city": "Mexico City",
                "country": "Mexico",
                "neutral": True,
            }
        ],
    )
    elo_path = _write_elo_input(tmp_path / "elo.csv")
    training_frame_path = _write_training_frame(tmp_path / "training.csv")
    availability_path = _write_availability_input(
        tmp_path / "availability.csv",
        [
            {
                "match_date": "2026-06-15",
                "team": "France",
                "expected_starter_count": 11,
                "expected_starter_availability_weight_sum": 11.0,
                "unavailable_expected_starter_count": 0,
                "doubtful_expected_starter_count": 0,
                "expected_goalkeeper_count": 1,
                "goalkeeper_starter_available": True,
                "lineup_confidence": 1.0,
            },
            {
                "match_date": "2026-06-15",
                "team": "Senegal",
                "expected_starter_count": 11,
                "expected_starter_availability_weight_sum": 9.0,
                "unavailable_expected_starter_count": 2,
                "doubtful_expected_starter_count": 0,
                "expected_goalkeeper_count": 1,
                "goalkeeper_starter_available": False,
                "lineup_confidence": 1.0,
            },
        ],
    )
    output_path = tmp_path / "predictions.csv"
    summary_output_path = tmp_path / "summary.json"

    predictions, summary = save_world_cup_fixture_predictions(
        model_input=model_path,
        fixtures_input=fixtures_path,
        elo_ratings_input=elo_path,
        training_frame_input=training_frame_path,
        output=output_path,
        summary_output=summary_output_path,
        availability_input=availability_path,
    )

    assert len(predictions) == 1
    row = predictions.iloc[0]
    assert row["availability_elo_diff_pre"] == pytest.approx(60.0)
    assert row["adjusted_elo_diff_pre"] == pytest.approx(210.0)
    assert row["home_availability_elo_adjustment"] == pytest.approx(0.0)
    assert row["away_availability_elo_adjustment"] == pytest.approx(-60.0)
    assert row["home_expected_goals"] == pytest.approx(1.71)
    assert row["away_expected_goals"] == pytest.approx(0.89)
    assert summary["fixture_count"] == 1
    assert summary["fixtures_with_availability_count"] == 1
    assert summary["highest_home_win_probability"] == {
        "match_date": "2026-06-15",
        "home_team": "France",
        "away_team": "Senegal",
        "home_win_probability": 0.55,
    }

    saved_predictions = pd.read_csv(output_path)
    saved_summary = json.loads(summary_output_path.read_text(encoding="utf-8"))
    assert saved_predictions.loc[0, "home_team"] == "France"
    assert saved_summary["highest_away_win_probability"]["away_team"] == "Senegal"


def test_predict_world_cup_fixtures_handles_empty_fixture_set(tmp_path: Path) -> None:
    model_path = _write_model_pickle(tmp_path / "model.pkl")
    fixtures_path = _write_fixture_input(
        tmp_path / "fixtures.csv",
        [
            {
                "date": "2026-06-10",
                "home_team": "France",
                "away_team": "Senegal",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
                "city": "Mexico City",
                "country": "Mexico",
                "neutral": True,
            }
        ],
    )
    elo_path = _write_elo_input(tmp_path / "elo.csv")
    training_frame_path = _write_training_frame(tmp_path / "training.csv")

    predictions, summary = predict_world_cup_fixtures(
        model_input=model_path,
        fixtures_input=fixtures_path,
        elo_ratings_input=elo_path,
        training_frame_input=training_frame_path,
        start_date="2026-06-12",
    )

    assert predictions.empty
    assert "home_win_probability" in predictions.columns
    assert summary["fixture_count"] == 0
    assert summary["fixtures_with_availability_count"] == 0
    assert summary["highest_home_win_probability"] is None
    assert summary["highest_draw_probability"] is None
    assert summary["highest_away_win_probability"] is None


def _write_model_pickle(path: Path) -> Path:
    with path.open("wb") as file_handle:
        pickle.dump(
            _StubModel(
                probabilities=ThreeWayProbabilities(
                    home=0.55,
                    draw=0.25,
                    away=0.20,
                )
            ),
            file_handle,
        )
    return path


def _write_fixture_input(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_elo_input(path: Path) -> Path:
    pd.DataFrame(
        [
            {"team": "France", "elo_rating": 1900.0},
            {"team": "Senegal", "elo_rating": 1750.0},
        ]
    ).to_csv(path, index=False)
    return path


def _write_training_frame(path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "home_team": "France",
                "away_team": "Senegal",
                "neutral": True,
                "elo_diff_pre": 150.0,
                "home_result": "home",
            }
        ]
    ).to_csv(path, index=False)
    return path


def _write_availability_input(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
