from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping


OUTCOME_HOME = "home"
OUTCOME_DRAW = "draw"
OUTCOME_AWAY = "away"
THREE_WAY_OUTCOMES = (OUTCOME_HOME, OUTCOME_DRAW, OUTCOME_AWAY)


@dataclass(frozen=True)
class MatchInfo:
    match_id: str
    home_team: str
    away_team: str
    kickoff_utc: datetime | None = None
    competition: str | None = None
    season: str | None = None
    is_neutral_site: bool = False


@dataclass(frozen=True)
class ThreeWayProbabilities:
    home: float
    draw: float
    away: float

    def as_dict(self) -> dict[str, float]:
        return {
            OUTCOME_HOME: self.home,
            OUTCOME_DRAW: self.draw,
            OUTCOME_AWAY: self.away,
        }


@dataclass(frozen=True)
class PredictionRecord:
    match: MatchInfo
    probabilities: ThreeWayProbabilities
    model_name: str
    generated_at_utc: datetime | None = None


@dataclass(frozen=True)
class BookmakerQuote:
    bookmaker: str
    home_odds: float
    draw_odds: float
    away_odds: float
    captured_at_utc: datetime | None = None


@dataclass(frozen=True)
class MarketComparison:
    match: MatchInfo
    model_name: str
    model_probabilities: ThreeWayProbabilities
    fair_market_probabilities: ThreeWayProbabilities
    raw_market_implied_probabilities: ThreeWayProbabilities
    quote: BookmakerQuote
    edges: Mapping[str, float]

