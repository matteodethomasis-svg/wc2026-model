"""Build the static web dashboard payload (web/data.json).

Reads the model's existing report artifacts, normalizes them, and emits a single
self-contained JSON that the static site (web/index.html) renders. No server: the
generated site can be opened locally or deployed to GitHub Pages / Netlify.

Everything here is read-only aggregation of files already produced by the model
pipeline, plus light edge/EV math via wc2026_model.markets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------- helpers --------------------------------------


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _json_default(value: Any) -> Any:
    # Coerce numpy scalar types (bool_, int64, float64) to native Python.
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _pct(value: Any) -> float | None:
    f = _num(value)
    return None if f is None else round(f * 100.0, 1)


def _fair_odds(prob: float | None) -> float | None:
    if prob is None or prob <= 0.0:
        return None
    return round(1.0 / prob, 2)


# ----------------------------- predictions ----------------------------------


def build_predictions(fixtures: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in fixtures.itertuples(index=False):
        home_p = _num(getattr(row, "home_win_probability", None))
        draw_p = _num(getattr(row, "draw_probability", None))
        away_p = _num(getattr(row, "away_win_probability", None))
        if None in (home_p, draw_p, away_p):
            continue
        # The model's most confident pick, for the "call" badge.
        picks = {"home": home_p, "draw": draw_p, "away": away_p}
        pick = max(picks, key=picks.get)
        rows.append(
            {
                "match_id": str(getattr(row, "match_id", "")),
                "date": str(getattr(row, "match_date", "")),
                "home": str(getattr(row, "home_team", "")),
                "away": str(getattr(row, "away_team", "")),
                "city": str(getattr(row, "city", "")),
                "country": str(getattr(row, "country", "")),
                "neutral": bool(getattr(row, "neutral", False) is True
                or str(getattr(row, "neutral", "")).lower() == "true"),
                "home_elo": _num(getattr(row, "home_elo", None)),
                "away_elo": _num(getattr(row, "away_elo", None)),
                "elo_diff": _num(getattr(row, "elo_diff_pre", None)),
                "home_xg": _num(getattr(row, "home_expected_goals", None)),
                "away_xg": _num(getattr(row, "away_expected_goals", None)),
                "p_home": _pct(home_p),
                "p_draw": _pct(draw_p),
                "p_away": _pct(away_p),
                "odds_home": _fair_odds(home_p),
                "odds_draw": _fair_odds(draw_p),
                "odds_away": _fair_odds(away_p),
                "pick": pick,
                "confidence": _pct(picks[pick]),
            }
        )
    rows.sort(key=lambda r: (r["date"], r["home"]))
    return rows


# ----------------------------- market edges ---------------------------------


def build_match_edges(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in comparison.itertuples(index=False):
        outcomes = []
        for key, label in (("home", "1"), ("draw", "X"), ("away", "2")):
            prob_col = "draw_probability" if key == "draw" else f"{key}_win_probability"
            model_p = _num(getattr(row, prob_col, None))
            market_p = _num(getattr(row, f"{key}_no_vig_probability", None))
            edge = _num(getattr(row, f"{key}_edge_vs_no_vig", None))
            ev = _num(getattr(row, f"{key}_expected_value", None))
            dec = _num(getattr(row, f"{key}_decimal_odds", None))
            outcomes.append(
                {
                    "label": label,
                    "side": key,
                    "model_p": _pct(model_p),
                    "market_p": _pct(market_p),
                    "edge": None if edge is None else round(edge * 100.0, 1),
                    "ev": None if ev is None else round(ev * 100.0, 1),
                    "book_odds": dec,
                }
            )
        best_edge = _num(getattr(row, "best_model_edge_no_vig", None))
        best_ev = _num(getattr(row, "best_model_ev", None))
        rows.append(
            {
                "home": str(getattr(row, "home_team", "")),
                "away": str(getattr(row, "away_team", "")),
                "date": str(getattr(row, "match_date", "")),
                "bookmaker": str(getattr(row, "bookmaker", "")),
                "source_title": str(getattr(row, "source_title", "")),
                "source_url": str(getattr(row, "source_url", "")),
                "outcomes": outcomes,
                "best_edge": None if best_edge is None else round(best_edge * 100.0, 1),
                "best_ev": None if best_ev is None else round(best_ev * 100.0, 1),
            }
        )
    rows.sort(key=lambda r: (r["best_ev"] is None, -(r["best_ev"] or 0)))
    return rows


def build_outright_edges(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in comparison.itertuples(index=False):
        model_p = _num(getattr(row, "champion_probability", None))
        market_p = _num(getattr(row, "market_probability", None))
        edge = _num(getattr(row, "edge_vs_market", None))
        ev = _num(getattr(row, "expected_value_per_yes_share", None))
        if model_p is None:
            continue
        rows.append(
            {
                "team": str(getattr(row, "team", "")),
                "model_p": _pct(model_p),
                "market_p": _pct(market_p),
                "edge": None if edge is None else round(edge * 100.0, 1),
                "ev": None if ev is None else round(ev * 100.0, 1),
                "model_odds": _fair_odds(model_p),
                "market_odds": _num(getattr(row, "market_fair_decimal_odds", None)),
                "volume": _num(getattr(row, "volume", None)),
            }
        )
    rows.sort(key=lambda r: (r["ev"] is None, -(r["ev"] or 0)))
    return rows


# ----------------------------- reliability ----------------------------------


_BENCHMARK_LABELS = {
    "elo_multinomial_xg_h2h": "Our model (xG + H2H)",
    "elo_multinomial_xg": "xG model",
    "elo_multinomial_form": "Recent form",
    "elo_multinomial": "Elo only",
    "dixon_coles_elo": "Dixon-Coles",
    "historical_prior": "Historical average",
    "uniform": "Random guess",
}


def build_reliability(summary: pd.DataFrame | None, ablation: pd.DataFrame | None) -> dict[str, Any]:
    table = ablation if ablation is not None else summary
    models: list[dict[str, Any]] = []
    if table is not None:
        # Keep only the human-readable storyline models (skip technical blend variants).
        table = table[table["model_name"].isin(_BENCHMARK_LABELS.keys())]
        best_ll = table["mean_log_loss"].min()
        for row in table.itertuples(index=False):
            name = str(getattr(row, "model_name", ""))
            ll = _num(getattr(row, "mean_log_loss", None))
            models.append(
                {
                    "name": name,
                    "label": _BENCHMARK_LABELS.get(name, name.replace("_", " ")),
                    "log_loss": None if ll is None else round(ll, 4),
                    "brier": _num_round(getattr(row, "mean_brier_score", None), 4),
                    "rps": _num_round(getattr(row, "mean_ranked_probability_score", None), 4),
                    "ece": _num_round(getattr(row, "expected_calibration_error", None), 4),
                    "is_ours": name == "elo_multinomial_xg_h2h",
                    "is_best": ll is not None and abs(ll - best_ll) < 1e-9,
                }
            )
        models.sort(key=lambda m: (m["log_loss"] is None, m["log_loss"] or 9e9))
    return {"models": models}


def _num_round(value: Any, ndigits: int) -> float | None:
    f = _num(value)
    return None if f is None else round(f, ndigits)


# ----------------------------- tournament -----------------------------------


def build_track_record(summary: dict[str, Any] | None, scored: pd.DataFrame | None) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    if scored is not None:
        for row in scored.itertuples(index=False):
            matches.append({
                "home": str(getattr(row, "home_team", "")),
                "away": str(getattr(row, "away_team", "")),
                "result": str(getattr(row, "result_side", "")),
                "model_p": _pct((getattr(row, "model_p_result", None))),
                "market_p": _pct((getattr(row, "market_p_result", None))),
                "model_ll": _num_round(getattr(row, "model_log_loss", None), 3),
                "market_ll": _num_round(getattr(row, "market_log_loss", None), 3),
                "winner": str(getattr(row, "winner", "")),
            })
    return {"summary": summary or {"resolved_matches": 0}, "matches": matches}


def build_tournament(sim: pd.DataFrame) -> dict[str, Any]:
    teams: list[dict[str, Any]] = []
    for row in sim.itertuples(index=False):
        teams.append(
            {
                "team": str(getattr(row, "team", "")),
                "group": str(getattr(row, "group", "")),
                "champion": _pct(getattr(row, "champion_probability", None)),
                "final": _pct(getattr(row, "reach_final_probability", None)),
                "semi": _pct(getattr(row, "reach_semifinal_probability", None)),
                "quarter": _pct(getattr(row, "reach_quarterfinal_probability", None)),
                "group_winner": _pct(getattr(row, "group_winner_probability", None)),
            }
        )
    teams.sort(key=lambda t: (t["champion"] is None, -(t["champion"] or 0)))
    return {"teams": teams}


# ----------------------------- main -----------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static web dashboard data payload.")
    # Default to the squad-aware recipe (expected XI + goalkeeper): it applies the
    # roster/lineup/availability layer, so e.g. France is rated on its real squad
    # strength, not just national Elo. The plain fixture file leaves that layer at
    # scale 0 (adjusted_elo == elo) and under-rates strong-squad teams.
    parser.add_argument(
        "--fixtures",
        default="reports/wc2026_fixture_predictions_expected_xi_plus_goalkeeper_recipe_2026-06-13.csv",
    )
    parser.add_argument(
        "--match-edges",
        default="reports/bookmaker_match_odds_expected_xi_plus_goalkeeper_live_sample_comparison.csv",
    )
    parser.add_argument(
        "--outright-edges",
        default="reports/polymarket_world_cup_winner_live_comparison.csv",
    )
    parser.add_argument("--backtest-summary", default="reports/benchmark_backtest_summary.csv")
    parser.add_argument(
        "--ablation-summary", default="reports/benchmark_backtest_summary_xg_ablation.csv"
    )
    parser.add_argument(
        "--tournament",
        default="reports/wc2026_simulation_expected_xi_plus_goalkeeper_probabilities.csv",
    )
    parser.add_argument("--track-summary", default="reports/track_record_summary.json")
    parser.add_argument("--track-scored", default="reports/track_record_match_scored.csv")
    parser.add_argument("--output", default="web/data.json")
    args = parser.parse_args()

    fixtures = _read_csv(ROOT / args.fixtures)
    match_cmp = _read_csv(ROOT / args.match_edges)
    outright_cmp = _read_csv(ROOT / args.outright_edges)
    summary = _read_csv(ROOT / args.backtest_summary)
    ablation = _read_csv(ROOT / args.ablation_summary)
    sim = _read_csv(ROOT / args.tournament)

    payload: dict[str, Any] = {
        "meta": {
            "generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
            "model": "Elo + Dixon-Coles + xG + per-player squad layer (PlayerElo XI + GK)",
        },
        "predictions": build_predictions(fixtures) if fixtures is not None else [],
        "match_edges": build_match_edges(match_cmp) if match_cmp is not None else [],
        "outright_edges": build_outright_edges(outright_cmp) if outright_cmp is not None else [],
        "reliability": build_reliability(summary, ablation),
        "tournament": build_tournament(sim) if sim is not None else {"teams": []},
        "track_record": build_track_record(
            _read_json(ROOT / args.track_summary),
            _read_csv(ROOT / args.track_scored),
        ),
    }

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    print(f"Wrote {out_path}")
    print(
        f"  predictions={len(payload['predictions'])}"
        f" match_edges={len(payload['match_edges'])}"
        f" outright={len(payload['outright_edges'])}"
        f" benchmarks={len(payload['reliability']['models'])}"
        f" teams={len(payload['tournament']['teams'])}"
        f" tracked={len(payload['track_record']['matches'])}"
    )


if __name__ == "__main__":
    main()
