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


def _played_results_lookup(results: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, Any]]:
    if results is None or results.empty:
        return {}
    res = results.copy()
    res["match_date"] = pd.to_datetime(res["match_date"], errors="coerce")
    wc = res[(res["tournament"] == "FIFA World Cup") & (res["match_date"].dt.year == 2026)]
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for r in wc.itertuples(index=False):
        hg, ag = _num(r.home_goals), _num(r.away_goals)
        if hg is None or ag is None:
            continue
        side = "home" if hg > ag else "away" if hg < ag else "draw"
        lookup[(str(r.home_team), str(r.away_team))] = {
            "home_goals": int(hg), "away_goals": int(ag), "result_side": side,
        }
    return lookup


def _kickoff_lookup(kickoffs: pd.DataFrame | None) -> dict[tuple[str, str], pd.Timestamp]:
    if kickoffs is None or kickoffs.empty:
        return {}
    return {
        (str(r.home_team), str(r.away_team)): r.kickoff_ts
        for r in kickoffs.itertuples(index=False)
    }


_POSITION_ORDER = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}


def _lineups_by_team(lineups: pd.DataFrame | None) -> dict[str, list[dict[str, str]]]:
    """Map team -> its starting XI (ordered GK→DF→MF→FW) from the ESPN lineups feed.
    Keyed by team only (one upcoming fixture per team at a time)."""
    if lineups is None or lineups.empty:
        return {}
    out: dict[str, list[dict[str, str]]] = {}
    starters = lineups[lineups["is_expected_starter"].astype(str).str.lower().isin(["true", "1"])]
    for team, grp in starters.groupby("team"):
        players = [
            {"name": str(r.player), "pos": str(getattr(r, "position", "") or "")}
            for r in grp.itertuples(index=False)
        ]
        players.sort(key=lambda p: _POSITION_ORDER.get(p["pos"], 9))
        out[str(team)] = players
    return out


def build_predictions(
    fixtures: pd.DataFrame,
    results: pd.DataFrame | None = None,
    kickoffs: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    played = _played_results_lookup(results)
    ko = _kickoff_lookup(kickoffs)
    xi = _lineups_by_team(lineups)
    now = pd.Timestamp.now(tz="UTC")
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
        home_team = str(getattr(row, "home_team", ""))
        away_team = str(getattr(row, "away_team", ""))
        result = played.get((home_team, away_team))
        # A match is "started" once its real kickoff (UTC) has passed — timezone-correct,
        # so a fixture can disappear from the upcoming tools the moment it kicks off,
        # even before ESPN posts the final result.
        kickoff = ko.get((home_team, away_team))
        started = bool(result is not None or (kickoff is not None and now >= kickoff))
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
                "played": result is not None,
                "started": started,
                # Official starting XI (from ESPN), shown on the upcoming card once it's
                # published ~1h pre-kickoff. Only attached while the match is still upcoming.
                "home_xi": (xi.get(home_team, []) if not started else []),
                "away_xi": (xi.get(away_team, []) if not started else []),
                "result_home_goals": (result or {}).get("home_goals"),
                "result_away_goals": (result or {}).get("away_goals"),
                "result_side": (result or {}).get("result_side"),
                "pick_correct": (None if result is None else pick == result["result_side"]),
            }
        )
    # Upcoming matches first (sorted by date), played matches pushed to the bottom.
    rows.sort(key=lambda r: (r["played"], r["date"], r["home"]))
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
        # Our most confident outcome, used to sort the table by OUR call (not by edge).
        our_top = max((o["model_p"] or 0) for o in outcomes) if outcomes else 0
        rows.append(
            {
                "home": str(getattr(row, "home_team", "")),
                "away": str(getattr(row, "away_team", "")),
                "date": str(getattr(row, "match_date", "")),
                "bookmaker": str(getattr(row, "bookmaker", "Polymarket")),
                "source_title": str(getattr(row, "source_title", "")),
                "source_url": str(getattr(row, "source_url", "")),
                "outcomes": outcomes,
                "best_edge": None if best_edge is None else round(best_edge * 100.0, 1),
                "best_ev": None if best_ev is None else round(best_ev * 100.0, 1),
                "_our_top": our_top,
            }
        )
    # Sort by our most confident pick (descending), then date. Keeps the strongest
    # model calls on top rather than amplifying noisy high-EV longshots.
    rows.sort(key=lambda r: (-r["_our_top"], r["date"]))
    for r in rows:
        r.pop("_our_top", None)
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
    # Order by OUR probability (descending), not by edge — the market table should read
    # as "our title board", with the edge shown alongside, not driving the sort.
    rows.sort(key=lambda r: (r["model_p"] is None, -(r["model_p"] or 0)))
    return rows


def _per_team_market_rows(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows for a per-team Yes-market comparison (rounds / group winner), already
    sorted by our probability upstream. Market prob is the RAW Polymarket Yes price."""
    rows: list[dict[str, Any]] = []
    for row in comparison.itertuples(index=False):
        model_p = _num(getattr(row, "model_probability", None))
        if model_p is None:
            continue
        market_p = _num(getattr(row, "market_probability", None))
        edge = _num(getattr(row, "edge_vs_market", None))
        rows.append(
            {
                "team": str(getattr(row, "team", "")),
                "model_p": _pct(model_p),
                "market_p": _pct(market_p),
                "edge": None if edge is None else round(edge * 100.0, 1),
                "model_odds": _fair_odds(model_p),
                "market_odds": _fair_odds(market_p),
                "volume": _num(getattr(row, "volume", None)),
            }
        )
    return rows


# Human labels for the round-market dropdown.
_ROUND_LABELS = {
    "advance": "Advance past group",
    "r16": "Reach Round of 16",
    "quarter": "Reach Quarterfinals",
    "semi": "Reach Semifinals",
    "final": "Reach Final",
}


def build_market_comparisons(
    matches: pd.DataFrame | None,
    rounds: pd.DataFrame | None,
    groups: pd.DataFrame | None,
) -> dict[str, Any]:
    """Everything for the dropdown-driven 'Edge vs Market' tab: ante-post per-match
    1X2, per-team round markets, and per-group winner markets — all vs Polymarket."""
    out: dict[str, Any] = {"matches": [], "rounds": [], "groups": []}

    # Per-match 1X2 (already ante-post + sorted by our pick upstream).
    if matches is not None and not matches.empty:
        out["matches"] = build_match_edges(matches)

    # Round markets: a list of {key,label,rows} for the dropdown.
    if rounds is not None and not rounds.empty and "market_key" in rounds.columns:
        for key, label in _ROUND_LABELS.items():
            sub = rounds[rounds["market_key"] == key]
            if sub.empty:
                continue
            out["rounds"].append({"key": key, "label": label, "rows": _per_team_market_rows(sub)})

    # Group winner: one entry per group letter.
    if groups is not None and not groups.empty and "group" in groups.columns:
        for letter in sorted(groups["group"].dropna().unique()):
            sub = groups[groups["group"] == letter]
            out["groups"].append({"key": str(letter), "label": f"Group {letter} winner",
                                  "rows": _per_team_market_rows(sub)})
    return out


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


def build_sim_engine(
    elo: pd.DataFrame | None,
    squad: pd.DataFrame | None,
    groups: pd.DataFrame | None,
    fixtures: pd.DataFrame | None,
    results: pd.DataFrame | None,
    *,
    squad_scale: float,
    gk_scale: float,
) -> dict[str, Any]:
    """Ingredients for an in-browser Monte Carlo: each team's combined rating (the
    same national-Elo + squad layers the Python sim uses), the groups, the remaining
    fixtures, and results already played. The JS samples match scores from the rating
    gap (Poisson) and propagates through groups + the knockout bracket.
    """
    if elo is None or groups is None:
        return {}
    elo_map = {str(r.team): _num(r.elo_rating) for r in elo.itertuples(index=False)}
    xi_map, gk_map = {}, {}
    if squad is not None:
        for r in squad.itertuples(index=False):
            xi_map[str(r.team)] = _num(getattr(r, "expected_xi_player_elo_rating", None))
            gk_map[str(r.team)] = _num(getattr(r, "expected_xi_goalkeeper_player_elo_rating", None))

    team_ratings: dict[str, float] = {}
    group_of: dict[str, str] = {}
    for r in groups.itertuples(index=False):
        team = str(r.team)
        group_of[team] = str(r.group)
        base = elo_map.get(team)
        if base is None:
            continue
        rating = base
        if xi_map.get(team) is not None:
            rating += squad_scale * xi_map[team]
        if gk_map.get(team) is not None:
            rating += gk_scale * gk_map[team]
        team_ratings[team] = round(rating, 1)

    # Results already played (fixed in the sim).
    played = _played_results_lookup(results)
    played_list = [
        {"home": h, "away": a, "home_goals": v["home_goals"], "away_goals": v["away_goals"]}
        for (h, a), v in played.items()
    ]
    # Group-stage fixtures still to play.
    remaining = []
    if fixtures is not None:
        for r in fixtures.itertuples(index=False):
            h, a = str(getattr(r, "home_team", "")), str(getattr(r, "away_team", ""))
            if (h, a) in played:
                continue
            if h in group_of and a in group_of and group_of[h] == group_of[a]:
                remaining.append({"home": h, "away": a, "group": group_of[h]})

    teams = [{"team": t, "group": group_of.get(t, ""), "rating": team_ratings[t]}
             for t in sorted(team_ratings)]
    return {
        "teams": teams,
        "played": played_list,
        "remaining_group_fixtures": remaining,
        "home_advantage": 60.0,  # modest; most WC games are at neutral venues
    }


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
        # Ante-post per-match 1X2 vs Polymarket (future kickoffs only).
        default="reports/polymarket_match_edges_antepost_comparison.csv",
    )
    parser.add_argument(
        "--round-edges", default="reports/polymarket_round_comparison.csv",
    )
    parser.add_argument(
        "--group-edges", default="reports/polymarket_group_winner_comparison.csv",
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
    parser.add_argument("--results", default="data/interim/international_results_augmented.csv")
    parser.add_argument("--lineups", default="data/interim/wc2026_expected_lineups_espn.csv")
    parser.add_argument("--elo", default="reports/baseline_latest_elo_ratings.csv")
    parser.add_argument("--squad", default="reports/wc2026_squad_strength_player_elo_ratings.csv")
    parser.add_argument("--groups", default="data/reference/wc2026_groups_actual.csv")
    parser.add_argument("--squad-scale", type=float, default=0.52)
    parser.add_argument("--gk-scale", type=float, default=0.276)
    parser.add_argument("--output", default="web/data.json")
    args = parser.parse_args()

    fixtures = _read_csv(ROOT / args.fixtures)
    match_cmp = _read_csv(ROOT / args.match_edges)
    round_cmp = _read_csv(ROOT / args.round_edges)
    group_cmp = _read_csv(ROOT / args.group_edges)
    outright_cmp = _read_csv(ROOT / args.outright_edges)
    summary = _read_csv(ROOT / args.backtest_summary)
    ablation = _read_csv(ROOT / args.ablation_summary)
    sim = _read_csv(ROOT / args.tournament)
    results = _read_csv(ROOT / args.results)
    elo = _read_csv(ROOT / args.elo)
    squad = _read_csv(ROOT / args.squad)
    groups = _read_csv(ROOT / args.groups)

    # Kickoff times (UTC) let us mark a fixture "started" the moment it kicks off —
    # timezone-correct — so the upcoming-only tools (predict-a-game, edge match list)
    # drop it immediately, not only once ESPN posts the final score. Free ESPN call;
    # non-fatal if it hiccups (then nothing is flagged started early, only `played`).
    kickoffs = None
    try:
        from datetime import date as _date, timedelta as _timedelta

        from wc2026_model.data import fetch_world_cup_kickoffs
        end = (_date.today() + _timedelta(days=2)).isoformat()
        kickoffs = fetch_world_cup_kickoffs("2026-06-08", end)
    except Exception as exc:  # pragma: no cover - network best-effort
        print(f"  (kickoff fetch skipped: {exc})")

    # Official XIs (from the ESPN lineups feed produced by the refresh) for the upcoming cards.
    lineups = _read_csv(ROOT / args.lineups)

    payload: dict[str, Any] = {
        "meta": {
            "generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
            "model": "Elo + Dixon-Coles + xG + per-player squad layer (PlayerElo XI + GK)",
        },
        "predictions": build_predictions(fixtures, results, kickoffs, lineups) if fixtures is not None else [],
        "match_edges": build_match_edges(match_cmp) if match_cmp is not None else [],
        "outright_edges": build_outright_edges(outright_cmp) if outright_cmp is not None else [],
        "market_comparisons": build_market_comparisons(match_cmp, round_cmp, group_cmp),
        "reliability": build_reliability(summary, ablation),
        "tournament": build_tournament(sim) if sim is not None else {"teams": []},
        "track_record": build_track_record(
            _read_json(ROOT / args.track_summary),
            _read_csv(ROOT / args.track_scored),
        ),
        "sim_engine": build_sim_engine(
            elo, squad, groups, fixtures, results,
            squad_scale=args.squad_scale, gk_scale=args.gk_scale,
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
