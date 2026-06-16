"""Measure whether h2h (and, on the StatsBomb panel, xG) add edge ON TOP OF the
LIVE model regime — i.e. Dixon-Coles + Elo blend (alpha) + squad/GK strength +
power calibration — rather than on the bare logistic benchmark where they were
originally ablated.

Why this script exists (diagnosis 2026-06-15): the validated wins (h2h = only
regressor with edge in the ablation; the whole xG layer) live ONLY in the
`EloMultinomialBenchmark` / xG-challenger track. The LIVE model
(`baseline_dixon_coles_elo.pkl`, served by refresh_model_and_site.py) never sees
them — every signal funnels through the single scalar `adjusted_elo_diff`. Before
wiring anything, we measure each candidate IN THE LIVE REGIME, leak-free.

Two faithful caveats this script is built to expose:
  1. The benchmark validated h2h as LOGISTIC FEATURES with a fitted weight. Here it
     enters as a scalar Elo-like adjustment with a SWEPT scale (like the squad
     layer). A null result could mean "h2h doesn't help the live model" OR "the
     scalar encoding is weaker than the logistic one". We sweep the scale so a real
     signal has a chance to show; if the best scale is 0, h2h adds nothing here.
  2. xG rolling features only exist on the StatsBomb panel, so the xG arm runs on
     that panel; the h2h arm runs on the full results file (the true WC backtest).

Outputs a single JSON: baseline vs +h2h (and vs +xg on the panel), same folds,
same recipe, so the decision is data-driven.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wc2026_model.data import canonicalize_team_name, load_international_results
from wc2026_model.evaluation import EloMultinomialBenchmark
from wc2026_model.evaluation.scoring import (
    brier_score_three_way,
    log_loss_three_way,
    ranked_probability_score,
)
from wc2026_model.evaluation.backtest import generate_rolling_cutoffs
from wc2026_model.features import augment_with_pre_match_elo
from wc2026_model.features.international_context import augment_with_pre_match_h2h_features
from wc2026_model.features.world_cup_xg import (
    WorldCupXGConfig,
    augment_with_pre_match_xg_features,
)
from wc2026_model.models import blend_three_way_probabilities, power_calibrate_probabilities
from wc2026_model.pipeline import BaselineTrainingConfig, train_baseline_model


class _ScaledEloRow:
    def __init__(self, *, elo_diff_pre: float, neutral: bool) -> None:
        self.elo_diff_pre = elo_diff_pre
        self.neutral = neutral


def _build_team_strength_lookup(
    squad_strengths: pd.DataFrame, *, year: int, rating_column: str
) -> dict[str, float]:
    if rating_column not in squad_strengths.columns:
        raise ValueError(f"Squad-strength frame missing column '{rating_column}'.")
    year_frame = squad_strengths.loc[squad_strengths["tournament_year"] == year]
    return {
        canonicalize_team_name(str(row.team)): float(getattr(row, rating_column))
        for row in year_frame.loc[:, ["team", rating_column]].itertuples(index=False)
        if pd.notna(getattr(row, rating_column))
    }


def _strength_adjust(base: float, home: str, away: str, lookup: dict[str, float], scale: float) -> float:
    h = float(lookup.get(canonicalize_team_name(home), 0.0))
    a = float(lookup.get(canonicalize_team_name(away), 0.0))
    return float(base) + float(scale) * (h - a)


def _h2h_net(row: object) -> float:
    """Directional, leak-free h2h advantage in [-1, 1]; 0 when no prior meetings."""
    count = getattr(row, "h2h_match_count", 0.0)
    if count is None or pd.isna(count) or float(count) <= 0.0:
        return 0.0
    home_rate = float(getattr(row, "h2h_home_win_rate", 0.0) or 0.0)
    away_rate = float(getattr(row, "h2h_away_win_rate", 0.0) or 0.0)
    return home_rate - away_rate


def _xg_net(row: object) -> float:
    """Leak-free rolling net-xG differential; 0 when either team lacks history."""
    vals = [
        getattr(row, c, np.nan)
        for c in ("home_xg_for_per_match", "home_xg_against_per_match",
                  "away_xg_for_per_match", "away_xg_against_per_match")
    ]
    if any(v is None or pd.isna(v) for v in vals):
        return 0.0
    home_for, home_against, away_for, away_against = (float(v) for v in vals)
    return (home_for - home_against) - (away_for - away_against)


def _score(probabilities, outcome: str) -> dict[str, float]:
    return {
        "log_loss": log_loss_three_way(probabilities, outcome),
        "brier_score": brier_score_three_way(probabilities, outcome),
        "ranked_probability_score": ranked_probability_score(probabilities, outcome),
    }


def _run_panel_mode(args) -> dict:
    """Expanding-window over the StatsBomb panel, live regime (DC + blend + gamma),
    sweeping an xG scalar and an h2h scalar. No squad layer (no panel-wide squad
    ratings exist). Answers: do xG/h2h survive the live wrapper on the panel where
    the ablation originally found h2h's edge?"""
    panel = load_international_results(args.results_input)
    # Bring the raw per-match xG columns across (loader may drop unknown cols).
    raw = pd.read_csv(args.results_input)
    raw["match_date"] = pd.to_datetime(raw["match_date"], errors="raise")
    panel["match_date"] = pd.to_datetime(panel["match_date"], errors="raise")
    for col in ("home_xg", "away_xg"):
        if col not in panel.columns and col in raw.columns:
            panel = panel.merge(
                raw[["match_date", "home_team", "away_team", col]],
                on=["match_date", "home_team", "away_team"], how="left",
            )
    panel = panel.sort_values(["match_date", "home_team", "away_team"], kind="stable").reset_index(drop=True)

    full = augment_with_pre_match_elo(panel, config=BaselineTrainingConfig().elo_config)
    full = augment_with_pre_match_h2h_features(full)
    full = augment_with_pre_match_xg_features(full, config=WorldCupXGConfig(window_size=3))

    cutoffs = generate_rolling_cutoffs(
        full, start_date=args.backtest_start, end_date=args.backtest_end, step_days=args.step_days
    )
    xg_scales = [float(s) for s in args.xg_scale_grid.split(",") if s.strip()]
    h2h_scales = [float(s) for s in args.h2h_scale_grid.split(",") if s.strip()]

    rows: list[dict[str, object]] = []
    for cutoff in cutoffs:
        test_end = cutoff + pd.Timedelta(days=args.test_window_days)
        model, training_frame = train_baseline_model(
            panel,
            config=BaselineTrainingConfig(
                min_match_date=args.min_match_date,
                training_cutoff=cutoff.strftime("%Y-%m-%d"),
                min_team_matches=1,
            ),
        )
        if training_frame.empty:
            continue
        test = full[(full["match_date"] >= cutoff) & (full["match_date"] < test_end)].copy()
        test = test[test["home_team"].isin(model.teams) & test["away_team"].isin(model.teams)]
        if test.empty:
            continue
        elo_benchmark = EloMultinomialBenchmark.fit(training_frame)
        for xg_scale in xg_scales:
            for h2h_scale in h2h_scales:
                for row in test.itertuples(index=False):
                    adjusted = float(row.elo_diff_pre)
                    adjusted += xg_scale * _xg_net(row)
                    adjusted += h2h_scale * _h2h_net(row)
                    base = model.predict_outcome_probabilities(
                        row.home_team, row.away_team, neutral_site=bool(row.neutral),
                        elo_diff_pre=adjusted, max_goals=args.max_goals,
                    )
                    blended = blend_three_way_probabilities(
                        base,
                        elo_benchmark.predict_proba(
                            _ScaledEloRow(elo_diff_pre=adjusted, neutral=bool(row.neutral))
                        ),
                        alpha_on_base=args.hybrid_alpha,
                    )
                    probs = power_calibrate_probabilities(
                        blended, gamma_home=args.gamma_home,
                        gamma_draw=args.gamma_draw, gamma_away=args.gamma_away,
                    )
                    rows.append({
                        "xg_scale": xg_scale, "h2h_scale": h2h_scale,
                        **_score(probs, str(row.home_result)),
                    })

    frame = pd.DataFrame(rows)
    arms = []
    for (xg_scale, h2h_scale), grp in frame.groupby(["xg_scale", "h2h_scale"], sort=True):
        arms.append({
            "xg_scale": float(xg_scale), "h2h_scale": float(h2h_scale),
            "predictions": int(len(grp)),
            "mean_log_loss": float(grp["log_loss"].mean()),
            "mean_brier_score": float(grp["brier_score"].mean()),
            "mean_rps": float(grp["ranked_probability_score"].mean()),
        })
    arms.sort(key=lambda a: a["mean_log_loss"])
    baseline = next(a for a in arms if a["xg_scale"] == 0.0 and a["h2h_scale"] == 0.0)
    best = arms[0]
    return {
        "regime": "live panel (DC + elo-blend + power-calibration, NO squad layer)",
        "results_input": args.results_input,
        "cutoffs": [c.strftime("%Y-%m-%d") for c in cutoffs],
        "baseline_no_xg_no_h2h": baseline,
        "best_arm": best,
        "delta_log_loss_vs_baseline": best["mean_log_loss"] - baseline["mean_log_loss"],
        "verdict": (
            "xG/h2h ADD edge in live regime"
            if (best["xg_scale"] != 0.0 or best["h2h_scale"] != 0.0)
            and best["mean_log_loss"] < baseline["mean_log_loss"] - 1e-6
            else "xG/h2h add NOTHING in live regime (best = baseline)"
        ),
        "all_arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["wc", "panel"], default="wc",
                        help="wc = faithful WC18/22 live regime (h2h only). "
                             "panel = StatsBomb panel folds, xG + h2h in live regime.")
    parser.add_argument("--results-input", default="data/interim/international_results_augmented.csv")
    parser.add_argument("--squad-strength-input",
                        default="reports/historical_world_cup_real_xi_squad_strength.csv")
    parser.add_argument("--rating-column", default="real_xi_club_elo_rating",
                        help="Squad layer held FIXED across arms (the only changing var is h2h).")
    parser.add_argument("--squad-scale", type=float, default=1.5,
                        help="Squad scale for the fixed squad layer (validated real-XI optimum).")
    parser.add_argument("--years", default="2018,2022")
    parser.add_argument("--hybrid-alpha", type=float, default=0.75)
    parser.add_argument("--gamma-home", type=float, default=1.05)
    parser.add_argument("--gamma-draw", type=float, default=1.0)
    parser.add_argument("--gamma-away", type=float, default=1.1)
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--min-match-date", default="2010-01-01")
    parser.add_argument("--min-team-matches", type=int, default=10)
    parser.add_argument("--h2h-scale-grid", default="0,15,30,60,120",
                        help="Elo-points per unit of h2h net win-rate. 0 = baseline (no h2h).")
    parser.add_argument("--xg-scale-grid", default="0,20,40,80,160",
                        help="(panel mode) Elo-points per unit of net-xG diff. 0 = no xG.")
    parser.add_argument("--backtest-start", default="2021-06-01")
    parser.add_argument("--backtest-end", default=None)
    parser.add_argument("--step-days", type=int, default=540)
    parser.add_argument("--test-window-days", type=int, default=45)
    parser.add_argument("--output", default="reports/live_regime_h2h_xg_eval.json")
    args = parser.parse_args()

    if args.mode == "panel":
        out = _run_panel_mode(args)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    years = [int(y) for y in args.years.split(",") if y.strip()]
    h2h_scales = [float(s) for s in args.h2h_scale_grid.split(",") if s.strip()]

    results = load_international_results(args.results_input)
    results["match_date"] = pd.to_datetime(results["match_date"], errors="raise")
    results = results.sort_values(["match_date", "home_team", "away_team"], kind="stable").reset_index(drop=True)
    squad_strengths = pd.read_csv(args.squad_strength_input)

    config = BaselineTrainingConfig(
        min_match_date=args.min_match_date,
        min_team_matches=args.min_team_matches,
    )
    # Pre-match Elo + pre-match h2h (both leak-free: computed from history only).
    full = augment_with_pre_match_elo(results, config=config.elo_config)
    full = augment_with_pre_match_h2h_features(full)

    rows: list[dict[str, object]] = []
    for year in years:
        tournament_frame = full[
            (full["tournament"] == "FIFA World Cup") & (full["match_date"].dt.year == year)
        ].copy()
        if tournament_frame.empty:
            continue
        start = pd.Timestamp(tournament_frame["match_date"].min())
        model, training_frame = train_baseline_model(
            results,
            config=BaselineTrainingConfig(
                min_match_date=args.min_match_date,
                training_cutoff=start.strftime("%Y-%m-%d"),
                min_team_matches=args.min_team_matches,
                elo_config=config.elo_config,
            ),
        )
        tournament_frame = tournament_frame[
            tournament_frame["home_team"].isin(model.teams)
            & tournament_frame["away_team"].isin(model.teams)
        ].reset_index(drop=True)
        if tournament_frame.empty or training_frame.empty:
            continue
        elo_benchmark = EloMultinomialBenchmark.fit(training_frame)
        squad_lookup = _build_team_strength_lookup(
            squad_strengths, year=year, rating_column=args.rating_column
        )

        for h2h_scale in h2h_scales:
            for row in tournament_frame.itertuples(index=False):
                adjusted = _strength_adjust(
                    float(row.elo_diff_pre), str(row.home_team), str(row.away_team),
                    squad_lookup, args.squad_scale,
                )
                adjusted += h2h_scale * _h2h_net(row)
                base_probs = model.predict_outcome_probabilities(
                    row.home_team, row.away_team,
                    neutral_site=bool(row.neutral),
                    elo_diff_pre=adjusted, max_goals=args.max_goals,
                )
                blended = blend_three_way_probabilities(
                    base_probs,
                    elo_benchmark.predict_proba(
                        _ScaledEloRow(elo_diff_pre=adjusted, neutral=bool(row.neutral))
                    ),
                    alpha_on_base=args.hybrid_alpha,
                )
                probs = power_calibrate_probabilities(
                    blended, gamma_home=args.gamma_home,
                    gamma_draw=args.gamma_draw, gamma_away=args.gamma_away,
                )
                rows.append({
                    "h2h_scale": h2h_scale,
                    "year": year,
                    "h2h_net": _h2h_net(row),
                    "outcome": str(row.home_result),
                    **_score(probs, str(row.home_result)),
                })

    frame = pd.DataFrame(rows)
    arms = []
    for h2h_scale, grp in frame.groupby("h2h_scale", sort=True):
        arms.append({
            "h2h_scale": float(h2h_scale),
            "predictions": int(len(grp)),
            "mean_log_loss": float(grp["log_loss"].mean()),
            "mean_brier_score": float(grp["brier_score"].mean()),
            "mean_rps": float(grp["ranked_probability_score"].mean()),
        })
    arms.sort(key=lambda a: a["mean_log_loss"])
    baseline = next(a for a in arms if a["h2h_scale"] == 0.0)
    best = arms[0]

    out = {
        "regime": "live (DC + elo-blend alpha + squad + power-calibration)",
        "results_input": args.results_input,
        "squad_rating_column": args.rating_column,
        "squad_scale": args.squad_scale,
        "years": years,
        "matches_with_prior_h2h": int((frame.loc[frame["h2h_scale"] == 0.0, "h2h_net"] != 0.0).sum()),
        "total_matches": int(len(frame.loc[frame["h2h_scale"] == 0.0])),
        "baseline_no_h2h": baseline,
        "best_arm": best,
        "h2h_delta_log_loss": best["mean_log_loss"] - baseline["mean_log_loss"],
        "verdict": (
            "h2h ADDS edge in live regime" if best["h2h_scale"] != 0.0
            and best["mean_log_loss"] < baseline["mean_log_loss"] - 1e-6
            else "h2h adds NOTHING in live regime (best scale = 0)"
        ),
        "all_arms": arms,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
