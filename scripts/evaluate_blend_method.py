"""Compare the LINEAR convex blend (current live) vs LOG-OPINION POOLING (the blend
DrElegantia's ensemble uses) on our WC 2018+2022 backtest, in the full live regime
(Dixon-Coles + squad/GK + Elo-multinomial blend + power calibration).

Idea borrowed from https://github.com/DrElegantia/worldcup-2026-model (their `_pool`):
weighted geometric mean of the two probability vectors instead of arithmetic. We hold
EVERYTHING else fixed (same squad layer, same gamma, same alpha grid) so the only
variable is the blend operator. If log-pool beats linear at the same alpha, it's a
free upgrade to wire into the live recipe.

Outputs reports/blend_method_eval.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.data import canonicalize_team_name, load_international_results
from wc2026_model.evaluation import EloMultinomialBenchmark
from wc2026_model.evaluation.scoring import (
    brier_score_three_way,
    log_loss_three_way,
    ranked_probability_score,
)
from wc2026_model.features import augment_with_pre_match_elo
from wc2026_model.models import (
    blend_three_way_probabilities,
    log_pool_three_way_probabilities,
    power_calibrate_probabilities,
)
from wc2026_model.pipeline import BaselineTrainingConfig, train_baseline_model


class _ScaledEloRow:
    def __init__(self, *, elo_diff_pre: float, neutral: bool) -> None:
        self.elo_diff_pre = elo_diff_pre
        self.neutral = neutral


def _lookup(squad: pd.DataFrame, *, year: int, col: str) -> dict[str, float]:
    yf = squad.loc[squad["tournament_year"] == year]
    return {
        canonicalize_team_name(str(r.team)): float(getattr(r, col))
        for r in yf.loc[:, ["team", col]].itertuples(index=False)
        if pd.notna(getattr(r, col))
    }


def _adj(base: float, home: str, away: str, lk: dict[str, float], scale: float) -> float:
    h = float(lk.get(canonicalize_team_name(home), 0.0))
    a = float(lk.get(canonicalize_team_name(away), 0.0))
    return float(base) + float(scale) * (h - a)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-input", default="data/interim/international_results_augmented.csv")
    p.add_argument("--squad-input", default="reports/historical_world_cup_real_xi_squad_strength.csv")
    p.add_argument("--rating-column", default="real_xi_club_elo_rating")
    p.add_argument("--squad-scale", type=float, default=1.5)
    p.add_argument("--years", default="2018,2022")
    p.add_argument("--alpha-grid", default="0.5,0.65,0.75,0.85")
    p.add_argument("--temperature-grid", default="0.7,0.85,1.0,1.2")
    p.add_argument("--gamma-home", type=float, default=1.05)
    p.add_argument("--gamma-draw", type=float, default=1.0)
    p.add_argument("--gamma-away", type=float, default=1.1)
    p.add_argument("--max-goals", type=int, default=10)
    p.add_argument("--output", default="reports/blend_method_eval.json")
    args = p.parse_args()

    years = [int(y) for y in args.years.split(",") if y.strip()]
    alphas = [float(a) for a in args.alpha_grid.split(",") if a.strip()]
    temps = [float(t) for t in args.temperature_grid.split(",") if t.strip()]

    results = load_international_results(args.results_input)
    results["match_date"] = pd.to_datetime(results["match_date"], errors="raise")
    results = results.sort_values(["match_date", "home_team", "away_team"], kind="stable").reset_index(drop=True)
    squad = pd.read_csv(args.squad_input)
    config = BaselineTrainingConfig(min_match_date="2010-01-01", min_team_matches=10)
    full = augment_with_pre_match_elo(results, config=config.elo_config)

    # Pre-compute, per year, the (adjusted_elo_diff, base DC probs, elo-bench probs, outcome)
    # so the blend sweep doesn't refit anything.
    precomputed: list[dict] = []
    for year in years:
        tf = full[(full["tournament"] == "FIFA World Cup") & (full["match_date"].dt.year == year)].copy()
        if tf.empty:
            continue
        start = pd.Timestamp(tf["match_date"].min())
        model, training_frame = train_baseline_model(
            results,
            config=BaselineTrainingConfig(
                min_match_date="2010-01-01", training_cutoff=start.strftime("%Y-%m-%d"),
                min_team_matches=10, elo_config=config.elo_config,
            ),
        )
        tf = tf[tf["home_team"].isin(model.teams) & tf["away_team"].isin(model.teams)].reset_index(drop=True)
        if tf.empty or training_frame.empty:
            continue
        bench = EloMultinomialBenchmark.fit(training_frame)
        lk = _lookup(squad, year=year, col=args.rating_column)
        for row in tf.itertuples(index=False):
            adj = _adj(float(row.elo_diff_pre), str(row.home_team), str(row.away_team), lk, args.squad_scale)
            base = model.predict_outcome_probabilities(
                row.home_team, row.away_team, neutral_site=bool(row.neutral),
                elo_diff_pre=adj, max_goals=args.max_goals,
            )
            ov = bench.predict_proba(_ScaledEloRow(elo_diff_pre=adj, neutral=bool(row.neutral)))
            precomputed.append({"base": base, "overlay": ov, "outcome": str(row.home_result)})

    def _score_arm(blend_fn, alpha: float, temperature: float | None) -> dict:
        lls, briers, rpss = [], [], []
        for rec in precomputed:
            if temperature is None:
                probs = blend_fn(rec["base"], rec["overlay"], alpha_on_base=alpha)
            else:
                probs = blend_fn(rec["base"], rec["overlay"], alpha_on_base=alpha, temperature=temperature)
            probs = power_calibrate_probabilities(
                probs, gamma_home=args.gamma_home, gamma_draw=args.gamma_draw, gamma_away=args.gamma_away,
            )
            lls.append(log_loss_three_way(probs, rec["outcome"]))
            briers.append(brier_score_three_way(probs, rec["outcome"]))
            rpss.append(ranked_probability_score(probs, rec["outcome"]))
        return {"mean_log_loss": sum(lls) / len(lls),
                "mean_brier": sum(briers) / len(briers),
                "mean_rps": sum(rpss) / len(rpss)}

    arms: list[dict] = []
    for alpha in alphas:
        m = _score_arm(blend_three_way_probabilities, alpha, None)
        arms.append({"method": "linear", "alpha": alpha, "temperature": None, **m})
    for alpha in alphas:
        for t in temps:
            m = _score_arm(log_pool_three_way_probabilities, alpha, t)
            arms.append({"method": "log_pool", "alpha": alpha, "temperature": t, **m})

    arms.sort(key=lambda a: a["mean_log_loss"])
    best = arms[0]
    best_linear = min((a for a in arms if a["method"] == "linear"), key=lambda a: a["mean_log_loss"])
    out = {
        "regime": "live (DC + real-XI squad + elo-blend + power-calibration)",
        "years": years, "n_matches": len(precomputed),
        "best_overall": best,
        "best_linear": best_linear,
        "log_pool_beats_linear": best["method"] == "log_pool"
        and best["mean_log_loss"] < best_linear["mean_log_loss"] - 1e-4,
        "improvement_log_loss": round(best["mean_log_loss"] - best_linear["mean_log_loss"], 5),
        "all_arms": arms,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
          ("best_overall", "best_linear", "log_pool_beats_linear", "improvement_log_loss")}, indent=2))


if __name__ == "__main__":
    main()
