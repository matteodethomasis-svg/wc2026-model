"""Validate the tournament Elo-temperature on real historical knockout matches.

The tournament simulation over-rates favourites because small per-match edges
compound across a 7-game knockout. We soften the Elo gap with a temperature T
(elo_diff / T). To pick T without overfitting to the betting market, we score how
well the Dixon-Coles model — at each T — predicts the *actual* knockout results of
past World Cups (2018, 2022).

Knockout games are decisive (no draws): a game level after extra time is settled on
penalties, which is close to a coin flip and which the Elo gap barely informs. So
we score on the two-way win/lose outcome (penalty results count as the side that
advanced), using log loss. The best T is the one that best predicts who actually
went through — a direct, market-free check on favourite-vs-underdog calibration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wc2026_model.features import augment_with_pre_match_elo
from wc2026_model.pipeline import BaselineTrainingConfig, train_baseline_model

KNOCKOUT_STAGES = {
    "Round of 16",
    "Quarter-finals",
    "Semi-finals",
    "3rd Place Final",
    "Final",
}


def _two_way_log_loss(p_home_advances: float, home_advanced: bool) -> float:
    eps = 1e-12
    p = min(max(p_home_advances, eps), 1.0 - eps)
    return -np.log(p if home_advanced else (1.0 - p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-input", default="data/interim/international_results_augmented.csv")
    parser.add_argument(
        "--world-cup-input", default="data/interim/statsbomb_world_cup_match_features.csv"
    )
    parser.add_argument(
        "--temperatures", default="1.0,1.25,1.5,1.75,2.0,2.5,3.0",
        help="Comma-separated Elo temperatures to test.",
    )
    parser.add_argument("--output", default="reports/knockout_temperature_backtest.csv")
    args = parser.parse_args()

    temperatures = [float(t) for t in args.temperatures.split(",")]

    # Train the same Dixon-Coles baseline used live, on all results.
    results = pd.read_csv(args.results_input)
    model, _ = train_baseline_model(results)

    # Pre-match Elo for every international game, then keep WC knockout games.
    elo_frame = augment_with_pre_match_elo(results)
    elo_lookup = {
        (str(r.match_date), str(r.home_team), str(r.away_team)): float(r.elo_diff_pre)
        for r in elo_frame.itertuples(index=False)
    }

    wc = pd.read_csv(args.world_cup_input)
    ko = wc[wc["competition_stage"].isin(KNOCKOUT_STAGES)].copy()

    rows = []
    matched = 0
    for game in ko.itertuples(index=False):
        key = (str(game.match_date), str(game.home_team), str(game.away_team))
        elo_diff = elo_lookup.get(key)
        if elo_diff is None:
            continue
        matched += 1
        # "home advanced" = home won in regulation, or won the tie overall.
        # In this dataset home_result is the regulation result; a draw was settled
        # on penalties. We can't see the shootout, so a regulation draw is dropped
        # from scoring (penalties ≈ coin flip, uninformative for Elo calibration).
        hg, ag = float(game.home_goals), float(game.away_goals)
        if hg == ag:
            decisive = None
        else:
            decisive = hg > ag
        rows.append({"elo_diff": elo_diff, "home_advanced": decisive,
                     "home_team": game.home_team, "away_team": game.away_team,
                     "season": game.source_season_name})

    scored = [r for r in rows if r["home_advanced"] is not None]
    print(f"Knockout games matched: {matched}, decisive (scored): {len(scored)}")

    summary = []
    for T in temperatures:
        total_ll = 0.0
        for r in scored:
            probs = model.predict_outcome_probabilities(
                r["home_team"], r["away_team"], neutral_site=True,
                elo_diff_pre=r["elo_diff"] / T, max_goals=10,
            )
            non_draw = probs.home + probs.away
            p_home_adv = probs.home / non_draw if non_draw > 0 else 0.5
            total_ll += _two_way_log_loss(p_home_adv, r["home_advanced"])
        mean_ll = total_ll / len(scored)
        summary.append({"temperature": T, "scored_games": len(scored), "mean_two_way_log_loss": mean_ll})
        print(f"  T={T:<5} mean two-way log loss = {mean_ll:.4f}")

    out = pd.DataFrame(summary).sort_values("mean_two_way_log_loss").reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    best = out.iloc[0]
    print(f"\nBest temperature: T={best['temperature']} (log loss {best['mean_two_way_log_loss']:.4f})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
