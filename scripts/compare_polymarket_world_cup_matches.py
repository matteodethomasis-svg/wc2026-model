"""Compare our model against the live Polymarket per-match (1X2) and per-team
advancement/group markets.

Two honesty rules baked in:
  - ANTE-POST ONLY for the site's "edge" view: a match enters the edge comparison
    only if its real kickoff (UTC) is still in the future vs the snapshot moment.
    Timezone-correct (kickoffs are in UTC), so a game that has kicked off — even if
    its slug still says "today" — is excluded. Started/finished matches flow to the
    track record instead, where model vs Polymarket is scored on the real result.
  - Round/group markets use the RAW Polymarket Yes price as the market probability
    (these are independent Yes/No bets, not a mutually-exclusive 1X2 — no cross-team
    renormalization).

Outputs:
  - reports/polymarket_match_edges_antepost_comparison.csv   (future kickoffs only)
  - reports/polymarket_match_full_comparison.csv             (all merged matches)
  - reports/polymarket_round_comparison.csv                  (model vs market, per round)
  - reports/polymarket_group_winner_comparison.csv           (model vs market, per group)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wc2026_model.data import canonicalize_team_name, fetch_world_cup_kickoffs
from wc2026_model.markets.match_odds import compare_match_probabilities

ROOT = Path(__file__).resolve().parents[1]

# market_key -> sim probability column
_ROUND_SIM_COLUMN = {
    "advance": "reach_round_of_32_probability",   # advancing past the group stage
    "r16": "reach_round_of_16_probability",
    "quarter": "reach_quarterfinal_probability",
    "semi": "reach_semifinal_probability",
    "final": "reach_final_probability",
}


def _read(rel: str) -> pd.DataFrame | None:
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _ante_post_mask(comparison: pd.DataFrame, kickoffs: pd.DataFrame | None, now: datetime) -> pd.Series:
    """True for matches whose kickoff is still in the future (ante-post)."""
    if comparison.empty:
        return pd.Series([], dtype=bool)
    if kickoffs is None or kickoffs.empty:
        # No kickoff data: be conservative and keep everything (can't prove it started).
        return pd.Series([True] * len(comparison), index=comparison.index)
    ko = {
        (canonicalize_team_name(str(r.home_team)), canonicalize_team_name(str(r.away_team))): r.kickoff_ts
        for r in kickoffs.itertuples(index=False)
    }
    keep = []
    for row in comparison.itertuples(index=False):
        kickoff = ko.get((str(row.home_team), str(row.away_team)))
        # Future kickoff -> ante-post. Unknown kickoff -> keep (can't prove it started).
        keep.append(True if kickoff is None else now < kickoff)
    return pd.Series(keep, index=comparison.index)


def _compare_round(market: pd.DataFrame, sim: pd.DataFrame, sim_column: str) -> pd.DataFrame:
    sim_small = sim[["team", sim_column]].copy()
    sim_small["team"] = sim_small["team"].astype(str).map(canonicalize_team_name)
    m = market.copy()
    m["team"] = m["team"].astype(str).map(canonicalize_team_name)
    merged = sim_small.merge(m, on="team", how="inner")
    merged = merged.rename(columns={sim_column: "model_probability"})
    merged["edge_vs_market"] = merged["model_probability"] - merged["market_probability"]
    # Sort by OUR probability descending (not by edge).
    return merged.sort_values(
        ["model_probability", "team"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        default="reports/wc2026_fixture_predictions_expected_xi_plus_goalkeeper_recipe_2026-06-13.csv")
    parser.add_argument(
        "--sim", default="reports/wc2026_simulation_expected_xi_plus_goalkeeper_probabilities.csv")
    parser.add_argument("--matches", default="data/interim/polymarket_world_cup_matches.csv")
    parser.add_argument("--rounds", default="data/interim/polymarket_world_cup_rounds.csv")
    parser.add_argument("--groups", default="data/interim/polymarket_world_cup_group_winner.csv")
    parser.add_argument(
        "--match-antepost-output",
        default="reports/polymarket_match_edges_antepost_comparison.csv")
    parser.add_argument(
        "--match-full-output", default="reports/polymarket_match_full_comparison.csv")
    parser.add_argument("--round-output", default="reports/polymarket_round_comparison.csv")
    parser.add_argument(
        "--group-output", default="reports/polymarket_group_winner_comparison.csv")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    fixtures = _read(args.fixtures)
    sim = _read(args.sim)
    matches = _read(args.matches)

    # --- per-match 1X2 ---
    if fixtures is not None and matches is not None and not matches.empty:
        full = compare_match_probabilities(fixtures, matches)
        # Re-sort the FULL file by date (chronological); the edge file is ante-post only.
        full = full.sort_values(["match_date", "home_team", "away_team"], kind="stable").reset_index(drop=True)
        full.to_csv(ROOT / args.match_full_output, index=False)

        try:
            kickoffs = fetch_world_cup_kickoffs("2026-06-10", now.date().isoformat())
        except Exception:
            kickoffs = None
        ante = full[_ante_post_mask(full, kickoffs, now)].copy()
        # Sort the ante-post edge view by OUR top pick probability (most confident first).
        if not ante.empty:
            ante["_our_top"] = ante[
                ["home_win_probability", "draw_probability", "away_win_probability"]
            ].max(axis=1)
            ante = ante.sort_values(["_our_top", "match_date"], ascending=[False, True]).drop(columns="_our_top")
        ante.to_csv(ROOT / args.match_antepost_output, index=False)
        print(f"Matches: {len(full)} merged, {len(ante)} ante-post (future kickoff).")
    else:
        print("Matches: no fixtures/market data — skipping per-match comparison.")

    # --- per-team round markets ---
    rounds = _read(args.rounds)
    if sim is not None and rounds is not None and not rounds.empty:
        out = []
        for key, sim_col in _ROUND_SIM_COLUMN.items():
            sub = rounds[rounds["market_key"] == key]
            if sub.empty or sim_col not in sim.columns:
                continue
            cmp = _compare_round(sub, sim, sim_col)
            # market_key already rides along from the round file via the merge; make
            # sure it's the leading column for readability.
            cmp["market_key"] = key
            out.append(cmp)
        round_cmp = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        round_cmp.to_csv(ROOT / args.round_output, index=False)
        print(f"Round markets: {len(round_cmp)} model-vs-market rows.")

    # --- per-team group winner ---
    groups = _read(args.groups)
    if sim is not None and groups is not None and not groups.empty and "group_winner_probability" in sim.columns:
        out = []
        for letter, sub in groups.groupby("group"):
            sim_g = sim[["team", "group", "group_winner_probability"]].copy()
            sim_g["team"] = sim_g["team"].astype(str).map(canonicalize_team_name)
            m = sub.copy()
            m["team"] = m["team"].astype(str).map(canonicalize_team_name)
            merged = sim_g.merge(m.drop(columns="group"), on="team", how="inner")
            merged = merged.rename(columns={"group_winner_probability": "model_probability"})
            merged["edge_vs_market"] = merged["model_probability"] - merged["market_probability"]
            merged["group"] = letter
            out.append(merged)
        group_cmp = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        if not group_cmp.empty:
            group_cmp = group_cmp.sort_values(
                ["group", "model_probability"], ascending=[True, False], kind="stable"
            ).reset_index(drop=True)
        group_cmp.to_csv(ROOT / args.group_output, index=False)
        print(f"Group winner: {len(group_cmp)} model-vs-market rows.")


if __name__ == "__main__":
    main()
