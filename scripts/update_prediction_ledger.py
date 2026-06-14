"""Update the model-vs-market track record: log a timestamped snapshot, then score
every prediction whose match has now been played.

Run after each model refresh. The ledger is append-only, so over the tournament it
builds a real head-to-head: who predicts better (model vs market) and which flagged
edges actually paid off.

Inputs (all already produced by the refresh):
  - fixture predictions (model 1X2 for every scheduled match)
  - bookmaker match comparison (market no-vig 1X2, where odds exist)
  - polymarket outright comparison (model vs market champion %)
  - the augmented results dataset (to resolve played matches via ESPN-fed results)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wc2026_model.data import fetch_world_cup_kickoffs
from wc2026_model.evaluation import (
    append_match_snapshot,
    append_outright_snapshot,
    score_match_ledger,
    summarize_track_record,
)

ROOT = Path(__file__).resolve().parents[1]

MATCH_LEDGER = "reports/track_record_match_ledger.csv"
OUTRIGHT_LEDGER = "reports/track_record_outright_ledger.csv"
SCORED_OUT = "reports/track_record_match_scored.csv"
SUMMARY_OUT = "reports/track_record_summary.json"


def _read(path: str) -> pd.DataFrame | None:
    p = ROOT / path
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def build_match_snapshot(fixtures: pd.DataFrame, market: pd.DataFrame | None) -> pd.DataFrame:
    snap = pd.DataFrame({
        "match_date": fixtures["match_date"],
        "home_team": fixtures["home_team"],
        "away_team": fixtures["away_team"],
        "model_home": fixtures["home_win_probability"],
        "model_draw": fixtures["draw_probability"],
        "model_away": fixtures["away_win_probability"],
        "market_home": pd.NA, "market_draw": pd.NA, "market_away": pd.NA,
    })
    if market is not None and "home_no_vig_probability" in market.columns:
        mkt = market.set_index(["home_team", "away_team"])
        for i, row in snap.iterrows():
            key = (row["home_team"], row["away_team"])
            if key in mkt.index:
                m = mkt.loc[key]
                m = m.iloc[0] if isinstance(m, pd.DataFrame) else m
                snap.at[i, "market_home"] = m["home_no_vig_probability"]
                snap.at[i, "market_draw"] = m["draw_no_vig_probability"]
                snap.at[i, "market_away"] = m["away_no_vig_probability"]
    return snap


def _blank_market_for_started_matches(snapshot, kickoffs, now):
    """Null out market odds for any match that has already kicked off.

    Once a game starts, Polymarket is live (in-play), so logging its odds would make
    an unfair comparison vs our fixed ante-post prediction. We keep the model probs
    (they're pre-match by construction) but drop the market columns for started games.
    The last snapshot logged BEFORE kickoff already carried the ante-post market.
    """
    if kickoffs is None or kickoffs.empty:
        return snapshot
    ko = {
        (str(r.home_team), str(r.away_team)): r.kickoff_ts
        for r in kickoffs.itertuples(index=False)
    }
    out = snapshot.copy()
    for i, row in out.iterrows():
        kickoff = ko.get((str(row["home_team"]), str(row["away_team"])))
        if kickoff is not None and now >= kickoff:
            out.at[i, "market_home"] = pd.NA
            out.at[i, "market_draw"] = pd.NA
            out.at[i, "market_away"] = pd.NA
    return out


def build_outright_snapshot(outright: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "team": outright["team"],
        "model_champion": outright["champion_probability"],
        "market_champion": outright.get("market_probability"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        default="reports/wc2026_fixture_predictions_expected_xi_plus_goalkeeper_recipe_2026-06-13.csv")
    parser.add_argument(
        "--market-matches",
        # Full per-match Polymarket comparison (all merged matches, with no-vig 1X2).
        # The ledger only KEEPS market odds snapshotted strictly before kickoff
        # (_blank_market_for_started_matches), so this is ante-post by construction.
        default="reports/polymarket_match_full_comparison.csv")
    parser.add_argument(
        "--outright", default="reports/polymarket_world_cup_winner_live_comparison.csv")
    parser.add_argument(
        "--results", default="data/interim/international_results_augmented.csv")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%MZ")

    # Kickoff times make the model-vs-market comparison fair: we must never log or score
    # market odds captured at/after kickoff (the live market has moved on the in-play
    # action, our prediction hasn't). Fetched once, reused for capture + scoring.
    try:
        kickoffs = fetch_world_cup_kickoffs("2026-06-10", now.date().isoformat())
    except Exception:
        kickoffs = None

    fixtures = _read(args.fixtures)
    if fixtures is not None:
        snapshot = build_match_snapshot(fixtures, _read(args.market_matches))
        snapshot = _blank_market_for_started_matches(snapshot, kickoffs, now)
        append_match_snapshot(ROOT / MATCH_LEDGER, snapshot, snapshot_ts=ts)

    outright = _read(args.outright)
    if outright is not None:
        append_outright_snapshot(
            ROOT / OUTRIGHT_LEDGER, build_outright_snapshot(outright), snapshot_ts=ts)

    # Score the match ledger against real results (WC2026 played matches).
    ledger = _read(MATCH_LEDGER)
    results = _read(args.results)
    summary: dict = {"snapshot_ts": ts, "resolved_matches": 0}
    if ledger is not None and results is not None:
        results = results.copy()
        results["match_date"] = pd.to_datetime(results["match_date"], errors="coerce")
        played = results[
            (results["tournament"] == "FIFA World Cup")
            & (results["match_date"].dt.year == 2026)
        ]
        scored = score_match_ledger(ledger, played, kickoffs=kickoffs)
        if not scored.empty:
            scored.to_csv(ROOT / SCORED_OUT, index=False)
        summary = {"snapshot_ts": ts} | summarize_track_record(scored)

    (ROOT / SUMMARY_OUT).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Track record updated @ {ts}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
