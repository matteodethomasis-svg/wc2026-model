"""Pre-compute the per-player club goals/90 rate (365d window before WC2026 kickoff)
from the heavy raw Transfermarkt appearances file, and save a SMALL committable
artifact (data/reference/wc2026_player_goal_rate.csv, ~250KB).

Why: the goalscorer model needs each player's club goal rate, but the raw
appearances.csv is 142MB and gitignored — so it does NOT exist in CI. This script
distills it once to a tiny CSV that IS committed, so build_wc2026_goalscorer_predictions
(and the live refresh / deploy) can run without the heavy file. Re-run locally
whenever the Transfermarkt appearances are refreshed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

WC2026_KICKOFF = pd.Timestamp("2026-06-08")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearances-input", default="data/raw/transfermarkt/appearances.csv")
    parser.add_argument("--window-days", type=int, default=365)
    parser.add_argument("--min-minutes", type=int, default=270)
    parser.add_argument("--output", default="data/reference/wc2026_player_goal_rate.csv")
    args = parser.parse_args()

    lo = WC2026_KICKOFF - pd.Timedelta(days=args.window_days)
    ap = pd.read_csv(
        args.appearances_input,
        usecols=["player_id", "player_name", "date", "goals", "minutes_played"],
    )
    ap["date"] = pd.to_datetime(ap["date"], errors="coerce")
    w = ap[(ap["date"] >= lo) & (ap["date"] < WC2026_KICKOFF)]
    g = w.groupby("player_id").agg(
        name=("player_name", "first"), goals=("goals", "sum"), mins=("minutes_played", "sum"),
    ).reset_index()
    g = g[g["mins"] >= args.min_minutes].copy()
    g["per90"] = g["goals"] / g["mins"] * 90.0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    g[["player_id", "name", "goals", "mins", "per90"]].to_csv(args.output, index=False)
    size_kb = Path(args.output).stat().st_size / 1024
    print(f"Wrote {args.output}: {len(g)} players ({size_kb:.0f} KB).")


if __name__ == "__main__":
    main()
