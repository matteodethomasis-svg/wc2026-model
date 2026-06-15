"""Fetch the REAL starting XIs for the StatsBomb backtest panel matches.

The backtested "expected XI" so far is a HEURISTIC top-11 of the squad (by club-Elo /
caps / goals / age). StatsBomb open data — which we already use for xG — actually carries
the real teamsheets, so this pulls the actual starting XI per match. That lets us validate
the lineup layer on real lineups instead of the heuristic (the missing leak-free test).

Input: a panel CSV with `source_match_id` (the StatsBomb match id).
Output: one row per (match, team, starter) with player + position + an `is_starter` flag,
keyed by match_id so it joins back to the panel.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from wc2026_model.data import fetch_statsbomb_match_lineup

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        default="data/interim/statsbomb_men_major_tournaments_match_features.csv",
        help="Backtest panel CSV carrying a source_match_id column.")
    parser.add_argument("--match-id-column", default="source_match_id")
    parser.add_argument(
        "--output", default="data/interim/statsbomb_real_lineups.csv")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Optional delay between requests (be polite to the CDN).")
    args = parser.parse_args()

    panel = pd.read_csv(ROOT / args.panel)
    match_ids = (
        pd.to_numeric(panel[args.match_id_column], errors="coerce")
        .dropna().astype(int).unique().tolist()
    )
    print(f"Fetching real lineups for {len(match_ids)} StatsBomb matches…")

    frames: list[pd.DataFrame] = []
    failed: list[int] = []
    for i, mid in enumerate(match_ids, 1):
        try:
            frames.append(fetch_statsbomb_match_lineup(mid))
        except Exception:
            failed.append(mid)
        if args.sleep:
            time.sleep(args.sleep)
        if i % 50 == 0:
            print(f"  …{i}/{len(match_ids)}")

    lineups = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    lineups.to_csv(out, index=False)

    starters = int(lineups["is_starter"].sum()) if not lineups.empty else 0
    print(f"\nWrote {out}")
    print(f"  matches_with_lineups={lineups['match_id'].nunique() if not lineups.empty else 0}"
          f" rows={len(lineups)} starters={starters} failed={len(failed)}")
    if failed:
        print(f"  failed match ids (no open-data lineup): {failed[:20]}"
              + (" …" if len(failed) > 20 else ""))


if __name__ == "__main__":
    main()
