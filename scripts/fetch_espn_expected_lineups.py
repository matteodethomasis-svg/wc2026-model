"""Fetch official/confirmed starting XIs for WC2026 fixtures from ESPN's free,
keyless summary API and write the flat CSV the squad-intelligence layer consumes.

ESPN publishes the probable XI ~1h before kickoff and the confirmed XI at kickoff, so
running this on the same 20-min cadence as the model refresh picks up each lineup as soon
as it appears. These are OFFICIAL lineups (late but certain); predicted lineups with more
lead time come from a separate source — see memory live-lineups-injuries-plan.

Output schema (consumed by load_expected_lineups_feed via the flat-file path):
  team, player, position, is_expected_starter, lineup_confidence, match_date, fixture_id
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from wc2026_model.data import fetch_world_cup_expected_lineups

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default window: a couple of days back (confirmed XIs) through two days ahead
    # (catches probable XIs that post shortly before kickoff).
    parser.add_argument("--start-date", default=(date.today() - timedelta(days=2)).isoformat())
    parser.add_argument("--end-date", default=(date.today() + timedelta(days=2)).isoformat())
    parser.add_argument("--output", default="data/interim/wc2026_expected_lineups_espn.csv")
    args = parser.parse_args()

    frame = fetch_world_cup_expected_lineups(args.start_date, args.end_date)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    teams = frame["team"].nunique() if not frame.empty else 0
    starters = int(frame["is_expected_starter"].sum()) if not frame.empty else 0
    print(f"Wrote {out}")
    print(f"  rows={len(frame)} teams={teams} starters={starters} "
          f"window={args.start_date}..{args.end_date}")


if __name__ == "__main__":
    main()
