"""Fetch live Polymarket World Cup per-match (1X2) and per-team advancement/group
markets, and save snapshots ready for the model-vs-market comparison.

These markets live under the broader 'fifa-world-cup' tag (NOT '2026-fifa-world-cup',
which is winner-only). One HTTP call to the free Gamma API, no key.

Outputs:
  - data/interim/polymarket_world_cup_matches.csv     (per-match 1X2 decimal odds)
  - data/interim/polymarket_world_cup_rounds.csv      (per-team reach-round Yes prices)
  - data/interim/polymarket_world_cup_group_winner.csv(per-team group-winner Yes prices)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.markets.polymarket import (
    extract_match_odds_frame,
    extract_per_team_yes_market_frame,
    fetch_polymarket_world_cup_events,
)

ROOT = Path(__file__).resolve().parents[1]

# Which round-market title maps to which sim probability column (filled downstream by
# the comparison script). Recorded here as the canonical 'market_key'.
_ROUND_MARKETS = {
    "World Cup: Team to advance to Knockout Stages": "advance",
    "World Cup: Nation To Reach Round of 16": "r16",
    "World Cup: Nation To Reach Quarterfinals": "quarter",
    "World Cup: Nation To Reach Semifinals": "semi",
    "World Cup: Nation to Reach Final": "final",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-slug", default="fifa-world-cup")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--matches-output", default="data/interim/polymarket_world_cup_matches.csv")
    parser.add_argument("--rounds-output", default="data/interim/polymarket_world_cup_rounds.csv")
    parser.add_argument("--groups-output", default="data/interim/polymarket_world_cup_group_winner.csv")
    parser.add_argument("--summary-output", default="reports/polymarket_world_cup_matches_summary.json")
    args = parser.parse_args()

    events = fetch_polymarket_world_cup_events(tag_slug=args.tag_slug, limit=args.limit)

    # 1. Per-match 1X2.
    matches = extract_match_odds_frame(events)

    # 2. Per-team round markets (one frame, tagged by market_key).
    round_frames = []
    for event in events:
        title = str(event.get("title", "")).strip()
        key = _ROUND_MARKETS.get(title)
        if key is None:
            continue
        frame = extract_per_team_yes_market_frame(event)
        if frame.empty:
            continue
        frame.insert(0, "market_key", key)
        frame.insert(1, "market_title", title)
        round_frames.append(frame)
    rounds = pd.concat(round_frames, ignore_index=True) if round_frames else pd.DataFrame(
        columns=["market_key", "market_title", "team", "market_probability", "volume"])

    # 3. Per-team group winner (12 groups, tagged by group letter).
    group_frames = []
    for event in events:
        title = str(event.get("title", "")).strip()
        if not (title.startswith("World Cup Group ") and title.endswith(" Winner")):
            continue
        letter = title.removeprefix("World Cup Group ").removesuffix(" Winner").strip()
        frame = extract_per_team_yes_market_frame(event)
        if frame.empty:
            continue
        frame.insert(0, "group", letter)
        group_frames.append(frame)
    groups = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame(
        columns=["group", "team", "market_probability", "volume"])

    for rel, frame in (
        (args.matches_output, matches),
        (args.rounds_output, rounds),
        (args.groups_output, groups),
    ):
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    summary = {
        "tag_slug": args.tag_slug,
        "match_count": int(len(matches)),
        "round_markets": {k: int((rounds["market_key"] == k).sum()) for k in _ROUND_MARKETS.values()}
        if not rounds.empty else {},
        "group_winner_rows": int(len(groups)),
    }
    (ROOT / args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved per-match odds  -> {args.matches_output} ({len(matches)} matches)")
    print(f"Saved round markets   -> {args.rounds_output} ({len(rounds)} rows)")
    print(f"Saved group winners   -> {args.groups_output} ({len(groups)} rows)")


if __name__ == "__main__":
    main()
