"""Fetch the Polymarket WC2026 goalscorer markets (Golden Boot winner +
Player-to-score) and save per-player price snapshots. These are the two player
markets that exist on Polymarket and map to our individual-goals model
(see memory individual-goal-share-result).

Golden Boot: ~80 Yes/No-per-player markets (vol ~$9.9M). Player to score: ~155
Yes/No-per-player markets (anytime scorer over the tournament).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wc2026_model.markets.polymarket import (
    extract_per_player_yes_market_frame,
    fetch_polymarket_world_cup_events,
    find_polymarket_event,
)

_MARKETS = {
    "golden_boot": "world-cup-golden-boot-winner",
    "player_to_score": "world-cup-player-to-score",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-slug", default="fifa-world-cup")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--golden-boot-output",
                        default="data/interim/polymarket_golden_boot.csv")
    parser.add_argument("--player-to-score-output",
                        default="data/interim/polymarket_player_to_score.csv")
    parser.add_argument("--summary-output",
                        default="reports/polymarket_goalscorer_markets_summary.json")
    args = parser.parse_args()

    events = fetch_polymarket_world_cup_events(tag_slug=args.tag_slug, limit=args.limit)
    outputs = {"golden_boot": args.golden_boot_output,
               "player_to_score": args.player_to_score_output}

    summary: dict[str, object] = {}
    for key, slug in _MARKETS.items():
        try:
            event = find_polymarket_event(events, event_slug=slug)
        except KeyError:
            print(f"  (Polymarket event '{slug}' not found; skipping {key})")
            summary[key] = {"found": False}
            continue
        frame = extract_per_player_yes_market_frame(event)
        out = Path(outputs[key])
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        summary[key] = {
            "found": True,
            "slug": slug,
            "player_count": int(len(frame)),
            "top_players": frame.head(8)[["player", "market_probability"]]
            .to_dict(orient="records"),
        }
        print(f"Saved {key}: {len(frame)} players -> {out}")

    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
