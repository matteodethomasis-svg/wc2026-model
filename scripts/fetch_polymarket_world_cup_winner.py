from __future__ import annotations

import argparse
import json
from pathlib import Path

from wc2026_model.markets.polymarket import (
    PolymarketEventQuery,
    extract_world_cup_winner_market_frame,
    fetch_polymarket_events,
    find_polymarket_event,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch the active Polymarket World Cup winner market and save a team-level price table."
    )
    parser.add_argument("--tag-slug", default="2026-fifa-world-cup")
    parser.add_argument("--event-slug", default="world-cup-winner")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", default="data/interim/polymarket_world_cup_winner.csv")
    parser.add_argument(
        "--summary-output",
        default="reports/polymarket_world_cup_winner_summary.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    query = PolymarketEventQuery(
        tag_slug=args.tag_slug,
        event_slug=args.event_slug,
        limit=args.limit,
    )
    events = fetch_polymarket_events(query)
    event = find_polymarket_event(events, event_slug=query.event_slug)
    market_frame = extract_world_cup_winner_market_frame(event)

    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    market_frame.to_csv(output_path, index=False)

    summary = {
        "tag_slug": query.tag_slug,
        "event_slug": query.event_slug,
        "team_count": int(len(market_frame)),
        "top_market_probabilities": market_frame.head(10).loc[
            :, ["market_team", "market_probability"]
        ].to_dict(orient="records"),
    }
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved Polymarket winner market snapshot to {output_path}")
    print(f"Saved summary to {summary_output_path}")


if __name__ == "__main__":
    main()
