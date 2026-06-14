from __future__ import annotations

import argparse
import json

from wc2026_model.data.public_match_odds import search_talksport_posts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search talkSPORT posts via the public WordPress search endpoint."
    )
    parser.add_argument("query", help="Search query to send to talkSPORT.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    results = search_talksport_posts(args.query)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
