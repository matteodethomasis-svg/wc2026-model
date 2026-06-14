from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.data.public_match_odds import (
    extract_talksport_article_match_odds,
    extract_talksport_widget_match_odds,
    extract_the_sun_match_odds,
    load_public_match_odds_sources,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a bookmaker match-odds snapshot from public article/widget sources."
    )
    parser.add_argument(
        "--sources-input",
        default="data/reference/public_bookmaker_match_sources_2026-06-13.json",
        help="JSON list of public source definitions to fetch and parse.",
    )
    parser.add_argument(
        "--output",
        default="data/interim/public_bookmaker_match_odds_snapshot.csv",
        help="CSV path for the parsed public match odds snapshot.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/public_bookmaker_match_odds_snapshot_summary.json",
        help="JSON path for a compact parsing summary.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    sources = load_public_match_odds_sources(args.sources_input)

    records: list[dict[str, object]] = []
    for source in sources:
        source_type = str(source["source_type"])
        if source_type == "talksport_widget":
            record = extract_talksport_widget_match_odds(
                str(source["url"]),
                match_date=str(source["match_date"]),
                home_team=_optional_str(source.get("home_team")),
                away_team=_optional_str(source.get("away_team")),
                source_title=_optional_str(source.get("source_title")),
            )
        elif source_type == "talksport_article":
            record = extract_talksport_article_match_odds(
                str(source["url"]),
                match_date=str(source["match_date"]),
                home_team=_optional_str(source.get("home_team")),
                away_team=_optional_str(source.get("away_team")),
                source_title=_optional_str(source.get("source_title")),
            )
        elif source_type == "the_sun_article":
            record = extract_the_sun_match_odds(
                str(source["url"]),
                match_date=str(source["match_date"]),
                home_team=str(source["home_team"]),
                away_team=str(source["away_team"]),
                source_title=_optional_str(source.get("source_title")),
            )
        else:
            raise ValueError(f"Unsupported public source type: {source_type!r}")
        records.append(record.to_dict())

    snapshot = pd.DataFrame(records).sort_values(
        ["match_date", "home_team", "away_team"],
        kind="stable",
    )
    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(output_path, index=False)

    summary = {
        "source_count": len(sources),
        "parsed_match_count": int(len(snapshot)),
        "matches": snapshot.loc[:, ["match_date", "home_team", "away_team", "bookmaker", "source_type"]]
        .to_dict(orient="records"),
    }
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved public bookmaker match snapshot to {output_path}")
    print(f"Saved summary to {summary_output_path}")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":
    main()
