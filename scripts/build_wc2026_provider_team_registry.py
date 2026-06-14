from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.data import (
    build_wc2026_provider_team_registry,
    build_wc2026_provider_team_registry_summary,
    load_expected_lineups_feed,
    load_injuries_feed,
    load_world_cup_squads_from_wikipedia,
)
from wc2026_model.data.provider_team_registry import (
    enrich_registry_with_api_football_feed,
    enrich_registry_with_sportmonks_feed,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a World Cup 2026 provider team-ID registry and optionally enrich it "
            "from Sportmonks expected lineups or API-Football injuries feeds."
        )
    )
    parser.add_argument(
        "--groups-input",
        default="data/reference/wc2026_groups_actual.csv",
        help="CSV containing the real WC2026 groups and teams.",
    )
    parser.add_argument(
        "--squads-input",
        default="data/interim/wc2026_squads_from_wikipedia.csv",
        help=(
            "Fallback squad list used when groups input is missing. "
            "If absent, the script falls back to the Wikipedia squad parser."
        ),
    )
    parser.add_argument(
        "--sportmonks-input",
        default=None,
        help="Optional Sportmonks expected-lineups CSV/JSON used to auto-fill Sportmonks team IDs.",
    )
    parser.add_argument(
        "--api-football-input",
        default=None,
        help="Optional API-Football injuries CSV/JSON used to auto-fill API-Football team IDs.",
    )
    parser.add_argument(
        "--output",
        default="data/reference/wc2026_provider_team_registry.csv",
        help="Path to save the provider registry CSV.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/wc2026_provider_team_registry_summary.json",
        help="Path to save a compact summary JSON.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    groups = _load_optional_csv(Path(args.groups_input))
    squads = _load_optional_squads(Path(args.squads_input))

    registry = build_wc2026_provider_team_registry(groups=groups, squads=squads)
    if args.sportmonks_input:
        expected_lineups = load_expected_lineups_feed(args.sportmonks_input, provider="auto")
        registry = enrich_registry_with_sportmonks_feed(registry, expected_lineups)
    if args.api_football_input:
        injuries = load_injuries_feed(args.api_football_input, provider="auto")
        registry = enrich_registry_with_api_football_feed(registry, injuries)

    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    registry.to_csv(output_path, index=False)
    summary = build_wc2026_provider_team_registry_summary(registry)
    summary["output"] = str(output_path)
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved provider team registry to {output_path}")


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_optional_squads(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return load_world_cup_squads_from_wikipedia()


if __name__ == "__main__":
    main()
