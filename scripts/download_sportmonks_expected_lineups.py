from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.data import (
    DEFAULT_SPORTMONKS_EXPECTED_LINEUPS_INCLUDES,
    get_sportmonks_api_token,
    read_provider_team_ids,
    save_sportmonks_expected_lineups_outputs,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Sportmonks expected lineups and save raw JSON plus normalized CSV."
    )
    parser.add_argument(
        "--team-ids",
        default=None,
        help="Comma-separated Sportmonks team IDs.",
    )
    parser.add_argument(
        "--registry-input",
        default=None,
        help=(
            "Optional provider registry CSV. When supplied, Sportmonks IDs are read from its "
            "'sportmonks_team_id' column."
        ),
    )
    parser.add_argument(
        "--team-ids-input",
        default=None,
        help="Optional CSV/JSON file containing a 'team_id' or 'id' column.",
    )
    parser.add_argument(
        "--include",
        default=",".join(DEFAULT_SPORTMONKS_EXPECTED_LINEUPS_INCLUDES),
        help="Comma-separated Sportmonks includes.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=None,
        help="Optional per-page request size.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Optional pause between paginated requests for the same team.",
    )
    parser.add_argument(
        "--raw-output",
        default="data/interim/sportmonks_expected_lineups_raw.json",
        help="Where to save the raw JSON payload.",
    )
    parser.add_argument(
        "--csv-output",
        default="data/interim/sportmonks_expected_lineups.csv",
        help="Where to save the normalized CSV.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/sportmonks_expected_lineups_download_summary.json",
        help="Where to save a compact JSON summary.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    team_ids = _resolve_team_ids(
        csv_team_ids=args.team_ids,
        registry_input=Path(args.registry_input) if args.registry_input else None,
        team_ids_input=Path(args.team_ids_input) if args.team_ids_input else None,
    )
    if not team_ids:
        raise ValueError("Provide at least one team ID via --team-ids or --team-ids-input.")

    include = [item.strip() for item in str(args.include).split(",") if item.strip()]
    api_token = get_sportmonks_api_token()

    raw_output = Path(args.raw_output)
    csv_output = Path(args.csv_output)
    summary_output = Path(args.summary_output)
    for path in (raw_output, csv_output, summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    save_sportmonks_expected_lineups_outputs(
        raw_destination=raw_output,
        csv_destination=csv_output,
        team_ids=team_ids,
        api_token=api_token,
        include=include,
        per_page=args.per_page,
        request_pause_seconds=args.pause_seconds,
    )

    dataframe = pd.read_csv(csv_output)
    summary = {
        "team_id_count": len(team_ids),
        "team_ids": team_ids,
        "row_count": int(len(dataframe)),
        "expected_starter_rows": int(
            dataframe["is_expected_starter"].fillna(False).astype(bool).sum()
            if "is_expected_starter" in dataframe.columns
            else 0
        ),
        "fixture_count": int(dataframe["fixture_id"].nunique()) if "fixture_id" in dataframe.columns else 0,
        "teams_found": sorted(dataframe["team"].dropna().astype(str).unique().tolist())
        if "team" in dataframe.columns
        else [],
        "raw_output": str(raw_output),
        "csv_output": str(csv_output),
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved raw Sportmonks expected lineups to {raw_output}")
    print(f"Saved normalized Sportmonks expected lineups to {csv_output}")


def _resolve_team_ids(
    *,
    csv_team_ids: str | None,
    registry_input: Path | None,
    team_ids_input: Path | None,
) -> list[int]:
    resolved: list[int] = []
    if csv_team_ids:
        resolved.extend(
            int(team_id.strip())
            for team_id in str(csv_team_ids).split(",")
            if team_id.strip() != ""
        )

    if registry_input is not None:
        resolved.extend(read_provider_team_ids(registry_input, provider="sportmonks"))

    if team_ids_input is not None:
        frame = _load_id_frame(team_ids_input)
        id_column = "team_id" if "team_id" in frame.columns else "id" if "id" in frame.columns else None
        if id_column is None:
            raise ValueError("Team IDs input must contain a 'team_id' or 'id' column.")
        resolved.extend(int(value) for value in frame[id_column].dropna().tolist())

    return list(dict.fromkeys(resolved))


def _load_id_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame.from_records(payload)
        if isinstance(payload, dict):
            for key in ("data", "response", "rows", "items"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame.from_records(payload[key])
            return pd.DataFrame.from_records([payload])
    raise ValueError(f"Unsupported team IDs input type for {path}. Use CSV or JSON.")


if __name__ == "__main__":
    main()
