from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests


WIKIPEDIA_KNOCKOUT_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and normalize the World Cup 2026 round-of-32 third-place lookup."
    )
    parser.add_argument(
        "--output",
        default="data/reference/wc2026_round_of_32_lookup.csv",
        help="Path to save the normalized lookup CSV.",
    )
    return parser


def _download_lookup_table() -> pd.DataFrame:
    response = requests.get(
        WIKIPEDIA_KNOCKOUT_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    if not tables:
        raise ValueError("No HTML tables found on the Wikipedia knockout-stage page.")
    return tables[0]


def _normalize_lookup_table(raw_table: pd.DataFrame) -> pd.DataFrame:
    assignment_columns = [column for column in raw_table.columns if str(column).endswith("vs")]
    advance_columns = [
        column
        for column in raw_table.columns
        if column not in {"No.", "Unnamed: 13"} and column not in assignment_columns
    ]
    if len(assignment_columns) != 8:
        raise ValueError(f"Expected 8 assignment columns, found {len(assignment_columns)}.")
    if len(advance_columns) != 12:
        raise ValueError(f"Expected 12 advancing-group columns, found {len(advance_columns)}.")

    output_rows: list[dict[str, object]] = []
    for row in raw_table.itertuples(index=False):
        # The tuple returned by itertuples sanitizes column names, so read values by position.
        row_values = dict(zip(raw_table.columns, row, strict=True))
        advancing_groups = "".join(
            str(row_values[column]).strip()
            for column in advance_columns
            if pd.notna(row_values[column]) and str(row_values[column]).strip()
        )
        output_rows.append(
            {
                "combination_number": int(row_values["No."]),
                "advancing_third_place_groups": advancing_groups,
                "third_for_1A": _normalize_third_place_seed(row_values["1A vs"]),
                "third_for_1B": _normalize_third_place_seed(row_values["1B vs"]),
                "third_for_1D": _normalize_third_place_seed(row_values["1D vs"]),
                "third_for_1E": _normalize_third_place_seed(row_values["1E vs"]),
                "third_for_1G": _normalize_third_place_seed(row_values["1G vs"]),
                "third_for_1I": _normalize_third_place_seed(row_values["1I vs"]),
                "third_for_1K": _normalize_third_place_seed(row_values["1K vs"]),
                "third_for_1L": _normalize_third_place_seed(row_values["1L vs"]),
            }
        )

    normalized = pd.DataFrame.from_records(output_rows)
    normalized = normalized.sort_values("advancing_third_place_groups", kind="stable").reset_index(
        drop=True
    )
    if normalized["advancing_third_place_groups"].duplicated().any():
        raise ValueError("Duplicate advancing-group combinations found in normalized lookup.")
    return normalized


def _normalize_third_place_seed(seed: object) -> str:
    value = str(seed).strip()
    if not value.startswith("3") or len(value) != 2:
        raise ValueError(f"Unexpected third-place seed format: {value}")
    return value[1:]


def main() -> None:
    args = _build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_table = _download_lookup_table()
    normalized = _normalize_lookup_table(raw_table)
    normalized.to_csv(output_path, index=False)

    print(f"Saved {len(normalized)} round-of-32 combinations to {output_path}")


if __name__ == "__main__":
    main()
