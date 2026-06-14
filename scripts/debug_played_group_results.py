from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wc2026_model.data import canonicalize_team_name, load_international_results
from wc2026_model.tournament.simulation import build_group_stage_schedule


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug the played World Cup group results extracted from a results file."
    )
    parser.add_argument(
        "--groups-input",
        default="data/reference/wc2026_groups_actual.csv",
    )
    parser.add_argument(
        "--results-input",
        default="data/interim/international_results_augmented.csv",
    )
    parser.add_argument(
        "--as-of-date",
        default="2026-06-12",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    groups_frame = pd.read_csv(args.groups_input)
    groups_frame["group"] = groups_frame["group"].astype(str)
    groups_frame["team"] = groups_frame["team"].astype(str).map(canonicalize_team_name)
    if "slot" in groups_frame.columns:
        groups_frame = groups_frame.sort_values(["group", "slot"], kind="stable")
    groups = groups_frame.groupby("group", sort=True)["team"].apply(list).to_dict()

    results = load_international_results(Path(args.results_input))
    results = results.loc[results["match_date"] <= pd.Timestamp(args.as_of_date)].copy()
    world_cup_results = results.loc[results["tournament"] == "FIFA World Cup"].copy()

    schedule = build_group_stage_schedule(groups).loc[:, ["group", "home_team", "away_team"]].copy()
    played_group_results = world_cup_results.merge(
        schedule,
        on=["home_team", "away_team"],
        how="inner",
    )
    played_group_results = played_group_results.loc[
        :, ["group", "home_team", "away_team", "home_goals", "away_goals", "match_date"]
    ].drop_duplicates(subset=["group", "home_team", "away_team"], keep="last")

    print(played_group_results.to_string(index=False))
    print(f"count={len(played_group_results)}")


if __name__ == "__main__":
    main()
