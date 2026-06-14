from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


GROUP_STAGE_PAIRINGS = (
    (0, 1),
    (2, 3),
    (0, 2),
    (3, 1),
    (3, 0),
    (1, 2),
)
WC2026_GROUPS = tuple("ABCDEFGHIJKL")
WC2026_WINNER_VS_THIRD_GROUPS = ("A", "B", "D", "E", "G", "I", "K", "L")


def build_group_stage_schedule(groups: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for group_name, teams in sorted(groups.items(), key=lambda item: item[0]):
        team_list = list(teams)
        if len(team_list) != 4:
            raise ValueError(f"Group {group_name} must contain exactly four teams.")
        if len(set(team_list)) != 4:
            raise ValueError(f"Group {group_name} contains duplicate team names.")

        for match_index, (home_slot, away_slot) in enumerate(GROUP_STAGE_PAIRINGS, start=1):
            records.append(
                {
                    "group": group_name,
                    "group_match_number": match_index,
                    "home_team": team_list[home_slot],
                    "away_team": team_list[away_slot],
                    "home_slot": home_slot + 1,
                    "away_slot": away_slot + 1,
                    "stage": "group",
                }
            )
    return pd.DataFrame.from_records(records)


def rank_group_standings(results: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"group", "home_team", "away_team", "home_goals", "away_goals"}
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns for group ranking: {missing}")

    dataframe = results.copy()
    dataframe["group"] = dataframe["group"].astype(str)

    standings_frames: list[pd.DataFrame] = []
    for group_name, group_matches in dataframe.groupby("group", sort=True):
        overall_stats = _aggregate_group_statistics(group_matches)
        ordered_teams = _order_group_teams(overall_stats, group_matches)

        group_rows: list[dict[str, object]] = []
        for rank, team in enumerate(ordered_teams, start=1):
            row = dict(overall_stats[team])
            row["group"] = group_name
            row["team"] = team
            row["group_rank"] = rank
            group_rows.append(row)
        standings_frames.append(pd.DataFrame.from_records(group_rows))

    if not standings_frames:
        return pd.DataFrame(
            columns=[
                "group",
                "team",
                "played",
                "wins",
                "draws",
                "losses",
                "goals_for",
                "goals_against",
                "goal_difference",
                "points",
                "group_rank",
            ]
        )

    return pd.concat(standings_frames, ignore_index=True)[
        [
            "group",
            "team",
            "played",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "goal_difference",
            "points",
            "group_rank",
        ]
    ]


def rank_third_place_teams(standings: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"group", "team", "points", "goal_difference", "goals_for", "group_rank"}
    missing_columns = required_columns.difference(standings.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns for third-place ranking: {missing}")

    third_place = standings.loc[standings["group_rank"] == 3].copy()
    if third_place.empty:
        return third_place.assign(third_place_rank=pd.Series(dtype=int))

    third_place = third_place.sort_values(
        ["points", "goal_difference", "goals_for", "group", "team"],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    third_place["third_place_rank"] = np.arange(1, len(third_place) + 1, dtype=int)
    third_place["qualifies_for_round_of_32"] = third_place["third_place_rank"] <= 8
    return third_place


@lru_cache(maxsize=4)
def load_round_of_32_lookup(
    lookup_path: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    resolved_path = (
        Path(lookup_path)
        if lookup_path is not None
        else Path(__file__).resolve().parents[3]
        / "data"
        / "reference"
        / "wc2026_round_of_32_lookup.csv"
    )
    lookup_frame = pd.read_csv(resolved_path)
    required_columns = {
        "advancing_third_place_groups",
        "third_for_1A",
        "third_for_1B",
        "third_for_1D",
        "third_for_1E",
        "third_for_1G",
        "third_for_1I",
        "third_for_1K",
        "third_for_1L",
    }
    missing_columns = required_columns.difference(lookup_frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Round-of-32 lookup is missing columns: {missing}")

    lookup: dict[str, dict[str, str]] = {}
    for row in lookup_frame.itertuples(index=False):
        lookup[str(row.advancing_third_place_groups)] = {
            "A": str(row.third_for_1A),
            "B": str(row.third_for_1B),
            "D": str(row.third_for_1D),
            "E": str(row.third_for_1E),
            "G": str(row.third_for_1G),
            "I": str(row.third_for_1I),
            "K": str(row.third_for_1K),
            "L": str(row.third_for_1L),
        }
    return lookup


def resolve_round_of_32_matchups(
    standings: pd.DataFrame,
    third_place_ranking: pd.DataFrame,
    *,
    lookup_path: str | Path | None = None,
) -> pd.DataFrame:
    required_standings_columns = {"group", "team", "group_rank"}
    missing_standings = required_standings_columns.difference(standings.columns)
    if missing_standings:
        missing = ", ".join(sorted(missing_standings))
        raise ValueError(f"Missing required columns for standings: {missing}")

    qualified_third_places = third_place_ranking.loc[
        third_place_ranking["third_place_rank"] <= 8
    ].copy()
    if len(qualified_third_places) != 8:
        raise ValueError("Exactly eight third-placed teams must qualify for the round of 32.")

    placements = _build_qualification_placements(standings)
    third_place_lookup = load_round_of_32_lookup(lookup_path)
    combination_key = "".join(sorted(qualified_third_places["group"].astype(str).tolist()))
    if combination_key not in third_place_lookup:
        raise KeyError(
            f"No round-of-32 mapping found for advancing third-place combination {combination_key}."
        )

    third_place_mapping = third_place_lookup[combination_key]
    qualified_third_teams = {
        str(row.group): str(row.team) for row in qualified_third_places.itertuples(index=False)
    }

    dynamic_third_seeds = {
        "3_for_1E": qualified_third_teams[third_place_mapping["E"]],
        "3_for_1I": qualified_third_teams[third_place_mapping["I"]],
        "3_for_1A": qualified_third_teams[third_place_mapping["A"]],
        "3_for_1L": qualified_third_teams[third_place_mapping["L"]],
        "3_for_1D": qualified_third_teams[third_place_mapping["D"]],
        "3_for_1G": qualified_third_teams[third_place_mapping["G"]],
        "3_for_1B": qualified_third_teams[third_place_mapping["B"]],
        "3_for_1K": qualified_third_teams[third_place_mapping["K"]],
    }

    seeds = placements | dynamic_third_seeds
    round_of_32 = [
        (73, "2A", "2B"),
        (74, "1E", "3_for_1E"),
        (75, "1F", "2C"),
        (76, "1C", "2F"),
        (77, "1I", "3_for_1I"),
        (78, "2E", "2I"),
        (79, "1A", "3_for_1A"),
        (80, "1L", "3_for_1L"),
        (81, "1D", "3_for_1D"),
        (82, "1G", "3_for_1G"),
        (83, "2K", "2L"),
        (84, "1H", "2J"),
        (85, "1B", "3_for_1B"),
        (86, "1J", "2H"),
        (87, "1K", "3_for_1K"),
        (88, "2D", "2G"),
    ]

    records = []
    for match_number, home_seed, away_seed in round_of_32:
        records.append(
            {
                "stage": "round_of_32",
                "match_number": match_number,
                "home_seed": home_seed,
                "away_seed": away_seed,
                "home_team": seeds[home_seed],
                "away_team": seeds[away_seed],
            }
        )
    return pd.DataFrame.from_records(records)


def sample_scoreline_from_matrix(
    score_matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    probabilities = np.asarray(score_matrix, dtype=float)
    flattened = probabilities.ravel()
    flattened = flattened / flattened.sum()
    sampled_index = int(rng.choice(probabilities.size, p=flattened))
    home_goals, away_goals = divmod(sampled_index, probabilities.shape[1])
    return int(home_goals), int(away_goals)


def simulate_world_cup_2026(
    *,
    model: object,
    groups: Mapping[str, Sequence[str]],
    elo_ratings: Mapping[str, float] | None = None,
    played_group_results: pd.DataFrame | None = None,
    simulations: int = 1000,
    random_state: int | None = None,
    max_goals: int = 10,
    extra_time_scale: float = 1.0 / 3.0,
    elo_temperature: float = 1.0,
) -> pd.DataFrame:
    if simulations <= 0:
        raise ValueError("simulations must be positive.")
    if elo_temperature <= 0.0:
        raise ValueError("elo_temperature must be positive.")

    schedule = build_group_stage_schedule(groups)
    _validate_wc2026_groups(schedule["group"].unique().tolist())
    played_group_results = _normalize_played_group_results(played_group_results, schedule)
    remaining_group_schedule = _remove_played_group_matches(schedule, played_group_results)
    fixed_group_result_records = played_group_results.loc[
        :, ["group", "home_team", "away_team", "home_goals", "away_goals"]
    ].to_dict(orient="records")

    elo_ratings = elo_ratings or {}
    rng = np.random.default_rng(random_state)

    teams = sorted({team for team_list in groups.values() for team in team_list})
    group_by_team = {
        team: group_name for group_name, team_list in groups.items() for team in team_list
    }
    counts = {
        team: {
            "group_winner": 0,
            "group_runner_up": 0,
            "group_third": 0,
            "group_fourth": 0,
            "reach_round_of_32": 0,
            "reach_round_of_16": 0,
            "reach_quarterfinal": 0,
            "reach_semifinal": 0,
            "reach_final": 0,
            "champion": 0,
            "group_points_total": 0.0,
        }
        for team in teams
    }

    for _ in range(simulations):
        simulated_group_results = list(fixed_group_result_records)
        for match in remaining_group_schedule.itertuples(index=False):
            home_team = str(match.home_team)
            away_team = str(match.away_team)
            score_matrix = model.predict_score_matrix(
                home_team,
                away_team,
                neutral_site=True,
                elo_diff_pre=_tempered_elo_diff(
                    elo_ratings, home_team, away_team, elo_temperature=elo_temperature
                ),
                max_goals=max_goals,
            )
            home_goals, away_goals = sample_scoreline_from_matrix(score_matrix, rng)
            simulated_group_results.append(
                {
                    "group": match.group,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                }
            )

        simulated_group_results_frame = pd.DataFrame.from_records(simulated_group_results)
        standings = rank_group_standings(simulated_group_results_frame)
        third_place_ranking = rank_third_place_teams(standings)

        for row in standings.itertuples(index=False):
            team_counts = counts[str(row.team)]
            if row.group_rank == 1:
                team_counts["group_winner"] += 1
                team_counts["reach_round_of_32"] += 1
            elif row.group_rank == 2:
                team_counts["group_runner_up"] += 1
                team_counts["reach_round_of_32"] += 1
            elif row.group_rank == 3:
                team_counts["group_third"] += 1
            else:
                team_counts["group_fourth"] += 1
            team_counts["group_points_total"] += float(row.points)

        for row in third_place_ranking.loc[
            third_place_ranking["third_place_rank"] <= 8
        ].itertuples(index=False):
            counts[str(row.team)]["reach_round_of_32"] += 1

        round_of_32 = resolve_round_of_32_matchups(standings, third_place_ranking)
        round_of_32_winners = _simulate_knockout_round(
            model=model,
            matchups=round_of_32,
            elo_ratings=elo_ratings,
            rng=rng,
            max_goals=max_goals,
            extra_time_scale=extra_time_scale,
            elo_temperature=elo_temperature,
        )
        for team in round_of_32_winners.values():
            counts[team]["reach_round_of_16"] += 1

        round_of_16 = _build_next_round_matchups(
            "round_of_16",
            [
                (89, 73, 75),
                (90, 74, 77),
                (91, 76, 78),
                (92, 79, 80),
                (93, 83, 84),
                (94, 81, 82),
                (95, 86, 88),
                (96, 85, 87),
            ],
            round_of_32_winners,
        )
        round_of_16_winners = _simulate_knockout_round(
            model=model,
            matchups=round_of_16,
            elo_ratings=elo_ratings,
            rng=rng,
            max_goals=max_goals,
            extra_time_scale=extra_time_scale,
            elo_temperature=elo_temperature,
        )
        for team in round_of_16_winners.values():
            counts[team]["reach_quarterfinal"] += 1

        quarterfinals = _build_next_round_matchups(
            "quarterfinal",
            [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)],
            round_of_16_winners,
        )
        quarterfinal_winners = _simulate_knockout_round(
            model=model,
            matchups=quarterfinals,
            elo_ratings=elo_ratings,
            rng=rng,
            max_goals=max_goals,
            extra_time_scale=extra_time_scale,
            elo_temperature=elo_temperature,
        )
        for team in quarterfinal_winners.values():
            counts[team]["reach_semifinal"] += 1

        semifinals = _build_next_round_matchups(
            "semifinal",
            [(101, 97, 98), (102, 99, 100)],
            quarterfinal_winners,
        )
        semifinal_winners = _simulate_knockout_round(
            model=model,
            matchups=semifinals,
            elo_ratings=elo_ratings,
            rng=rng,
            max_goals=max_goals,
            extra_time_scale=extra_time_scale,
            elo_temperature=elo_temperature,
        )
        for team in semifinal_winners.values():
            counts[team]["reach_final"] += 1

        final = _build_next_round_matchups(
            "final",
            [(104, 101, 102)],
            semifinal_winners,
        )
        final_winners = _simulate_knockout_round(
            model=model,
            matchups=final,
            elo_ratings=elo_ratings,
            rng=rng,
            max_goals=max_goals,
            extra_time_scale=extra_time_scale,
            elo_temperature=elo_temperature,
        )
        champion = final_winners[104]
        counts[champion]["champion"] += 1

    output_rows = []
    for team in teams:
        team_counts = counts[team]
        output_rows.append(
            {
                "team": team,
                "group": group_by_team[team],
                "average_group_points": team_counts["group_points_total"] / simulations,
                "group_winner_probability": team_counts["group_winner"] / simulations,
                "group_runner_up_probability": team_counts["group_runner_up"] / simulations,
                "group_third_probability": team_counts["group_third"] / simulations,
                "group_fourth_probability": team_counts["group_fourth"] / simulations,
                "reach_round_of_32_probability": team_counts["reach_round_of_32"]
                / simulations,
                "reach_round_of_16_probability": team_counts["reach_round_of_16"]
                / simulations,
                "reach_quarterfinal_probability": team_counts["reach_quarterfinal"]
                / simulations,
                "reach_semifinal_probability": team_counts["reach_semifinal"] / simulations,
                "reach_final_probability": team_counts["reach_final"] / simulations,
                "champion_probability": team_counts["champion"] / simulations,
            }
        )
    return pd.DataFrame.from_records(output_rows).sort_values(
        ["champion_probability", "reach_final_probability", "team"],
        ascending=[False, False, True],
        kind="stable",
    )


def _normalize_played_group_results(
    played_group_results: pd.DataFrame | None,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    if played_group_results is None or played_group_results.empty:
        return pd.DataFrame(
            columns=["group", "home_team", "away_team", "home_goals", "away_goals"]
        )

    required_columns = {"group", "home_team", "away_team", "home_goals", "away_goals"}
    missing_columns = required_columns.difference(played_group_results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Played group results are missing columns: {missing}")

    normalized = played_group_results.loc[
        :, ["group", "home_team", "away_team", "home_goals", "away_goals"]
    ].copy()
    normalized["group"] = normalized["group"].astype(str)
    normalized["home_team"] = normalized["home_team"].astype(str)
    normalized["away_team"] = normalized["away_team"].astype(str)
    normalized["home_goals"] = normalized["home_goals"].astype(int)
    normalized["away_goals"] = normalized["away_goals"].astype(int)
    normalized = normalized.drop_duplicates(
        subset=["group", "home_team", "away_team"],
        keep="last",
    ).reset_index(drop=True)

    schedule_keys = schedule.loc[:, ["group", "home_team", "away_team"]].drop_duplicates().copy()
    schedule_keys["_is_scheduled"] = True
    validated = normalized.merge(
        schedule_keys,
        on=["group", "home_team", "away_team"],
        how="left",
    )
    invalid_rows = validated.loc[validated["_is_scheduled"].isna()]
    if not invalid_rows.empty:
        row = invalid_rows.iloc[0]
        raise ValueError(
            "Played group result does not match the World Cup group schedule: "
            f"{row['group']} {row['home_team']} vs {row['away_team']}."
        )

    return normalized


def _remove_played_group_matches(
    schedule: pd.DataFrame,
    played_group_results: pd.DataFrame,
) -> pd.DataFrame:
    if played_group_results.empty:
        return schedule.copy()

    played_keys = played_group_results.loc[:, ["group", "home_team", "away_team"]].copy()
    played_keys["_already_played"] = True
    remaining = schedule.merge(
        played_keys,
        on=["group", "home_team", "away_team"],
        how="left",
    )
    remaining = remaining.loc[remaining["_already_played"].isna()].drop(columns="_already_played")
    return remaining.reset_index(drop=True)


def _aggregate_group_statistics(group_matches: pd.DataFrame) -> dict[str, dict[str, int]]:
    teams = sorted(
        set(group_matches["home_team"].astype(str)).union(group_matches["away_team"].astype(str))
    )
    statistics = {
        team: {
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "points": 0,
        }
        for team in teams
    }

    for row in group_matches.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        home_goals = int(row.home_goals)
        away_goals = int(row.away_goals)

        home_stats = statistics[home_team]
        away_stats = statistics[away_team]
        home_stats["played"] += 1
        away_stats["played"] += 1
        home_stats["goals_for"] += home_goals
        home_stats["goals_against"] += away_goals
        away_stats["goals_for"] += away_goals
        away_stats["goals_against"] += home_goals

        if home_goals > away_goals:
            home_stats["wins"] += 1
            away_stats["losses"] += 1
            home_stats["points"] += 3
        elif home_goals < away_goals:
            away_stats["wins"] += 1
            home_stats["losses"] += 1
            away_stats["points"] += 3
        else:
            home_stats["draws"] += 1
            away_stats["draws"] += 1
            home_stats["points"] += 1
            away_stats["points"] += 1

    for team in teams:
        statistics[team]["goal_difference"] = (
            statistics[team]["goals_for"] - statistics[team]["goals_against"]
        )
    return statistics


def _order_group_teams(
    overall_stats: dict[str, dict[str, int]],
    group_matches: pd.DataFrame,
) -> list[str]:
    grouped_teams: dict[tuple[int, int, int], list[str]] = {}
    for team, stats in overall_stats.items():
        key = (stats["points"], stats["goal_difference"], stats["goals_for"])
        grouped_teams.setdefault(key, []).append(team)

    ordered_teams: list[str] = []
    for key in sorted(grouped_teams.keys(), reverse=True):
        tied_teams = grouped_teams[key]
        if len(tied_teams) == 1:
            ordered_teams.extend(tied_teams)
        else:
            ordered_teams.extend(_break_tie_by_head_to_head(tied_teams, group_matches, overall_stats))
    return ordered_teams


def _break_tie_by_head_to_head(
    tied_teams: Sequence[str],
    group_matches: pd.DataFrame,
    overall_stats: Mapping[str, Mapping[str, int]],
) -> list[str]:
    if len(tied_teams) <= 1:
        return list(tied_teams)

    mini_matches = group_matches[
        group_matches["home_team"].isin(tied_teams) & group_matches["away_team"].isin(tied_teams)
    ]
    head_to_head_stats = _aggregate_group_statistics(mini_matches)

    grouped_teams: dict[tuple[int, int, int], list[str]] = {}
    for team in tied_teams:
        stats = head_to_head_stats.get(
            team,
            {
                "points": 0,
                "goal_difference": 0,
                "goals_for": 0,
            },
        )
        key = (stats["points"], stats["goal_difference"], stats["goals_for"])
        grouped_teams.setdefault(key, []).append(team)

    if len(grouped_teams) == 1:
        return sorted(
            tied_teams,
            key=lambda team: (
                -int(overall_stats[team]["goal_difference"]),
                -int(overall_stats[team]["goals_for"]),
                str(team),
            ),
        )

    ordered_teams: list[str] = []
    for key in sorted(grouped_teams.keys(), reverse=True):
        subgroup = grouped_teams[key]
        if len(subgroup) == 1:
            ordered_teams.extend(subgroup)
        else:
            ordered_teams.extend(_break_tie_by_head_to_head(subgroup, group_matches, overall_stats))
    return ordered_teams


def _build_qualification_placements(standings: pd.DataFrame) -> dict[str, str]:
    placements: dict[str, str] = {}
    for group_name in WC2026_GROUPS:
        group_rows = standings.loc[standings["group"] == group_name]
        if len(group_rows) != 4:
            raise ValueError(f"Standings must contain exactly four rows for group {group_name}.")

        winner = group_rows.loc[group_rows["group_rank"] == 1, "team"]
        runner_up = group_rows.loc[group_rows["group_rank"] == 2, "team"]
        if winner.empty or runner_up.empty:
            raise ValueError(f"Standings are missing winner or runner-up for group {group_name}.")

        placements[f"1{group_name}"] = str(winner.iloc[0])
        placements[f"2{group_name}"] = str(runner_up.iloc[0])
    return placements


def _build_next_round_matchups(
    stage: str,
    pairings: Sequence[tuple[int, int, int]],
    winners_by_match_number: Mapping[int, str],
) -> pd.DataFrame:
    records = []
    for match_number, home_match, away_match in pairings:
        records.append(
            {
                "stage": stage,
                "match_number": match_number,
                "home_seed": f"W{home_match}",
                "away_seed": f"W{away_match}",
                "home_team": winners_by_match_number[home_match],
                "away_team": winners_by_match_number[away_match],
            }
        )
    return pd.DataFrame.from_records(records)


def _simulate_knockout_round(
    *,
    model: object,
    matchups: pd.DataFrame,
    elo_ratings: Mapping[str, float],
    rng: np.random.Generator,
    max_goals: int,
    extra_time_scale: float,
    elo_temperature: float = 1.0,
) -> dict[int, str]:
    winners: dict[int, str] = {}
    for matchup in matchups.itertuples(index=False):
        home_team = str(matchup.home_team)
        away_team = str(matchup.away_team)
        elo_diff = _tempered_elo_diff(
            elo_ratings, home_team, away_team, elo_temperature=elo_temperature
        )
        score_matrix = model.predict_score_matrix(
            home_team,
            away_team,
            neutral_site=True,
            elo_diff_pre=elo_diff,
            max_goals=max_goals,
        )
        home_goals, away_goals = sample_scoreline_from_matrix(score_matrix, rng)
        if home_goals != away_goals:
            winners[int(matchup.match_number)] = home_team if home_goals > away_goals else away_team
            continue

        lambda_home, lambda_away = model.predict_expected_goals(
            home_team,
            away_team,
            neutral_site=True,
            elo_diff_pre=elo_diff,
        )
        extra_home = int(rng.poisson(max(lambda_home * extra_time_scale, 0.0)))
        extra_away = int(rng.poisson(max(lambda_away * extra_time_scale, 0.0)))
        home_total = home_goals + extra_home
        away_total = away_goals + extra_away
        if home_total != away_total:
            winners[int(matchup.match_number)] = home_team if home_total > away_total else away_team
            continue

        regulation_probabilities = model.predict_outcome_probabilities(
            home_team,
            away_team,
            neutral_site=True,
            elo_diff_pre=elo_diff,
            max_goals=max_goals,
        )
        non_draw_mass = regulation_probabilities.home + regulation_probabilities.away
        if non_draw_mass <= 0.0:
            home_penalty_probability = 0.5
        else:
            home_penalty_probability = regulation_probabilities.home / non_draw_mass
        winners[int(matchup.match_number)] = (
            home_team if rng.random() < home_penalty_probability else away_team
        )
    return winners


def _team_elo(elo_ratings: Mapping[str, float], team: str) -> float:
    return float(elo_ratings.get(team, 1500.0))


def _tempered_elo_diff(
    elo_ratings: Mapping[str, float],
    home_team: str,
    away_team: str,
    *,
    elo_temperature: float,
) -> float:
    """Elo gap softened by a temperature.

    Per-match the Dixon-Coles model is well calibrated, but across a 7-match
    knockout those small favourite edges compound into over-confident title odds.
    Dividing the Elo gap by a temperature > 1 injects tournament-level variance
    (more upsets) without touching the validated per-match model. T = 1 is a no-op.
    """
    raw_diff = _team_elo(elo_ratings, home_team) - _team_elo(elo_ratings, away_team)
    return raw_diff / float(elo_temperature)


def _validate_wc2026_groups(group_names: Sequence[str]) -> None:
    observed_groups = sorted(str(group_name) for group_name in group_names)
    expected_groups = list(WC2026_GROUPS)
    if observed_groups != expected_groups:
        raise ValueError(
            f"World Cup 2026 groups must be {expected_groups}, received {observed_groups}."
        )
