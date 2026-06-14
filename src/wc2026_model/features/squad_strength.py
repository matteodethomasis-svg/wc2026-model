from __future__ import annotations

from difflib import SequenceMatcher, get_close_matches
import math
import re
import unicodedata

import pandas as pd

from wc2026_model.data import canonicalize_team_name

_CLUB_NAME_ALIASES = {
    "ac milan": "milan",
    "arsenal fc": "arsenal",
    "atletico de madrid": "atletico",
    "atletico madrid": "atletico",
    "bayern munich": "bayern",
    "bayer 04 leverkusen": "bayer leverkusen",
    "borussia monchengladbach": "mgladbach",
    "brighton and hove albion": "brighton",
    "club brugge kv": "club brugge",
    "internazionale": "inter",
    "inter milan": "inter",
    "juventus fc": "juventus",
    "manchester city": "man city",
    "manchester united": "man united",
    "newcastle united": "newcastle",
    "olympique lyonnais": "lyon",
    "olympique de marseille": "marseille",
    "paris saint germain": "paris sg",
    "paris saint-germain": "paris sg",
    "psg": "paris sg",
    "psv eindhoven": "psv",
    "sporting cp": "sporting",
    "sporting lisbon": "sporting",
    "tottenham hotspur": "tottenham",
    "wolverhampton wanderers": "wolves",
}

_GENERIC_CLUB_TOKENS = {
    "1907",
    "1913",
    "afc",
    "athletic",
    "association",
    "atletico",
    "cf",
    "club",
    "de",
    "del",
    "deportivo",
    "fc",
    "fk",
    "football",
    "sc",
    "sd",
    "sv",
}

_EXPECTED_XI_FORMATIONS = (
    {"DF": 4, "MF": 3, "FW": 3},
    {"DF": 4, "MF": 4, "FW": 2},
    {"DF": 3, "MF": 4, "FW": 3},
    {"DF": 5, "MF": 3, "FW": 2},
    {"DF": 3, "MF": 5, "FW": 2},
)


def build_squad_player_club_elo_frame(
    squads: pd.DataFrame,
    club_elo: pd.DataFrame,
    *,
    fuzzy_cutoff: float = 0.84,
) -> pd.DataFrame:
    required_squad_columns = {"team", "player", "club"}
    missing_squad_columns = required_squad_columns.difference(squads.columns)
    if missing_squad_columns:
        missing = ", ".join(sorted(missing_squad_columns))
        raise ValueError(f"Missing required squad columns: {missing}")

    required_club_elo_columns = {"club", "club_elo"}
    missing_club_elo_columns = required_club_elo_columns.difference(club_elo.columns)
    if missing_club_elo_columns:
        missing = ", ".join(sorted(missing_club_elo_columns))
        raise ValueError(f"Missing required club Elo columns: {missing}")

    squad_frame = squads.copy()
    squad_frame["team"] = squad_frame["team"].astype(str).map(canonicalize_team_name)
    squad_frame["club"] = squad_frame["club"].astype(str)

    lookup = _build_club_elo_lookup(club_elo)

    matched_club_names: list[str | None] = []
    matched_club_elos: list[float | None] = []
    matched_methods: list[str | None] = []
    matched_scores: list[float | None] = []

    for club_name in squad_frame["club"]:
        match = _match_club_name_to_elo(club_name, lookup=lookup, fuzzy_cutoff=fuzzy_cutoff)
        matched_club_names.append(match["matched_club"])
        matched_club_elos.append(match["club_elo"])
        matched_methods.append(match["match_method"])
        matched_scores.append(match["match_score"])

    squad_frame["matched_club"] = matched_club_names
    squad_frame["club_elo"] = matched_club_elos
    squad_frame["club_elo_match_method"] = matched_methods
    squad_frame["club_elo_match_score"] = matched_scores
    return squad_frame


def aggregate_team_squad_strength(
    squad_players: pd.DataFrame,
    *,
    top_player_count: int = 15,
    core_player_count: int = 11,
    star_player_count: int = 3,
    fallback_club_elo: float | None = None,
) -> pd.DataFrame:
    required_columns = {"team", "club_elo"}
    missing_columns = required_columns.difference(squad_players.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required squad player columns: {missing}")

    if fallback_club_elo is None:
        numeric_club_elo = pd.to_numeric(squad_players["club_elo"], errors="coerce").dropna()
        fallback_club_elo = (
            float(numeric_club_elo.median()) if not numeric_club_elo.empty else float("nan")
        )

    rows: list[dict[str, object]] = []
    for team, group in squad_players.groupby("team", sort=True):
        mapped = group[group["club_elo"].notna()].copy()
        mapped = mapped.sort_values("club_elo", ascending=False, kind="stable")
        caps = _numeric_series(group, "caps")
        ages = _numeric_series(group, "age")

        top_rating = _coverage_adjusted_mean(
            mapped["club_elo"],
            target_count=top_player_count,
            fallback_value=fallback_club_elo,
        )
        core_rating = _coverage_adjusted_mean(
            mapped["club_elo"],
            target_count=core_player_count,
            fallback_value=fallback_club_elo,
        )
        star_rating = _safe_mean(mapped["club_elo"].head(star_player_count))
        all_rating = _safe_mean(mapped["club_elo"])
        mapped_only_top_rating = _safe_mean(mapped["club_elo"].head(top_player_count))
        mapped_only_core_rating = _safe_mean(mapped["club_elo"].head(core_player_count))
        expected_xi = _select_expected_xi(
            group,
            fallback_club_elo=float(fallback_club_elo),
        )

        rows.append(
            {
                "team": team,
                "squad_player_count": int(len(group)),
                "mapped_player_count": int(mapped["club_elo"].notna().sum()),
                "mapped_player_share": float(
                    0.0 if len(group) == 0 else mapped["club_elo"].notna().sum() / len(group)
                ),
                "squad_club_elo_rating": float(top_rating),
                "squad_club_elo_core_rating": float(core_rating),
                "squad_club_elo_star_rating": float(star_rating),
                "squad_club_elo_all_rating": float(all_rating),
                "mapped_only_squad_club_elo_rating": float(mapped_only_top_rating),
                "mapped_only_squad_club_elo_core_rating": float(mapped_only_core_rating),
                "mean_caps": float(caps.mean()),
                "median_caps": float(caps.median()),
                "mean_age": float(ages.mean()),
                "expected_xi_club_elo_rating": float(expected_xi["club_elo_rating"]),
                "expected_xi_mapped_only_club_elo_rating": float(
                    expected_xi["mapped_only_club_elo_rating"]
                ),
                "expected_xi_selection_score": float(expected_xi["selection_score"]),
                "expected_xi_mapped_player_share": float(expected_xi["mapped_player_share"]),
                "expected_xi_mean_caps": float(expected_xi["mean_caps"]),
                "expected_xi_mean_age": float(expected_xi["mean_age"]),
                "expected_xi_formation": str(expected_xi["formation"]),
                "expected_xi_goalkeeper_club_elo_rating": float(
                    expected_xi["goalkeeper_club_elo_rating"]
                ),
                "expected_xi_defense_club_elo_rating": float(
                    expected_xi["defense_club_elo_rating"]
                ),
                "expected_xi_midfield_club_elo_rating": float(
                    expected_xi["midfield_club_elo_rating"]
                ),
                "expected_xi_attack_club_elo_rating": float(
                    expected_xi["attack_club_elo_rating"]
                ),
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values("squad_club_elo_rating", ascending=False, kind="stable").reset_index(
        drop=True
    )


def attach_team_strength_ratings(
    matches: pd.DataFrame,
    team_strengths: pd.DataFrame,
    *,
    rating_column: str = "squad_club_elo_rating",
) -> pd.DataFrame:
    required_match_columns = {"home_team", "away_team"}
    missing_match_columns = required_match_columns.difference(matches.columns)
    if missing_match_columns:
        missing = ", ".join(sorted(missing_match_columns))
        raise ValueError(f"Missing required match columns: {missing}")
    if "team" not in team_strengths.columns:
        raise ValueError("Team strengths frame must contain a 'team' column.")
    if rating_column not in team_strengths.columns:
        raise ValueError(f"Team strengths frame is missing rating column '{rating_column}'.")

    rating_lookup = {
        canonicalize_team_name(str(row.team)): float(row[rating_column])
        for _, row in team_strengths.loc[:, ["team", rating_column]].iterrows()
        if pd.notna(row[rating_column])
    }

    enriched = matches.copy()
    enriched["home_team"] = enriched["home_team"].astype(str).map(canonicalize_team_name)
    enriched["away_team"] = enriched["away_team"].astype(str).map(canonicalize_team_name)
    enriched["home_team_strength_rating"] = enriched["home_team"].map(rating_lookup)
    enriched["away_team_strength_rating"] = enriched["away_team"].map(rating_lookup)
    enriched["team_strength_rating_diff"] = (
        enriched["home_team_strength_rating"] - enriched["away_team_strength_rating"]
    )
    return enriched


def _build_club_elo_lookup(club_elo: pd.DataFrame) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for row in club_elo.itertuples(index=False):
        normalized_name = normalize_club_name(getattr(row, "club"))
        if not normalized_name:
            continue
        current = lookup.get(normalized_name)
        current_elo = float(getattr(row, "club_elo"))
        if current is None or current_elo > float(current["club_elo"]):
            lookup[normalized_name] = {
                "club": str(getattr(row, "club")),
                "club_elo": current_elo,
            }
    return lookup


def _match_club_name_to_elo(
    club_name: str,
    *,
    lookup: dict[str, dict[str, object]],
    fuzzy_cutoff: float,
) -> dict[str, object | None]:
    normalized_name = normalize_club_name(club_name)
    if normalized_name in lookup:
        matched = lookup[normalized_name]
        return {
            "matched_club": matched["club"],
            "club_elo": matched["club_elo"],
            "match_method": "exact",
            "match_score": 1.0,
        }

    if normalized_name in _CLUB_NAME_ALIASES and _CLUB_NAME_ALIASES[normalized_name] in lookup:
        matched = lookup[_CLUB_NAME_ALIASES[normalized_name]]
        return {
            "matched_club": matched["club"],
            "club_elo": matched["club_elo"],
            "match_method": "alias",
            "match_score": 1.0,
        }

    lookup_keys = list(lookup.keys())
    close_matches = get_close_matches(normalized_name, lookup_keys, n=1, cutoff=fuzzy_cutoff)
    if close_matches:
        matched_key = close_matches[0]
        matched = lookup[matched_key]
        return {
            "matched_club": matched["club"],
            "club_elo": matched["club_elo"],
            "match_method": "fuzzy",
            "match_score": SequenceMatcher(None, normalized_name, matched_key).ratio(),
        }

    substring_matches = [
        key for key in lookup_keys if normalized_name in key or key in normalized_name
    ]
    if len(substring_matches) == 1:
        matched_key = substring_matches[0]
        matched = lookup[matched_key]
        return {
            "matched_club": matched["club"],
            "club_elo": matched["club_elo"],
            "match_method": "substring",
            "match_score": SequenceMatcher(None, normalized_name, matched_key).ratio(),
        }

    return {
        "matched_club": None,
        "club_elo": None,
        "match_method": None,
        "match_score": None,
    }


def normalize_club_name(club_name: object) -> str:
    text = unicodedata.normalize("NFKD", str(club_name)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token and token not in _GENERIC_CLUB_TOKENS]
    normalized = " ".join(tokens)
    return _CLUB_NAME_ALIASES.get(normalized, normalized)


def _safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())


def _numeric_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(dataframe[column], errors="coerce")


def _coverage_adjusted_mean(
    series: pd.Series,
    *,
    target_count: int,
    fallback_value: float,
) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna().head(target_count).tolist()
    if not numeric:
        return float(fallback_value)
    if len(numeric) < target_count:
        numeric.extend([float(fallback_value)] * (target_count - len(numeric)))
    return float(sum(numeric) / len(numeric))


def _select_expected_xi(
    squad_players: pd.DataFrame,
    *,
    fallback_club_elo: float,
) -> dict[str, object]:
    candidates = squad_players.copy()
    if "position" not in candidates.columns:
        candidates["position"] = ""
    candidates["club_elo_filled"] = pd.to_numeric(
        candidates.get("club_elo"), errors="coerce"
    ).fillna(float(fallback_club_elo))
    candidates["caps_numeric"] = _numeric_series(candidates, "caps").fillna(0.0)
    candidates["goals_numeric"] = _numeric_series(candidates, "goals").fillna(0.0)
    candidates["age_numeric"] = _numeric_series(candidates, "age")
    candidates["starter_selection_score"] = candidates.apply(
        lambda row: _starter_selection_score(
            position=str(row.get("position", "")),
            club_elo=float(row["club_elo_filled"]),
            caps=float(row["caps_numeric"]),
            goals=float(row["goals_numeric"]),
            age=(
                None
                if pd.isna(row.get("age_numeric"))
                else float(row.get("age_numeric"))
            ),
        ),
        axis=1,
    )

    goalkeeper_pool = _sort_expected_xi_candidates(candidates, position="GK")
    if goalkeeper_pool.empty:
        fallback_lineup = candidates.sort_values(
            ["starter_selection_score", "club_elo_filled", "caps_numeric", "goals_numeric"],
            ascending=[False, False, False, False],
            kind="stable",
        ).head(min(11, len(candidates)))
        return _summarize_expected_xi(
            fallback_lineup,
            formation_label="fallback",
        )

    goalkeeper = goalkeeper_pool.head(1)
    best_lineup = None
    best_formation = None
    best_total_score = float("-inf")

    for formation in _EXPECTED_XI_FORMATIONS:
        lineup_parts = [goalkeeper]
        valid = True
        for position, count in formation.items():
            pool = _sort_expected_xi_candidates(candidates, position=position)
            if len(pool) < count:
                valid = False
                break
            lineup_parts.append(pool.head(count))
        if not valid:
            continue
        lineup = pd.concat(lineup_parts, ignore_index=False)
        total_score = float(lineup["starter_selection_score"].sum())
        if total_score > best_total_score:
            best_lineup = lineup
            best_formation = formation
            best_total_score = total_score

    if best_lineup is None:
        outfield = candidates.loc[candidates["position"].astype(str) != "GK"].copy()
        outfield = outfield.sort_values(
            ["starter_selection_score", "club_elo_filled", "caps_numeric"],
            ascending=[False, False, False],
            kind="stable",
        )
        best_lineup = pd.concat([goalkeeper, outfield.head(10)], ignore_index=False)
        best_formation = {"DF": -1, "MF": -1, "FW": -1}

    formation_label = (
        "fallback"
        if min(best_formation.values()) < 0
        else f"{best_formation['DF']}-{best_formation['MF']}-{best_formation['FW']}"
    )
    return _summarize_expected_xi(
        best_lineup,
        formation_label=formation_label,
    )


def _summarize_expected_xi(
    lineup: pd.DataFrame,
    *,
    formation_label: str,
) -> dict[str, object]:
    lineup_caps = pd.to_numeric(lineup.get("caps_numeric"), errors="coerce")
    lineup_ages = pd.to_numeric(lineup.get("age_numeric"), errors="coerce")
    mapped_mask = pd.to_numeric(lineup.get("club_elo"), errors="coerce").notna()
    return {
        "club_elo_rating": float(lineup["club_elo_filled"].mean()),
        "mapped_only_club_elo_rating": float(
            pd.to_numeric(lineup.loc[mapped_mask, "club_elo"], errors="coerce").mean()
        ),
        "selection_score": float(lineup["starter_selection_score"].mean()),
        "mapped_player_share": float(mapped_mask.mean()),
        "mean_caps": float(lineup_caps.mean()),
        "mean_age": float(lineup_ages.mean()),
        "formation": formation_label,
        "goalkeeper_club_elo_rating": _lineup_position_mean(
            lineup,
            position="GK",
            value_column="club_elo_filled",
        ),
        "defense_club_elo_rating": _lineup_position_mean(
            lineup,
            position="DF",
            value_column="club_elo_filled",
        ),
        "midfield_club_elo_rating": _lineup_position_mean(
            lineup,
            position="MF",
            value_column="club_elo_filled",
        ),
        "attack_club_elo_rating": _lineup_position_mean(
            lineup,
            position="FW",
            value_column="club_elo_filled",
        ),
    }


def _sort_expected_xi_candidates(
    squad_players: pd.DataFrame,
    *,
    position: str,
) -> pd.DataFrame:
    return squad_players.loc[squad_players["position"].astype(str) == position].sort_values(
        ["starter_selection_score", "club_elo_filled", "caps_numeric", "goals_numeric"],
        ascending=[False, False, False, False],
        kind="stable",
    )


def _lineup_position_mean(
    lineup: pd.DataFrame,
    *,
    position: str,
    value_column: str,
) -> float:
    return _safe_mean(
        lineup.loc[lineup["position"].astype(str) == position, value_column]
    )


def _starter_selection_score(
    *,
    position: str,
    club_elo: float,
    caps: float,
    goals: float,
    age: float | None,
) -> float:
    normalized_position = str(position).upper()
    capped_caps = max(float(caps), 0.0)
    capped_goals = max(float(goals), 0.0)
    caps_bonus = 30.0 * (math.log1p(capped_caps) / math.log(101.0))
    goal_weight = {
        "GK": 0.0,
        "DF": 1.0,
        "MF": 3.0,
        "FW": 5.0,
    }.get(normalized_position, 2.0)
    goals_bonus = goal_weight * math.log1p(capped_goals)
    if age is None or not math.isfinite(age):
        age_bonus = 0.0
    else:
        age_bonus = max(0.0, 8.0 - abs(float(age) - 27.0))
    return float(club_elo + caps_bonus + goals_bonus + age_bonus)
