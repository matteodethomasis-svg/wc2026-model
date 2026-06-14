"""Build WC2026 squad strength from individual player Elo (PlayerElo snapshot).

The default squad-strength layer uses each player's *club* Elo as a proxy for the
player's quality. That misrates stars at weak clubs (Maignan at AC Milan) and can't
tell a star from a benchwarmer in the same club (every PSG player shared PSG's Elo).
This script replaces the proxy with per-player Elo from PlayerElo
(`data/raw/archive/players.csv`), then reuses the existing expected-XI aggregation.

IMPORTANT: PlayerElo here is a STATIC 2026 snapshot, so this is leak-free only for
the live WC2026 fixtures. Do not use it to re-validate on historical backtests.
See memory note player-elo-static-snapshot. The "correct" version needs a per-date
player Elo time series, deferred like the coach-Elo work.

Matching: PlayerElo names look like "O. Dembélé" (initial + surname); WC squads use
full names. We match on (surname, first-initial) and disambiguate homonyms by club.
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd

from wc2026_model.features import aggregate_team_squad_strength
from wc2026_model.features.squad_strength import normalize_club_name


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _norm_token(text: str) -> str:
    return _strip_accents(str(text)).lower().strip()


def _surname_key(full_name: str) -> str:
    parts = _norm_token(full_name).replace(".", " ").split()
    return parts[-1] if parts else ""


def _first_initial(full_name: str) -> str:
    parts = _norm_token(full_name).replace(".", " ").split()
    return parts[0][0] if parts and parts[0] else ""


def _build_player_elo_index(player_elo: pd.DataFrame) -> dict[tuple[str, str], list[dict]]:
    index: dict[tuple[str, str], list[dict]] = {}
    for row in player_elo.itertuples(index=False):
        name = str(getattr(row, "player_name", ""))
        key = (_surname_key(name), _first_initial(name))
        index.setdefault(key, []).append(
            {
                "elo": float(getattr(row, "elo")),
                "club": normalize_club_name(str(getattr(row, "current_team", ""))),
            }
        )
    return index


def _match_player_elo(
    squad_player: str,
    squad_club: str,
    index: dict[tuple[str, str], list[dict]],
) -> tuple[float | None, str]:
    key = (_surname_key(squad_player), _first_initial(squad_player))
    candidates = index.get(key, [])
    if not candidates:
        return None, "unmatched"
    if len(candidates) == 1:
        return candidates[0]["elo"], "name"
    # Disambiguate homonyms by club.
    target_club = normalize_club_name(squad_club)
    club_hits = [c for c in candidates if c["club"] and c["club"] == target_club]
    if len(club_hits) == 1:
        return club_hits[0]["elo"], "name+club"
    if club_hits:
        # Several at the same club: take the strongest (the actual international).
        return max(club_hits, key=lambda c: c["elo"])["elo"], "name+club+max"
    # No club match: fall back to the strongest namesake (usually the famous one).
    return max(candidates, key=lambda c: c["elo"])["elo"], "name+max"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--squad-players-input",
        default="reports/wc2026_squad_players_with_club_elo.csv",
        help="Player-level squad file (already mapped to club Elo).",
    )
    parser.add_argument(
        "--player-elo-input", default="data/raw/archive/players.csv",
        help="PlayerElo per-player snapshot.",
    )
    parser.add_argument("--groups-input", default="data/reference/wc2026_groups_actual.csv")
    parser.add_argument(
        "--players-output", default="reports/wc2026_squad_players_with_player_elo.csv"
    )
    parser.add_argument(
        "--teams-output", default="reports/wc2026_squad_strength_player_elo_ratings.csv"
    )
    args = parser.parse_args()

    squad = pd.read_csv(args.squad_players_input)
    player_elo = pd.read_csv(args.player_elo_input)
    index = _build_player_elo_index(player_elo)

    elos: list[float | None] = []
    methods: list[str] = []
    for row in squad.itertuples(index=False):
        elo, method = _match_player_elo(
            str(getattr(row, "player", "")), str(getattr(row, "club", "")), index
        )
        elos.append(elo)
        methods.append(method)

    squad = squad.copy()
    squad["player_elo"] = elos
    squad["player_elo_match_method"] = methods

    matched = sum(1 for e in elos if e is not None)
    print(f"Matched {matched}/{len(squad)} players to individual PlayerElo "
          f"({matched / len(squad):.0%})")

    # Reuse the expected-XI aggregation by swapping club_elo -> player_elo.
    agg_input = squad.copy()
    agg_input["club_elo"] = squad["player_elo"]
    team_strength = aggregate_team_squad_strength(agg_input)
    # Rename the club-Elo-named outputs to make the source explicit.
    team_strength = team_strength.rename(
        columns=lambda c: c.replace("club_elo", "player_elo")
    )

    if Path(args.groups_input).exists():
        groups = pd.read_csv(args.groups_input)
        if {"team", "group"}.issubset(groups.columns):
            team_strength = team_strength.merge(
                groups[["team", "group"]], on="team", how="left"
            )

    Path(args.players_output).parent.mkdir(parents=True, exist_ok=True)
    squad.to_csv(args.players_output, index=False)
    team_strength.to_csv(args.teams_output, index=False)
    print(f"Wrote {args.players_output} and {args.teams_output}")

    # Quick sanity print: France key players + expected XI vs club-Elo baseline.
    fr = squad[squad["team"] == "France"]
    print("\nFrance star check (player Elo):")
    for nm in ("Maignan", "Dembélé", "Mbappé"):
        hit = fr[fr["player"].str.contains(nm, na=False)]
        for r in hit.itertuples(index=False):
            print(f"  {r.player:<22} club_elo={getattr(r, 'club_elo', float('nan')):.0f}"
                  f"  player_elo={r.player_elo}")


if __name__ == "__main__":
    main()
