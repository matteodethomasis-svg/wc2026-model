"""Build a (team, tournament_year) squad-strength CSV from the REAL starting XIs.

Joins the StatsBomb real lineups (who actually started each match) to the historical
players-with-club-Elo frame by name, then averages the club-Elo of the real starters per
team-tournament. The output mirrors the heuristic squad-strength schema so the existing
backtest harness (evaluate_historical_world_cup_squad_scale.py) can consume it with
``--rating-column real_xi_club_elo_rating`` — a like-for-like comparison of REAL XI vs the
heuristic top-11 "expected XI".

Name matching mirrors build_world_cup_player_elo_strength: surname + first initial, with
club as a tie-breaker isn't available here (StatsBomb has no club), so we match on
(surname, first-initial) within the same team-year, which is unique enough in practice.
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# StatsBomb competition/season id -> tournament year, to map lineups back to the year the
# players-with-club-Elo frame is keyed on.
_SB_SEASON_TO_YEAR = {
    (43, 3): 2018,    # FIFA World Cup 2018
    (43, 106): 2022,  # FIFA World Cup 2022
    (43, 51): 2014,   # FIFA World Cup 2014 (if present)
}


def _strip(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c)).lower().strip()


def _surname_key(name: str) -> str:
    parts = _strip(name).replace(".", " ").split()
    return parts[-1] if parts else ""


def _first_initial(name: str) -> str:
    parts = _strip(name).replace(".", " ").split()
    return parts[0][0] if parts and parts[0] else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineups", default="data/interim/statsbomb_real_lineups.csv")
    parser.add_argument(
        "--panel", default="data/interim/statsbomb_men_major_tournaments_match_features.csv")
    parser.add_argument(
        "--players-elo", default="reports/historical_world_cup_squad_players_with_club_elo.csv")
    parser.add_argument(
        "--output", default="reports/historical_world_cup_real_xi_squad_strength.csv")
    args = parser.parse_args()

    lineups = pd.read_csv(ROOT / args.lineups)
    panel = pd.read_csv(ROOT / args.panel)
    players = pd.read_csv(ROOT / args.players_elo)

    # Map each match_id -> tournament_year via the panel's competition/season ids.
    panel = panel.copy()
    panel["tournament_year"] = [
        _SB_SEASON_TO_YEAR.get((int(c), int(s)))
        for c, s in zip(panel["source_competition_id"], panel["source_season_id"], strict=True)
    ]
    mid_to_year = dict(zip(panel["source_match_id"].astype(int), panel["tournament_year"], strict=True))

    starters = lineups[lineups["is_starter"].astype(str).str.lower().isin(["true", "1"])].copy()
    starters["tournament_year"] = starters["match_id"].map(mid_to_year)
    starters = starters.dropna(subset=["tournament_year"])
    starters["tournament_year"] = starters["tournament_year"].astype(int)

    # Index Wikipedia players per (year, team) with their club-Elo + name tokens. We match
    # on TOKEN OVERLAP (any shared name token), then fall back to surname+initial. Token
    # overlap handles "José Ignacio Fernández Iglesias" -> "Nacho" only via fallback, but
    # catches most multi-name cases that surname-only misses. Brazilian/Spanish nicknames
    # (Casemiro, Isco, Paulinho) genuinely don't overlap and stay unmatched — accepted.
    players = players.copy()
    players["club_elo"] = pd.to_numeric(players["club_elo"], errors="coerce")
    players = players.dropna(subset=["club_elo"])
    wiki_index: dict[tuple, list[dict]] = {}
    for r in players.itertuples(index=False):
        toks = set(_strip(r.player).replace(".", " ").split())
        entry = {"tokens": toks, "surname": _surname_key(r.player),
                 "initial": _first_initial(r.player), "elo": float(r.club_elo)}
        wiki_index.setdefault((int(r.tournament_year), str(r.team)), []).append(entry)

    matched_rows = []
    unmatched = 0
    for r in starters.itertuples(index=False):
        key = (int(r.tournament_year), str(r.team))
        candidates = wiki_index.get(key, [])
        sb_tokens = set(_strip(r.player).replace(".", " ").split())
        sb_surname, sb_initial = _surname_key(r.player), _first_initial(r.player)
        hit = None
        for c in candidates:                       # 1) shared name token
            if sb_tokens & c["tokens"]:
                hit = c
                break
        if hit is None:                            # 2) surname + first initial
            for c in candidates:
                if c["surname"] == sb_surname and c["initial"] == sb_initial:
                    hit = c
                    break
        if hit is None:
            unmatched += 1
            continue
        matched_rows.append({
            "tournament_year": int(r.tournament_year),
            "team": str(r.team),
            "player": str(r.player),
            "club_elo": hit["elo"],
        })
    matched = pd.DataFrame(matched_rows)

    # Aggregate to (team, year): mean club-Elo of the real starters across the tournament.
    agg = (
        matched.groupby(["tournament_year", "team"])
        .agg(real_xi_club_elo_rating=("club_elo", "mean"),
             real_xi_starter_sample=("club_elo", "size"))
        .reset_index()
    )
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)

    total = len(starters)
    print(f"Wrote {out}")
    print(f"  team-years={len(agg)} matched_starter_rows={len(matched)}/{total} "
          f"unmatched={unmatched} ({100*unmatched/max(total,1):.0f}%)")
    if not agg.empty:
        print(agg.sort_values("real_xi_club_elo_rating", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
