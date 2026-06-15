"""Aggregate the corrected Transfermarkt per-player rating over the REAL starting XI to a
(team, tournament_year) squad strength — the combination of both upgrades:
real lineups ([[real-xi-beats-heuristic-validated]]) + a better per-player rating
([[transfermarkt-rating-adjustments]]).

For each WC backtest match we read each real starter's Transfermarkt rating AT THE MATCH
DATE (leak-free), average per team, then average across the team's tournament matches.
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

_SB_SEASON_TO_YEAR = {(43, 3): 2018, (43, 106): 2022, (43, 51): 2014}


def _strip(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(t)) if not unicodedata.combining(c)).lower().strip()


def _toks(t: str) -> set[str]:
    return set(_strip(t).replace(".", " ").replace("-", " ").split())


def _match_candidates(name: str, team: str, tm_index) -> list[int]:
    """Return ALL plausible TM player_ids for a lineup name, nationality-gated. The caller
    picks among homonyms by market value at the match date (the star, not a 3rd-tier
    namesake) — this is what fixes mononyms like Brazil's Fred / Casemiro / Hulk."""
    nat = tm_index["team_to_country"].get(team)
    s = _strip(name)
    nt = _toks(name)
    parts = s.split()
    if not parts:
        return []
    first = parts[0]

    def _collect(require_nat: bool) -> list[int]:
        out: list[int] = []
        pid = tm_index["by_name"].get(s)
        if pid is not None and (not require_nat or tm_index["nat_of"].get(pid) == nat):
            out.append(pid)
        need = 1 if (require_nat and nat) else 2
        for cpid, ptoks in tm_index["all"]:
            if require_nat and tm_index["nat_of"].get(cpid) != nat:
                continue
            if len(nt & ptoks) >= need:
                out.append(cpid)
        for tok in (parts[-1], *nt):
            for cpid, pfirst in tm_index["by_last_first"].get(tok, []):
                if require_nat and tm_index["nat_of"].get(cpid) != nat:
                    continue
                if pfirst and first and pfirst[0] == first[0]:
                    out.append(cpid)
        return list(dict.fromkeys(out))

    if nat:
        hits = _collect(require_nat=True)
        if hits:
            return hits
    return _collect(require_nat=False)


def build_tm_index(tm_players: pd.DataFrame, ratings: pd.DataFrame, lineups: pd.DataFrame) -> dict:
    """Name/nationality index over TM players who have a rating, plus a team->country map.
    Shared by the value and composite strength builders."""
    rated_ids = set(ratings["player_id"].unique())
    by_name, by_last_first, all_players, nat_of = {}, {}, [], {}
    for r in tm_players.itertuples(index=False):
        if r.player_id not in rated_ids:
            continue
        nm = _strip(r.name)
        parts = nm.split()
        if not parts:
            continue
        by_name.setdefault(nm, r.player_id)
        nat_of[r.player_id] = _strip(getattr(r, "country_of_citizenship", "") or "")
        by_last_first.setdefault(parts[-1], []).append((r.player_id, parts[0]))
        for tok in set(parts):
            by_last_first.setdefault(tok, []).append((r.player_id, parts[0]))
        all_players.append((r.player_id, _toks(r.name)))

    _TEAM_ALIASES = {
        "south korea": "korea, south", "ivory coast": "cote d'ivoire",
        "iran": "iran", "usa": "united states", "united states": "united states",
        "czech republic": "czech republic", "china pr": "china",
    }
    tm_countries = {_strip(c) for c in tm_players["country_of_citizenship"].dropna().unique()}
    team_to_country = {}
    for team in lineups["team"].dropna().unique():
        ts = _strip(team)
        cand = _TEAM_ALIASES.get(ts, ts)
        team_to_country[str(team)] = cand if cand in tm_countries else (ts if ts in tm_countries else None)

    return {"by_name": by_name, "by_last_first": by_last_first, "all": all_players,
            "nat_of": nat_of, "team_to_country": team_to_country}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineups", default="data/interim/statsbomb_real_lineups.csv")
    parser.add_argument(
        "--panel", default="data/interim/statsbomb_men_major_tournaments_match_features.csv")
    parser.add_argument("--ratings", default="data/interim/transfermarkt_player_rating.csv")
    parser.add_argument("--players", default="data/raw/transfermarkt/players.csv")
    parser.add_argument(
        "--output", default="reports/historical_world_cup_real_xi_transfermarkt_strength.csv")
    args = parser.parse_args()

    lineups = pd.read_csv(ROOT / args.lineups)
    panel = pd.read_csv(ROOT / args.panel)
    ratings = pd.read_csv(ROOT / args.ratings, parse_dates=["date"])
    tm_players = pd.read_csv(ROOT / args.players)

    # match_id -> (year, match_date)
    panel = panel[panel["source_competition_id"] == 43].copy()
    panel["year"] = [_SB_SEASON_TO_YEAR.get((43, int(s))) for s in panel["source_season_id"]]
    panel["match_date"] = pd.to_datetime(panel["match_date"], errors="coerce")
    mid_info = {int(r.source_match_id): (r.year, r.match_date) for r in panel.itertuples(index=False)}

    tm_index = build_tm_index(tm_players, ratings, lineups)

    # Ratings keyed for fast "value at date" lookup.
    ratings = ratings.sort_values(["player_id", "date"])
    rating_by_player = {pid: g for pid, g in ratings.groupby("player_id")}

    def rating_at(pid: int, when) -> float | None:
        g = rating_by_player.get(pid)
        if g is None:
            return None
        s = g[g["date"] <= when]
        return None if s.empty else float(s.iloc[-1]["rating"])

    starters = lineups[lineups["is_starter"].astype(str).str.lower().isin(["true", "1"])]
    rows = []
    matched = unmatched = 0
    name_cache: dict[str, int | None] = {}
    for r in starters.itertuples(index=False):
        info = mid_info.get(int(r.match_id))
        if info is None or info[0] is None:
            continue
        year, mdate = int(info[0]), info[1]
        cache_key = f"{r.team}|{r.player}"
        if cache_key not in name_cache:
            name_cache[cache_key] = _match_candidates(r.player, str(r.team), tm_index)
        candidates = name_cache[cache_key]
        # Disambiguate homonyms by market value AT THE MATCH DATE: a WC starter is almost
        # always the most valuable of the same-named, same-nationality players.
        rat = None
        for cpid in candidates:
            cr = rating_at(cpid, mdate)
            if cr is not None and (rat is None or cr > rat):
                rat = cr
        if rat is None:
            unmatched += 1
            continue
        matched += 1
        rows.append({"tournament_year": year, "team": r.team, "rating": rat})

    df = pd.DataFrame(rows)
    agg = (df.groupby(["tournament_year", "team"])
           .agg(real_xi_tm_rating=("rating", "mean"), sample=("rating", "size"))
           .reset_index())
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)
    total = matched + unmatched
    print(f"Wrote {out}  team-years={len(agg)} matched={matched}/{total} "
          f"({100*matched/max(total,1):.0f}%)")
    print(agg.sort_values("real_xi_tm_rating", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
