"""Composite real-XI squad strength: blend market value + club performance + (optionally)
club-Elo per player, aggregated over the REAL starting XI at each match date.

Rationale (user): each signal captures a different facet — market value = durable talent,
performance (minutes + G/A) = demonstrated club form, club-Elo = team level — so a blend
should estimate strength better than any single one, with errors partly cancelling.

Emits (team, year) columns: composite + each component, so the backtest can weight them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SB_SEASON_TO_YEAR = {(43, 3): 2018, (43, 106): 2022}


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineups", default="data/interim/statsbomb_real_lineups.csv")
    parser.add_argument(
        "--panel", default="data/interim/statsbomb_men_major_tournaments_match_features.csv")
    parser.add_argument("--value-rating", default="data/interim/transfermarkt_player_rating.csv")
    parser.add_argument("--perf-rating", default="data/interim/transfermarkt_performance_rating.csv")
    parser.add_argument("--players", default="data/raw/transfermarkt/players.csv")
    # We reuse the matched (team, player) -> player_id mapping logic from the value script
    # by importing it.
    parser.add_argument(
        "--matched-strength",
        default="reports/historical_world_cup_real_xi_transfermarkt_strength.csv",
        help="Not used directly; kept for parity.")
    parser.add_argument(
        "--output", default="reports/historical_world_cup_real_xi_composite_strength.csv")
    parser.add_argument("--w-value", type=float, default=1.0)
    parser.add_argument("--w-perf", type=float, default=1.0)
    args = parser.parse_args()

    # Reuse the value script's matcher to get (team, player) -> player_id.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vstr", ROOT / "scripts" / "build_real_xi_transfermarkt_strength.py")
    vstr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vstr)

    lineups = pd.read_csv(ROOT / args.lineups)
    panel = pd.read_csv(ROOT / args.panel)
    players = pd.read_csv(ROOT / args.players)
    value = pd.read_csv(ROOT / args.value_rating, parse_dates=["date"])
    perf = pd.read_csv(ROOT / args.perf_rating, parse_dates=["anchor_date"])

    panel = panel[panel["source_competition_id"] == 43].copy()
    panel["year"] = [_SB_SEASON_TO_YEAR.get((43, int(s))) for s in panel["source_season_id"]]
    panel["match_date"] = pd.to_datetime(panel["match_date"], errors="coerce")
    mid_info = {int(r.source_match_id): (r.year, r.match_date) for r in panel.itertuples(index=False)}

    tm_index = vstr.build_tm_index(players, value, lineups)

    value = value.sort_values(["player_id", "date"])
    val_by_player = {pid: g for pid, g in value.groupby("player_id")}
    perf = perf.sort_values(["player_id", "anchor_date"])
    perf_by_player = {pid: g for pid, g in perf.groupby("player_id")}

    def latest(g, col, when, datecol):
        s = g[g[datecol] <= when]
        return None if s.empty else float(s.iloc[-1][col])

    starters = lineups[lineups["is_starter"].astype(str).str.lower().isin(["true", "1"])]
    rows = []
    cache: dict[str, list[int]] = {}
    for r in starters.itertuples(index=False):
        info = mid_info.get(int(r.match_id))
        if info is None or info[0] is None:
            continue
        year, mdate = int(info[0]), info[1]
        key = f"{r.team}|{r.player}"
        if key not in cache:
            cache[key] = vstr._match_candidates(r.player, str(r.team), tm_index)
        # pick the homonym with the highest value at the date
        best_pid, best_val = None, None
        for cpid in cache[key]:
            g = val_by_player.get(cpid)
            if g is None:
                continue
            vv = latest(g, "rating", mdate, "date")
            if vv is not None and (best_val is None or vv > best_val):
                best_pid, best_val = cpid, vv
        if best_pid is None:
            continue
        pv = (latest(perf_by_player[best_pid], "perf_rating", mdate, "anchor_date")
              if best_pid in perf_by_player else None)
        rows.append({"year": year, "team": r.team, "player_id": best_pid,
                     "value_rating": best_val, "perf_rating": pv})

    df = pd.DataFrame(rows)
    # Players with no perf snapshot (e.g. non-top-league) get the median perf so they're
    # not dropped — value still carries them.
    df["perf_rating"] = df["perf_rating"].fillna(df["perf_rating"].median())
    # Z-score each component across the whole pool, then blend.
    df["v_z"] = _zscore(df["value_rating"])
    df["p_z"] = _zscore(df["perf_rating"])
    df["composite"] = args.w_value * df["v_z"] + args.w_perf * df["p_z"]

    agg = (df.groupby(["year", "team"])
           .agg(composite=("composite", "mean"),
                value_only=("v_z", "mean"),
                perf_only=("p_z", "mean"),
                sample=("composite", "size"))
           .reset_index()
           .rename(columns={"year": "tournament_year"}))
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)
    print(f"Wrote {out}  team-years={len(agg)}")
    for y in (2018, 2022):
        print(f"\nWC {y} by composite:")
        sub = agg[agg.tournament_year == y].sort_values("composite", ascending=False).head(8)
        for x in sub.itertuples():
            print(f"  {x.team:16} comp={x.composite:+.2f} (val={x.value_only:+.2f} perf={x.perf_only:+.2f})")


if __name__ == "__main__":
    main()
