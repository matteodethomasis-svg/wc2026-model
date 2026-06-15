"""Turn Transfermarkt historical market values into a per-player, per-date quality rating.

Three data-driven corrections (see memory transfermarkt-rating-adjustments):
  1. AGE: market value peaks ~28 and decays for older players even when still elite. We
     estimate the inflation-removed log-value-vs-age curve and ADD BACK the age penalty so
     the rating reflects current strength, not resale value (up-rates Modric/Messi).
  2. INFLATION: the market drifts over time. We index on years with a FULL sample only
     (n >= MIN_YEAR_N, ~through 2023); sparse recent years (2024-26, esp. 2026 with ~49
     rows) anchor to the last full year — never divide by the empty-2026 median.
  3. ROLE: GK/DF are valued below FW at equal quality. We normalize within position so the
     keeper layer isn't under-weighted.

Output: a tidy CSV (player_id, name, date, age, position, raw_value, rating) where `rating`
is a corrected log-value on a comparable scale across ages, eras, and positions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MIN_YEAR_N = 10_000          # a year needs this many valuations to define the inflation index
PEAK_AGE = 28                # career-value peak from the data
AGE_MIN, AGE_MAX = 16, 40


def _coarse_position(sub_position: str, position: str) -> str:
    p = str(position or sub_position or "").lower()
    if "keeper" in p or p == "gk" or "goalkeeper" in p:
        return "GK"
    if "back" in p or "defen" in p or p == "df":
        return "DF"
    if "mid" in p or p == "mf":
        return "MF"
    if "forward" in p or "wing" in p or "strik" in p or "attack" in p or p == "fw":
        return "FW"
    return "MF"


def build_rating_table(
    valuations: pd.DataFrame, players: pd.DataFrame, *,
    correct_age: bool = True, correct_inflation: bool = True, correct_role: bool = True,
) -> pd.DataFrame:
    players = players.copy()
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    players["coarse_pos"] = [
        _coarse_position(sp, p)
        for sp, p in zip(players.get("sub_position", ""), players.get("position", ""), strict=False)
    ]
    meta = players.set_index("player_id")[["name", "date_of_birth", "coarse_pos"]]

    v = valuations.copy()
    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    v = v[v["market_value_in_eur"] > 0].dropna(subset=["date"])
    v = v.join(meta, on="player_id")
    v = v.dropna(subset=["date_of_birth"])
    v["age"] = (v["date"] - v["date_of_birth"]).dt.days / 365.25
    v = v[(v["age"] >= AGE_MIN) & (v["age"] <= AGE_MAX)]
    v["log_mv"] = np.log(v["market_value_in_eur"])
    v["year"] = v["date"].dt.year

    # --- inflation index on FULL-sample years only ---
    counts = v.groupby("year")["log_mv"].transform("size")
    full = v[counts >= MIN_YEAR_N]
    year_index = full.groupby("year")["log_mv"].median()
    last_full_year = int(year_index.index.max())
    def infl(year: int) -> float:
        return float(year_index.get(year, year_index.loc[last_full_year]))
    v["infl"] = v["year"].map(infl) if correct_inflation else 0.0
    v["log_deinfl"] = v["log_mv"] - v["infl"]

    # --- age curve from de-inflated values; ADD BACK the penalty vs the peak ---
    if correct_age:
        age_curve = full.assign(
            ld=full["log_mv"] - full["year"].map(infl)
        ).groupby(full["age"].round())["ld"].mean()
        peak = float(age_curve.get(PEAK_AGE, age_curve.max()))
        def age_adj(age: float) -> float:
            a = float(np.clip(round(age), AGE_MIN, AGE_MAX))
            return peak - float(age_curve.get(a, age_curve.min()))
        v["log_age_corrected"] = v["log_deinfl"] + v["age"].map(age_adj)
    else:
        v["log_age_corrected"] = v["log_deinfl"]

    # --- role normalization: center each position to the same mean ---
    if correct_role:
        pos_mean = v.groupby("coarse_pos")["log_age_corrected"].transform("mean")
        v["rating"] = v["log_age_corrected"] - pos_mean + v["log_age_corrected"].mean()
    else:
        v["rating"] = v["log_age_corrected"]

    out = v[[
        "player_id", "name", "date", "year", "age", "coarse_pos",
        "market_value_in_eur", "rating",
    ]].rename(columns={"coarse_pos": "position", "market_value_in_eur": "raw_value"})
    return out.sort_values(["player_id", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valuations", default="data/raw/transfermarkt/player_valuations.csv")
    parser.add_argument("--players", default="data/raw/transfermarkt/players.csv")
    parser.add_argument("--output", default="data/interim/transfermarkt_player_rating.csv")
    parser.add_argument("--no-age", action="store_true")
    parser.add_argument("--no-inflation", action="store_true")
    parser.add_argument("--no-role", action="store_true")
    args = parser.parse_args()

    valuations = pd.read_csv(ROOT / args.valuations)
    players = pd.read_csv(ROOT / args.players)
    table = build_rating_table(
        valuations, players,
        correct_age=not args.no_age, correct_inflation=not args.no_inflation,
        correct_role=not args.no_role,
    )

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {out}  rows={len(table)} players={table['player_id'].nunique()}")
    print(f"  rating: mean={table['rating'].mean():.2f} sd={table['rating'].std():.2f} "
          f"min={table['rating'].min():.2f} max={table['rating'].max():.2f}")


if __name__ == "__main__":
    main()
