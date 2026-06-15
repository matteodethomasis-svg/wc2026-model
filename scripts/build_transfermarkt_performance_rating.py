"""Per-player, per-date PERFORMANCE rating from Transfermarkt appearances (club games).

Market value prices hype/potential; this measures DEMONSTRATED club output. For a given
date we look back over a trailing window and summarize each player's club form:
  - minutes played (trust / starter status — works for GK/DF too)
  - goals + assists per 90 (attacking output — weak for GK/DF, but blended later)
  - competition strength weight (a goal in a top-5 league > a lower league)

Output: player_id, date-anchored rating snapshots we can read AT A MATCH DATE (leak-free).
Because computing a trailing window for every (player, date) is heavy, we emit a rating per
player per CALENDAR HALF-YEAR anchor; the consumer takes the latest anchor <= match date.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Rough competition-strength multipliers (top-5 leagues + UCL highest).
_COMP_WEIGHT = {
    "GB1": 1.00, "ES1": 1.00, "IT1": 0.95, "L1": 0.95, "FR1": 0.90,  # top-5 leagues
    "CL": 1.10, "EL": 0.85,  # European cups
    "PO1": 0.75, "NL1": 0.75, "TR1": 0.70, "BE1": 0.65, "RU1": 0.65,
}
_DEFAULT_COMP_WEIGHT = 0.60
WINDOW_DAYS = 365


def build_performance_rating(appearances: pd.DataFrame) -> pd.DataFrame:
    a = appearances.copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    a = a.dropna(subset=["date"])
    a["w"] = a["competition_id"].map(_COMP_WEIGHT).fillna(_DEFAULT_COMP_WEIGHT)
    a["minutes_played"] = pd.to_numeric(a["minutes_played"], errors="coerce").fillna(0.0)
    a["goals"] = pd.to_numeric(a["goals"], errors="coerce").fillna(0.0)
    a["assists"] = pd.to_numeric(a["assists"], errors="coerce").fillna(0.0)
    a["ga"] = a["goals"] + a["assists"]
    a["half"] = a["date"].dt.year * 2 + (a["date"].dt.month > 6).astype(int)

    # Anchor dates: end of each half-year present in the data.
    anchors = sorted(a["half"].unique())
    half_to_date = {h: pd.Timestamp(year=h // 2, month=12 if h % 2 else 6, day=28) for h in anchors}

    rows: list[dict] = []
    # For each anchor, summarize the trailing WINDOW_DAYS of club games per player.
    a = a.sort_values("date")
    for h in anchors:
        end = half_to_date[h]
        start = end - pd.Timedelta(days=WINDOW_DAYS)
        win = a[(a["date"] > start) & (a["date"] <= end)]
        if win.empty:
            continue
        g = win.groupby("player_id")
        agg = g.agg(
            minutes=("minutes_played", "sum"),
            wminutes=("minutes_played", lambda s: float((s * win.loc[s.index, "w"]).sum())),
            ga=("ga", "sum"),
            wga=("ga", lambda s: float((s * win.loc[s.index, "w"]).sum())),
        )
        agg = agg[agg["minutes"] >= 270]  # >=3 full games to be meaningful
        if agg.empty:
            continue
        agg["ga_per90"] = 90.0 * agg["wga"] / agg["minutes"].clip(lower=1)
        # Performance score: log minutes (playing time / trust) + attacking output.
        agg["perf"] = np.log1p(agg["wminutes"]) + 3.0 * agg["ga_per90"]
        for pid, r in agg.iterrows():
            rows.append({"player_id": int(pid), "anchor_date": end,
                         "minutes": float(r["minutes"]), "ga_per90": float(r["ga_per90"]),
                         "perf": float(r["perf"])})

    out = pd.DataFrame(rows)
    # Standardize perf to a comparable scale (z-score across all snapshots).
    if not out.empty:
        out["perf_rating"] = (out["perf"] - out["perf"].mean()) / out["perf"].std()
    return out.sort_values(["player_id", "anchor_date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearances", default="data/raw/transfermarkt/appearances.csv")
    parser.add_argument("--output", default="data/interim/transfermarkt_performance_rating.csv")
    args = parser.parse_args()

    appearances = pd.read_csv(
        ROOT / args.appearances,
        usecols=["player_id", "date", "competition_id", "minutes_played", "goals", "assists"],
    )
    table = build_performance_rating(appearances)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {out}  rows={len(table)} players={table['player_id'].nunique() if not table.empty else 0}")


if __name__ == "__main__":
    main()
