"""Build the REAL per-match goal-scorer target from StatsBomb open-data events,
for the World Cup matches in the major-tournaments panel (WC 2018 + 2022).

This is the leak-free TARGET for validating an individual-goals term: who actually
scored each goal. A goal is a Shot with outcome=Goal (open play / penalty / free
kick), plus 'Own Goal For' (credited to the benefiting team, no real scorer). We
record team + scorer + goal type; own goals are tagged so they can be excluded
from scorer attribution.

Output: data/interim/statsbomb_wc_goal_scorers.csv with one row per goal.
Columns: source_match_id, season, team, scorer, scorer_id, goal_type, is_own_goal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wc2026_model.data.statsbomb_open_data import fetch_statsbomb_events


def _extract_goals(match_id: int) -> list[dict[str, object]]:
    events = fetch_statsbomb_events(int(match_id))
    goals: list[dict[str, object]] = []
    for event in events:
        etype = (event.get("type") or {}).get("name")
        team = (event.get("team") or {}).get("name")
        player = event.get("player") or {}
        if etype == "Shot":
            shot = event.get("shot") or {}
            if (shot.get("outcome") or {}).get("name") == "Goal":
                goals.append({
                    "team": team,
                    "scorer": player.get("name"),
                    "scorer_id": player.get("id"),
                    "goal_type": (shot.get("type") or {}).get("name"),
                    "is_own_goal": False,
                })
        elif etype == "Own Goal For":
            # Credited to the team that benefits; no attributable scorer for our purpose.
            goals.append({
                "team": team,
                "scorer": None,
                "scorer_id": None,
                "goal_type": "Own Goal",
                "is_own_goal": True,
            })
    return goals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-input",
                        default="data/interim/statsbomb_men_major_tournaments_match_features.csv")
    parser.add_argument("--competition-filter", default="World Cup")
    parser.add_argument("--output", default="data/interim/statsbomb_wc_goal_scorers.csv")
    args = parser.parse_args()

    panel = pd.read_csv(args.panel_input)
    wc = panel[panel["source_competition_name"].str.contains(
        args.competition_filter, case=False, na=False)]
    match_ids = wc[["source_match_id", "source_season_name"]].drop_duplicates()

    rows: list[dict[str, object]] = []
    for i, mrow in enumerate(match_ids.itertuples(index=False), 1):
        mid = int(mrow.source_match_id)
        try:
            goals = _extract_goals(mid)
        except Exception as exc:  # one bad match shouldn't kill the build
            print(f"  [skip] match {mid}: {exc}")
            continue
        for g in goals:
            rows.append({"source_match_id": mid,
                         "season": mrow.source_season_name, **g})
        if i % 20 == 0:
            print(f"  ...{i}/{len(match_ids)} matches")

    frame = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    real = frame[~frame["is_own_goal"]]
    print(f"\nWrote {args.output}: {len(frame)} goals "
          f"({len(real)} attributable, {len(frame) - len(real)} own goals) "
          f"across {frame['source_match_id'].nunique()} matches.")
    print("Top scorers (attributable):")
    print(real.groupby("scorer").size().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
