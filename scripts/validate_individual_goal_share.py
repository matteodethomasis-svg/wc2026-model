"""Validate the individual-goals term p_i (predicted share of a team's goals per
player) against REAL World Cup goal-scorers (WC 2018 + 2022), leak-free.

Design (the test the user approved 2026-06-15): the Dixon-Coles already gives
lambda_team (team goals) — that's NOT what we test. We test the RATIO: given a goal
was scored by a team, which player scored it? p_i is predicted from each player's
PRE-TOURNAMENT CLUB goal rate (goals/90 from appearances.csv, 365d window before
kickoff — the user is right that club stats carry ~90% of the signal). Aggregated
over the team's REAL starting XI (StatsBomb).

Target: data/interim/statsbomb_wc_goal_scorers.csv (real scorers). For each
attributable goal, the model assigns each XI player a probability p_i; we score
the multinomial "who scored this goal" with log loss + top-1 accuracy, vs two
baselines: uniform (1/XI) and a fixed position prior (FW>MF>DF>GK).

A goal is only scored if its scorer is in the matched XI (so we measure share
quality on the covered set). Outputs reports/individual_goal_share_validation.json.
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

WC_KICKOFF = {"2018": pd.Timestamp("2018-06-14"), "2022": pd.Timestamp("2022-11-20")}
# Fixed position prior (relative goal propensity) — a sane, non-fit baseline.
POSITION_PRIOR = {"GK": 0.02, "DF": 0.5, "MF": 1.0, "FW": 2.2}


def _norm(text: str) -> str:
    t = "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c))
    return t.lower().strip()


def _tokens(name: str) -> set[str]:
    return {tok for tok in _norm(name).replace(".", " ").split() if len(tok) > 1}


def _surname(name: str) -> str:
    parts = _norm(name).replace(".", " ").split()
    return parts[-1] if parts else ""


def _coarse_position(pos: str) -> str:
    p = str(pos).lower()
    if "goalkeeper" in p:
        return "GK"
    if "back" in p or "defen" in p:
        return "DF"
    if "forward" in p or "striker" in p or "wing" in p and "back" not in p:
        return "FW"
    if "midfield" in p or "center" in p or "centre" in p:
        return "MF"
    return "MF"


def _build_club_rate(appearances: pd.DataFrame, *, cutoff: pd.Timestamp,
                     window_days: int = 365) -> pd.DataFrame:
    lo = cutoff - pd.Timedelta(days=window_days)
    w = appearances[(appearances["date"] >= lo) & (appearances["date"] < cutoff)]
    g = w.groupby("player_id").agg(
        name=("player_name", "first"),
        goals=("goals", "sum"),
        mins=("minutes_played", "sum"),
    ).reset_index()
    g = g[g["mins"] >= 270]  # >= ~3 full matches of club football to be meaningful
    g["per90"] = g["goals"] / g["mins"] * 90.0
    return g


def _match_rate(player: str, rate_index_tok: dict, rate_index_sur: dict) -> float | None:
    toks = _tokens(player)
    best = None
    best_overlap = 0
    for key_toks, rec in rate_index_tok:
        ov = len(toks & key_toks)
        if ov >= 2 and ov > best_overlap:
            best_overlap = ov
            best = rec
    if best is not None:
        return best
    # surname fallback
    return rate_index_sur.get(_surname(player))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineups-input", default="data/interim/statsbomb_real_lineups.csv")
    parser.add_argument("--scorers-input", default="data/interim/statsbomb_wc_goal_scorers.csv")
    parser.add_argument("--panel-input",
                        default="data/interim/statsbomb_men_major_tournaments_match_features.csv")
    parser.add_argument("--appearances-input", default="data/raw/transfermarkt/appearances.csv")
    parser.add_argument("--output", default="reports/individual_goal_share_validation.json")
    args = parser.parse_args()

    lineups = pd.read_csv(args.lineups_input)
    scorers = pd.read_csv(args.scorers_input)
    panel = pd.read_csv(args.panel_input)
    appearances = pd.read_csv(
        args.appearances_input,
        usecols=["player_id", "player_name", "date", "goals", "minutes_played"],
    )
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")

    # match_id -> season, to pick the right pre-tournament cutoff
    season_by_match = dict(zip(panel["source_match_id"], panel["source_season_name"].astype(str)))

    # Pre-build club rate lookups per season (one cutoff each).
    rate_lookup: dict[str, dict] = {}
    for season, cutoff in WC_KICKOFF.items():
        rate = _build_club_rate(appearances, cutoff=cutoff)
        tok_index = [(_tokens(r.name), float(r.per90)) for r in rate.itertuples(index=False)]
        sur_index: dict[str, float] = {}
        for r in rate.itertuples(index=False):
            sur_index.setdefault(_surname(r.name), float(r.per90))
        rate_lookup[season] = {"tok": tok_index, "sur": sur_index}

    # Only attributable goals whose scorer we can place in the starting XI.
    real_goals = scorers[~scorers["is_own_goal"]].copy()

    eval_rows = []
    matched_goals = 0
    unmatched_scorer = 0
    for goal in real_goals.itertuples(index=False):
        mid = int(goal.source_match_id)
        season = season_by_match.get(mid)
        if season not in WC_KICKOFF:
            continue
        team = goal.team
        xi = lineups[(lineups["match_id"] == mid) & (lineups["team"] == team)
                     & (lineups["is_starter"] == True)]  # noqa: E712
        if xi.empty:
            continue
        # locate the scorer within this XI (by name)
        scorer_tokens = _tokens(goal.scorer)
        xi_players = list(xi["player"])
        scorer_idx = None
        for i, p in enumerate(xi_players):
            if len(_tokens(p) & scorer_tokens) >= 2 or _surname(p) == _surname(goal.scorer):
                scorer_idx = i
                break
        if scorer_idx is None:
            unmatched_scorer += 1
            continue
        matched_goals += 1

        # build p_i for the XI from club rate + position prior
        rl = rate_lookup[season]
        rates = []
        priors = []
        for _, prow in xi.iterrows():
            r = _match_rate(prow["player"], rl["tok"], rl["sur"])
            rates.append(0.0 if r is None else max(r, 0.0))
            priors.append(POSITION_PRIOR[_coarse_position(prow["position"])])
        rates = np.array(rates, dtype=float)
        priors = np.array(priors, dtype=float)

        # Model p_i: club rate, smoothed by the position prior so a 0-rate FW isn't 0.
        # (rate + small prior floor), then normalize.
        model_w = rates + 0.15 * priors
        if model_w.sum() <= 0:
            model_w = priors.copy()
        p_model = model_w / model_w.sum()
        p_uniform = np.full(len(xi), 1.0 / len(xi))
        p_prior = priors / priors.sum()

        eval_rows.append({
            "ll_model": -math.log(max(p_model[scorer_idx], 1e-12)),
            "ll_uniform": -math.log(max(p_uniform[scorer_idx], 1e-12)),
            "ll_prior": -math.log(max(p_prior[scorer_idx], 1e-12)),
            "top1_model": int(np.argmax(p_model) == scorer_idx),
            "top1_prior": int(np.argmax(p_prior) == scorer_idx),
            "top3_model": int(scorer_idx in np.argsort(p_model)[-3:]),
            "top3_prior": int(scorer_idx in np.argsort(p_prior)[-3:]),
        })

    frame = pd.DataFrame(eval_rows)
    out = {
        "real_attributable_goals": int(len(real_goals)),
        "goals_scorer_found_in_XI": matched_goals,
        "goals_scorer_not_in_matched_XI": unmatched_scorer,
        "eval_goals": int(len(frame)),
        "mean_log_loss": {
            "model_club_rate": round(float(frame["ll_model"].mean()), 4),
            "position_prior": round(float(frame["ll_prior"].mean()), 4),
            "uniform_1_over_XI": round(float(frame["ll_uniform"].mean()), 4),
        },
        "top1_accuracy": {
            "model_club_rate": round(float(frame["top1_model"].mean()), 3),
            "position_prior": round(float(frame["top1_prior"].mean()), 3),
        },
        "top3_accuracy": {
            "model_club_rate": round(float(frame["top3_model"].mean()), 3),
            "position_prior": round(float(frame["top3_prior"].mean()), 3),
        },
        "verdict": (
            "p_i ADDS value: club-rate beats both uniform and position-prior on log loss"
            if frame["ll_model"].mean() < frame["ll_prior"].mean() - 1e-3
            and frame["ll_model"].mean() < frame["ll_uniform"].mean() - 1e-3
            else "p_i does NOT clearly beat the position prior — club rate adds little over 'forwards score'"
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
