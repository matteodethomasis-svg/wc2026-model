"""Track record: an append-only ledger of model-vs-market predictions over time,
scored against real results as matches finish.

The model produces snapshots (fixture 1X2 probabilities, outright champion
probabilities). The market (bookmaker / Polymarket) produces its own. We log each
snapshot with a timestamp so that, once a match is played, we can ask the only
question that matters: who predicted it better, and were the flagged edges real?

Two ledgers, same shape:
  - match ledger: one row per (snapshot_ts, match, outcome_side) for scheduled games
  - outright ledger: one row per (snapshot_ts, team) for the title market

Scoring uses log loss / Brier on the resolved outcome. Edges are "resolved" by
checking whether the side the model favoured over the market actually happened.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

MATCH_LEDGER_COLUMNS = [
    "snapshot_ts", "match_date", "home_team", "away_team",
    "model_home", "model_draw", "model_away",
    "market_home", "market_draw", "market_away",
]
OUTRIGHT_LEDGER_COLUMNS = [
    "snapshot_ts", "team", "model_champion", "market_champion",
]


def append_match_snapshot(
    ledger_path: str | Path,
    snapshot: pd.DataFrame,
    *,
    snapshot_ts: str,
) -> pd.DataFrame:
    """Append one model-vs-market match snapshot, de-duplicating on (ts, teams)."""
    rows = snapshot.copy()
    rows.insert(0, "snapshot_ts", snapshot_ts)
    out_cols = [c for c in MATCH_LEDGER_COLUMNS if c in rows.columns]
    rows = rows.loc[:, out_cols]
    return _append_dedup(ledger_path, rows, key=["snapshot_ts", "home_team", "away_team"])


def append_outright_snapshot(
    ledger_path: str | Path,
    snapshot: pd.DataFrame,
    *,
    snapshot_ts: str,
) -> pd.DataFrame:
    rows = snapshot.copy()
    rows.insert(0, "snapshot_ts", snapshot_ts)
    out_cols = [c for c in OUTRIGHT_LEDGER_COLUMNS if c in rows.columns]
    rows = rows.loc[:, out_cols]
    return _append_dedup(ledger_path, rows, key=["snapshot_ts", "team"])


def _append_dedup(ledger_path: str | Path, rows: pd.DataFrame, *, key: list[str]) -> pd.DataFrame:
    path = Path(ledger_path)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, rows], ignore_index=True)
    else:
        combined = rows
    combined = combined.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def _log_loss(prob: float, occurred: bool) -> float:
    eps = 1e-12
    p = min(max(float(prob), eps), 1.0 - eps)
    return -math.log(p if occurred else (1.0 - p))


def _result_side(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def score_match_ledger(
    ledger: pd.DataFrame,
    results: pd.DataFrame,
    *,
    kickoffs: pd.DataFrame | None = None,
    lineup_lead_minutes: int = 75,
) -> pd.DataFrame:
    """Score each match snapshot against the real result, PER MATCH.

    The honest comparison is game-by-game: for each match, who gave the outcome that
    actually happened the better full-distribution score (log loss over 1/X/2)? Then
    count how many games each side wins. This is deliberately NOT aggregated by
    outcome type (all draws, all home wins, ...): that would reward a biased model —
    e.g. one that over-predicts draws looks "good on draws" just by predicting many,
    not by being better calibrated on the actual games.

    FAIRNESS: the market (Polymarket) is live and moves during a game, while our
    prediction is fixed ante-post. To compare like-with-like we must use the market's
    ANTE-POST snapshot — but more precisely, the last one captured before the OFFICIAL
    LINEUP dropped. Official XIs publish ~1h before kickoff and move both our model and
    the market; scoring after that would compare a lineup-informed snapshot. So we cut at
    ``kickoff − lineup_lead_minutes`` (default 75 min) and keep, per match, the latest
    snapshot before that cutoff. This is the "equal information" test: who predicted
    better with LESS info, before anyone saw the teamsheet. Pass `kickoffs`
    (home_team, away_team, kickoff_ts). Without kickoffs we fall back to the latest
    snapshot (only safe before any of the matches start).
    """
    if ledger.empty or results.empty:
        return pd.DataFrame()

    work = ledger.copy()
    work["_snap_ts"] = pd.to_datetime(work["snapshot_ts"], errors="coerce", utc=True)

    if kickoffs is not None and not kickoffs.empty:
        ko = {
            (str(r.home_team), str(r.away_team)): r.kickoff_ts
            for r in kickoffs.itertuples(index=False)
        }
        cutoff_delta = pd.Timedelta(minutes=lineup_lead_minutes)
        kept = []
        for (home, away), grp in work.groupby(["home_team", "away_team"]):
            kickoff = ko.get((str(home), str(away)))
            # Cut at kickoff − lineup lead so the scored snapshot predates the official XI.
            cutoff = None if kickoff is None else kickoff - cutoff_delta
            ante = grp if cutoff is None else grp[grp["_snap_ts"] < cutoff]
            if ante.empty:
                continue  # no pre-lineup snapshot -> can't fairly score this match
            kept.append(ante.sort_values("_snap_ts").iloc[[-1]])
        latest = pd.concat(kept, ignore_index=True) if kept else work.iloc[0:0]
    else:
        latest = (
            work.sort_values("_snap_ts")
            .drop_duplicates(subset=["home_team", "away_team"], keep="last")
        )

    res = results.copy()
    res["result_side"] = [
        _result_side(h, a) for h, a in zip(res["home_goals"], res["away_goals"], strict=True)
    ]
    res_lookup = {
        (str(r.home_team), str(r.away_team)): str(r.result_side)
        for r in res.itertuples(index=False)
    }

    rows = []
    for r in latest.itertuples(index=False):
        side = res_lookup.get((str(r.home_team), str(r.away_team)))
        if side is None:
            continue  # not played yet
        model_probs = {"home": r.model_home, "draw": r.model_draw, "away": r.model_away}
        market_probs = {"home": r.market_home, "draw": r.market_draw, "away": r.market_away}
        model_ll = sum(_log_loss(model_probs[s], s == side) for s in ("home", "draw", "away"))
        has_market = all(pd.notna(market_probs[s]) for s in ("home", "draw", "away"))
        market_ll = (
            sum(_log_loss(market_probs[s], s == side) for s in ("home", "draw", "away"))
            if has_market else float("nan")
        )
        rows.append({
            "match_date": r.match_date,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "result_side": side,
            "model_log_loss": round(model_ll, 4),
            "market_log_loss": None if math.isnan(market_ll) else round(market_ll, 4),
            "model_p_result": round(float(model_probs[side]), 4),
            "market_p_result": (None if not has_market else round(float(market_probs[side]), 4)),
            "winner": (
                "tie" if not has_market or abs(model_ll - market_ll) < 1e-9
                else ("model" if model_ll < market_ll else "market")
            ),
        })
    return pd.DataFrame(rows)


def summarize_track_record(scored: pd.DataFrame) -> dict:
    """Aggregate the head-to-head: who predicts better across resolved matches."""
    if scored.empty:
        return {"resolved_matches": 0}
    with_market = scored.dropna(subset=["market_log_loss"])
    summary = {
        "resolved_matches": int(len(scored)),
        "model_mean_log_loss": round(float(scored["model_log_loss"].mean()), 4),
    }
    if not with_market.empty:
        model_ll = float(with_market["model_log_loss"].mean())
        market_ll = float(with_market["market_log_loss"].mean())
        # Aggregate PRECISION index: a log-loss skill score. Because log loss punishes
        # confident misses much harder than it rewards small wins, one 10% blunder
        # outweighs many 1% edges — exactly the weighting the user wants. >0 = we beat
        # the market overall; e.g. +0.05 = 5% lower (better) average log loss than market.
        skill = (market_ll - model_ll) / market_ll if market_ll > 0 else 0.0
        # Brier on the realized outcome: (1 - p_assigned_to_actual)^2, also error-weighted
        # (quadratic), as a second aggregate accuracy measure.
        model_brier = float(((1.0 - with_market["model_p_result"]) ** 2).mean())
        market_brier = float(((1.0 - with_market["market_p_result"]) ** 2).mean())
        summary.update({
            "matches_vs_market": int(len(with_market)),
            "market_mean_log_loss": round(market_ll, 4),
            "model_mean_log_loss_vs_market_subset": round(model_ll, 4),
            "model_wins": int((with_market["winner"] == "model").sum()),
            "market_wins": int((with_market["winner"] == "market").sum()),
            # Aggregate precision (the headline single number the user asked for).
            "log_loss_skill_vs_market": round(skill, 4),
            "model_mean_brier_vs_market": round(model_brier, 4),
            "market_mean_brier_vs_market": round(market_brier, 4),
        })
    return summary
