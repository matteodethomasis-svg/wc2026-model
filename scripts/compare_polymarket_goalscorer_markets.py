"""Compare our individual-goals model vs the Polymarket goalscorer markets
(Golden Boot winner + Player-to-score), per player, ante-post only.

Join is by player NAME (token overlap, surname fallback) since Polymarket has no
stable player id. Honest market-comparison rules from memory:
  - Golden Boot: nobody has won it yet -> fully ante-post, compare all live rows.
  - Player to score: some markets have already RESOLVED to 1.0 (the player scored)
    or 0.0 (eliminated). Those are NOT ante-post; we keep only live prices
    (0 < p < 1) and flag the rest as resolved/non-comparable, so we never grade the
    pre-tournament model against in-play information.

Outputs reports/polymarket_goalscorer_{golden_boot,player_to_score}_comparison.csv.
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd


def _norm(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(t)) if not unicodedata.combining(c)).lower().strip()


def _tokens(n: str) -> set[str]:
    return {x for x in _norm(n).replace(".", " ").split() if len(x) > 1}


def _surname(n: str) -> str:
    p = _norm(n).replace(".", " ").split()
    return p[-1] if p else ""


def _match_player(market_name: str, model_index: list[tuple[set, str, str, float]]):
    """Return (model_player, team, model_prob) for the best name match, or None."""
    tk = _tokens(market_name)
    best, best_ov = None, 0
    for mtok, mname, mteam, mprob in model_index:
        ov = len(tk & mtok)
        if ov >= 2 and ov > best_ov:
            best_ov, best = ov, (mname, mteam, mprob)
    if best is not None:
        return best
    sur = _surname(market_name)
    for mtok, mname, mteam, mprob in model_index:
        if _surname(mname) == sur:
            return (mname, mteam, mprob)
    return None


def _compare(market: pd.DataFrame, model: pd.DataFrame, *, model_col: str,
             antepost_only: bool) -> pd.DataFrame:
    model_index = [
        (_tokens(r.player), str(r.player), str(r.team), float(getattr(r, model_col)))
        for r in model.itertuples(index=False)
    ]
    rows: list[dict[str, object]] = []
    for mr in market.itertuples(index=False):
        market_p = float(mr.market_probability)
        resolved = market_p <= 0.0 or market_p >= 1.0
        if antepost_only and resolved:
            continue  # never compare against a settled / in-play price
        hit = _match_player(str(mr.player), model_index)
        if hit is None:
            continue
        model_player, team, model_p = hit
        rows.append({
            "player": str(mr.player),
            "model_player": model_player,
            "team": team,
            "model_probability": round(model_p, 4),
            "market_probability": round(market_p, 4),
            "edge_vs_market": round(model_p - market_p, 4),
            "model_fair_odds": round(1.0 / model_p, 2) if model_p > 0 else None,
            "market_fair_odds": round(1.0 / market_p, 2) if market_p > 0 else None,
            "volume": float(getattr(mr, "volume", 0.0) or 0.0),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["model_probability", "edge_vs_market"],
                           ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input",
                        default="reports/wc2026_goalscorer_model_predictions.csv")
    parser.add_argument("--golden-boot-market",
                        default="data/interim/polymarket_golden_boot.csv")
    parser.add_argument("--player-to-score-market",
                        default="data/interim/polymarket_player_to_score.csv")
    parser.add_argument("--golden-boot-output",
                        default="reports/polymarket_goalscorer_golden_boot_comparison.csv")
    parser.add_argument("--player-to-score-output",
                        default="reports/polymarket_goalscorer_player_to_score_comparison.csv")
    args = parser.parse_args()

    model = pd.read_csv(args.model_input)

    if Path(args.golden_boot_market).exists():
        gb = pd.read_csv(args.golden_boot_market)
        gb_cmp = _compare(gb, model, model_col="golden_boot_probability", antepost_only=True)
        gb_cmp.to_csv(args.golden_boot_output, index=False)
        print(f"Golden Boot: {len(gb_cmp)} comparable players -> {args.golden_boot_output}")
        if not gb_cmp.empty:
            print(gb_cmp.head(8).to_string(index=False))

    if Path(args.player_to_score_market).exists():
        pts = pd.read_csv(args.player_to_score_market)
        pts_cmp = _compare(pts, model, model_col="anytime_score_probability", antepost_only=True)
        pts_cmp.to_csv(args.player_to_score_output, index=False)
        print(f"\nPlayer to score (live prices only): {len(pts_cmp)} comparable "
              f"-> {args.player_to_score_output}")
        if not pts_cmp.empty:
            print(pts_cmp.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
