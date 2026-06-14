"""One-shot refresh: pull latest results, rebuild the model, re-condition the
tournament on matches already played, and regenerate the web dashboard.

Designed to be idempotent and cheap enough to run after every match. The results
source is a free public GitHub-raw JSON (no API key, no rate limit), so the only
external call is a single file download. Heavy steps (retrain, sim) are skipped when
no new results arrived, unless --force is passed.

Pipeline:
  1. download latest results (free cup26_open feed)
  2. rebuild the deduped augmented dataset (fixes cross-source duplicate matches)
  3. if new matches vs the previous dataset (or --force): retrain Elo + Dixon-Coles
  4. rebuild player-Elo squad strength
  5. simulate WC2026, CONDITIONED on matches already played (as-of today)
  6. regenerate fixture predictions + Polymarket comparison
  7. rebuild web/data.json

Scales/recipe are the validated defaults: squad XI 0.52, GK 0.276, blend alpha 0.75,
calibration gammas 1.05 / 1.10, friendly Elo K=10.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable]

AUGMENTED = "data/interim/international_results_augmented.csv"
SQUAD_STRENGTH = "reports/wc2026_squad_strength_player_elo_ratings.csv"
SIM_OUT = "reports/wc2026_simulation_expected_xi_plus_goalkeeper_probabilities.csv"
SIM_SUMMARY = "reports/wc2026_simulation_expected_xi_plus_goalkeeper_summary.json"
FIXTURES_OUT = "reports/wc2026_fixture_predictions_expected_xi_plus_goalkeeper_recipe_2026-06-13.csv"
POLY_CMP = "reports/polymarket_world_cup_winner_live_comparison.csv"
POLY_MARKET = "data/interim/polymarket_world_cup_winner.csv"
POLY_SUMMARY = "reports/polymarket_world_cup_winner_live_comparison_summary.json"

# The tuned recipe is the edge: these scales/weights are the distilled result of all
# the backtesting. We keep them OUT of the public repo. The real values live in a
# local, gitignored `model_config.json` (or the MODEL_CONFIG_JSON env var, injected as
# a GitHub Actions secret for our own deploy). Without them, the model falls back to a
# de-tuned baseline — a working shell that does NOT reproduce our calibrated numbers.
_DETUNED_FALLBACK = {
    "squad_scale": "0.0",      # no per-player squad layer
    "gk_scale": "0.0",         # no goalkeeper layer
    "blend_alpha": "1.0",      # pure Dixon-Coles, no Elo blend
    "gamma_home": "1.0",       # no calibration
    "gamma_away": "1.0",
}


def _load_recipe() -> dict[str, str]:
    raw = os.environ.get("MODEL_CONFIG_JSON")
    if raw:
        try:
            return {**_DETUNED_FALLBACK, **json.loads(raw)}
        except Exception:
            pass
    cfg_path = ROOT / "model_config.json"
    if cfg_path.exists():
        try:
            return {**_DETUNED_FALLBACK, **json.loads(cfg_path.read_text(encoding="utf-8"))}
        except Exception:
            pass
    print("  [recipe] No model_config.json / MODEL_CONFIG_JSON found — using DE-TUNED "
          "baseline (public fallback, not the calibrated model).")
    return dict(_DETUNED_FALLBACK)


_RECIPE = _load_recipe()
SQUAD_SCALE = _RECIPE["squad_scale"]
GK_SCALE = _RECIPE["gk_scale"]
BLEND_ALPHA = _RECIPE["blend_alpha"]
GAMMA_HOME = _RECIPE["gamma_home"]
GAMMA_AWAY = _RECIPE["gamma_away"]


def run(cmd: list[str], desc: str) -> None:
    print(f"\n>>> {desc}")
    result = subprocess.run(PY + cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {desc}")


def _row_count(path: Path) -> int:
    if not path.exists():
        return -1
    try:
        return len(pd.read_csv(path))
    except Exception:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-date", default=date.today().isoformat(),
                        help="Cutoff for conditioning the sim on played matches (default: today).")
    parser.add_argument("--simulations", default="2000")
    parser.add_argument("--force", action="store_true",
                        help="Retrain/re-simulate even if no new results were found.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Use the results already on disk (offline / testing).")
    parser.add_argument("--skip-market-fetch", action="store_true",
                        help="Don't re-fetch live Polymarket odds (use the snapshot on disk).")
    args = parser.parse_args()

    before = _row_count(ROOT / AUGMENTED)

    # 1-2. Fetch latest results and rebuild the deduped augmented dataset.
    if not args.skip_download:
        run(["scripts/build_augmented_results_dataset.py"], "Fetch + merge + dedupe latest results")

    after = _row_count(ROOT / AUGMENTED)
    new_matches = after - before if before >= 0 else after
    print(f"\nDataset rows: {before} -> {after}  (new: {new_matches})")

    if new_matches <= 0 and not args.force:
        print("No new matches. Re-conditioning the sim only (cheap), skipping retrain.")
    else:
        # 3. Retrain Elo + Dixon-Coles on the refreshed data.
        run(["scripts/train_baseline.py", "--input", AUGMENTED],
            "Retrain Elo + Dixon-Coles")
        # 4. Rebuild player-Elo squad strength.
        run(["scripts/build_world_cup_player_elo_strength.py"],
            "Rebuild player-Elo squad strength")

    # 5. Simulate, CONDITIONED on matches already played up to as-of-date.
    run([
        "scripts/simulate_world_cup.py",
        "--groups-input", "data/reference/wc2026_groups_actual.csv",
        "--squad-strength-input", SQUAD_STRENGTH,
        "--squad-strength-column", "expected_xi_player_elo_rating", "--squad-elo-scale", SQUAD_SCALE,
        "--secondary-squad-strength-column", "expected_xi_goalkeeper_player_elo_rating",
        "--secondary-squad-elo-scale", GK_SCALE,
        "--elo-blend-alpha", BLEND_ALPHA,
        "--calibration-gamma-home", GAMMA_HOME, "--calibration-gamma-away", GAMMA_AWAY,
        "--simulations", args.simulations, "--random-state", "2026",
        "--results-input", AUGMENTED, "--as-of-date", args.as_of_date,
        "--output", SIM_OUT, "--summary-output", SIM_SUMMARY,
    ], f"Simulate WC2026 (conditioned on played matches as-of {args.as_of_date})")

    # 6. Fixture predictions + market comparison.
    run([
        "scripts/predict_world_cup_fixtures.py",
        "--squad-strength-input", SQUAD_STRENGTH,
        "--squad-strength-column", "expected_xi_player_elo_rating", "--squad-elo-scale", SQUAD_SCALE,
        "--secondary-squad-strength-column", "expected_xi_goalkeeper_player_elo_rating",
        "--secondary-squad-elo-scale", GK_SCALE,
        "--elo-blend-alpha", BLEND_ALPHA,
        "--calibration-gamma-home", GAMMA_HOME, "--calibration-gamma-away", GAMMA_AWAY,
        "--output", FIXTURES_OUT,
    ], "Regenerate fixture predictions")

    # Refresh live Polymarket winner odds (free Gamma API) so the edge/track record
    # compares against current prices, not a stale snapshot. Non-fatal on hiccup.
    if not args.skip_market_fetch:
        try:
            run(["scripts/fetch_polymarket_world_cup_winner.py"], "Fetch live Polymarket odds")
        except SystemExit:
            print("  (Polymarket fetch failed; keeping the existing odds snapshot.)")

    if (ROOT / POLY_MARKET).exists():
        run([
            "scripts/compare_polymarket_world_cup_winner.py",
            "--model-probabilities-input", SIM_OUT,
            "--market-probabilities-input", POLY_MARKET,
            "--output", POLY_CMP, "--summary-output", POLY_SUMMARY,
        ], "Refresh Polymarket comparison")

    # 7. Update the model-vs-market track record (append snapshot + score played).
    run(["scripts/update_prediction_ledger.py"], "Update model-vs-market track record")

    # 8. Rebuild the web dashboard payload.
    run(["scripts/build_web_dashboard.py"], "Rebuild web/data.json")

    print("\n[OK] Refresh complete. web/data.json is up to date.")


if __name__ == "__main__":
    main()
