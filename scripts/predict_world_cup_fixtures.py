from __future__ import annotations

import argparse
import json

from wc2026_model.pipeline.live import (
    _availability_elo_adjustment,
    _coerce_bool,
    _coerce_float,
    _get_team_availability_record,
    _load_team_availability_lookup,
    _load_team_availability_lookup_from_frame,
    _top_fixture,
    predict_world_cup_fixtures,
    save_world_cup_fixture_predictions,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict upcoming FIFA World Cup fixtures using the fitted baseline model."
    )
    parser.add_argument(
        "--model-input",
        default="models/baseline_dixon_coles_elo.pkl",
        help="Path to the fitted model pickle.",
    )
    parser.add_argument(
        "--fixtures-input",
        default="data/raw/international_results.csv",
        help="Path to the raw results/fixtures CSV containing scheduled World Cup matches.",
    )
    parser.add_argument(
        "--elo-ratings-input",
        default="reports/baseline_latest_elo_ratings.csv",
        help="Path to the latest Elo ratings CSV.",
    )
    parser.add_argument(
        "--training-frame-input",
        default="reports/baseline_training_frame.csv",
        help="Path to the training frame used to fit the optional Elo benchmark blend.",
    )
    parser.add_argument(
        "--elo-blend-alpha",
        type=float,
        default=1.0,
        help="Weight on the baseline Dixon-Coles probabilities. Set below 1.0 to blend with Elo multinomial probabilities.",
    )
    parser.add_argument(
        "--calibration-gamma-home",
        type=float,
        default=1.0,
        help="Outcome-specific power calibration for home-win probability.",
    )
    parser.add_argument(
        "--calibration-gamma-draw",
        type=float,
        default=1.0,
        help="Outcome-specific power calibration for draw probability.",
    )
    parser.add_argument(
        "--calibration-gamma-away",
        type=float,
        default=1.0,
        help="Outcome-specific power calibration for away-win probability.",
    )
    parser.add_argument(
        "--squad-strength-input",
        default=None,
        help="Optional CSV containing team-level squad strength ratings.",
    )
    parser.add_argument(
        "--squad-strength-column",
        default="squad_club_elo_rating",
        help="Column in --squad-strength-input used as the squad rating.",
    )
    parser.add_argument(
        "--secondary-squad-strength-column",
        default=None,
        help="Optional second squad-strength column used as an additional adjustment.",
    )
    parser.add_argument(
        "--squad-elo-scale",
        type=float,
        default=0.0,
        help="Adds scale * squad_strength_diff to elo_diff_pre before prediction.",
    )
    parser.add_argument(
        "--secondary-squad-elo-scale",
        type=float,
        default=0.0,
        help="Adds scale * secondary_squad_strength_diff to elo_diff_pre before prediction.",
    )
    parser.add_argument(
        "--availability-input",
        default=None,
        help=(
            "Optional CSV containing team-level live availability features, typically built by "
            "scripts/build_live_squad_intelligence.py."
        ),
    )
    parser.add_argument(
        "--availability-starter-absence-elo",
        type=float,
        default=18.0,
        help="Penalty applied per missing expected-starter equivalent (unavailable=1, doubtful=0.5).",
    )
    parser.add_argument(
        "--availability-goalkeeper-absence-elo",
        type=float,
        default=24.0,
        help="Additional penalty when the expected starting goalkeeper is unavailable.",
    )
    parser.add_argument(
        "--tournament",
        default="FIFA World Cup",
        help="Tournament filter for scheduled fixtures.",
    )
    parser.add_argument(
        "--start-date",
        default="2026-06-12",
        help="Only include fixtures on or after this date.",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=10,
        help="Maximum goals used in probability integration.",
    )
    parser.add_argument(
        "--output",
        default="reports/wc2026_fixture_predictions.csv",
        help="Path to save the fixture-level probability table.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/wc2026_fixture_predictions_summary.json",
        help="Path to save a compact prediction summary.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    _, summary = save_world_cup_fixture_predictions(
        model_input=args.model_input,
        fixtures_input=args.fixtures_input,
        elo_ratings_input=args.elo_ratings_input,
        training_frame_input=args.training_frame_input,
        output=args.output,
        summary_output=args.summary_output,
        elo_blend_alpha=args.elo_blend_alpha,
        calibration_gamma_home=args.calibration_gamma_home,
        calibration_gamma_draw=args.calibration_gamma_draw,
        calibration_gamma_away=args.calibration_gamma_away,
        squad_strength_input=args.squad_strength_input,
        squad_strength_column=args.squad_strength_column,
        secondary_squad_strength_column=args.secondary_squad_strength_column,
        squad_elo_scale=args.squad_elo_scale,
        secondary_squad_elo_scale=args.secondary_squad_elo_scale,
        availability_input=args.availability_input,
        availability_starter_absence_elo=args.availability_starter_absence_elo,
        availability_goalkeeper_absence_elo=args.availability_goalkeeper_absence_elo,
        tournament=args.tournament,
        start_date=args.start_date,
        max_goals=args.max_goals,
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved fixture predictions to {args.output}")
    print(f"Saved prediction summary to {args.summary_output}")


if __name__ == "__main__":
    main()
