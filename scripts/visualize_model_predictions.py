from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pandas as pd

from wc2026_model.pipeline.live import predict_world_cup_fixtures


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a simple HTML report for World Cup prediction probabilities."
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
        help="Path to the training frame used for the optional Elo benchmark blend.",
    )
    parser.add_argument(
        "--elo-blend-alpha",
        type=float,
        default=1.0,
        help="Weight on the baseline model before blending with Elo benchmark probabilities.",
    )
    parser.add_argument(
        "--calibration-gamma-home",
        type=float,
        default=1.0,
        help="Power calibration for home-win probability.",
    )
    parser.add_argument(
        "--calibration-gamma-draw",
        type=float,
        default=1.0,
        help="Power calibration for draw probability.",
    )
    parser.add_argument(
        "--calibration-gamma-away",
        type=float,
        default=1.0,
        help="Power calibration for away-win probability.",
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
        help="Scale applied to squad strength difference before prediction.",
    )
    parser.add_argument(
        "--secondary-squad-elo-scale",
        type=float,
        default=0.0,
        help="Scale applied to the secondary squad strength difference before prediction.",
    )
    parser.add_argument(
        "--availability-input",
        default=None,
        help="Optional CSV with live team availability features.",
    )
    parser.add_argument(
        "--availability-starter-absence-elo",
        type=float,
        default=18.0,
        help="Elo penalty for each unavailable expected starter.",
    )
    parser.add_argument(
        "--availability-goalkeeper-absence-elo",
        type=float,
        default=24.0,
        help="Elo penalty for a missing goalkeeper starter.",
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
        "--team",
        action="append",
        default=[],
        help="Optional team name to filter the report to a specific team. Repeat for multiple teams.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many top predicted fixtures to show in the report.",
    )
    parser.add_argument(
        "--output",
        default="reports/wc2026_model_visualization.html",
        help="Output path for the generated HTML report.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML report in the default browser.",
    )
    return parser


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fair_odds(probability: float) -> str:
    if probability <= 0.0:
        return "N/A"
    return f"{1.0 / probability:.2f}"


def _build_report(predictions: pd.DataFrame, summary: dict[str, object], config: dict[str, object], top_n: int) -> str:
    if predictions.empty:
        return "<html><body><h1>Nessuna previsione disponibile</h1></body></html>"

    probabilities = predictions.loc[:, ["home_win_probability", "draw_probability", "away_win_probability"]].to_numpy()
    sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
    predictions = predictions.copy()
    predictions["favorite_probability"] = predictions[["home_win_probability", "draw_probability", "away_win_probability"]].max(axis=1)
    predictions["favorite_outcome"] = predictions[["home_win_probability", "draw_probability", "away_win_probability"]].idxmax(axis=1).replace(
        {
            "home_win_probability": "Home win",
            "draw_probability": "Draw",
            "away_win_probability": "Away win",
        }
    )
    predictions["confidence"] = sorted_probs[:, 0] - sorted_probs[:, 1]
    predictions["max_probability"] = sorted_probs[:, 0]
    predictions["probability_spread"] = sorted_probs[:, 0] - sorted_probs[:, 2]

    top_matches = predictions.sort_values("favorite_probability", ascending=False).head(top_n)
    total_fixtures = len(predictions)
    average_favorite = float(predictions["favorite_probability"].mean())
    strong_favorites = int((predictions["favorite_probability"] >= 0.60).sum())
    draw_favorites = int((predictions["favorite_outcome"] == "Draw").sum())
    top_home = predictions.sort_values("home_win_probability", ascending=False).head(1).iloc[0]
    top_away = predictions.sort_values("away_win_probability", ascending=False).head(1).iloc[0]
    top_draw = predictions.sort_values("draw_probability", ascending=False).head(1).iloc[0]

    header = f"<h1>WC2026 Model Preview</h1>\n"
    header += f"<p>Generato con modello <strong>{Path(config['model_input']).name}</strong></p>\n"
    header += "<div class='config'>\n"
    header += "<h2>Parametri di previsione</h2>\n"
    header += "<pre>"
    header += json.dumps(config, indent=2, ensure_ascii=False)
    header += "</pre>\n"
    header += "</div>\n"

    cards = f"<div class='cards'>\n"
    cards += _render_card("Fixture totali", f"{total_fixtures}")
    cards += _render_card("Favorite medie", _format_pct(average_favorite))
    cards += _render_card("Favorite ≥ 60%", str(strong_favorites))
    cards += _render_card("Pareggi preferiti", str(draw_favorites))
    cards += _render_card("Match più probabile (Home)", f"{top_home.home_team} vs {top_home.away_team} – {_format_pct(top_home.home_win_probability)}")
    cards += _render_card("Match più probabile (Away)", f"{top_away.home_team} vs {top_away.away_team} – {_format_pct(top_away.away_win_probability)}")
    cards += _render_card("Match più probabile (Draw)", f"{top_draw.home_team} vs {top_draw.away_team} – {_format_pct(top_draw.draw_probability)}")
    cards += "</div>\n"

    table_rows = []
    for _, row in top_matches.iterrows():
        table_rows.append(
            f"<tr>"
            f"<td>{row.match_date}</td>"
            f"<td>{row.home_team}</td>"
            f"<td>{row.away_team}</td>"
            f"<td>{_format_pct(row.home_expected_goals)}</td>"
            f"<td>{_format_pct(row.away_expected_goals)}</td>"
            f"<td>{row.favorite_outcome}</td>"
            f"<td>{_format_pct(row.favorite_probability)}</td>"
            f"<td>{_format_pct(row.confidence)}</td>"
            f"<td>{_format_pct(row.home_win_probability)}</td>"
            f"<td>{_format_pct(row.draw_probability)}</td>"
            f"<td>{_format_pct(row.away_win_probability)}</td>"
            f"<td>{_fair_odds(row.home_win_probability)}</td>"
            f"<td>{_fair_odds(row.draw_probability)}</td>"
            f"<td>{_fair_odds(row.away_win_probability)}</td>"
            f"</tr>"
        )

    table_html = (
        "<table>"
        "<thead>"
        "<tr>"
        "<th>Date</th>"
        "<th>Home</th>"
        "<th>Away</th>"
        "<th>Home xG</th>"
        "<th>Away xG</th>"
        "<th>Favorite</th>"
        "<th>Favorite %</th>"
        "<th>Confidence</th>"
        "<th>Home %</th>"
        "<th>Draw %</th>"
        "<th>Away %</th>"
        "<th>Home fair</th>"
        "<th>Draw fair</th>"
        "<th>Away fair</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "\n".join(table_rows)
        + "</tbody>"
        "</table>\n"
    )

    html = f"""
<html>
<head>
  <meta charset="utf-8">
  <title>WC2026 Model Visualization</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111; }}
    h1 {{ margin-bottom: 8px; }}
    .config pre {{ background: #f5f5f5; padding: 12px; border-radius: 8px; overflow-x: auto; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 16px 0; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    .card strong {{ display: block; margin-bottom: 8px; color: #333; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th, td {{ padding: 8px 10px; border: 1px solid #ddd; text-align: left; font-size: 0.95rem; }}
    th {{ background: #f0f0f0; }}
    tbody tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
{header}
{cards}
<h2>Top {top_n} predicted fixtures</h2>
{table_html}
</body>
</html>
"""
    return html


def _render_card(title: str, value: str) -> str:
    return f"<div class='card'><strong>{title}</strong><div>{value}</div></div>\n"


def _normalize_team_filters(team_filters: list[str]) -> list[str]:
    return [team.strip().lower() for team in team_filters if team.strip()]


def _filter_by_teams(predictions: pd.DataFrame, team_filters: list[str]) -> pd.DataFrame:
    if not team_filters:
        return predictions
    filters = _normalize_team_filters(team_filters)
    mask = predictions["home_team"].str.lower().isin(filters) | predictions["away_team"].str.lower().isin(filters)
    return predictions.loc[mask].copy()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    predictions, summary = predict_world_cup_fixtures(
        model_input=args.model_input,
        fixtures_input=args.fixtures_input,
        elo_ratings_input=args.elo_ratings_input,
        training_frame_input=args.training_frame_input,
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

    predictions = _filter_by_teams(predictions, args.team)
    report_config = {
        "model_input": args.model_input,
        "fixtures_input": args.fixtures_input,
        "elo_ratings_input": args.elo_ratings_input,
        "training_frame_input": args.training_frame_input,
        "elo_blend_alpha": args.elo_blend_alpha,
        "calibration_gamma_home": args.calibration_gamma_home,
        "calibration_gamma_draw": args.calibration_gamma_draw,
        "calibration_gamma_away": args.calibration_gamma_away,
        "squad_strength_input": args.squad_strength_input,
        "squad_strength_column": args.squad_strength_column,
        "secondary_squad_strength_column": args.secondary_squad_strength_column,
        "squad_elo_scale": args.squad_elo_scale,
        "secondary_squad_elo_scale": args.secondary_squad_elo_scale,
        "availability_input": args.availability_input,
        "availability_starter_absence_elo": args.availability_starter_absence_elo,
        "availability_goalkeeper_absence_elo": args.availability_goalkeeper_absence_elo,
        "tournament": args.tournament,
        "start_date": args.start_date,
        "max_goals": args.max_goals,
        "team_filter": args.team,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _build_report(predictions, summary, report_config, args.top)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report generato: {output_path}")
    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
