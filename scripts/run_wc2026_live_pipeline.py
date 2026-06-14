from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.data import (
    aggregate_team_availability_features,
    build_live_squad_intelligence,
    build_live_squad_intelligence_summary,
    build_wc2026_provider_team_registry,
    build_wc2026_provider_team_registry_summary,
    get_api_football_api_key,
    get_api_football_api_key_header,
    get_api_football_host,
    get_sportmonks_api_token,
    load_expected_lineups_feed,
    load_flat_file_table,
    load_injuries_feed,
    load_provider_team_registry,
    select_provider_team_ids_for_fixture_window,
    load_world_cup_squads_from_wikipedia,
    save_api_football_injuries_by_team_ids_outputs,
    save_sportmonks_expected_lineups_outputs,
)
from wc2026_model.data.provider_team_registry import (
    enrich_registry_with_api_football_feed,
    enrich_registry_with_sportmonks_feed,
    normalize_provider_team_registry,
)
from wc2026_model.pipeline import save_world_cup_fixture_predictions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the World Cup 2026 live pipeline end-to-end: registry, provider feeds, "
            "live squad intelligence, and fixture predictions."
        )
    )
    parser.add_argument(
        "--groups-input",
        default="data/reference/wc2026_groups_actual.csv",
    )
    parser.add_argument(
        "--official-squads-input",
        default="data/interim/wc2026_squads_from_wikipedia.csv",
    )
    parser.add_argument(
        "--registry-input",
        default="data/reference/wc2026_provider_team_registry.csv",
    )
    parser.add_argument(
        "--refresh-registry",
        action="store_true",
        help="Rebuild the provider registry from groups/squads before running the rest of the pipeline.",
    )
    parser.add_argument(
        "--sportmonks-team-ids",
        default=None,
        help="Optional comma-separated Sportmonks team IDs used when the registry is still empty.",
    )
    parser.add_argument(
        "--api-football-team-ids",
        default=None,
        help="Optional comma-separated API-Football team IDs used when the registry is still empty.",
    )
    parser.add_argument(
        "--sportmonks-input",
        default=None,
        help="Optional existing Sportmonks JSON/CSV feed to reuse instead of downloading.",
    )
    parser.add_argument(
        "--api-football-input",
        default=None,
        help="Optional existing API-Football JSON/CSV feed to reuse instead of downloading.",
    )
    parser.add_argument(
        "--api-football-free-plan",
        action="store_true",
        help="Convenience mode for API-Football Free: skip premium lineup pulls and target only upcoming World Cup teams.",
    )
    parser.add_argument(
        "--skip-sportmonks-download",
        action="store_true",
    )
    parser.add_argument(
        "--skip-api-football-download",
        action="store_true",
    )
    parser.add_argument(
        "--sportmonks-raw-output",
        default="data/interim/sportmonks_expected_lineups_raw.json",
    )
    parser.add_argument(
        "--sportmonks-csv-output",
        default="data/interim/sportmonks_expected_lineups.csv",
    )
    parser.add_argument(
        "--sportmonks-per-page",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--sportmonks-pause-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--api-football-raw-output",
        default="data/interim/api_football_injuries_raw.json",
    )
    parser.add_argument(
        "--api-football-csv-output",
        default="data/interim/api_football_injuries.csv",
    )
    parser.add_argument(
        "--api-football-league",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--api-football-season",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--api-football-fixture",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--api-football-date",
        default=None,
    )
    parser.add_argument(
        "--api-football-timezone",
        default=None,
    )
    parser.add_argument(
        "--api-football-page",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--api-football-upcoming-window-days",
        type=int,
        default=None,
        help="Optional rolling fixture window used to limit API-Football requests to the next set of World Cup teams.",
    )
    parser.add_argument(
        "--api-football-max-teams",
        type=int,
        default=None,
        help="Optional cap on the number of targeted API-Football teams after fixture-window filtering.",
    )
    parser.add_argument(
        "--player-output",
        default="reports/wc2026_live_squad_intelligence.csv",
    )
    parser.add_argument(
        "--team-output",
        default="reports/wc2026_team_availability_features.csv",
    )
    parser.add_argument(
        "--live-summary-output",
        default="reports/wc2026_live_squad_intelligence_summary.json",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
    )
    parser.add_argument(
        "--model-input",
        default="models/baseline_dixon_coles_elo.pkl",
    )
    parser.add_argument(
        "--fixtures-input",
        default="data/raw/international_results.csv",
    )
    parser.add_argument(
        "--elo-ratings-input",
        default="reports/baseline_latest_elo_ratings.csv",
    )
    parser.add_argument(
        "--training-frame-input",
        default="reports/baseline_training_frame.csv",
    )
    parser.add_argument(
        "--squad-strength-input",
        default="reports/wc2026_squad_strength_ratings.csv",
    )
    parser.add_argument(
        "--squad-strength-column",
        default="squad_club_elo_rating",
    )
    parser.add_argument(
        "--secondary-squad-strength-column",
        default=None,
    )
    parser.add_argument(
        "--squad-elo-scale",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--secondary-squad-elo-scale",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--availability-starter-absence-elo",
        type=float,
        default=18.0,
    )
    parser.add_argument(
        "--availability-goalkeeper-absence-elo",
        type=float,
        default=24.0,
    )
    parser.add_argument(
        "--tournament",
        default="FIFA World Cup",
    )
    parser.add_argument(
        "--start-date",
        default="2026-06-12",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--prediction-output",
        default="reports/wc2026_fixture_predictions.csv",
    )
    parser.add_argument(
        "--prediction-summary-output",
        default="reports/wc2026_fixture_predictions_summary.json",
    )
    parser.add_argument(
        "--pipeline-summary-output",
        default="reports/wc2026_live_pipeline_summary.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.api_football_free_plan:
        args.skip_sportmonks_download = True
        if args.api_football_upcoming_window_days is None:
            args.api_football_upcoming_window_days = 2
        if args.api_football_max_teams is None:
            args.api_football_max_teams = 16

    summary_output = Path(args.pipeline_summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    groups = _load_optional_csv(Path(args.groups_input))
    official_squads = _load_official_squads(Path(args.official_squads_input))
    registry = _load_or_build_registry(
        registry_input=Path(args.registry_input),
        groups=groups,
        official_squads=official_squads,
        refresh_registry=args.refresh_registry,
    )

    step_summary: dict[str, object] = {}

    sportmonks_input = _resolve_existing_feed_path(
        args.sportmonks_input,
        args.sportmonks_raw_output,
        args.sportmonks_csv_output,
    )
    if sportmonks_input is not None:
        registry = _enrich_registry_from_sportmonks_input(registry, sportmonks_input)
        step_summary["sportmonks"] = _step_status(
            "used_existing",
            input_path=str(sportmonks_input),
        )
    elif args.skip_sportmonks_download:
        step_summary["sportmonks"] = _step_status("skipped", reason="download_disabled")
    else:
        sportmonks_team_ids = _resolve_provider_team_ids(
            csv_team_ids=args.sportmonks_team_ids,
            registry=registry,
            provider="sportmonks",
        )
        sportmonks_input, step_summary["sportmonks"] = _maybe_download_sportmonks(
            team_ids=sportmonks_team_ids,
            raw_output=Path(args.sportmonks_raw_output),
            csv_output=Path(args.sportmonks_csv_output),
            per_page=args.sportmonks_per_page,
            pause_seconds=args.sportmonks_pause_seconds,
        )
        if sportmonks_input is not None:
            registry = _enrich_registry_from_sportmonks_input(registry, sportmonks_input)

    api_football_input = _resolve_existing_feed_path(
        args.api_football_input,
        args.api_football_raw_output,
        args.api_football_csv_output,
    )
    if api_football_input is not None:
        registry = _enrich_registry_from_api_football_input(registry, api_football_input)
        step_summary["api_football"] = _step_status(
            "used_existing",
            input_path=str(api_football_input),
        )
    elif args.skip_api_football_download:
        step_summary["api_football"] = _step_status("skipped", reason="download_disabled")
    else:
        api_football_team_ids = _resolve_provider_team_ids(
            csv_team_ids=args.api_football_team_ids,
            registry=registry,
            provider="api_football",
            include_registry=not args.api_football_free_plan,
        )
        if args.api_football_upcoming_window_days is not None:
            targeted = select_provider_team_ids_for_fixture_window(
                registry,
                provider="api_football",
                fixtures_input=args.fixtures_input,
                tournament=args.tournament,
                start_date=args.start_date,
                window_days=args.api_football_upcoming_window_days,
                max_teams=args.api_football_max_teams,
            )
            targeted_ids = [int(team_id) for team_id in targeted["team_ids"]]
            api_football_team_ids = _merge_targeted_team_ids(
                base_team_ids=api_football_team_ids,
                targeted_team_ids=targeted_ids,
                free_plan=args.api_football_free_plan,
                explicit_csv_team_ids=args.api_football_team_ids,
            )
            step_summary["api_football_targeting"] = _step_status(
                "selected",
                **{key: value for key, value in targeted.items() if key != "window_frame"},
            )
        api_football_input, step_summary["api_football"] = _maybe_download_api_football(
            team_ids=api_football_team_ids,
            raw_output=Path(args.api_football_raw_output),
            csv_output=Path(args.api_football_csv_output),
            league=args.api_football_league,
            season=args.api_football_season,
            fixture=args.api_football_fixture,
            date=args.api_football_date,
            timezone=args.api_football_timezone,
            page=args.api_football_page,
        )
        if api_football_input is not None:
            registry = _enrich_registry_from_api_football_input(registry, api_football_input)

    registry = normalize_provider_team_registry(registry)
    registry_path = Path(args.registry_input)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(registry_path, index=False)
    registry_summary = build_wc2026_provider_team_registry_summary(registry)
    step_summary["registry"] = _step_status(
        "saved",
        output_path=str(registry_path),
        summary=registry_summary,
    )

    expected_lineups = (
        load_expected_lineups_feed(sportmonks_input, provider="auto")
        if sportmonks_input is not None
        else None
    )
    injuries = (
        load_injuries_feed(api_football_input, provider="auto")
        if api_football_input is not None
        else None
    )

    player_intelligence = build_live_squad_intelligence(
        official_squads,
        expected_lineups=expected_lineups,
        injuries=injuries,
        official_source_label="official_wikipedia_squads",
        expected_lineup_source_label="sportmonks_expected_lineups",
        injury_source_label="api_football_injuries",
    )
    team_features = aggregate_team_availability_features(player_intelligence)
    live_summary = build_live_squad_intelligence_summary(
        player_intelligence,
        expected_lineups=expected_lineups,
        injuries=injuries,
    )
    live_summary["official_squad_input_rows"] = int(len(official_squads))
    live_summary["team_feature_rows"] = int(len(team_features))

    player_output = Path(args.player_output)
    team_output = Path(args.team_output)
    live_summary_output = Path(args.live_summary_output)
    for path in (player_output, team_output, live_summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    player_intelligence.to_csv(player_output, index=False)
    team_features.to_csv(team_output, index=False)
    live_summary_output.write_text(json.dumps(live_summary, indent=2), encoding="utf-8")
    step_summary["live_squad_intelligence"] = _step_status(
        "saved",
        player_output=str(player_output),
        team_output=str(team_output),
        summary_output=str(live_summary_output),
        summary=live_summary,
    )

    if args.skip_predictions:
        step_summary["predictions"] = _step_status("skipped", reason="prediction_disabled")
    else:
        prediction_requirements = [
            Path(args.model_input),
            Path(args.fixtures_input),
            Path(args.elo_ratings_input),
            Path(args.training_frame_input),
        ]
        missing_prediction_inputs = [
            str(path) for path in prediction_requirements if not path.exists()
        ]
        if missing_prediction_inputs:
            step_summary["predictions"] = _step_status(
                "skipped",
                reason="missing_required_inputs",
                missing_inputs=missing_prediction_inputs,
            )
        else:
            squad_strength_input = (
                Path(args.squad_strength_input)
                if args.squad_strength_input and Path(args.squad_strength_input).exists()
                else None
            )
            _, prediction_summary = save_world_cup_fixture_predictions(
                model_input=args.model_input,
                fixtures_input=args.fixtures_input,
                elo_ratings_input=args.elo_ratings_input,
                training_frame_input=args.training_frame_input,
                output=args.prediction_output,
                summary_output=args.prediction_summary_output,
                squad_strength_input=squad_strength_input,
                squad_strength_column=args.squad_strength_column,
                secondary_squad_strength_column=args.secondary_squad_strength_column,
                squad_elo_scale=args.squad_elo_scale,
                secondary_squad_elo_scale=args.secondary_squad_elo_scale,
                availability_input=team_output,
                availability_starter_absence_elo=args.availability_starter_absence_elo,
                availability_goalkeeper_absence_elo=args.availability_goalkeeper_absence_elo,
                tournament=args.tournament,
                start_date=args.start_date,
                max_goals=args.max_goals,
            )
            step_summary["predictions"] = _step_status(
                "saved",
                output_path=args.prediction_output,
                summary_output=args.prediction_summary_output,
                summary=prediction_summary,
            )

    pipeline_summary = {
        "registry_input": args.registry_input,
        "sportmonks_input": str(sportmonks_input) if sportmonks_input is not None else None,
        "api_football_input": str(api_football_input) if api_football_input is not None else None,
        "steps": step_summary,
    }
    summary_output.write_text(json.dumps(pipeline_summary, indent=2), encoding="utf-8")

    print(json.dumps(pipeline_summary, indent=2))
    print(f"Saved live pipeline summary to {summary_output}")


def _load_or_build_registry(
    *,
    registry_input: Path,
    groups: pd.DataFrame | None,
    official_squads: pd.DataFrame,
    refresh_registry: bool,
) -> pd.DataFrame:
    if registry_input.exists() and not refresh_registry:
        return load_provider_team_registry(registry_input)
    return build_wc2026_provider_team_registry(
        groups=groups,
        squads=official_squads,
    )


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_official_squads(path: Path) -> pd.DataFrame:
    if path.exists():
        return load_flat_file_table(path)
    return load_world_cup_squads_from_wikipedia()


def _resolve_existing_feed_path(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _resolve_provider_team_ids(
    *,
    csv_team_ids: str | None,
    registry: pd.DataFrame,
    provider: str,
    include_registry: bool = True,
) -> list[int]:
    resolved: list[int] = []
    if csv_team_ids:
        resolved.extend(
            int(team_id.strip())
            for team_id in str(csv_team_ids).split(",")
            if team_id.strip() != ""
        )

    provider_key = provider.strip().lower()
    if provider_key == "sportmonks":
        column = "sportmonks_team_id"
    elif provider_key == "api_football":
        column = "api_football_team_id"
    else:
        raise ValueError(f"Unsupported provider '{provider}'.")

    if include_registry and column in registry.columns:
        resolved.extend(
            pd.to_numeric(registry[column], errors="coerce").dropna().astype(int).tolist()
        )
    return list(dict.fromkeys(resolved))


def _merge_targeted_team_ids(
    *,
    base_team_ids: list[int],
    targeted_team_ids: list[int],
    free_plan: bool,
    explicit_csv_team_ids: str | None,
) -> list[int]:
    if free_plan and not explicit_csv_team_ids:
        return list(dict.fromkeys(targeted_team_ids))
    return list(dict.fromkeys(targeted_team_ids + base_team_ids))


def _maybe_download_sportmonks(
    *,
    team_ids: list[int],
    raw_output: Path,
    csv_output: Path,
    per_page: int | None,
    pause_seconds: float,
) -> tuple[Path | None, dict[str, object]]:
    if not team_ids:
        return None, _step_status("skipped", reason="no_team_ids")
    try:
        api_token = get_sportmonks_api_token()
    except RuntimeError as exc:
        return None, _step_status("skipped", reason="missing_token", detail=str(exc))

    save_sportmonks_expected_lineups_outputs(
        raw_destination=raw_output,
        csv_destination=csv_output,
        team_ids=team_ids,
        api_token=api_token,
        per_page=per_page,
        request_pause_seconds=pause_seconds,
    )
    return raw_output, _step_status(
        "downloaded",
        team_ids=team_ids,
        raw_output=str(raw_output),
        csv_output=str(csv_output),
    )


def _maybe_download_api_football(
    *,
    team_ids: list[int],
    raw_output: Path,
    csv_output: Path,
    league: int | None,
    season: int | None,
    fixture: int | None,
    date: str | None,
    timezone: str | None,
    page: int | None,
) -> tuple[Path | None, dict[str, object]]:
    if not team_ids:
        return None, _step_status("skipped", reason="no_team_ids")
    try:
        api_key = get_api_football_api_key()
    except RuntimeError as exc:
        return None, _step_status("skipped", reason="missing_api_key", detail=str(exc))

    save_api_football_injuries_by_team_ids_outputs(
        raw_destination=raw_output,
        csv_destination=csv_output,
        team_ids=team_ids,
        api_key=api_key,
        league=league,
        season=season,
        fixture=fixture,
        date=date,
        timezone=timezone,
        page=page,
        api_key_header=get_api_football_api_key_header(),
        api_host=get_api_football_host(),
    )
    return raw_output, _step_status(
        "downloaded",
        team_ids=team_ids,
        raw_output=str(raw_output),
        csv_output=str(csv_output),
    )


def _enrich_registry_from_sportmonks_input(
    registry: pd.DataFrame,
    feed_input: Path,
) -> pd.DataFrame:
    expected_lineups = load_expected_lineups_feed(feed_input, provider="auto")
    return enrich_registry_with_sportmonks_feed(registry, expected_lineups)


def _enrich_registry_from_api_football_input(
    registry: pd.DataFrame,
    feed_input: Path,
) -> pd.DataFrame:
    injuries = load_injuries_feed(feed_input, provider="auto")
    return enrich_registry_with_api_football_feed(registry, injuries)


def _step_status(status: str, **details: object) -> dict[str, object]:
    return {"status": status, **details}


if __name__ == "__main__":
    main()
