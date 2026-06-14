from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.data import (
    apply_provider_team_match_suggestions,
    build_api_football_team_search_queries,
    build_provider_team_match_suggestions,
    build_team_search_queries,
    get_api_football_api_key,
    get_api_football_api_key_header,
    get_api_football_host,
    get_sportmonks_api_token,
    load_provider_team_registry,
    load_provider_team_search_candidates,
    save_api_football_team_search_candidates_json,
    save_sportmonks_team_search_candidates_json,
    score_provider_team_candidates,
    standardize_api_football_team_search_payload,
    standardize_sportmonks_team_search_payload,
)
from wc2026_model.data.international_results import canonicalize_team_name

_PROVIDER_ID_COLUMNS = {
    "sportmonks": "sportmonks_team_id",
    "api_football": "api_football_team_id",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search provider team candidates for the WC2026 registry, rank them, "
            "and optionally apply auto-selected matches back into the registry."
        )
    )
    parser.add_argument(
        "--registry-input",
        default="data/reference/wc2026_provider_team_registry.csv",
    )
    parser.add_argument(
        "--provider",
        default="sportmonks",
        choices=sorted(_PROVIDER_ID_COLUMNS),
    )
    parser.add_argument(
        "--candidates-input",
        default=None,
        help="Optional pre-existing raw JSON or flattened CSV candidates file to reuse.",
    )
    parser.add_argument(
        "--team-filter",
        default=None,
        help="Optional comma-separated subset of WC2026 teams to search.",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only search teams that are still missing the selected provider ID.",
    )
    parser.add_argument(
        "--raw-output",
        default=None,
    )
    parser.add_argument(
        "--candidates-output",
        default=None,
    )
    parser.add_argument(
        "--ranked-output",
        default="reports/wc2026_provider_team_match_candidates.csv",
    )
    parser.add_argument(
        "--suggestions-output",
        default="reports/wc2026_provider_team_match_suggestions.csv",
    )
    parser.add_argument(
        "--apply-output",
        default=None,
        help="Optional path to save an updated registry with auto-selected matches applied.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow auto-selected matches to replace existing provider IDs in the registry.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.75,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    provider = args.provider.strip().lower()
    raw_output = _resolve_candidate_output_path(args.raw_output, provider=provider, kind="raw")
    candidates_output = _resolve_candidate_output_path(
        args.candidates_output,
        provider=provider,
        kind="csv",
    )

    registry = load_provider_team_registry(args.registry_input)
    selected_teams = _select_teams(
        registry,
        provider=provider,
        team_filter=args.team_filter,
        missing_only=args.missing_only,
    )

    candidates, candidate_source = _load_or_fetch_candidates(
        provider=provider,
        selected_teams=selected_teams,
        candidates_input=args.candidates_input,
        raw_output=raw_output,
        candidates_output=candidates_output,
        per_page=args.per_page,
        pause_seconds=args.pause_seconds,
    )
    ranked_candidates = score_provider_team_candidates(candidates)
    suggestions = build_provider_team_match_suggestions(
        registry,
        ranked_candidates,
        provider=provider,
        min_score=args.min_score,
        min_margin=args.min_margin,
    )
    if selected_teams:
        suggestions = suggestions.loc[suggestions["team"].isin(selected_teams)].copy()
    else:
        suggestions = suggestions.iloc[0:0].copy()

    ranked_output = Path(args.ranked_output)
    suggestions_output = Path(args.suggestions_output)
    ranked_output.parent.mkdir(parents=True, exist_ok=True)
    suggestions_output.parent.mkdir(parents=True, exist_ok=True)
    ranked_candidates.to_csv(ranked_output, index=False)
    suggestions.to_csv(suggestions_output, index=False)

    apply_output = Path(args.apply_output) if args.apply_output else None
    applied_summary: dict[str, object] | None = None
    if apply_output is not None:
        apply_output.parent.mkdir(parents=True, exist_ok=True)
        updated_registry = apply_provider_team_match_suggestions(
            registry,
            suggestions,
            provider=provider,
            overwrite=args.overwrite_existing,
        )
        updated_registry.to_csv(apply_output, index=False)
        applied_summary = {
            "output_path": str(apply_output),
            "applied_team_count": int(suggestions["selected"].fillna(False).sum()),
        }

    summary = {
        "provider": provider,
        "candidate_source": candidate_source,
        "selected_team_count": len(selected_teams),
        "candidate_row_count": int(len(ranked_candidates)),
        "auto_selected_team_count": int(suggestions["selected"].fillna(False).sum()),
        "review_required_team_count": int(
            suggestions["selection_status"].fillna("").astype(str).str.startswith("review_required").sum()
        ),
        "no_candidate_team_count": int((suggestions["selection_status"] == "no_candidates").sum()),
        "ranked_output": str(ranked_output),
        "suggestions_output": str(suggestions_output),
        "apply": applied_summary,
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved ranked candidates to {ranked_output}")
    print(f"Saved match suggestions to {suggestions_output}")
    if apply_output is not None:
        print(f"Saved updated registry to {apply_output}")


def _select_teams(
    registry: pd.DataFrame,
    *,
    provider: str,
    team_filter: str | None,
    missing_only: bool,
) -> list[str]:
    frame = registry.copy()
    provider_column = _PROVIDER_ID_COLUMNS[provider]
    if missing_only:
        frame = frame.loc[frame[provider_column].isna()].copy()
    if team_filter:
        allowed_teams = {
            canonicalize_team_name(team)
            for team in str(team_filter).split(",")
            if team.strip() != ""
        }
        frame = frame.loc[frame["team"].isin(allowed_teams)].copy()
    return frame["team"].astype(str).drop_duplicates().tolist()


def _load_or_fetch_candidates(
    *,
    provider: str,
    selected_teams: list[str],
    candidates_input: str | None,
    raw_output: Path,
    candidates_output: Path,
    per_page: int,
    pause_seconds: float,
) -> tuple[pd.DataFrame, str]:
    if candidates_input:
        input_path = Path(candidates_input)
        if not input_path.exists():
            raise FileNotFoundError(f"Candidates input not found: {input_path}")
        frame = load_provider_team_search_candidates(candidates_input, provider=provider)
        if provider == "sportmonks" and candidates_output != Path(candidates_input):
            candidates_output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(candidates_output, index=False)
        return frame, str(candidates_input)

    if candidates_output.exists():
        return (
            load_provider_team_search_candidates(candidates_output, provider=provider),
            str(candidates_output),
        )
    if raw_output.exists():
        frame = load_provider_team_search_candidates(raw_output, provider=provider)
        candidates_output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(candidates_output, index=False)
        return frame, str(raw_output)

    if not selected_teams:
        return pd.DataFrame(), "no_selected_teams"

    searches = [
        {"target_team": team, "search_query": query}
        for team in selected_teams
        for query in (
            build_api_football_team_search_queries(team)
            if provider == "api_football"
            else build_team_search_queries(team)
        )
    ]
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    candidates_output.parent.mkdir(parents=True, exist_ok=True)
    if provider == "sportmonks":
        save_sportmonks_team_search_candidates_json(
            raw_output,
            searches=searches,
            api_token=get_sportmonks_api_token(),
            per_page=per_page,
            request_pause_seconds=pause_seconds,
        )
        frame = standardize_sportmonks_team_search_payload(
            json.loads(raw_output.read_text(encoding="utf-8"))
        )
    elif provider == "api_football":
        save_api_football_team_search_candidates_json(
            raw_output,
            searches=searches,
            api_key=get_api_football_api_key(),
            api_key_header=get_api_football_api_key_header(),
            api_host=get_api_football_host(),
            request_pause_seconds=pause_seconds,
        )
        frame = standardize_api_football_team_search_payload(
            json.loads(raw_output.read_text(encoding="utf-8"))
        )
    else:
        raise ValueError(f"Unsupported provider '{provider}'.")
    frame.to_csv(candidates_output, index=False)
    return frame, str(raw_output)


def _resolve_candidate_output_path(
    value: str | None,
    *,
    provider: str,
    kind: str,
) -> Path:
    if value:
        return Path(value)
    if kind == "raw":
        filename = f"{provider}_team_search_candidates_raw.json"
    else:
        filename = f"{provider}_team_search_candidates.csv"
    return Path("data/interim") / filename


if __name__ == "__main__":
    main()
