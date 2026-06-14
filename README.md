# World Cup 2026 Prediction Model

Quant-oriented pipeline for pre-match World Cup forecasting, market comparison, and EV extraction.

## Project Goal

Build a production-style football model that:

1. predicts scorelines and match outcome probabilities for every World Cup match;
2. compares model probabilities with bookmaker prices and Polymarket prices;
3. tracks calibration, log loss, Brier score, ranked probability score, and EV;
4. flags potential value bets after removing bookmaker margin.

## Recommended Base Stack

We are starting from scratch, but not from zero knowledge.

- `penaltyblog` as the main modelling base for Poisson, bivariate Poisson, Dixon-Coles, Elo-style ratings, implied probabilities, and betting utilities.
- `soccerdata` as the ingestion layer for public football data such as Club Elo, FBref, Football-Data.co.uk, Understat, and more.
- `international_results` as a clean historical base for national-team match results.
- `StatsBomb open-data` plus `socceraction` as the optional richer feature layer when we want event-level features such as xT/VAEP.

## Current Data Base

The project now supports two recent-result sources on top of the modelling stack:

- `martj42/international_results` as the main historical/live base;
- the open `cup26matches` recent-results JSON as a supplemental recent-cycle feed for extra 2025-2026 coverage;
- optional `football-data.org` API integration for structured competition-level pulls when an API token is available.

For player-level World Cup intelligence, the repo now also supports a normalized live layer that combines:

- `official squads` as the high-trust roster base;
- `expected lineups` from external providers when available;
- `injury / availability reports` from external providers when available.

## Modelling Roadmap

### Phase 1

Strong baseline:

- dynamic Elo / Pi / strength ratings;
- Dixon-Coles or bivariate Poisson score model;
- pre-match features: neutral-site flag, rest days, confederation interactions, host effects, squad-strength proxies, recent form decay.

### Phase 2

Research-grade upgrade:

- hierarchical Bayesian or dynamic attack/defense model;
- market-aware features;
- ensemble of score model + classifier + rating model.

### Phase 3

Trading / market layer:

- bookmaker odds ingestion;
- vig removal;
- Polymarket market probabilities;
- CLV tracking and EV extraction rules;
- backtest with realistic timestamp discipline.

## Repository Layout

```text
.
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- docs/
|   `-- research_notes.md
|-- models/
|-- reports/
|-- src/
|   `-- wc2026_model/
|       `-- __init__.py
`-- pyproject.toml
```

## Immediate Next Build Steps

1. refresh the international-match data source beyond the current historical CSV cutoff;
2. add squad-strength / roster-value features and host-effect refinements;
3. extend the market layer with bookmaker and Polymarket ingestion;
4. calibrate the knockout extra-time / penalties module;
5. add tournament-level forecast reports and notebook-style diagnostics.

## Current Build Status

Implemented:

- international results loader and standardizer;
- tournament and team-name canonicalization across multiple recent sources;
- open recent-results loader for a public WC2026 dataset;
- `football-data.org` API client for World Cup / qualification pulls;
- augmented-results merge pipeline with filtering for non-national-team rows;
- rolling pre-match Elo feature generation;
- latest team-level Elo rating export for downstream simulation;
- Dixon-Coles baseline with Elo covariate and time-decay weights;
- betting-market probability utilities;
- scoring metrics and tests;
- expanding-window backtest harness;
- World Cup 2026 group-stage ranking, best-third-place ranking, and exact round-of-32 lookup;
- World Cup 2026 tournament simulator and CLI scripts to generate outright probabilities.
- live squad-intelligence normalization for official squads, expected lineups, and injury reports;
- team-level availability feature generation for starter completeness and availability risk.

## Quick Start

Download the international results dataset:

```bash
uv run python scripts/download_international_results.py
```

Download the open recent-results supplement:

```bash
uv run python scripts/download_cup26_open_results.py
```

Build an augmented historical + recent dataset:

```bash
uv run python scripts/build_augmented_results_dataset.py --auto-download-historical
```

Train the baseline model:

```bash
uv run python scripts/train_baseline.py --input data/interim/international_results_augmented.csv
```

Run an expanding-window backtest:

```bash
uv run python scripts/backtest_baseline.py --auto-download
```

Rebuild the official 2026 round-of-32 third-place lookup:

```bash
uv run python scripts/build_wc2026_round_of_32_lookup.py
```

Simulate the World Cup once you have a groups CSV with columns `group,team[,slot]`:

```bash
uv run python scripts/simulate_world_cup.py --groups-input data/reference/wc2026_groups_template.csv
```

Generate probabilities for upcoming scheduled World Cup fixtures from the live feed:

```bash
uv run python scripts/predict_world_cup_fixtures.py
```

Add the live availability overlay once you have built team-level availability features:

```bash
uv run python scripts/predict_world_cup_fixtures.py \
  --availability-input reports/wc2026_team_availability_features.csv
```

Build the live player-availability layer from official squads plus optional lineup / injury exports:

```bash
uv run python scripts/build_live_squad_intelligence.py \
  --expected-lineups-input data/interim/sportmonks_expected_lineups.json \
  --expected-lineups-provider sportmonks \
  --injuries-input data/interim/api_football_injuries.json \
  --injuries-provider api_football
```

The script also still accepts already-flattened CSV / JSON exports; with `--*-provider auto` it will detect nested
Sportmonks expected-lineup payloads and nested API-Football injury payloads automatically.

Build a reusable WC2026 provider team-ID registry from the real groups file:

```bash
uv run python scripts/build_wc2026_provider_team_registry.py
```

This creates `data/reference/wc2026_provider_team_registry.csv`, which can later be enriched from provider feeds and
reused by the Sportmonks downloader instead of passing team IDs manually.

Download Sportmonks expected lineups directly when you have a token and the provider team IDs:

```bash
$env:SPORTMONKS_API_TOKEN="your-token"
uv run python scripts/download_sportmonks_expected_lineups.py \
  --registry-input data/reference/wc2026_provider_team_registry.csv \
  --raw-output data/interim/sportmonks_expected_lineups_raw.json \
  --csv-output data/interim/sportmonks_expected_lineups.csv
```

Download API-Football injuries directly when you have a key:

```bash
$env:API_FOOTBALL_API_KEY="your-key"
uv run python scripts/download_api_football_injuries.py \
  --registry-input data/reference/wc2026_provider_team_registry.csv \
  --league 1 \
  --season 2026 \
  --raw-output data/interim/api_football_injuries_raw.json \
  --csv-output data/interim/api_football_injuries.csv
```

To stay inside the API-Football Free plan, target only the next World Cup match window instead of all 48 teams:

```bash
$env:API_FOOTBALL_API_KEY="your-key"
uv run python scripts/download_api_football_injuries.py \
  --registry-input data/reference/wc2026_provider_team_registry.csv \
  --free-plan
```

`--free-plan` automatically narrows the pull to the next two matchdays and caps the request set to a small team
window, which is much more realistic for daily refreshes on the free tier.

If your API-Football account uses a different header name, set `API_FOOTBALL_API_KEY_HEADER`.
The default is `x-apisports-key`; RapidAPI users can override it with `x-rapidapi-key`.

Populate the Sportmonks team-ID side of the WC2026 provider registry by searching candidate teams, ranking them, and
optionally applying only the high-confidence matches back into the registry:

```bash
$env:SPORTMONKS_API_TOKEN="your-token"
uv run python scripts/search_wc2026_provider_team_candidates.py \
  --provider sportmonks \
  --missing-only \
  --apply-output data/reference/wc2026_provider_team_registry.csv
```

The same command also works offline if you already have a cached candidates JSON / CSV:

```bash
uv run python scripts/search_wc2026_provider_team_candidates.py \
  --provider sportmonks \
  --missing-only \
  --candidates-input data/interim/sportmonks_team_search_candidates.csv
```

This writes both the full ranked candidate table and the one-row-per-team suggestion table, so ambiguous matches stay
reviewable instead of being silently forced into the registry.

You can populate the API-Football side of the same registry with the exact same workflow:

```bash
$env:API_FOOTBALL_API_KEY="your-key"
uv run python scripts/search_wc2026_provider_team_candidates.py \
  --provider api_football \
  --missing-only \
  --apply-output data/reference/wc2026_provider_team_registry.csv
```

That uses the `teams?search=` endpoint, ranks the returned national-team candidates, and only auto-applies the
high-confidence matches back into the registry.

Run the whole live pipeline in one shot while reusing the local squads / registry / model artifacts already in the repo:

```bash
uv run python scripts/run_wc2026_live_pipeline.py \
  --skip-sportmonks-download \
  --skip-api-football-download
```

Run the live pipeline in API-Football free-plan mode:

```bash
$env:API_FOOTBALL_API_KEY="your-key"
uv run python scripts/run_wc2026_live_pipeline.py \
  --api-football-free-plan
```

This mode skips premium lineup pulls and targets only the next short World Cup fixture window for injuries, which
keeps the request budget under control while we validate whether the live signal is worth paying up for.

Once your provider team IDs are populated in `data/reference/wc2026_provider_team_registry.csv`, the same script can
pull fresh expected lineups and injuries, rebuild team availability, and refresh fixture probabilities in one pass:

```bash
$env:SPORTMONKS_API_TOKEN="your-token"
$env:API_FOOTBALL_API_KEY="your-key"
uv run python scripts/run_wc2026_live_pipeline.py \
  --api-football-league 1 \
  --api-football-season 2026
```

The live pipeline automatically reuses existing provider exports from either the raw JSON files or the flattened CSVs
when they are already present, so we can run fast offline refreshes between live-data pulls.

Artifacts will be written to:

- `models/baseline_dixon_coles_elo.pkl`
- `reports/baseline_team_strengths.csv`
- `reports/baseline_training_frame.csv`
- `reports/baseline_latest_elo_ratings.csv`
- `reports/baseline_fit_summary.json`
- `reports/backtest_predictions.csv`
- `reports/backtest_summary.csv`
- `reports/backtest_aggregate.json`
- `data/raw/cup26_open_results.json`
- `data/interim/international_results_augmented.csv`
- `reports/augmented_results_summary.json`
- `data/reference/wc2026_round_of_32_lookup.csv`
- `data/reference/wc2026_groups_template.csv`
- `reports/wc2026_simulation_probabilities.csv`
- `reports/wc2026_simulation_summary.json`
- `reports/wc2026_fixture_predictions.csv`
- `reports/wc2026_fixture_predictions_summary.json`
- `reports/wc2026_live_squad_intelligence.csv`
- `reports/wc2026_team_availability_features.csv`
- `reports/wc2026_live_squad_intelligence_summary.json`
- `reports/wc2026_live_pipeline_summary.json`
- `reports/wc2026_provider_team_match_candidates.csv`
- `reports/wc2026_provider_team_match_suggestions.csv`
