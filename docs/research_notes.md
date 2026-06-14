# Research Notes

## Best Starting Points Found

### 1. penaltyblog

Best all-round Python base for this project.

Why it matters:

- Poisson, bivariate Poisson, Dixon-Coles, and Bayesian match models;
- rating systems including Elo-style methods;
- bookmaker implied-probability utilities;
- examples for match prediction, ratings, xT, and betting workflows.

### 2. soccerdata

Best ingestion backbone for a Python-first workflow.

Why it matters:

- pulls public football data into Pandas DataFrames;
- supports Club Elo, FBref, Football-Data.co.uk, Understat, WhoScored, Sofascore, and more;
- useful for historical results, team strength proxies, and historical odds.

### 3. international_results

Very strong seed dataset for national teams.

Why it matters:

- long history of international matches;
- clean CSV format;
- good fit for World Cup-specific modelling.

### 4. StatsBomb open-data + socceraction

Optional feature-engineering layer.

Why it matters:

- event and lineup data;
- xT / VAEP style features for richer team profiling;
- useful if we want to move beyond pure score models.

## Working Recommendation

Build the first version in Python around:

1. `soccerdata` for ingestion;
2. `international_results` for national-team history;
3. `penaltyblog` for baseline models and odds utilities;
4. our own evaluation and market-comparison layer.

Then upgrade toward a dynamic Bayesian score model inspired by the more recent football-prediction literature.
