# WC2026 Model — Web Dashboard

A static, shareable site that turns the model's reports into something fun and
readable: match predictions, where the model beats the market, how reliable it is,
and a head-to-head / title-race playground.

No server, no build step, no framework — just HTML/CSS/vanilla JS fed by one JSON file.

## Build

```bash
# 1. Generate the data payload from the latest reports
python scripts/build_web_dashboard.py

# 2. Serve the web/ folder over http (NOT file://, fetch() needs http)
cd web
python -m http.server 8000
# open http://localhost:8000
```

`build_web_dashboard.py` reads (all optional — missing files degrade gracefully):

| Source report | Powers |
|---|---|
| `wc2026_fixture_predictions.csv` | 🏆 Predictions |
| `bookmaker_match_odds_*_comparison.csv` | 💰 Match edges |
| `polymarket_world_cup_winner_live_comparison.csv` | 💰 Title-winner edge |
| `benchmark_backtest_summary_xg_ablation.csv` | 📊 Reliability leaderboard |
| `wc2026_live_simulation_probabilities.csv` | 🎲 Title race + H2H |

Override any path with the matching `--flag` (see `--help`).

## One-command refresh (model + site)

```bash
# Pulls latest results (free GitHub-raw feed, no API key), rebuilds the model,
# re-conditions the tournament sim on matches already played, regenerates data.json.
python scripts/refresh_model_and_site.py
```

It's idempotent and cheap: if no new results arrived it skips the retrain and only
re-conditions the simulation. Run it after each matchday (or let CI do it, below).

## Auto-update + deploy (GitHub Pages, free, hands-off)

A workflow at `.github/workflows/refresh-and-deploy.yml` runs the refresh every
30 minutes, commits the updated `web/data.json`, and publishes `web/` to Pages.

One-time setup:

1. Create a GitHub repo and push this project:
   ```bash
   git init && git add . && git commit -m "WC2026 model + site"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. On GitHub: **Settings → Pages → Source: GitHub Actions**.
3. That's it. The site auto-refreshes during the tournament; the link under
   Settings → Pages is what you send to friends.

The results source is a free public JSON (GitHub-raw), so there are **no API-call
limits** — the 30-minute cadence costs nothing.

## Deploy elsewhere (alternative)

The `web/` folder is fully static, so Netlify / Vercel / Cloudflare Pages also work
(drag-and-drop the folder). Re-run the refresh and redeploy to update.

## Notes

- "Fair odds" = `1 / model_probability`. "Edge" = model probability minus the
  market's no-vig probability. "EV" = expected profit per unit staked if the model
  is right. All computed from files the model already produces.
- The reliability "rival models" are the project's own backtest baselines
  (Elo-only, recent form, Dixon-Coles, random) — the market is the rival in the
  Edge tab. (football-data.co.uk was evaluated as a third-party benchmark but it's
  club-only, so it's useless for an international tournament.)
