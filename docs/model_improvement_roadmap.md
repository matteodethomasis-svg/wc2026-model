# Model Improvement Roadmap

## Current Read

The project is no longer in the "find obvious bugs" phase.

We already know:

- The historical World Cup contamination bug in live simulation is fixed.
- Pure Dixon-Coles plus Elo is directionally good, but too conservative on elite favorites in some tournament contexts.
- A convex blend of Dixon-Coles and Elo benchmark improves out-of-sample log loss.
- For the current World Cup Group I example, the market is still more bullish on France than both the baseline model and the best backtested hybrid.

That means the next task is not another broad rebuild. The next task is controlled probability refinement.

## Best Next Step

The most sensible immediate path is:

1. Promote the `75% Dixon-Coles / 25% Elo` hybrid as the main candidate model.
2. Add post-hoc probability calibration on top of that hybrid.
3. Re-evaluate on rolling backtests.
4. Only then add richer structural features.

Why this order:

- It is the highest expected gain per unit of complexity.
- It preserves the current pipeline.
- It is fully measurable with the evaluation stack we already have.
- It helps with exactly the symptom we are seeing: favorite probabilities look too flat.

## Priority Order

### Phase 1: Lock the hybrid baseline

Goal:

- Treat the hybrid blend as the default challenger model, not as an experiment.

Actions:

- Re-run benchmark backtests including `dixon_coles_elo_blend`.
- Generate fresh fixture and tournament outputs for the hybrid once command execution is stable again.
- Compare baseline vs hybrid on:
  - overall log loss
  - elite-favorite matches
  - World Cup group-stage matches
  - current market-facing examples

Acceptance criteria:

- Hybrid remains better than pure Dixon-Coles on out-of-sample log loss.
- Hybrid closes at least part of the gap against live market probabilities without blowing up calibration.

### Phase 2: Calibrate the hybrid probabilities

Goal:

- Fine-tune `home/draw/away` probabilities without retraining the whole model.

Actions:

- Tune outcome-specific power calibration on the hybrid prediction set.
- Focus especially on whether the draw probability is slightly too high in strong-favorite matches.
- Save the best `gamma_home`, `gamma_draw`, `gamma_away` configuration.

Acceptance criteria:

- Calibration layer improves log loss again versus the raw hybrid.
- ECE does not worsen materially.
- Market-facing favorite examples become more realistic.

### Phase 3: Build a light meta-stacker

Goal:

- Learn how to combine model components instead of fixing the blend weight by hand.

Recommended stacker input set:

- Dixon-Coles `pred_home`, `pred_draw`, `pred_away`
- Elo benchmark `pred_home`, `pred_draw`, `pred_away`
- `elo_diff_pre`
- `abs_elo_diff_pre`
- `neutral`

Recommended first version:

- Multinomial logistic meta-model

Why:

- It is simple, robust, easy to backtest, and much cheaper than opening a completely new modeling family too early.

Acceptance criteria:

- Beats the calibrated hybrid on rolling backtest metrics.
- Stays stable across folds.

### Phase 4: Add new structural signals

Only do this after Phase 1-3 are measured.

Best candidates:

1. Squad-strength prior
   - market value
   - club minutes
   - top-player concentration

2. Tournament prior
   - recent major tournament performance
   - knockout experience proxy

3. Availability / shock features
   - injuries
   - suspensions
   - coach changes

4. Competition-context features
   - World Cup group vs qualifier vs friendly
   - confederation interaction

These are attractive, but they increase data maintenance cost. They should be added only after the cheaper calibration/stacking path is exhausted.

## What Not To Do Next

Avoid these as the immediate next move:

- manually boosting France or any single team
- manually inflating Elo differences at prediction time
- adding many new data sources before validating calibration and stacking
- rebuilding the core goal model from scratch

All of those can create movement, but not disciplined progress.

## Recommended Default Path

If we want the single most sensible plan, it is:

1. `dixon_coles_elo_blend` as candidate default
2. tune probability calibration on that hybrid
3. add a small stacker if calibration alone is not enough
4. then expand the feature set

## Concrete Next Deliverables

The next repo deliverables should be:

1. `benchmark_backtest.py` run with hybrid included in the official summary
2. calibration tuning report for the hybrid
3. fresh hybrid tournament simulation outputs
4. match-level bookmaker comparison table for Group I fixtures
5. decision memo:
   - baseline vs hybrid vs calibrated hybrid
   - which one becomes the main production candidate
