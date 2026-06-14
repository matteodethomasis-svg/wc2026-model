# External WC2026 Prediction Models

Date checked: 2026-06-12

## Bottom line

Yes, public World Cup 2026 prediction models do exist.

However, the useful public artifacts are mostly:

- explainers of model methodology;
- interactive simulators;
- media or research predictions;
- not robust open-source codebases that obviously save us large implementation time.

## Best external models found

### 1. EL PAIS statistical model

Date: 2026-06-11

Most useful external model found so far.

What it uses:

- current Elo;
- historical Elo / pedigree over the last decade;
- squad market value from Transfermarkt;
- host effects and match circumstances;
- GAM-Poisson match model;
- Dixon-Coles-style draw adjustment;
- 100,000 tournament simulations.

What we should steal:

- do not rely on current Elo alone for national teams;
- add a slow-moving historical-strength prior;
- add roster-quality proxies because national teams have few competitive matches;
- evaluate with temporal out-of-sample validation and ranked probability score.

### 2. talkSPORT AI betting experiment

Date: 2026-06-11

This is less transparent, but still useful as a signal of what public “supercomputer” models are leaning on.

What it says it uses:

- Elo ratings;
- squad values;
- historical metrics;
- many tournament simulations / brackets.

What we should steal:

- ensemble thinking;
- tournament-level simulation, not only single-match probabilities.

### 3. The Sun AI / supercomputer coverage

Dates found:

- 2026-06-08 article on AI betting predictions;
- 2025-11-20 article on a pre-tournament supercomputer simulation.

Useful only at high level.

What it suggests:

- public-facing models heavily lean on squad value, historical strength, and many repeated tournament simulations.

### 4. University of Reading supercomputer mention

Source mention found in a Guardian live blog on 2026-06-08.

Useful mainly as additional confirmation that:

- standard favorite set is stable across models;
- strong teams still have relatively low outright win probabilities because of tournament variance.

## Related non-WC2026 but directly relevant model ideas

### 5. Elo-based World Cup paper

Gilch and Müller (2018):

- Poisson regression with Elo covariates;
- neutral-ground focus for World Cup-style matches;
- tournament simulation.

### 6. World Cup 2022 zero-inflated generalized Poisson

Gilch (2022):

- Elo;
- attack and defense team skills;
- location;
- time and importance weighting.

### 7. Historical-data + bookmaker-odds Bayesian model

Egidi, Pauli, Torelli (2018):

- blend model-based and market-based information;
- directly relevant for our eventual EV and odds-comparison layer.

### 8. Recent market-calibrated forecasting work

Recent 2026 papers reinforce that:

- market calibration is extremely hard to beat;
- we should treat bookmaker and exchange prices as benchmark inputs, not just competitors.

## Practical implications for our build

We should move toward this ladder:

1. Elo + Dixon-Coles baseline;
2. temporal backtest and tournament simulator;
3. historical-Elo prior;
4. squad-value / roster-strength features;
5. market-aware calibration layer.

## Source links

- EL PAIS WC2026 model:
  https://elpais.com/deportes/mundial-futbol/2026-06-11/quien-ganara-el-mundial-asi-arrancan-nuestras-predicciones.html
- talkSPORT AI experiment:
  https://talksport.com/betting/4311569/world-cup-2026-ai-vs-human-betting-tips-predictions/
- The Sun AI betting predictions:
  https://www.thesun.co.uk/betting/39272679/world-cup-2026-ai-betting-predictions/
- The Sun supercomputer simulation:
  https://www.thesun.co.uk/sport/37381608/england-world-cup-2026-supercomputer-prediction/
- Elo-based World Cup prediction paper:
  https://arxiv.org/abs/1806.01930
- World Cup 2022 generalized Poisson paper:
  https://arxiv.org/abs/2205.04173
- Historical data + bookmaker odds paper:
  https://arxiv.org/abs/1802.08848
- Market-calibrated in-play model:
  https://arxiv.org/abs/2605.16066
- Odds-conversion / EMH paper:
  https://arxiv.org/abs/2604.17194
