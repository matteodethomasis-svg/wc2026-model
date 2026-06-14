# WC2026 Repo Scan

Date checked: 2026-06-12

## Conclusion

I did not find a strong public GitHub repository that is clearly:

- specific to World Cup 2026;
- focused on serious probabilistic forecasting;
- maintained enough to save meaningful engineering time.

What does exist publicly is much better described as:

- general football modelling libraries;
- academic papers for World Cup forecasting;
- media-facing predictor/simulator tools without reusable code.

## Practical Implication

Best path:

1. reuse strong generic football libraries;
2. build our own World Cup 2026 pipeline on top;
3. own the data, evaluation, market-comparison, and tournament-simulation layers.

## Reusable Bases Instead

### penaltyblog

Strongest Python modelling base found for this use case.

### soccerdata

Best Python ingestion backbone found for public football data.

### international_results

Useful historical international-match dataset.

### StatsBomb open-data + socceraction

Good optional enrichment layer, but not the best project backbone.
