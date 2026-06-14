import pandas as pd

from wc2026_model.evaluation import (
    append_match_snapshot,
    score_match_ledger,
    summarize_track_record,
)


def _snapshot():
    return pd.DataFrame([
        {"match_date": "2026-06-20", "home_team": "France", "away_team": "Senegal",
         "model_home": 0.57, "model_draw": 0.27, "model_away": 0.16,
         "market_home": 0.50, "market_draw": 0.28, "market_away": 0.22},
        {"match_date": "2026-06-20", "home_team": "Spain", "away_team": "Japan",
         "model_home": 0.70, "model_draw": 0.20, "model_away": 0.10,
         "market_home": 0.65, "market_draw": 0.22, "market_away": 0.13},
    ])


def test_append_is_idempotent_per_timestamp(tmp_path):
    ledger = tmp_path / "match_ledger.csv"
    append_match_snapshot(ledger, _snapshot(), snapshot_ts="2026-06-19T10:00Z")
    combined = append_match_snapshot(ledger, _snapshot(), snapshot_ts="2026-06-19T10:00Z")
    # Same timestamp + same matches => no duplication.
    assert len(combined) == 2
    # A new timestamp appends new rows.
    combined = append_match_snapshot(ledger, _snapshot(), snapshot_ts="2026-06-19T18:00Z")
    assert len(combined) == 4


def test_score_picks_winner_and_summary(tmp_path):
    ledger = tmp_path / "match_ledger.csv"
    led = append_match_snapshot(ledger, _snapshot(), snapshot_ts="2026-06-19T10:00Z")
    results = pd.DataFrame([
        # France win (model favoured home more than market -> model should win)
        {"home_team": "France", "away_team": "Senegal", "home_goals": 2, "away_goals": 0},
        # Japan upset win over Spain (both wrong, but market gave away more -> market better)
        {"home_team": "Spain", "away_team": "Japan", "home_goals": 0, "away_goals": 1},
    ])
    scored = score_match_ledger(led, results)
    assert len(scored) == 2
    fr = scored[scored["home_team"] == "France"].iloc[0]
    assert fr["result_side"] == "home"
    assert fr["winner"] == "model"  # model gave France more probability and was right
    sp = scored[scored["home_team"] == "Spain"].iloc[0]
    assert sp["result_side"] == "away"
    assert sp["winner"] == "market"  # market gave the upset more probability

    summary = summarize_track_record(scored)
    assert summary["resolved_matches"] == 2
    assert summary["model_wins"] + summary["market_wins"] == 2


def test_per_match_comparison_is_bias_proof(tmp_path):
    # A model that BLINDLY over-predicts draws must NOT win per-match on a non-draw.
    # This is the property the per-match (full-distribution) score guarantees.
    ledger = tmp_path / "match_ledger.csv"
    snap = pd.DataFrame([{
        "match_date": "2026-06-20", "home_team": "Spain", "away_team": "Japan",
        # draw-biased model vs a sharp market that nailed the home win
        "model_home": 0.33, "model_draw": 0.50, "model_away": 0.17,
        "market_home": 0.70, "market_draw": 0.20, "market_away": 0.10,
    }])
    led = append_match_snapshot(ledger, snap, snapshot_ts="2026-06-19T10:00Z")
    results = pd.DataFrame([
        {"home_team": "Spain", "away_team": "Japan", "home_goals": 2, "away_goals": 0}])
    scored = score_match_ledger(led, results)
    row = scored.iloc[0]
    assert row["result_side"] == "home"
    # The sharp market gave the home win 0.70 vs the draw-biased model's 0.33 -> market wins.
    assert row["winner"] == "market"


def test_scoring_uses_ante_post_snapshot_not_live(tmp_path):
    # Two snapshots for the same match: one ante-post (before kickoff) with the real
    # market, and one logged AFTER kickoff that (hypothetically) carries moved odds.
    # The scorer must use the ante-post one for a fair comparison.
    import pandas as pd
    ledger = tmp_path / "match_ledger.csv"
    ante = pd.DataFrame([{
        "match_date": "2026-06-20", "home_team": "France", "away_team": "Senegal",
        "model_home": 0.57, "model_draw": 0.27, "model_away": 0.16,
        "market_home": 0.50, "market_draw": 0.28, "market_away": 0.22,
    }])
    live = pd.DataFrame([{
        "match_date": "2026-06-20", "home_team": "France", "away_team": "Senegal",
        "model_home": 0.57, "model_draw": 0.27, "model_away": 0.16,
        # live market has 'seen' France score -> would unfairly look great
        "market_home": 0.95, "market_draw": 0.04, "market_away": 0.01,
    }])
    append_match_snapshot(ledger, ante, snapshot_ts="2026-06-20T16:00Z")  # before KO
    led = append_match_snapshot(ledger, live, snapshot_ts="2026-06-20T19:30Z")  # after KO

    kickoffs = pd.DataFrame([{
        "home_team": "France", "away_team": "Senegal",
        "kickoff_ts": pd.Timestamp("2026-06-20T18:00Z"),
    }])
    results = pd.DataFrame([
        {"home_team": "France", "away_team": "Senegal", "home_goals": 2, "away_goals": 0}])
    scored = score_match_ledger(led, results, kickoffs=kickoffs)
    row = scored.iloc[0]
    # Must reflect the ante-post market (0.50 on home), NOT the live 0.95.
    assert abs(row["market_p_result"] - 0.50) < 1e-9


def test_unplayed_matches_are_skipped(tmp_path):
    ledger = tmp_path / "match_ledger.csv"
    led = append_match_snapshot(ledger, _snapshot(), snapshot_ts="2026-06-19T10:00Z")
    scored = score_match_ledger(led, pd.DataFrame(
        columns=["home_team", "away_team", "home_goals", "away_goals"]))
    assert scored.empty
