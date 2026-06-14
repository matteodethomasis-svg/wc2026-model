from pathlib import Path


def test_dump_live_market_workflow_targets() -> None:
    targets = [
        "scripts/compare_bookmaker_match_odds.py",
        "src/wc2026_model/markets/match_odds.py",
        "scripts/predict_world_cup_fixtures.py",
        "reports/wc2026_squad_strength_summary.json",
    ]
    for target in targets:
        path = Path(target)
        print(f"\n===== {target} =====")
        if path.suffix == ".json":
            print(path.read_text(encoding="utf-8"))
        else:
            print(path.read_text(encoding="utf-8"))
