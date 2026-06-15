"""Download the Transfermarkt player valuations + players tables (historical market
values) from the free davidcariboo/player-scores mirror — no scraping, no Kaggle login.

player_valuations.csv is the FULL per-player market-value TIME SERIES (player_id, date,
market_value_in_eur, club). Combined with players.csv (id -> name, nationality, dob) it
lets us read each player's market value AT ANY MATCH DATE — a leak-free, club-based per-
player quality signal that replaces the crude "every player gets their club's Elo" proxy.

The R2 mirror is published by dcaribou/transfermarkt-datasets and refreshed weekly.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]

R2_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
FILES = ("player_valuations", "players")
_HEADERS = {"User-Agent": "Mozilla/5.0 (wc2026-model transfermarkt fetcher)"}


def _download_gz(name: str, out_dir: Path) -> Path:
    url = f"{R2_BASE}/{name}.csv.gz"
    gz_path = out_dir / f"{name}.csv.gz"
    csv_path = out_dir / f"{name}.csv"
    request = Request(url, headers=_HEADERS)
    with urlopen(request, timeout=120) as response, gz_path.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    with gzip.open(gz_path, "rb") as gz, csv_path.open("wb") as out:
        shutil.copyfileobj(gz, out)
    gz_path.unlink(missing_ok=True)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw/transfermarkt")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = _download_gz(name, out_dir)
        size_mb = path.stat().st_size / 1e6
        print(f"  {path.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    print("Transfermarkt valuations + players tables downloaded.")


if __name__ == "__main__":
    main()
