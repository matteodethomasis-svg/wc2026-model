from __future__ import annotations

import argparse

from wc2026_model.data import (
    DEFAULT_INTERNATIONAL_RESULTS_URL,
    download_international_results_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the international football results dataset."
    )
    parser.add_argument(
        "--output",
        default="data/raw/international_results.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_INTERNATIONAL_RESULTS_URL,
        help="Source CSV URL.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args()

    output_path = download_international_results_csv(
        args.output,
        source_url=args.source_url,
        overwrite=args.overwrite,
    )
    print(f"Downloaded international results to {output_path}")


if __name__ == "__main__":
    main()
