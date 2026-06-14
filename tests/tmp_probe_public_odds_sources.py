from __future__ import annotations

from urllib.request import Request, urlopen


def test_probe_public_odds_sources() -> None:
    urls = [
        "https://www.oddschecker.com/football/world-cup",
        "https://www.oddschecker.com/football/world-cup/winner",
        "https://www.oddschecker.com/football/world-cup/fixtures",
        "https://www.oddspedia.com/football",
    ]
    for url in urls:
        request = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; wc2026-model/0.1)"},
        )
        try:
            with urlopen(request, timeout=20) as response:
                html = response.read(1200).decode("utf-8", errors="ignore")
                print("URL", url, "STATUS_OK", True)
                print(html[:1200])
        except Exception as exc:  # pragma: no cover - probe only
            print("URL", url, "STATUS_OK", False, "ERROR", repr(exc))
