from __future__ import annotations

from wc2026_model.data.public_match_odds import (
    PublicMatchOddsRecord,
    _normalize_label,
    extract_talksport_article_match_odds,
    extract_talksport_widget_urls_from_article_html,
    extract_talksport_widget_match_odds,
    extract_the_sun_match_odds,
    search_talksport_posts,
)


def test_extract_talksport_widget_match_odds(monkeypatch) -> None:
    html = """
    <html>
      <body>
        <h2>Qatar vs Switzerland</h2>
        <div class="operator">
          <span>bet365</span>
        </div>
        <div class="odds">
          <a href="https://example.com?coupon_key=%26bs%3D123%7E12%2F1%7E0%26bet%3D1">
            <span class="type">1</span>
          </a>
          <a href="https://example.com?coupon_key=%26bs%3D124%7E6%2F1%7E0%26bet%3D1">
            <span class="type">X</span>
          </a>
          <a href="https://example.com?coupon_key=%26bs%3D125%7E1%2F5%7E0%26bet%3D1">
            <span class="type">2</span>
          </a>
          <span class="reverseBet">94.96%</span>
        </div>
      </body>
    </html>
    """
    monkeypatch.setattr(
        "wc2026_model.data.public_match_odds.fetch_public_html",
        lambda url, timeout=20: html,
    )

    record = extract_talksport_widget_match_odds(
        "https://bettingwidgets.talksport.com/event?token=test",
        match_date="2026-06-13",
    )

    assert isinstance(record, PublicMatchOddsRecord)
    assert record.home_team == "Qatar"
    assert record.away_team == "Switzerland"
    assert record.bookmaker == "bet365"
    assert record.home_fractional_odds == "12/1"
    assert record.draw_fractional_odds == "6/1"
    assert record.away_fractional_odds == "1/5"
    assert record.home_decimal_odds == 13.0
    assert record.draw_decimal_odds == 7.0
    assert record.away_decimal_odds == 1.2


def test_extract_the_sun_match_odds(monkeypatch) -> None:
    html = """
    <html>
      <body>
        <h2 class="wp-block-heading">Brazil vs Morocco odds</h2>
        <ul class="wp-block-list">
          <li><strong>Brazil 8/13 with <a>Betfair</a></strong></li>
        </ul>
        <ul class="wp-block-list">
          <li><strong>Draw 11/4 with <a>Paddy Power</a></strong></li>
        </ul>
        <ul class="wp-block-list">
          <li><strong>Morocco 4/1 with <a>Sky Bet</a></strong></li>
        </ul>
        <h2 class="wp-block-heading">Brazil vs Morocco tips and betting predictions</h2>
      </body>
    </html>
    """
    monkeypatch.setattr(
        "wc2026_model.data.public_match_odds.fetch_public_html",
        lambda url, timeout=20: html,
    )

    record = extract_the_sun_match_odds(
        "https://www.thesun.co.uk/sport/example",
        match_date="2026-06-13",
        home_team="Brazil",
        away_team="Morocco",
    )

    assert record.bookmaker == "best_listed_prices"
    assert record.home_fractional_odds == "8/13"
    assert record.draw_fractional_odds == "11/4"
    assert record.away_fractional_odds == "4/1"
    assert record.home_decimal_odds == 1 + (8 / 13)
    assert record.draw_decimal_odds == 3.75
    assert record.away_decimal_odds == 5.0


def test_extract_talksport_widget_urls_from_article_html() -> None:
    article_html = """
    <iframe src="https://bettingwidgets.talksport.com/event?token=abc123"></iframe>
    <iframe src="https://bettingwidgets.talksport.com/event?token=def456"></iframe>
    <iframe src="https://bettingwidgets.talksport.com/event?token=abc123"></iframe>
    """
    assert extract_talksport_widget_urls_from_article_html(article_html) == [
        "https://bettingwidgets.talksport.com/event?token=abc123",
        "https://bettingwidgets.talksport.com/event?token=def456",
    ]


def test_extract_talksport_article_match_odds(monkeypatch) -> None:
    article_html = """
    <html><body>
    <iframe src="https://bettingwidgets.talksport.com/event?token=abc123"></iframe>
    </body></html>
    """
    widget_html = """
    <html>
      <body>
        <h2>Haiti vs Scotland</h2>
        <div class="operator"><span>bet365</span></div>
        <div class="odds">
          <a href="https://example.com?coupon_key=%26bs%3D1%7E7%2F1%7E0%26bet%3D1"><span class="type">1</span></a>
          <a href="https://example.com?coupon_key=%26bs%3D2%7E16%2F5%7E0%26bet%3D1"><span class="type">X</span></a>
          <a href="https://example.com?coupon_key=%26bs%3D3%7E11%2F20%7E0%26bet%3D1"><span class="type">2</span></a>
          <span class="reverseBet">95.40%</span>
        </div>
      </body>
    </html>
    """

    def _fake_fetch(url: str, timeout: int = 20) -> str:
        if "token=abc123" in url:
            return widget_html
        return article_html

    monkeypatch.setattr(
        "wc2026_model.data.public_match_odds.fetch_public_html",
        _fake_fetch,
    )

    record = extract_talksport_article_match_odds(
        "https://talksport.com/betting/example",
        match_date="2026-06-13",
        home_team="Haiti",
        away_team="Scotland",
    )

    assert record.source_type == "talksport_article"
    assert record.source_url == "https://talksport.com/betting/example"
    assert record.home_fractional_odds == "7/1"
    assert record.draw_fractional_odds == "16/5"
    assert record.away_fractional_odds == "11/20"


def test_search_talksport_posts(monkeypatch) -> None:
    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return (
                b'[{"id": 4325440, "title": "Haiti vs Scotland", "url": '
                b'"https://talksport.com/betting/4325440/haiti-vs-scotland-odds-tips-2026/", '
                b'"type": "post", "subtype": "post"}]'
            )

    monkeypatch.setattr(
        "wc2026_model.data.public_match_odds.urlopen",
        lambda request, timeout=20: _FakeResponse(),
    )

    results = search_talksport_posts("Haiti Scotland odds")

    assert results == [
        {
            "id": 4325440,
            "title": "Haiti vs Scotland",
            "url": "https://talksport.com/betting/4325440/haiti-vs-scotland-odds-tips-2026/",
            "type": "post",
            "subtype": "post",
        }
    ]


def test_normalize_label() -> None:
    assert _normalize_label("  Ivory   Coast  ") == "ivory coast"
