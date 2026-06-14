from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape
from typing import Any
from urllib.parse import unquote
from urllib.request import Request, urlopen

from wc2026_model.markets import fractional_odds_to_decimal

USER_AGENT = "Mozilla/5.0 (compatible; wc2026-model/0.1)"

_TALKSPORT_MATCH_TITLE_PATTERN = re.compile(
    r"<h2[^>]*>(?P<home>[^<]+?) vs (?P<away>[^<]+?)</h2>",
    re.IGNORECASE,
)
_TALKSPORT_ARTICLE_WIDGET_URL_PATTERN = re.compile(
    r"https://bettingwidgets\.talksport\.com/event\?token=[A-Za-z0-9]+",
    re.IGNORECASE,
)
_TALKSPORT_OPERATOR_ROW_PATTERN = re.compile(
    r'<div class="[^"]*operator[^"]*">.*?<span>(?P<bookmaker>[^<]+)</span></div>'
    r'\s*<div class="[^"]*odds[^"]*">(?P<odds_block>.*?)'
    r'<span class="[^"]*reverseBet[^"]*"[^>]*>(?P<reverse_bet>[^<]+)</span>',
    re.IGNORECASE | re.DOTALL,
)
_TALKSPORT_BOOKMAKER_PATTERN = re.compile(
    r'<div class="[^"]*operator[^"]*"[^>]*>.*?<span>(?P<bookmaker>[^<]+)</span>',
    re.IGNORECASE | re.DOTALL,
)
_TALKSPORT_ODDS_BLOCK_PATTERN = re.compile(
    r'<div class="[^"]*odds[^"]*"[^>]*>(?P<odds_block>.*?)'
    r'<span class="[^"]*reverseBet[^"]*"[^>]*>(?P<reverse_bet>[^<]+)</span>',
    re.IGNORECASE | re.DOTALL,
)
_TALKSPORT_CELL_PATTERN = re.compile(
    r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>\s*'
    r'<span[^>]*class="[^"]*type[^"]*"[^>]*>(?P<slot>[12X])</span>',
    re.IGNORECASE | re.DOTALL,
)
_COUPON_KEY_ODDS_PATTERN = re.compile(r"~(?P<odds>(?:[0-9]+/[0-9]+|EVS|Evs|Evens))~0")
_SUN_ODDS_HEADING_TEMPLATE = r"<h2 class=\"wp-block-heading\">{home} vs {away} odds</h2>"
_SUN_LIST_ITEM_PATTERN = re.compile(
    r"<li><strong>(?P<label>[^<]+?)\s+"
    r"(?P<odds>(?:[0-9]+/[0-9]+|EVS|Evs|Evens))\s+with\s+"
    r"<a[^>]*>(?P<bookmaker>[^<]+)</a>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PublicMatchOddsRecord:
    match_date: str
    home_team: str
    away_team: str
    home_fractional_odds: str
    draw_fractional_odds: str
    away_fractional_odds: str
    home_decimal_odds: float
    draw_decimal_odds: float
    away_decimal_odds: float
    bookmaker: str
    source_type: str
    source_url: str
    source_title: str | None = None
    source_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "match_date": self.match_date,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_fractional_odds": self.home_fractional_odds,
            "draw_fractional_odds": self.draw_fractional_odds,
            "away_fractional_odds": self.away_fractional_odds,
            "home_decimal_odds": self.home_decimal_odds,
            "draw_decimal_odds": self.draw_decimal_odds,
            "away_decimal_odds": self.away_decimal_odds,
            "bookmaker": self.bookmaker,
            "source_type": self.source_type,
            "source_url": self.source_url,
        }
        if self.source_title is not None:
            payload["source_title"] = self.source_title
        if self.source_note is not None:
            payload["source_note"] = self.source_note
        return payload


def fetch_public_html(url: str, *, timeout: int = 20) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def extract_talksport_widget_match_odds(
    url: str,
    *,
    match_date: str,
    home_team: str | None = None,
    away_team: str | None = None,
    source_title: str | None = None,
) -> PublicMatchOddsRecord:
    html = fetch_public_html(url)
    if home_team is None or away_team is None:
        match = _TALKSPORT_MATCH_TITLE_PATTERN.search(html)
        if match is None:
            raise ValueError("Unable to determine teams from talkSPORT widget HTML.")
        home_team = home_team or unescape(match.group("home")).strip()
        away_team = away_team or unescape(match.group("away")).strip()

    row_match = _TALKSPORT_OPERATOR_ROW_PATTERN.search(html)
    if row_match is None:
        bookmaker_match = _TALKSPORT_BOOKMAKER_PATTERN.search(html)
        odds_block_match = _TALKSPORT_ODDS_BLOCK_PATTERN.search(html)
        if bookmaker_match is None or odds_block_match is None:
            raise ValueError("Unable to find a bookmaker row in talkSPORT widget HTML.")
        bookmaker = unescape(bookmaker_match.group("bookmaker")).strip()
        odds_block = odds_block_match.group("odds_block")
        reverse_bet = unescape(odds_block_match.group("reverse_bet")).strip()
    else:
        bookmaker = unescape(row_match.group("bookmaker")).strip()
        odds_block = row_match.group("odds_block")
        reverse_bet = unescape(row_match.group("reverse_bet")).strip()
    slot_to_odds: dict[str, str] = {}
    for cell_match in _TALKSPORT_CELL_PATTERN.finditer(odds_block):
        href = unquote(unescape(cell_match.group("href")))
        odds_match = _COUPON_KEY_ODDS_PATTERN.search(href)
        if odds_match is None:
            continue
        slot_to_odds[cell_match.group("slot")] = odds_match.group("odds")

    required_slots = {"1", "X", "2"}
    missing_slots = required_slots.difference(slot_to_odds)
    if missing_slots:
        missing = ", ".join(sorted(missing_slots))
        raise ValueError(f"Missing talkSPORT widget odds for slots: {missing}")
    return PublicMatchOddsRecord(
        match_date=match_date,
        home_team=home_team,
        away_team=away_team,
        home_fractional_odds=slot_to_odds["1"],
        draw_fractional_odds=slot_to_odds["X"],
        away_fractional_odds=slot_to_odds["2"],
        home_decimal_odds=fractional_odds_to_decimal(slot_to_odds["1"]),
        draw_decimal_odds=fractional_odds_to_decimal(slot_to_odds["X"]),
        away_decimal_odds=fractional_odds_to_decimal(slot_to_odds["2"]),
        bookmaker=bookmaker,
        source_type="talksport_widget",
        source_url=url,
        source_title=source_title,
        source_note=f"Extracted from the first visible 1X2 row in talkSPORT widget ({reverse_bet} reverse bet).",
    )


def extract_talksport_article_match_odds(
    article_url: str,
    *,
    match_date: str,
    home_team: str | None = None,
    away_team: str | None = None,
    source_title: str | None = None,
) -> PublicMatchOddsRecord:
    article_html = fetch_public_html(article_url)
    widget_urls = extract_talksport_widget_urls_from_article_html(article_html)
    if not widget_urls:
        raise ValueError("Unable to find a talkSPORT event widget URL in article HTML.")
    record = extract_talksport_widget_match_odds(
        widget_urls[0],
        match_date=match_date,
        home_team=home_team,
        away_team=away_team,
        source_title=source_title,
    )
    return PublicMatchOddsRecord(
        match_date=record.match_date,
        home_team=record.home_team,
        away_team=record.away_team,
        home_fractional_odds=record.home_fractional_odds,
        draw_fractional_odds=record.draw_fractional_odds,
        away_fractional_odds=record.away_fractional_odds,
        home_decimal_odds=record.home_decimal_odds,
        draw_decimal_odds=record.draw_decimal_odds,
        away_decimal_odds=record.away_decimal_odds,
        bookmaker=record.bookmaker,
        source_type="talksport_article",
        source_url=article_url,
        source_title=source_title,
        source_note=f"Extracted via embedded talkSPORT event widget {widget_urls[0]}.",
    )


def extract_talksport_widget_urls_from_article_html(article_html: str) -> list[str]:
    matches = _TALKSPORT_ARTICLE_WIDGET_URL_PATTERN.findall(article_html)
    unique_urls: list[str] = []
    seen: set[str] = set()
    for url in matches:
        if url not in seen:
            unique_urls.append(url)
            seen.add(url)
    return unique_urls


def extract_the_sun_match_odds(
    url: str,
    *,
    match_date: str,
    home_team: str,
    away_team: str,
    source_title: str | None = None,
) -> PublicMatchOddsRecord:
    html = fetch_public_html(url)
    heading = _SUN_ODDS_HEADING_TEMPLATE.format(
        home=re.escape(home_team),
        away=re.escape(away_team),
    )
    heading_match = re.search(heading, html, flags=re.IGNORECASE)
    if heading_match is None:
        raise ValueError(
            f"Unable to locate The Sun odds section for '{home_team} vs {away_team}'."
        )
    section_start = heading_match.end()
    next_heading = re.search(r"<h2[^>]*>", html[section_start:], flags=re.IGNORECASE)
    if next_heading is None:
        section_html = html[section_start:]
    else:
        section_html = html[section_start : section_start + next_heading.start()]

    label_to_odds: dict[str, tuple[str, str]] = {}
    for match in _SUN_LIST_ITEM_PATTERN.finditer(section_html):
        label = _normalize_label(unescape(match.group("label")))
        odds = match.group("odds")
        bookmaker = unescape(match.group("bookmaker")).strip()
        label_to_odds[label] = (odds, bookmaker)

    home_key = _normalize_label(home_team)
    away_key = _normalize_label(away_team)
    if "draw" not in label_to_odds or home_key not in label_to_odds or away_key not in label_to_odds:
        missing_labels = [
            label
            for label in (home_key, "draw", away_key)
            if label not in label_to_odds
        ]
        raise ValueError(
            "The Sun odds section is missing required entries: "
            + ", ".join(missing_labels)
        )

    home_fractional_odds, home_bookmaker = label_to_odds[home_key]
    draw_fractional_odds, draw_bookmaker = label_to_odds["draw"]
    away_fractional_odds, away_bookmaker = label_to_odds[away_key]
    source_note = (
        "Extracted from The Sun best-price bullets; "
        f"home via {home_bookmaker}, draw via {draw_bookmaker}, away via {away_bookmaker}."
    )
    return PublicMatchOddsRecord(
        match_date=match_date,
        home_team=home_team,
        away_team=away_team,
        home_fractional_odds=home_fractional_odds,
        draw_fractional_odds=draw_fractional_odds,
        away_fractional_odds=away_fractional_odds,
        home_decimal_odds=fractional_odds_to_decimal(home_fractional_odds),
        draw_decimal_odds=fractional_odds_to_decimal(draw_fractional_odds),
        away_decimal_odds=fractional_odds_to_decimal(away_fractional_odds),
        bookmaker="best_listed_prices",
        source_type="the_sun_article",
        source_url=url,
        source_title=source_title,
        source_note=source_note,
    )


def load_public_match_odds_sources(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Public match odds source config must be a JSON list.")
    return payload


def search_talksport_posts(query: str, *, timeout: int = 20) -> list[dict[str, Any]]:
    url = f"https://talksport.com/wp-json/wp/v2/search?search={query.replace(' ', '%20')}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    if not isinstance(payload, list):
        raise ValueError("talkSPORT search response must be a JSON list.")
    normalized_results: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        normalized_results.append(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "url": item.get("url"),
                "type": item.get("type"),
                "subtype": item.get("subtype"),
            }
        )
    return normalized_results


def _normalize_label(value: str) -> str:
    return " ".join(value.lower().split())
