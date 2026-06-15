"""Fetch live World Cup results from ESPN's free public JSON scoreboard API.

ESPN exposes a keyless JSON endpoint per date:
  https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=YYYYMMDD

It refreshes within minutes of full time and gives the final score + status, which is
all we need to move the Elo and condition the tournament sim. (It does not expose true
xG; that stays on the pre-tournament StatsBomb panel — see the xG-proxy follow-up.)

Team names are normalized with the project's canonicalize_team_name (handles
"Czechia"->"Czech Republic", "Türkiye"->"Turkey", "Bosnia-Herzegovina"->...).

Note on dates: ESPN tags events in UTC. A late kickoff can land on the next UTC day,
so callers should fetch an inclusive date range with a day of slack on each side; the
match is keyed by teams+date so a one-day shift still dedupes downstream.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from urllib.request import Request, urlopen

import pandas as pd

from wc2026_model.data.international_results import canonicalize_team_name

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={date}"
)
ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event={event}"
)
_USER_AGENT = "Mozilla/5.0 (wc2026-model results fetcher)"
_FINAL_STATUSES = {"STATUS_FULL_TIME", "STATUS_FINAL"}
# ESPN position abbreviations -> our coarse buckets.
_ESPN_POSITION = {"G": "GK", "D": "DF", "M": "MF", "F": "FW"}


def _fetch_scoreboard_json(yyyymmdd: str, *, timeout: int = 20) -> dict:
    url = ESPN_SCOREBOARD_URL.format(date=yyyymmdd)
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_summary_json(event_id: str, *, timeout: int = 20) -> dict:
    url = ESPN_SUMMARY_URL.format(event=event_id)
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_events(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for event in payload.get("events", []):
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        status = competition.get("status", {}).get("type", {}).get("name", "")
        if status not in _FINAL_STATUSES:
            continue  # skip scheduled / in-progress
        competitors = competition.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if home is None or away is None:
            continue
        try:
            home_goals = int(home["score"])
            away_goals = int(away["score"])
        except (KeyError, TypeError, ValueError):
            continue
        match_date = str(competition.get("date", event.get("date", "")))[:10]
        rows.append(
            {
                "match_date": match_date,
                "home_team": canonicalize_team_name(home["team"]["displayName"]),
                "away_team": canonicalize_team_name(away["team"]["displayName"]),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "tournament": "FIFA World Cup",
                "neutral": True,  # WC2026 group games are at neutral host venues
                "source": "espn",
            }
        )
    return rows


def fetch_world_cup_kickoffs(
    start_date: str,
    end_date: str,
    *,
    timeout: int = 20,
) -> pd.DataFrame:
    """Return every WC2026 fixture (any status) with its kickoff timestamp (UTC).

    Used to make the model-vs-market comparison fair: the market snapshot scored for a
    match must have been captured BEFORE this kickoff (ante-post), never during/after
    the game when the live market has already moved on the in-play action.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    rows: list[dict] = []
    day = start
    while day <= end:
        try:
            payload = _fetch_scoreboard_json(day.strftime("%Y%m%d"), timeout=timeout)
            for event in payload.get("events", []):
                competitions = event.get("competitions") or []
                if not competitions:
                    continue
                competition = competitions[0]
                competitors = competition.get("competitors", [])
                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if home is None or away is None:
                    continue
                rows.append({
                    "home_team": canonicalize_team_name(home["team"]["displayName"]),
                    "away_team": canonicalize_team_name(away["team"]["displayName"]),
                    "kickoff_ts": str(competition.get("date", event.get("date", ""))),
                })
        except Exception:
            pass
        day += timedelta(days=1)
    frame = pd.DataFrame(rows, columns=["home_team", "away_team", "kickoff_ts"])
    if not frame.empty:
        frame["kickoff_ts"] = pd.to_datetime(frame["kickoff_ts"], errors="coerce", utc=True)
        frame = frame.dropna(subset=["kickoff_ts"]).drop_duplicates(
            subset=["home_team", "away_team"], keep="last"
        ).reset_index(drop=True)
    return frame


def fetch_world_cup_results(
    start_date: str,
    end_date: str,
    *,
    timeout: int = 20,
) -> pd.DataFrame:
    """Return finished WC2026 matches between start_date and end_date (inclusive)."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    rows: list[dict] = []
    day = start
    while day <= end:
        try:
            payload = _fetch_scoreboard_json(day.strftime("%Y%m%d"), timeout=timeout)
            rows.extend(_parse_events(payload))
        except Exception:
            # A single bad day shouldn't kill the whole fetch.
            pass
        day += timedelta(days=1)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "match_date", "home_team", "away_team",
                "home_goals", "away_goals", "tournament", "neutral", "source",
            ]
        )
    # Match the other sources' dtype (Timestamp) so downstream merges/dedupe align.
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    frame = frame.dropna(subset=["match_date"])
    frame = frame.drop_duplicates(
        subset=["match_date", "home_team", "away_team", "home_goals", "away_goals"]
    ).reset_index(drop=True)
    return frame


def fetch_world_cup_expected_lineups(
    start_date: str,
    end_date: str,
    *,
    timeout: int = 20,
) -> pd.DataFrame:
    """Return expected/confirmed starting XIs for WC2026 fixtures in the date range.

    Source: the SAME free, keyless ESPN endpoints used for results — the per-event
    ``summary`` carries a ``rosters`` block with a ``starter`` flag and position per
    player. ESPN publishes the probable XI roughly an hour before kickoff and the
    confirmed XI at kickoff, so a frequent poll picks it up automatically.

    Output is the flat schema the squad-intelligence loader expects:
    ``team, player, position, is_expected_starter, lineup_confidence, match_date``.
    Only rows with a populated roster (>1 entry, i.e. a real lineup) are returned, so a
    fixture with no lineup yet simply contributes nothing.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    rows: list[dict] = []
    day = start
    while day <= end:
        try:
            payload = _fetch_scoreboard_json(day.strftime("%Y%m%d"), timeout=timeout)
        except Exception:
            day += timedelta(days=1)
            continue
        for event in payload.get("events", []):
            event_id = str(event.get("id", ""))
            match_date = str(event.get("date", ""))[:10]
            if not event_id:
                continue
            try:
                summary = _fetch_summary_json(event_id, timeout=timeout)
            except Exception:
                continue
            rosters = summary.get("rosters") or []
            for team_roster in rosters:
                roster = team_roster.get("roster") or []
                if len(roster) <= 1:
                    continue  # lineup not published yet
                team_name = canonicalize_team_name(
                    (team_roster.get("team") or {}).get("displayName", "")
                )
                # Confirmed XI (game started) is certain; a pre-match probable XI is less so.
                status = summary.get("header", {}).get("competitions", [{}])[0] \
                    .get("status", {}).get("type", {}).get("name", "")
                confidence = 1.0 if status not in ("STATUS_SCHEDULED", "") else 0.7
                for entry in roster:
                    athlete = entry.get("athlete") or {}
                    name = athlete.get("displayName") or athlete.get("fullName")
                    if not name:
                        continue
                    pos = (entry.get("position") or {}).get("abbreviation", "")
                    rows.append(
                        {
                            "match_date": match_date,
                            "team": team_name,
                            "player": str(name),
                            "position": _ESPN_POSITION.get(pos, pos),
                            "is_expected_starter": bool(entry.get("starter")),
                            "lineup_confidence": confidence,
                            "fixture_id": event_id,
                        }
                    )
        day += timedelta(days=1)

    columns = [
        "match_date", "team", "player", "position",
        "is_expected_starter", "lineup_confidence", "fixture_id",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].drop_duplicates(
        subset=["fixture_id", "team", "player"]
    ).reset_index(drop=True)
