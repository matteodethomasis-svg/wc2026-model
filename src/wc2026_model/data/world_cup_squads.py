from __future__ import annotations

from html import unescape
from io import StringIO
from pathlib import Path
import re
from urllib.request import Request, urlopen

import pandas as pd

DEFAULT_WC2026_SQUADS_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"

_TEAM_HEADING_PATTERN = re.compile(
    r"<h3[^>]*>(?P<heading>.*?)</h3>",
    flags=re.IGNORECASE | re.DOTALL,
)
_WIKITABLE_PATTERN = re.compile(
    r"(<table[^>]*class=\"[^\"]*wikitable[^\"]*\"[^>]*>.*?</table>)",
    flags=re.IGNORECASE | re.DOTALL,
)


def fetch_world_cup_squads_page_html(
    source_url: str = DEFAULT_WC2026_SQUADS_URL,
) -> str:
    request = Request(
        source_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; wc2026-model/0.1; +https://example.com)"
        },
    )
    with urlopen(request) as response:
        return response.read().decode("utf-8", errors="ignore")


def load_world_cup_squads_from_wikipedia(
    source_url: str = DEFAULT_WC2026_SQUADS_URL,
    *,
    html: str | None = None,
) -> pd.DataFrame:
    page_html = html if html is not None else fetch_world_cup_squads_page_html(source_url)
    team_frames: list[pd.DataFrame] = []

    heading_matches = list(_TEAM_HEADING_PATTERN.finditer(page_html))
    for index, match in enumerate(heading_matches):
        team_name = _extract_heading_text(match.group("heading"))
        next_start = (
            heading_matches[index + 1].start() if index + 1 < len(heading_matches) else len(page_html)
        )
        block = page_html[match.end() : next_start]
        table_match = _WIKITABLE_PATTERN.search(block)
        if table_match is None:
            continue

        tables = pd.read_html(StringIO(table_match.group(1)))
        if not tables:
            continue
        squad_table = _standardize_squad_table(tables[0], team_name=team_name)
        if squad_table is None:
            continue
        team_frames.append(squad_table)

    if not team_frames:
        raise ValueError("Could not parse any World Cup squad tables from the source page.")

    return pd.concat(team_frames, ignore_index=True)


def save_world_cup_squads_csv(
    destination: str | Path,
    *,
    source_url: str = DEFAULT_WC2026_SQUADS_URL,
    html: str | None = None,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    squads = load_world_cup_squads_from_wikipedia(source_url, html=html)
    squads.to_csv(destination_path, index=False)
    return destination_path


def _standardize_squad_table(
    table: pd.DataFrame,
    *,
    team_name: str,
) -> pd.DataFrame | None:
    flattened_columns = [_flatten_column_name(column) for column in table.columns]
    column_lookup = {column.lower(): column for column in flattened_columns}

    player_column = _find_column(flattened_columns, "player")
    club_column = _find_column(flattened_columns, "club")
    position_column = _find_column(flattened_columns, "pos")
    caps_column = _find_column(flattened_columns, "caps")
    goals_column = _find_column(flattened_columns, "goals")
    number_column = _find_column(flattened_columns, "no")
    birth_column = _find_column(flattened_columns, "date of birth")

    required_columns = [player_column, club_column, position_column, caps_column]
    if any(column is None for column in required_columns):
        return None

    renamed = table.copy()
    renamed.columns = flattened_columns

    standardized = pd.DataFrame(
        {
            "team": str(team_name).strip(),
            "squad_number": _to_numeric_or_na(renamed[number_column]) if number_column else pd.NA,
            "position": renamed[position_column].map(_normalize_position),
            "player": renamed[player_column].map(_clean_player_name),
            "caps": _to_numeric_or_na(renamed[caps_column]),
            "goals": _to_numeric_or_na(renamed[goals_column]) if goals_column else pd.NA,
            "club": renamed[club_column].map(_clean_cell_text),
        }
    )

    if birth_column is not None:
        birth_details = renamed[birth_column].map(_parse_birth_details)
        standardized["birth_date"] = birth_details.map(lambda item: item[0])
        standardized["age"] = birth_details.map(lambda item: item[1])
    else:
        standardized["birth_date"] = pd.NaT
        standardized["age"] = pd.NA

    standardized["team"] = standardized["team"].astype(str)
    standardized["player"] = standardized["player"].astype(str)
    standardized["club"] = standardized["club"].astype(str)
    standardized = standardized[
        standardized["player"].ne("") & standardized["club"].ne("") & standardized["position"].ne("")
    ].copy()
    return standardized.reset_index(drop=True)


def _flatten_column_name(column: object) -> str:
    if isinstance(column, tuple):
        parts = [str(part).strip() for part in column if str(part).strip() and str(part) != "nan"]
        return " ".join(parts)
    return str(column).strip()


def _extract_heading_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _find_column(columns: list[str], token: str) -> str | None:
    normalized_token = token.lower()
    for column in columns:
        lowered = column.lower()
        if lowered == normalized_token or lowered.startswith(normalized_token):
            return column
    for column in columns:
        if normalized_token in column.lower():
            return column
    return None


def _clean_player_name(value: object) -> str:
    text = _clean_cell_text(value)
    text = re.sub(r"\(\s*captain\s*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]+\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_cell_text(value: object) -> str:
    text = unescape(str(value)).strip()
    text = re.sub(r"\[[^\]]+\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_position(value: object) -> str:
    text = _clean_cell_text(value).upper()
    match = re.search(r"(GK|DF|MF|FW)", text)
    return match.group(1) if match else text


def _to_numeric_or_na(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_birth_details(value: object) -> tuple[pd.Timestamp | pd.NaT, float | pd.NA]:
    text = _clean_cell_text(value)
    text = re.sub(r"^\([^)]*\)", "", text).strip()
    age_match = re.search(r"aged\s+(\d+)", text, flags=re.IGNORECASE)
    age = float(age_match.group(1)) if age_match else pd.NA
    birth_text = re.sub(r"\(aged\s+\d+\)", "", text, flags=re.IGNORECASE).strip()
    birth_date = pd.to_datetime(birth_text, errors="coerce")
    return birth_date, age
