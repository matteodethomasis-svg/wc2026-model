from __future__ import annotations

from wc2026_model.data import load_world_cup_squads_from_wikipedia


_SAMPLE_HTML = """
<html>
  <body>
    <h2><span class="mw-headline">Group I</span></h2>
    <h3><span class="mw-headline" id="France">France</span></h3>
    <p>Coach: Didier Deschamps</p>
    <table class="wikitable">
      <tr>
        <th>No.</th>
        <th>Pos.</th>
        <th>Player</th>
        <th>Date of birth (age)</th>
        <th>Caps</th>
        <th>Goals</th>
        <th>Club</th>
      </tr>
      <tr>
        <td>10</td>
        <td>FW</td>
        <td>Kylian Mbappe (captain)</td>
        <td>(1998-12-20) December 20, 1998 (aged 27)</td>
        <td>94</td>
        <td>50</td>
        <td>Real Madrid</td>
      </tr>
      <tr>
        <td>1</td>
        <td>GK</td>
        <td>Mike Maignan</td>
        <td>(1995-07-03) July 3, 1995 (aged 30)</td>
        <td>29</td>
        <td>0</td>
        <td>Milan</td>
      </tr>
    </table>
    <h3><span class="mw-headline" id="Norway">Norway</span></h3>
    <p>Coach: Stale Solbakken</p>
    <table class="wikitable">
      <tr>
        <th>No.</th>
        <th>Pos.</th>
        <th>Player</th>
        <th>Date of birth (age)</th>
        <th>Caps</th>
        <th>Goals</th>
        <th>Club</th>
      </tr>
      <tr>
        <td>9</td>
        <td>FW</td>
        <td>Erling Haaland</td>
        <td>(2000-07-21) July 21, 2000 (aged 25)</td>
        <td>47</td>
        <td>41</td>
        <td>Manchester City</td>
      </tr>
    </table>
  </body>
</html>
"""


def test_load_world_cup_squads_from_wikipedia_parses_team_tables() -> None:
    squads = load_world_cup_squads_from_wikipedia(html=_SAMPLE_HTML)

    assert squads["team"].tolist() == ["France", "France", "Norway"]
    assert squads["player"].tolist() == ["Kylian Mbappe", "Mike Maignan", "Erling Haaland"]
    assert squads["club"].tolist() == ["Real Madrid", "Milan", "Manchester City"]
    assert squads["position"].tolist() == ["FW", "GK", "FW"]
    assert squads["caps"].tolist() == [94, 29, 47]
    assert squads["goals"].tolist() == [50, 0, 41]
    assert squads["age"].tolist() == [27.0, 30.0, 25.0]
