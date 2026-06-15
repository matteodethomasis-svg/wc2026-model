"use strict";

let DATA = null;

// ---------- utils ----------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 1) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
const fmtPct = (v) => (v === null || v === undefined ? "—" : `${Number(v).toFixed(1)}%`);
const fmtOdds = (v) => (v === null || v === undefined ? "—" : Number(v).toFixed(2));
const fmtSign = (v) => (v === null || v === undefined ? "—" : `${v > 0 ? "+" : ""}${Number(v).toFixed(1)}%`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------- flags ----------
// Emoji regional-indicator flags don't render on Windows (Segoe UI Emoji lacks them),
// so we map each nation to its ISO-3166 alpha-2 code and render a flagcdn.com SVG —
// free, no key, consistent on every platform.
const FLAG_CC = {
  "Argentina":"ar","Australia":"au","Austria":"at","Belgium":"be","Brazil":"br",
  "Cameroon":"cm","Canada":"ca","Chile":"cl","Colombia":"co","Costa Rica":"cr",
  "Croatia":"hr","Czechia":"cz","Czech Republic":"cz","Denmark":"dk","Ecuador":"ec",
  "Egypt":"eg","England":"gb-eng","France":"fr","Germany":"de","Ghana":"gh","Greece":"gr",
  "Hungary":"hu","Iran":"ir","IR Iran":"ir","Italy":"it","Ivory Coast":"ci","Côte d'Ivoire":"ci",
  "Japan":"jp","Jordan":"jo","Mexico":"mx","Morocco":"ma","Netherlands":"nl","New Zealand":"nz",
  "Nigeria":"ng","Norway":"no","Panama":"pa","Paraguay":"py","Peru":"pe","Poland":"pl",
  "Portugal":"pt","Qatar":"qa","Saudi Arabia":"sa","Scotland":"gb-sct","Senegal":"sn",
  "Serbia":"rs","Slovenia":"si","South Africa":"za","South Korea":"kr","Korea Republic":"kr",
  "Spain":"es","Sweden":"se","Switzerland":"ch","Tunisia":"tn","Turkey":"tr","Türkiye":"tr",
  "Ukraine":"ua","United States":"us","USA":"us","Uruguay":"uy","Uzbekistan":"uz","Wales":"gb-wls",
  "Algeria":"dz","Bolivia":"bo","Cape Verde":"cv","Cabo Verde":"cv","Curacao":"cw","Curaçao":"cw",
  "DR Congo":"cd","Haiti":"ht","Honduras":"hn","Jamaica":"jm","New Caledonia":"nc",
  "Iraq":"iq","UAE":"ae","United Arab Emirates":"ae","Bosnia and Herzegovina":"ba",
};
// Returns an <img> flag (or a neutral placeholder span when the nation is unknown).
function flag(team) {
  const cc = FLAG_CC[team];
  if (!cc) return `<span class="flag flag-unknown" title="${esc(team)}"></span>`;
  return `<img class="flag" loading="lazy" alt="${esc(team)}" `
    + `src="https://flagcdn.com/${cc}.svg" onerror="this.style.display='none'">`;
}

// ---------- tabs ----------
function initTabs() {
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab").forEach((b) => b.classList.remove("active"));
      $$(".panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $("#" + btn.dataset.tab).classList.add("active");
    });
  });
}

// ---------- predictions ----------
function pickLabel(pick) {
  return { home: ["1", "pill-home"], draw: ["X", "pill-draw"], away: ["2", "pill-away"] }[pick];
}

function upcomingCard(m) {
  const [pl, pc] = pickLabel(m.pick);
  const confTier = m.confidence >= 60 ? "high" : m.confidence >= 40 ? "med" : "low";
  const xg = (m.home_xg != null && m.away_xg != null)
    ? `📊 Expected goals: <b>${fmt(m.home_xg, 2)}</b> – <b>${fmt(m.away_xg, 2)}</b>` : "";
  return `
    <div class="match-card">
      <div class="mc-top">
        <span>${esc(m.date)}</span>
        <span>${m.neutral ? "neutral venue" : esc(m.city)}</span>
      </div>
      <div class="mc-teams">
        <span class="home">${flag(m.home)} ${esc(m.home)}</span>
        <span class="mc-vs">vs</span>
        <span class="away">${esc(m.away)} ${flag(m.away)}</span>
      </div>
      <div class="probbar">
        <span class="p-home" style="width:${m.p_home}%" title="Home win">${m.p_home >= 12 ? m.p_home + "%" : ""}</span>
        <span class="p-draw" style="width:${m.p_draw}%" title="Draw">${m.p_draw >= 12 ? m.p_draw + "%" : ""}</span>
        <span class="p-away" style="width:${m.p_away}%" title="Away win">${m.p_away >= 12 ? m.p_away + "%" : ""}</span>
      </div>
      <div class="mc-odds">
        <span>Fair odds &nbsp; 1 <b>${fmtOdds(m.odds_home)}</b></span>
        <span>X <b>${fmtOdds(m.odds_draw)}</b></span>
        <span>2 <b>${fmtOdds(m.odds_away)}</b></span>
      </div>
      <div class="mc-xg">${xg}</div>
      <div class="mc-call">
        <span>Model call <span class="pill ${pc}">${pl}</span></span>
        <span class="conf conf-${confTier}" title="How sure the model is">
          ${fmtPct(m.confidence)} confident
          <span class="conf-bar"><span style="width:${m.confidence}%"></span></span>
        </span>
      </div>
      ${lineupBlock(m)}
    </div>`;
}

// Official XI block — only renders once ESPN publishes the lineups (~1h pre-kickoff).
function lineupBlock(m) {
  const h = m.home_xi || [], a = m.away_xi || [];
  if (!h.length && !a.length) return "";
  const col = (team, xi) => `
    <div class="xi-col">
      <div class="xi-team">${flag(team)} ${esc(team)}</div>
      <ol class="xi-list">
        ${xi.map((p) => `<li><span class="xi-pos xi-${esc(p.pos)}">${esc(p.pos || "·")}</span> ${esc(p.name)}</li>`).join("")}
      </ol>
    </div>`;
  return `
    <details class="xi-details">
      <summary>📋 Official lineups out</summary>
      <div class="xi-grid">
        ${h.length ? col(m.home, h) : ""}
        ${a.length ? col(m.away, a) : ""}
      </div>
    </details>`;
}

function playedCard(m) {
  const [pl, pc] = pickLabel(m.pick);
  const ok = m.pick_correct;
  return `
    <div class="match-card played">
      <div class="mc-top">
        <span>${esc(m.date)} · full time</span>
        <span class="${ok ? "tick" : "cross"}">${ok ? "✓ model got it" : "✗ model missed"}</span>
      </div>
      <div class="mc-teams">
        <span class="home ${m.result_side === "home" ? "won" : ""}">${flag(m.home)} ${esc(m.home)}</span>
        <span class="mc-score">${m.result_home_goals}–${m.result_away_goals}</span>
        <span class="away ${m.result_side === "away" ? "won" : ""}">${esc(m.away)} ${flag(m.away)}</span>
      </div>
      <div class="mc-recap">
        Model had picked <span class="pill ${pc}">${pl}</span> at ${fmtPct(m.confidence)} confidence.
      </div>
    </div>`;
}

function renderPredictions() {
  const q = $("#pred-search").value.trim().toLowerCase();
  const sort = $("#pred-sort").value;
  let items = DATA.predictions.slice();
  if (q) items = items.filter((m) => (m.home + " " + m.away).toLowerCase().includes(q));

  const sortFn = (a, b) => {
    if (sort === "confidence") return (b.confidence ?? 0) - (a.confidence ?? 0);
    if (sort === "upset") return (a.confidence ?? 0) - (b.confidence ?? 0);
    return (a.date + a.home).localeCompare(b.date + b.home);
  };

  // Upcoming = not yet kicked off (timezone-correct). A started-but-unresolved match
  // is neither upcoming (prediction is locked) nor shown with a score yet, so it simply
  // waits out of the list until ESPN posts the result and it joins "played".
  const upcoming = items.filter((m) => !(m.started || m.played)).sort(sortFn);
  // Older matches: most recent first regardless of the chosen sort (latest result on top).
  const played = items.filter((m) => m.played)
    .sort((a, b) => (b.date + b.home).localeCompare(a.date + a.home));

  const upList = $("#pred-upcoming");
  const oldList = $("#pred-played");
  const oldHead = $("#played-toggle");
  const oldCount = $("#played-count");
  const upLabel = $("#upcoming-label");

  if (!upcoming.length && !played.length) {
    upLabel.style.display = "none";
    upList.innerHTML = `<div class="empty">No matches found.</div>`;
    oldHead.style.display = "none"; oldList.innerHTML = ""; return;
  }

  // Upcoming (next match first).
  upLabel.style.display = upcoming.length ? "" : "none";
  if (upcoming.length) {
    const [next, ...rest] = upcoming;
    upList.innerHTML =
      `<div class="next-up">
         <div class="next-badge">⏭️ Next up</div>
         ${upcomingCard(next)}
       </div>` +
      (rest.length ? `<div class="card-grid">${rest.map(upcomingCard).join("")}</div>` : "");
  } else {
    upList.innerHTML = "";
  }

  // Played (collapsible, latest first).
  if (played.length) {
    oldHead.style.display = "";
    oldCount.textContent = played.length;
    oldList.innerHTML = `<div class="card-grid">${played.map(playedCard).join("")}</div>`;
  } else {
    oldHead.style.display = "none";
    oldList.innerHTML = "";
  }
}

function initPredictionsCollapse() {
  const head = $("#played-toggle");
  if (!head) return;
  head.addEventListener("click", () => {
    const open = head.classList.toggle("open");
    $("#pred-played").classList.toggle("collapsed", !open);
  });
}

// ---------- edge: outright (collapsible, sorted by our prob) ----------
function renderOutright() {
  const wrap = $("#outright-table");
  const rows = DATA.outright_edges.filter((r) => r.market_p != null);
  if (!rows.length) { wrap.innerHTML = `<div class="empty">No market comparison available.</div>`; return; }
  wrap.innerHTML = perTeamTable(rows, { evCol: true });
}

// Shared renderer for "per-team Yes-market" tables (winner, rounds, group winner).
function perTeamTable(rows, { evCol = false } = {}) {
  return `
    <table>
      <thead><tr>
        <th>Team</th><th class="num">Model</th><th class="num">Polymarket</th>
        <th class="num">Edge</th>${evCol ? `<th class="num">EV / share</th>` : ""}
      </tr></thead>
      <tbody>
        ${rows.map((r) => `
        <tr>
          <td>${flag(r.team)} ${esc(r.team)}</td>
          <td class="num">${fmtPct(r.model_p)}</td>
          <td class="num">${fmtPct(r.market_p)}</td>
          <td class="num ${(r.edge ?? 0) > 0 ? "edge-pos" : "edge-neg"}">${fmtSign(r.edge)}</td>
          ${evCol ? `<td class="num ${(r.ev ?? 0) > 0 ? "edge-pos" : "edge-neg"}">${fmtSign(r.ev)}</td>` : ""}
        </tr>`).join("")}
      </tbody>
    </table>`;
}

// ---------- edge: per-match 1X2 (dropdown, one match at a time) ----------
// One row per outcome with two side-by-side bars (us vs Polymarket) so the
// agreement/disagreement reads at a glance.
function renderMatchEdge(m) {
  const wrap = $("#match-edges");
  if (!m) { wrap.innerHTML = `<div class="empty">No upcoming match odds on Polymarket right now — they appear before kickoff.</div>`; return; }
  const labelName = { "1": m.home, "X": "Draw", "2": m.away };
  const colorOf = { "1": "var(--home)", "X": "var(--draw)", "2": "var(--away)" };

  const rows = m.outcomes.map((o) => {
    const edgePos = (o.edge ?? 0) > 0;
    return `
    <div class="me-row">
      <div class="me-outcome"><span class="me-dot" style="background:${colorOf[o.label]}"></span>${esc(labelName[o.label] || o.label)}</div>
      <div class="me-bars">
        <div class="me-bar"><span class="me-tag">us</span><div class="me-track"><div class="me-fill us" style="width:${o.model_p}%"></div></div><span class="me-val">${fmtPct(o.model_p)}</span></div>
        <div class="me-bar"><span class="me-tag">poly</span><div class="me-track"><div class="me-fill poly" style="width:${o.market_p}%"></div></div><span class="me-val">${fmtPct(o.market_p)}</span></div>
      </div>
      <div class="me-edge ${edgePos ? "edge-pos" : "edge-neg"}">${fmtSign(o.edge)}</div>
    </div>`;
  }).join("");

  const ev = m.best_ev;
  const valueTone = (ev ?? -1) > 0 ? "edge-pos" : "edge-neg";
  wrap.innerHTML = `
    <div class="me-card">
      <div class="me-head">
        <div class="me-teams">${flag(m.home)} <b>${esc(m.home)}</b> <span class="me-vs">vs</span> <b>${esc(m.away)}</b> ${flag(m.away)}</div>
        <div class="me-date">${esc(m.date)}</div>
      </div>
      <div class="me-legend"><span><span class="me-swatch us"></span> our model</span><span><span class="me-swatch poly"></span> Polymarket</span><span class="me-edge-head">edge = us − poly</span></div>
      ${rows}
      <div class="me-foot">
        Best value for us: <span class="${valueTone}">${fmtSign(ev)} EV</span>
        ${m.source_url ? `· <a href="${esc(m.source_url)}" target="_blank" rel="noopener">view on Polymarket ↗</a>` : ""}
      </div>
    </div>`;
}

// Sort matches chronologically (soonest first) so the next game leads — and so a
// finished game naturally drops to/falls out of the list as the data refreshes.
function _matchEdgesSorted() {
  return (DATA.match_edges || []).slice().sort((a, b) => (a.date + a.home).localeCompare(b.date + b.home));
}

function initMatchEdges() {
  const sel = $("#me-match");
  const rows = _matchEdgesSorted();
  if (!rows.length) { renderMatchEdge(null); if (sel) sel.style.display = "none"; return; }
  sel.innerHTML = rows.map((m, i) =>
    `<option value="${i}">${esc(m.date)} — ${esc(m.home)} vs ${esc(m.away)}</option>`).join("");
  sel.addEventListener("change", () => renderMatchEdge(rows[parseInt(sel.value, 10) || 0]));
  renderMatchEdge(rows[0]);
}

// ---------- edge: round markets (dropdown) ----------
function initRoundMarkets() {
  const sel = $("#me-round");
  const groups = (DATA.market_comparisons && DATA.market_comparisons.rounds) || [];
  const wrap = $("#round-table");
  if (!groups.length) { wrap.innerHTML = `<div class="empty">No advancement markets available.</div>`; if (sel) sel.style.display = "none"; return; }
  sel.innerHTML = groups.map((g, i) => `<option value="${i}">${esc(g.label)}</option>`).join("");
  const render = () => {
    const g = groups[parseInt(sel.value, 10) || 0];
    wrap.innerHTML = g.rows.length ? perTeamTable(g.rows) : `<div class="empty">No data.</div>`;
  };
  sel.addEventListener("change", render);
  render();
}

// ---------- edge: group winner (dropdown) ----------
function initGroupMarkets() {
  const sel = $("#me-group");
  const groups = (DATA.market_comparisons && DATA.market_comparisons.groups) || [];
  const wrap = $("#group-table");
  if (!groups.length) { wrap.innerHTML = `<div class="empty">No group-winner markets available.</div>`; if (sel) sel.style.display = "none"; return; }
  sel.innerHTML = groups.map((g, i) => `<option value="${i}">${esc(g.label)}</option>`).join("");
  const render = () => {
    const g = groups[parseInt(sel.value, 10) || 0];
    wrap.innerHTML = g.rows.length ? perTeamTable(g.rows) : `<div class="empty">No data.</div>`;
  };
  sel.addEventListener("change", render);
  render();
}

// ---------- reliability ----------
function renderLeaderboard() {
  const wrap = $("#leaderboard");
  const models = DATA.reliability.models;
  if (!models.length) { wrap.innerHTML = `<div class="empty">No backtest data available.</div>`; return; }
  wrap.innerHTML = `
    <table>
      <thead><tr>
        <th>Model</th><th class="num">Log loss ↓</th><th class="num">Brier ↓</th>
        <th class="num">RPS ↓</th><th class="num">Calibration err ↓</th>
      </tr></thead>
      <tbody>
        ${models.map((m) => `
        <tr>
          <td>${esc(m.label)}
            ${m.is_ours ? `<span class="badge-ours">ours</span>` : ""}
            ${m.is_best && !m.is_ours ? `<span class="badge-best">best</span>` : ""}
            ${m.is_best && m.is_ours ? `<span class="badge-best">best</span>` : ""}
          </td>
          <td class="num">${fmt(m.log_loss, 4)}</td>
          <td class="num">${fmt(m.brier, 4)}</td>
          <td class="num">${fmt(m.rps, 4)}</td>
          <td class="num">${fmt(m.ece, 4)}</td>
        </tr>`).join("")}
      </tbody>
    </table>`;
}

// ---------- fun: H2H ----------
function teamProbForGroup(team) {
  return DATA.tournament.teams.find((t) => t.team === team);
}

function renderH2H() {
  const a = $("#h2h-a").value, b = $("#h2h-b").value;
  const out = $("#h2h-result");
  if (!a || !b || a === b) { out.innerHTML = `<div class="empty">Pick two different teams.</div>`; return; }
  // Use a match prediction if one exists, else derive a quick Elo-style estimate from champion odds.
  const m = DATA.predictions.find(
    (p) => (p.home === a && p.away === b) || (p.home === b && p.away === a));

  if (m) {
    const swap = m.home === b;
    const pa = swap ? m.p_away : m.p_home;
    const pb = swap ? m.p_home : m.p_away;
    const fav = pa === pb ? null : (pa > pb ? a : b);
    out.innerHTML = `
      <div class="h2h-card">
        <div class="h2h-headline">${fav ? `Model favours <b>${esc(fav)}</b>` : "Too close to call"}
          ${m.home_xg != null ? `· xG ${fmt(swap ? m.away_xg : m.home_xg, 2)}–${fmt(swap ? m.home_xg : m.away_xg, 2)}` : ""}</div>
        <div class="probbar" style="height:30px">
          <span class="p-home" style="width:${pa}%">${esc(a)} ${fmt(pa, 0)}%</span>
          <span class="p-draw" style="width:${m.p_draw}%">X ${fmt(m.p_draw, 0)}%</span>
          <span class="p-away" style="width:${pb}%">${esc(b)} ${fmt(pb, 0)}%</span>
        </div>
        <p class="hint" style="margin-top:10px">This is a real scheduled WC2026 fixture in the model.</p>
      </div>`;
    return;
  }

  // No direct fixture: compare title chances as a fun proxy.
  const ta = teamProbForGroup(a), tb = teamProbForGroup(b);
  const ca = ta?.champion ?? 0, cb = tb?.champion ?? 0;
  const fav = ca === cb ? null : (ca > cb ? a : b);
  out.innerHTML = `
    <div class="h2h-card">
      <div class="h2h-headline">${fav ? `Stronger side: <b>${esc(fav)}</b>` : "Evenly matched"}</div>
      <p class="hint">No direct fixture scheduled — comparing title-winning chances instead.</p>
      <div class="tr-row"><span class="tr-rank">🏆</span><span class="tr-team">${esc(a)}</span>
        <div class="tr-bar-track"><div class="tr-bar" style="width:${Math.min(100, ca * 4)}%"></div></div>
        <span class="tr-val">${fmtPct(ca)}</span></div>
      <div class="tr-row"><span class="tr-rank">🏆</span><span class="tr-team">${esc(b)}</span>
        <div class="tr-bar-track"><div class="tr-bar" style="width:${Math.min(100, cb * 4)}%"></div></div>
        <span class="tr-val">${fmtPct(cb)}</span></div>
    </div>`;
}

function initH2H() {
  const teams = DATA.tournament.teams.map((t) => t.team).sort();
  if (!teams.length) { $(".h2h").style.display = "none"; return; }
  const opts = (sel) => teams.map((t) => `<option ${t === sel ? "selected" : ""}>${esc(t)}</option>`).join("");
  $("#h2h-a").innerHTML = opts(teams[0]);
  $("#h2h-b").innerHTML = opts(teams[1]);
  $("#h2h-a").addEventListener("change", renderH2H);
  $("#h2h-b").addEventListener("change", renderH2H);
  renderH2H();
}

// ---------- fun: title race ----------
function renderTitleRace() {
  const wrap = $("#title-race");
  const teams = DATA.tournament.teams.filter((t) => t.champion != null).slice(0, 16);
  if (!teams.length) { wrap.innerHTML = `<div class="empty">No simulation data.</div>`; return; }
  const max = Math.max(...teams.map((t) => t.champion));
  wrap.innerHTML = teams.map((t, i) => `
    <div class="tr-row">
      <span class="tr-rank">${i + 1}</span>
      <span class="tr-team">${esc(t.team)}</span>
      <div class="tr-bar-track"><div class="tr-bar" style="width:${(t.champion / max) * 100}%"></div></div>
      <span class="tr-val">${fmtPct(t.champion)}</span>
    </div>`).join("");
}

// ---------- track record ----------
function renderTrackRecord() {
  const tr = DATA.track_record;
  const board = $("#track-scoreboard");
  const table = $("#track-table");
  if (!tr || !tr.matches || !tr.matches.length) {
    board.innerHTML = "";
    table.innerHTML = `<div class="empty">No matches scored yet — the head-to-head starts once games are played.</div>`;
    return;
  }
  const s = tr.summary || {};
  const cards = [];
  cards.push(`<div class="stat-card"><div class="big">${s.resolved_matches ?? 0}</div><div class="label">matches scored</div></div>`);
  if (s.model_mean_log_loss != null)
    cards.push(`<div class="stat-card model"><div class="big">${fmt(s.model_mean_log_loss, 3)}</div><div class="label">model avg log loss ↓</div></div>`);
  if (s.market_mean_log_loss != null)
    cards.push(`<div class="stat-card market"><div class="big">${fmt(s.market_mean_log_loss, 3)}</div><div class="label">market avg log loss ↓</div></div>`);
  if (s.model_wins != null)
    cards.push(`<div class="stat-card"><div class="big"><span class="w-model">${s.model_wins}</span>–<span class="w-market">${s.market_wins}</span></div><div class="label">games won head-to-head (model–market)</div></div>`);
  board.innerHTML = cards.join("");

  const resPill = (r) => `<span class="res-pill res-${r}">${{home:"1",draw:"X",away:"2"}[r]}</span>`;
  table.innerHTML = `
    <table>
      <thead><tr>
        <th>Match</th><th>Result</th><th class="num">Model said</th>
        <th class="num">Market said</th><th class="num">Model LL</th>
        <th class="num">Market LL</th><th>Better</th>
      </tr></thead>
      <tbody>
        ${tr.matches.map((m) => `
        <tr>
          <td>${esc(m.home)} v ${esc(m.away)}</td>
          <td>${resPill(m.result)}</td>
          <td class="num">${fmtPct(m.model_p)}</td>
          <td class="num">${fmtPct(m.market_p)}</td>
          <td class="num">${fmt(m.model_ll, 3)}</td>
          <td class="num">${fmt(m.market_ll, 3)}</td>
          <td class="w-${m.winner}">${m.winner === "tie" ? "—" : m.winner}</td>
        </tr>`).join("")}
      </tbody>
    </table>`;
}

// ---------- Monte Carlo simulator ----------
// Lightweight in-browser version of the model's tournament sim: scores come from each
// team's combined rating (Poisson means from the rating gap), groups then knockout.
const SIM = {
  // Expected goals for a team given a rating gap (calibrated to look like the model:
  // ~1.35 avg goals, swinging with the gap). Tuned for plausibility, not exactness.
  lambdas(gap) {
    const base = 1.35;
    const swing = gap / 400; // ~Elo scale
    return [Math.max(0.15, base + 0.55 * swing), Math.max(0.15, base - 0.55 * swing)];
  },
  poisson(lambda) {
    let L = Math.exp(-lambda), k = 0, p = 1;
    do { k++; p *= Math.random(); } while (p > L);
    return k - 1;
  },
  playMatch(a, b, ratings, homeAdv = 0) {
    const gap = (ratings[a] + homeAdv) - ratings[b];
    const [la, lb] = SIM.lambdas(gap);
    return [SIM.poisson(la), SIM.poisson(lb)];
  },
  // Decisive winner (knockout): if drawn, coin-flip weighted slightly by rating.
  knockoutWinner(a, b, ratings) {
    const [ga, gb] = SIM.playMatch(a, b, ratings);
    if (ga !== gb) return ga > gb ? a : b;
    const pa = 1 / (1 + Math.pow(10, (ratings[b] - ratings[a]) / 400));
    return Math.random() < pa ? a : b;
  },
  runOnce(engine) {
    const ratings = {};
    engine.teams.forEach((t) => { ratings[t.team] = t.rating; });
    const groups = {};
    engine.teams.forEach((t) => {
      (groups[t.group] = groups[t.group] || []).push(t.team);
    });

    // Standings seeded with already-played results, then remaining fixtures.
    const table = {};
    engine.teams.forEach((t) => { table[t.team] = { pts: 0, gf: 0, ga: 0, team: t.team, group: t.group }; });
    const applyResult = (h, a, hg, ag) => {
      table[h].gf += hg; table[h].ga += ag; table[a].gf += ag; table[a].ga += hg;
      if (hg > ag) table[h].pts += 3; else if (hg < ag) table[a].pts += 3;
      else { table[h].pts += 1; table[a].pts += 1; }
    };
    (engine.played || []).forEach((m) => applyResult(m.home, m.away, m.home_goals, m.away_goals));
    (engine.remaining_group_fixtures || []).forEach((m) => {
      const [hg, ag] = SIM.playMatch(m.home, m.away, ratings);
      applyResult(m.home, m.away, hg, ag);
    });

    // Rank each group; collect winners, runners-up, and the 8 best third-placed.
    const rankFn = (x, y) => (y.pts - x.pts) || ((y.gf - y.ga) - (x.gf - x.ga)) || (y.gf - x.gf) || (Math.random() - 0.5);
    const winners = {}, runners = {}, thirds = [];
    Object.keys(groups).forEach((g) => {
      const standing = groups[g].map((t) => table[t]).sort(rankFn);
      winners[g] = standing[0].team;
      runners[g] = standing[1].team;
      if (standing[2]) thirds.push(standing[2]);
    });
    const best8thirds = thirds.sort(rankFn).slice(0, 8).map((t) => t.team);

    // Knockout: pool everyone who advanced, pair them, play down to a champion.
    // (Bracket-faithful seeding is approximated by random pairing of qualified teams —
    //  enough for a fun run; the published title odds use the exact bracket.)
    let alive = [...Object.values(winners), ...Object.values(runners), ...best8thirds];
    const path = [];
    while (alive.length > 1) {
      shuffle(alive);
      const next = [];
      for (let i = 0; i < alive.length; i += 2) {
        if (i + 1 >= alive.length) { next.push(alive[i]); continue; }
        next.push(SIM.knockoutWinner(alive[i], alive[i + 1], ratings));
      }
      path.push(`${alive.length} → ${next.length}`);
      alive = next;
    }
    return { champion: alive[0], winners, runners };
  },
};

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function renderSimSingle(res) {
  $("#sim-many").innerHTML = "";
  $("#sim-single").innerHTML = `
    <div class="sim-result">
      <div class="champ">🏆 <b>${esc(res.champion)}</b> wins this simulation!</div>
      <p class="hint">Run it again — the model gives everyone a shot.</p>
    </div>`;
}

function renderSimMany(counts, runs) {
  $("#sim-single").innerHTML = "";
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 12);
  const max = sorted[0][1];
  $("#sim-many").innerHTML = `
    <div class="sim-result">
      <div class="champ">Across <b>${runs.toLocaleString()}</b> of your simulations:</div>
      <div class="title-race" style="margin-top:10px">
        ${sorted.map(([team, n]) => `
          <div class="tr-row">
            <span class="tr-rank">${((n / runs) * 100).toFixed(1)}%</span>
            <span class="tr-team">${esc(team)}</span>
            <div class="tr-bar-track"><div class="tr-bar" style="width:${(n / max) * 100}%"></div></div>
            <span class="tr-val">${n}</span>
          </div>`).join("")}
      </div>
      <p class="hint">Your own Monte Carlo — won't exactly match the published odds (those use the full bracket), but close.</p>
    </div>`;
}

// ---------- Predict a game ----------
function _ratingOf(team) {
  const t = (DATA.sim_engine?.teams || []).find((x) => x.team === team);
  return t ? t.rating : null;
}

function predictGame(match) {
  // Use the model's OFFICIAL calibrated 1/X/2 (precise), and add Monte-Carlo
  // scorelines from the team ratings for colour.
  const ra = _ratingOf(match.home), rb = _ratingOf(match.away);
  const scoreCounts = {};
  let simHome = 0, simDraw = 0, simAway = 0;
  const N = 4000;
  if (ra != null && rb != null) {
    for (let i = 0; i < N; i++) {
      const [hg, ag] = SIM.playMatch(match.home, match.away, { [match.home]: ra, [match.away]: rb }, DATA.sim_engine.home_advantage * (match.neutral ? 0 : 1));
      const key = `${Math.min(hg, 6)}–${Math.min(ag, 6)}`;
      scoreCounts[key] = (scoreCounts[key] || 0) + 1;
      if (hg > ag) simHome++; else if (hg < ag) simAway++; else simDraw++;
    }
  }
  const topScores = Object.entries(scoreCounts)
    .sort((a, b) => b[1] - a[1]).slice(0, 4)
    .map(([s, n]) => ({ score: s, pct: (n / N) * 100 }));
  return { topScores, simN: N };
}

let PG_BUSY = false;

// Build the full result card markup (revealed after the suspense animation).
function predictionCardHTML(match, mc) {
  const probs = { home: match.p_home, draw: match.p_draw, away: match.p_away };
  const pickSide = match.pick;
  const conf = match.confidence;
  const confTier = conf >= 60 ? "high" : conf >= 40 ? "med" : "low";
  const [pl] = pickLabel(pickSide);
  const pickName = pickSide === "home" ? match.home : pickSide === "away" ? match.away : "a draw";

  // Headline = the single most likely scoreline.
  const top = mc.topScores[0];
  const headlineScore = top ? top.score.replace("–", " – ") : "—";
  const rest = mc.topScores.slice(1, 4);

  return `
    <div class="pg-reveal">
      <div class="pg-fixture">
        <span class="pg-side"><span class="pg-flag">${flag(match.home)}</span><span class="pg-name">${esc(match.home)}</span></span>
        <span class="pg-bigscore">${headlineScore}</span>
        <span class="pg-side"><span class="pg-flag">${flag(match.away)}</span><span class="pg-name">${esc(match.away)}</span></span>
      </div>
      <div class="pg-headline-sub">most likely scoreline${top ? ` · <b>${top.pct.toFixed(0)}%</b> of sims` : ""}</div>

      ${rest.length ? `
      <div class="pg-alts">
        <span class="pg-alts-label">other likely results:</span>
        ${rest.map((s) => `<span class="score-chip">${esc(s.score.replace("–"," – "))} <b>${s.pct.toFixed(0)}%</b></span>`).join("")}
      </div>` : ""}

      <div class="probbar" style="height:32px;margin-top:16px">
        <span class="p-home" style="width:${probs.home}%">${probs.home >= 10 ? probs.home + "%" : ""}</span>
        <span class="p-draw" style="width:${probs.draw}%">${probs.draw >= 10 ? "X " + probs.draw + "%" : ""}</span>
        <span class="p-away" style="width:${probs.away}%">${probs.away >= 10 ? probs.away + "%" : ""}</span>
      </div>
      <div class="mc-odds" style="margin-top:6px">
        <span>${esc(match.home)} <b>${fmtPct(probs.home)}</b></span>
        <span>Draw <b>${fmtPct(probs.draw)}</b></span>
        <span>${esc(match.away)} <b>${fmtPct(probs.away)}</b></span>
      </div>
      <div class="pg-call">
        Model leans <span class="pill ${pickLabel(pickSide)[1]}">${pl}</span> <b>${esc(pickName)}</b>
        <span class="conf conf-${confTier}">· ${fmtPct(conf)} confident
          <span class="conf-bar"><span style="width:${conf}%"></span></span>
        </span>
      </div>
      ${match.home_xg != null ? `<div class="mc-xg" style="margin-top:8px">📊 Expected goals: <b>${fmt(match.home_xg,2)}</b> – <b>${fmt(match.away_xg,2)}</b></div>` : ""}
    </div>`;
}

// Suspense: spin a ball, flash random ticking scores, then reveal.
async function renderPrediction(match) {
  if (PG_BUSY) return;
  PG_BUSY = true;
  const out = $("#pg-result");
  const btn = $("#pg-run");
  btn.disabled = true;
  const mc = predictGame(match);

  // Build the "rolling" stage.
  out.innerHTML = `
    <div class="pg-rolling">
      <div class="pg-fixture">
        <span class="pg-side"><span class="pg-flag">${flag(match.home)}</span><span class="pg-name">${esc(match.home)}</span></span>
        <span class="pg-bigscore rolling" id="pg-roll">0 – 0</span>
        <span class="pg-side"><span class="pg-flag">${flag(match.away)}</span><span class="pg-name">${esc(match.away)}</span></span>
      </div>
      <div class="pg-spinner"><span class="pg-ball">⚽</span> simulating ${mc.simN.toLocaleString()} matches…</div>
    </div>`;

  // Flash random plausible scores for ~1.2s, slowing down.
  const roll = $("#pg-roll");
  const pool = mc.topScores.length ? mc.topScores : [{ score: "1–1" }];
  let delay = 55;
  for (let t = 0; t < 22; t++) {
    const s = pool[Math.floor(Math.random() * pool.length)].score;
    if (roll) roll.textContent = s.replace("–", " – ");
    await sleep(delay);
    delay = Math.min(220, delay * 1.12); // ease out
  }

  // Reveal.
  out.innerHTML = predictionCardHTML(match, mc);
  btn.disabled = false;
  PG_BUSY = false;
}

function initPredictGame() {
  const sel = $("#pg-match");
  if (!sel) return;
  // Only fixtures that haven't kicked off yet (timezone-correct via `started`); sorted
  // soonest-first so the next match leads and finished ones fall off automatically.
  const upcoming = DATA.predictions
    .filter((m) => !(m.started || m.played))
    .sort((a, b) => (a.date + a.home).localeCompare(b.date + b.home));
  if (!upcoming.length) { document.querySelectorAll(".sim-box")[1]?.style && (document.querySelectorAll(".sim-box")[1].style.display = "none"); return; }
  sel.innerHTML = upcoming.map((m, i) =>
    `<option value="${i}">${esc(m.date)} — ${esc(m.home)} vs ${esc(m.away)}</option>`).join("");
  const run = () => renderPrediction(upcoming[parseInt(sel.value, 10) || 0]);
  $("#pg-run").addEventListener("click", run);
  // Don't auto-run on dropdown change (would skip the suspense); just clear.
  sel.addEventListener("change", () => { if (!PG_BUSY) $("#pg-result").innerHTML = `<p class="hint">Hit <b>Predict</b> to simulate this match.</p>`; });
  $("#pg-result").innerHTML = `<p class="hint">Pick a match and hit <b>Predict</b> ⚽</p>`;
}

function initSimulator() {
  const engine = DATA.sim_engine;
  const box = document.querySelector(".sim-box");
  if (!engine || !engine.teams || !engine.teams.length) { if (box) box.style.display = "none"; return; }
  $("#sim-run").addEventListener("click", () => {
    $("#sim-status").textContent = "";
    renderSimSingle(SIM.runOnce(engine));
  });
  $("#sim-run-many").addEventListener("click", () => {
    const runs = 1000;
    $("#sim-status").textContent = "crunching…";
    setTimeout(() => {
      const counts = {};
      for (let i = 0; i < runs; i++) {
        const c = SIM.runOnce(engine).champion;
        counts[c] = (counts[c] || 0) + 1;
      }
      $("#sim-status").textContent = "";
      renderSimMany(counts, runs);
    }, 20);
  });
}

// ---------- boot ----------
async function boot() {
  initTabs();
  try {
    const res = await fetch("data.json");
    DATA = await res.json();
  } catch (e) {
    document.querySelector("main").innerHTML =
      `<div class="empty">Could not load <b>data.json</b>. Run <code>python scripts/build_web_dashboard.py</code> and serve this folder over http (not file://).</div>`;
    return;
  }
  $("#meta").innerHTML = `Model: ${esc(DATA.meta.model)}<br/>Updated ${esc(DATA.meta.generated_at)}`;

  renderPredictions();
  initPredictionsCollapse();
  $("#pred-search").addEventListener("input", renderPredictions);
  $("#pred-sort").addEventListener("change", renderPredictions);

  renderOutright();
  initMatchEdges();
  initRoundMarkets();
  initGroupMarkets();
  const ot = $("#outright-toggle");
  if (ot) ot.addEventListener("click", () => {
    const open = ot.classList.toggle("open");
    $("#outright-table").classList.toggle("collapsed", !open);
  });
  renderLeaderboard();
  renderTrackRecord();
  initSimulator();
  initPredictGame();
  initH2H();
  renderTitleRace();
}

boot();
