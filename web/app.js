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

function renderPredictions() {
  const list = $("#pred-list");
  const q = $("#pred-search").value.trim().toLowerCase();
  const sort = $("#pred-sort").value;
  let items = DATA.predictions.slice();

  if (q) items = items.filter((m) => (m.home + " " + m.away).toLowerCase().includes(q));

  if (sort === "confidence") items.sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  else if (sort === "upset") items.sort((a, b) => (a.confidence ?? 0) - (b.confidence ?? 0));
  else items.sort((a, b) => (a.date + a.home).localeCompare(b.date + b.home));

  if (!items.length) { list.innerHTML = `<div class="empty">No matches found.</div>`; return; }

  list.innerHTML = items.map((m) => {
    const [pl, pc] = pickLabel(m.pick);
    const confTier = m.confidence >= 60 ? "high" : m.confidence >= 40 ? "med" : "low";

    if (m.played) {
      // Final score + whether the model's pick was right.
      const ok = m.pick_correct;
      return `
      <div class="match-card played">
        <div class="mc-top">
          <span>${esc(m.date)} · full time</span>
          <span class="${ok ? "tick" : "cross"}">${ok ? "✓ model got it" : "✗ model missed"}</span>
        </div>
        <div class="mc-teams">
          <span class="home ${m.result_side === "home" ? "won" : ""}">${esc(m.home)}</span>
          <span class="mc-score">${m.result_home_goals}–${m.result_away_goals}</span>
          <span class="away ${m.result_side === "away" ? "won" : ""}">${esc(m.away)}</span>
        </div>
        <div class="mc-recap">
          Model had picked <span class="pill ${pc}">${pl}</span> at ${fmtPct(m.confidence)} confidence.
        </div>
      </div>`;
    }

    const xg = (m.home_xg != null && m.away_xg != null)
      ? `📊 Expected goals: <b>${fmt(m.home_xg, 2)}</b> – <b>${fmt(m.away_xg, 2)}</b>` : "";
    return `
    <div class="match-card">
      <div class="mc-top">
        <span>${esc(m.date)}</span>
        <span>${m.neutral ? "neutral venue" : esc(m.city)}</span>
      </div>
      <div class="mc-teams">
        <span class="home">${esc(m.home)}</span>
        <span class="mc-vs">vs</span>
        <span class="away">${esc(m.away)}</span>
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
    </div>`;
  }).join("");
}

// ---------- edge: outright ----------
function renderOutright() {
  const wrap = $("#outright-table");
  const rows = DATA.outright_edges.filter((r) => r.market_p != null);
  if (!rows.length) { wrap.innerHTML = `<div class="empty">No market comparison available.</div>`; return; }
  wrap.innerHTML = `
    <table>
      <thead><tr>
        <th>Team</th><th class="num">Model</th><th class="num">Market</th>
        <th class="num">Edge</th><th class="num">EV / share</th>
      </tr></thead>
      <tbody>
        ${rows.map((r) => `
        <tr>
          <td>${esc(r.team)}</td>
          <td class="num">${fmtPct(r.model_p)}</td>
          <td class="num">${fmtPct(r.market_p)}</td>
          <td class="num ${r.edge > 0 ? "edge-pos" : "edge-neg"}">${fmtSign(r.edge)}</td>
          <td class="num ${r.ev > 0 ? "edge-pos" : "edge-neg"}">${fmtSign(r.ev)}</td>
        </tr>`).join("")}
      </tbody>
    </table>`;
}

// ---------- edge: match cards ----------
function renderMatchEdges() {
  const wrap = $("#match-edges");
  const rows = DATA.match_edges;
  if (!rows.length) { wrap.innerHTML = `<div class="empty">No bookmaker match prices on file yet.</div>`; return; }
  wrap.innerHTML = rows.map((m) => {
    const hasValue = (m.best_ev ?? -1) > 0;
    const outcomeRows = m.outcomes.map((o) => `
      <div class="ec-row">
        <span class="o-label">${o.label}</span>
        <span>model <b>${fmtPct(o.model_p)}</b></span>
        <span>mkt ${fmtPct(o.market_p)} @ ${fmtOdds(o.book_odds)}</span>
        <span class="o-edge ${o.edge > 0 ? "edge-pos" : "edge-neg"}">${fmtSign(o.edge)}</span>
      </div>`).join("");
    return `
    <div class="edge-card ${hasValue ? "has-value" : ""}">
      <div class="ec-head">
        <span class="teams">${esc(m.home)} vs ${esc(m.away)}</span>
        <span class="book">${esc(m.bookmaker)}</span>
      </div>
      ${outcomeRows}
      <div style="margin-top:10px;font-size:13px">
        Best value: <span class="${hasValue ? "edge-pos" : "edge-neg"}">${fmtSign(m.best_ev)} EV</span>
        ${m.source_url ? `· <a href="${esc(m.source_url)}" target="_blank" rel="noopener" style="color:var(--accent)">source</a>` : ""}
      </div>
    </div>`;
  }).join("");
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
  $("#pred-search").addEventListener("input", renderPredictions);
  $("#pred-sort").addEventListener("change", renderPredictions);

  renderOutright();
  renderMatchEdges();
  renderLeaderboard();
  renderTrackRecord();
  initSimulator();
  initH2H();
  renderTitleRace();
}

boot();
