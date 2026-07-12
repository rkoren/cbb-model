"""Render the DASH-007 dashboard payload into one self-contained HTML string.

Pure function: ``render_html(payload) -> str``. No I/O, no file writes — the caller (the build
script) decides where to write it. The output embeds the payload as a single JSON blob and inlines
all CSS/JS (no CDN), so the file works offline and is shareable like ``monitoring/drift.html`` (the
DASH-007 decision).

Interaction shape (DASH-004): one ``activeDate`` clock drives two linked table views — the
FanMatch slate (DASH-002), showing each game's KenPom score, our score, the final, and the edge;
and the ratings view (DASH-003), showing the latest weekly snapshot on-or-before the clock. Both
re-render from one ``setDate`` call. ISO date strings sort chronologically, so the "latest ≤ clock"
lookup is a plain string comparison.
"""

from __future__ import annotations

import json
from typing import Any

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CBB Handicapper — us vs KenPom</title>
<style>
  :root {
    --bg:#fff; --panel:#fff; --fg:#0f172a; --muted:#64748b; --faint:#94a3b8;
    --line:#e5e7eb; --head:#f8fafc; --row-hover:#f8fafc;
    --pos:#059669; --neg:#dc2626; --accent:#4f46e5; --accent-soft:#eef2ff;
    --s1:#2a78d6; --s2:#1baf7a;   /* trajectory series: ours / theirs (validated CVD-safe pair) */
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0b1120; --panel:#0f172a; --fg:#e5e9f0; --muted:#94a3b8; --faint:#64748b;
      --line:#1e293b; --head:#131c2e; --row-hover:#131c2e; --accent:#818cf8; --accent-soft:#1e1b4b;
      --s1:#3987e5; --s2:#199e70; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif; -webkit-font-smoothing: antialiased; }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 1.75rem 1.25rem 4rem; }
  header { border-bottom: 1px solid var(--line); padding-bottom: 1rem; margin-bottom: 1.25rem; }
  h1 { font-size: 1.35rem; margin: 0; letter-spacing: -0.01em; }
  #meta { color: var(--muted); font-size: .82rem; margin-top: .3rem; }
  #controls { display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin-bottom: 1rem; }
  button, select { font: inherit; font-size: .85rem; padding: .35rem .6rem; border: 1px solid var(--line);
    border-radius: 8px; background: var(--panel); color: var(--fg); cursor: pointer; transition: .12s; }
  button:hover, select:hover { border-color: var(--accent); }
  #clock { font-weight: 650; min-width: 7.5rem; text-align: center; border: none; background: none;
    font-variant-numeric: tabular-nums; font-size: .95rem; }
  .tabs { margin-left: auto; display: flex; gap: .3rem; }
  button.tab { border-radius: 999px; padding: .35rem .9rem; }
  button.tab[aria-selected="true"] { background: var(--accent); color: #fff; border-color: var(--accent); }
  .scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }
  table { border-collapse: collapse; width: 100%; font-size: .82rem; font-variant-numeric: tabular-nums; }
  thead th { background: var(--head); color: var(--muted); font-weight: 600; font-size: .68rem;
    text-transform: uppercase; letter-spacing: .03em; text-align: right; padding: .5rem .5rem;
    position: sticky; top: 0; border-bottom: 1px solid var(--line); }
  thead th .sub { display: block; font-size: .6rem; font-weight: 500; text-transform: none;
    letter-spacing: 0; color: var(--faint); }
  tbody td { text-align: right; padding: .42rem .5rem; border-bottom: 1px solid var(--line); white-space: nowrap; }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: var(--row-hover); }
  th.l, td.l { text-align: left; }
  td.matchup { font-weight: 550; white-space: normal; line-height: 1.25; min-width: 8.5rem; max-width: 14rem; }
  .score b { font-weight: 700; }
  .wabbr { color: var(--accent); font-weight: 650; font-size: .72rem; margin-right: .1rem; }
  .wp { color: var(--muted); font-size: .74rem; margin-left: .12rem; }
  .badge { display: inline-block; min-width: 1.15rem; text-align: center; font-size: .65rem; font-weight: 700;
    padding: .05rem .3rem; border-radius: 5px; background: var(--accent-soft); color: var(--accent); margin-right: .45rem; }
  .pos { color: var(--pos); font-weight: 600; } .neg { color: var(--neg); font-weight: 600; }
  .muted { color: var(--faint); }
  .summary { margin: .8rem 0 .2rem; padding: .65rem .9rem; border: 1px solid var(--line);
    border-radius: 10px; background: var(--head); font-size: .82rem; line-height: 1.75; }
  .summary .lbl { color: var(--muted); }
  .good { color: var(--pos); font-weight: 650; } .bad { color: var(--neg); }
  .legend { color: var(--muted); font-size: .76rem; margin: .7rem .2rem 0; line-height: 1.5; }
  .empty { color: var(--muted); padding: 1.25rem; text-align: center; }
  .snapnote { color: var(--muted); font-size: .78rem; margin: 0 .2rem .5rem; }
  .mgroup { font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);
    margin: 1.4rem .2rem .5rem; font-weight: 700; }
  .mgroup:first-of-type { margin-top: .3rem; }
  .vs { color: var(--faint); }
  tr.clickable { cursor: pointer; }
  tr.clickable:hover td:first-child { box-shadow: inset 2px 0 0 var(--accent); }
  #detail { position: fixed; inset: 0; background: rgba(2,6,23,.55); display: flex;
    align-items: center; justify-content: center; padding: 1.5rem; z-index: 20; }
  #detail.hidden { display: none; }   /* ID beats .hidden's class selector, so override explicitly */
  #detailCard { background: var(--bg); border: 1px solid var(--line); border-radius: 14px;
    max-width: 760px; width: 100%; max-height: 88vh; overflow-y: auto; padding: 1.2rem 1.4rem; }
  #detailCard h2 { font-size: 1.05rem; margin: 0 0 .1rem; }
  #detailCard .close { float: right; border: 1px solid var(--line); border-radius: 8px;
    background: var(--panel); color: var(--fg); cursor: pointer; padding: .2rem .55rem; font-size: 1rem; }
  .chartgrid { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; margin-top: .8rem; }
  .chart { border: 1px solid var(--line); border-radius: 10px; padding: .3rem; }
  .chart svg { display: block; width: 100%; height: auto; }
  .clabel { fill: var(--muted); font-size: 10px; }
  .ctitle { fill: var(--fg); font-size: 11px; font-weight: 600; }
  .legend2 { display: flex; gap: 1rem; font-size: .78rem; margin-top: .3rem; color: var(--muted); }
  .legend2 i { display: inline-block; width: .8rem; height: .18rem; border-radius: 2px; vertical-align: middle; margin-right: .3rem; }
  .hidden { display: none; }
</style>
</head>
<body>
  <div class="wrap">
  <header>
    <h1>CBB Handicapper</h1>
    <div id="meta"></div>
  </header>
  <div id="controls">
    <button id="prevWk" title="back one week">« wk</button>
    <button id="prevDay" title="back one day">‹ day</button>
    <span id="clock">—</span>
    <button id="nextDay" title="forward one day">day ›</button>
    <button id="nextWk" title="forward one week">wk »</button>
    <select id="datePick" title="jump to date"></select>
    <select id="gender" title="filter by gender">
      <option value="all">All</option><option value="M">Men</option><option value="W">Women</option></select>
    <span class="tabs">
      <button class="tab" id="tabSlate" aria-selected="true">FanMatch</button>
      <button class="tab" id="tabRatings" aria-selected="false">Ratings</button>
      <button class="tab" id="tabMetrics" aria-selected="false">Metrics</button>
    </span>
  </div>
  <div id="slateView"></div>
  <div id="ratingsView" class="hidden"></div>
  <div id="metricsView" class="hidden"></div>
  </div>
  <div id="detail" class="hidden"><div id="detailCard"></div></div>

<script>
const DATA = __DATA__;
let activeDate = DATA.slate_dates.length ? DATA.slate_dates[DATA.slate_dates.length - 1] : null;
let gender = "all";
let view = "slate";

const $ = (id) => document.getElementById(id);
const DASH = "\\u2013";
const attr = (s) => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const num = (v, d) => (v === null || v === undefined) ? null : (typeof v === "number" ? v.toFixed(d) : v);
const em = (v, d) => { const s = num(v, d); return s === null ? '<span class="muted">\\u2014</span>' : s; };
const signed = (v, d) => {
  if (v === null || v === undefined) return '<span class="muted">\\u2014</span>';
  const c = v > 0 ? "pos" : (v < 0 ? "neg" : "");
  return '<span class="' + c + '">' + (v > 0 ? "+" : "") + v.toFixed(d) + "</span>";
};
// "SF 78–70 · 62%": abbreviated winner, then the score (winning side bolded), then optional win%.
function scoreCell(p, aAbbr, bAbbr, withProb) {
  if (!p || p.a === null || p.a === undefined) return '<span class="muted">\\u2014</span>';
  const win = p.a >= p.b ? aAbbr : bAbbr;
  const a = p.a >= p.b ? "<b>" + p.a + "</b>" : "" + p.a;
  const b = p.b > p.a ? "<b>" + p.b + "</b>" : "" + p.b;
  let s = '<span class="wabbr">' + win + '</span> <span class="score">' + a + DASH + b + "</span>";
  if (withProb && p.prob !== null && p.prob !== undefined)
    s += '<span class="wp">' + Math.round(p.prob * 100) + "%</span>";
  return s;
}

function latestRatingOnOrBefore(date) {
  let best = null;
  for (const rd of DATA.rating_dates) { if (rd <= date) best = rd; else break; }
  return best;
}
function stepDay(dir) {
  const i = DATA.slate_dates.indexOf(activeDate);
  const j = Math.min(Math.max(i + dir, 0), DATA.slate_dates.length - 1);
  if (j >= 0) setDate(DATA.slate_dates[j]);
}
function stepWeek(dir) { for (let k = 0; k < 7; k++) stepDay(dir); }

// Day accuracy vs KenPom: over completed men's games (KenPom + a final), how far each model's
// predicted total and margin landed from the final — signed avg (+/- vs final) and abs error.
function daySummary(games) {
  const g = games.filter(x => x.kp && x.kp.a !== null && x.actual.a !== null);
  if (!g.length) return "";
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const tot = (p) => p.a + p.b;
  const stat = (pick) => { const err = g.map(pick); return { pm: mean(err), mae: mean(err.map(Math.abs)) }; };
  const ourT = stat(x => tot(x.our) - tot(x.actual)), kpT = stat(x => tot(x.kp) - tot(x.actual));
  const ourM = stat(x => x.our.margin - x.actual.margin), kpM = stat(x => x.kp.margin - x.actual.margin);
  const sgn = (v) => (v > 0 ? "+" : "") + v.toFixed(1);
  const cell = (mine, theirs) => `<span class="${Math.abs(mine.mae) <= Math.abs(theirs.mae) ? "good" : "bad"}">${sgn(mine.pm)} <span class="wp">\\u00b1${mine.mae.toFixed(1)}</span></span>`;
  return `<div class="summary">
    <span class="lbl">This day \\u00b7 ${g.length} men's game${g.length === 1 ? "" : "s"} vs KenPom \\u00b7 avg predicted \\u2212 final (\\u00b1 = avg miss):</span><br>
    <span class="lbl">Total points:</span> us ${cell(ourT, kpT)} &nbsp; vs &nbsp; KenPom ${sgn(kpT.pm)} <span class="wp">\\u00b1${kpT.mae.toFixed(1)}</span><br>
    <span class="lbl">Margin:</span> us ${cell(ourM, kpM)} &nbsp; vs &nbsp; KenPom ${sgn(kpM.pm)} <span class="wp">\\u00b1${kpM.mae.toFixed(1)}</span>
  </div>`;
}

function renderSlate() {
  const games = (DATA.slate[activeDate] || []).filter(g => gender === "all" || g.gender === gender);
  if (!games.length) { $("slateView").innerHTML = '<p class="empty">No games on this date.</p>'; return; }
  const rows = games.map(g => `<tr>
    <td class="l matchup"><span class="badge">${g.gender}</span>${g.a} vs ${g.b}</td>
    <td>${scoreCell(g.kp, g.a_abbr, g.b_abbr, true)}</td>
    <td>${scoreCell(g.our, g.a_abbr, g.b_abbr, true)}</td>
    <td>${scoreCell(g.actual, g.a_abbr, g.b_abbr, false)}</td>
    <td>${signed(g.gap_margin, 1)}</td>
    <td>${signed(g.gap_total, 1)}</td></tr>`).join("");
  $("slateView").innerHTML = `<div class="scroll"><table>
    <thead><tr>
      <th class="l">Matchup</th>
      <th>KenPom <span class="sub">winner · score · win%</span></th>
      <th>Our model <span class="sub">winner · score · win%</span></th>
      <th>Final <span class="sub">winner · score</span></th>
      <th>Spread \\u0394 <span class="sub">us \\u2212 KP</span></th>
      <th>Total \\u0394 <span class="sub">us \\u2212 KP</span></th>
    </tr></thead><tbody>${rows}</tbody></table></div>
    ${daySummary(games)}
    <p class="legend"><b>Bold</b> / the leading abbreviation = predicted/actual winner \\u00b7
    <b>Spread \\u0394</b> = our margin \\u2212 KenPom's, <b>Total \\u0394</b> = our total \\u2212 KenPom's
    (rows sorted by the biggest spread disagreement).
    In the day summary, <span class="good">green</span> marks the model closer to the final (smaller avg miss).
    KenPom FanMatch covers men's games only \\u2014 women show \\u2014.</p>`;
}

function renderRatings() {
  const rd = latestRatingOnOrBefore(activeDate);
  const all = rd ? DATA.ratings[rd] : null;
  const teams = all ? all.filter(t => gender === "all" || t.gender === gender) : null;
  if (!teams || !teams.length) { $("ratingsView").innerHTML = '<p class="empty">No ratings snapshot on or before this date.</p>'; return; }
  const rows = teams.map(t => `<tr class="clickable" data-team="${attr(t.team)}" data-gen="${t.gender}">
    <td>${em(t.our.rank, 0)}</td><td><span class="badge">${t.gender}</span></td><td class="l matchup">${t.team}</td>
    <td>${em(t.our.em, 1)}</td><td>${em(t.our.oe, 1)}</td><td>${em(t.our.de, 1)}</td><td>${em(t.our.tempo, 1)}</td>
    <td>${em(t.kp.em, 1)}</td><td>${em(t.kp.rank, 0)}</td>
    <td>${signed(t.d_em, 1)}</td><td>${signed(t.d_rank, 0)}</td></tr>`).join("");
  $("ratingsView").innerHTML = `<p class="snapnote">Ratings as of ${rd} (latest snapshot \\u2264 ${activeDate}) \\u2014 our efficiency vs KenPom (men) / Torvik (women), ranked within gender. \\u0394 = us \\u2212 them. Click a team for its season trajectory.</p>
    <div class="scroll"><table class="ratings"><thead><tr>
      <th>Rk</th><th></th><th class="l">Team</th>
      <th>Our EM</th><th>OE</th><th>DE</th><th>Tempo</th>
      <th>Their EM <span class="sub">KP/Tvk</span></th><th>Their #</th>
      <th>\\u0394 EM</th><th>\\u0394 #</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

// Season accuracy vs the market (DASH-005): season-wide, independent of the date clock.
function renderMetrics() {
  const m = DATA.metrics;
  if (!m || !m.n_games) { $("metricsView").innerHTML = '<p class="empty">No KenPom-covered games to score.</p>'; return; }
  const cell = (u, k, digits, lowerBetter) => {
    if (u === null || u === undefined) return '<span class="muted">\\u2014</span>';
    let c = "";
    if (k !== null && k !== undefined && u !== k) c = (lowerBetter ? u < k : u > k) ? "good" : "bad";
    const kp = (k === null || k === undefined) ? "\\u2014" : k.toFixed(digits);
    return '<span class="' + c + '">' + u.toFixed(digits) + '</span> <span class="vs">/ ' + kp + "</span>";
  };
  const pct = (v) => v === null || v === undefined ? null : v * 100;
  let html = '<p class="snapnote">Season accuracy vs KenPom FanMatch \\u2014 men\\'s games with a market line ('
    + m.n_games + ' games). Each cell is <b>us</b> / KenPom; <span class="good">green</span> = we\\'re closer.</p>';
  for (const g of m.groups) {
    const rows = g.rows.map(r => `<tr>
      <td class="l">${r.label}</td><td>${r.n}</td>
      <td>${cell(r.us.margin, r.kp.margin, 1, true)}</td>
      <td>${cell(r.us.total, r.kp.total, 1, true)}</td>
      <td>${cell(r.us.brier, r.kp.brier, 3, true)}</td>
      <td>${cell(pct(r.us.acc), pct(r.kp.acc), 0, false)}</td></tr>`).join("");
    html += `<div class="mgroup">${g.name}</div><div class="scroll"><table>
      <thead><tr><th class="l">Segment</th><th>Games</th>
      <th>Margin MAE <span class="sub">us / KP</span></th>
      <th>Total MAE <span class="sub">us / KP</span></th>
      <th>Brier <span class="sub">us / KP</span></th>
      <th>Win% <span class="sub">us / KP</span></th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
  }
  $("metricsView").innerHTML = html;
}

// DASH-006 team trajectory: a small inline-SVG line chart, ours (s1) vs theirs (s2).
function lineChart(title, dates, us, them) {
  const w = 340, h = 150, pad = { t: 22, r: 12, b: 20, l: 34 };
  const all = us.concat(them).filter(v => v !== null && v !== undefined);
  if (!all.length) return "";
  let mn = Math.min(...all), mx = Math.max(...all);
  const r = (mx - mn) || 1; mn -= r * 0.12; mx += r * 0.12;
  const n = dates.length;
  const X = i => pad.l + (n < 2 ? (w - pad.l - pad.r) / 2 : (i / (n - 1)) * (w - pad.l - pad.r));
  const Y = v => pad.t + (1 - (v - mn) / (mx - mn)) * (h - pad.t - pad.b);
  const draw = (arr, col) => {
    let d = "", started = false, dots = "";
    arr.forEach((v, i) => {
      if (v === null || v === undefined) return;
      d += (started ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1) + " "; started = true;
      dots += '<circle cx="' + X(i).toFixed(1) + '" cy="' + Y(v).toFixed(1) + '" r="2.2" fill="'
        + col + '"><title>' + dates[i] + ": " + v.toFixed(1) + "</title></circle>";
    });
    return (d ? '<path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="2" stroke-linejoin="round"/>' : "") + dots;
  };
  const grid = [mn, (mn + mx) / 2, mx].map(v =>
    '<line x1="' + pad.l + '" x2="' + (w - pad.r) + '" y1="' + Y(v).toFixed(1) + '" y2="' + Y(v).toFixed(1) + '" stroke="var(--line)"/>'
    + '<text class="clabel" x="' + (pad.l - 4) + '" y="' + (Y(v) + 3).toFixed(1) + '" text-anchor="end">' + v.toFixed(0) + "</text>").join("");
  const xlab = '<text class="clabel" x="' + pad.l + '" y="' + (h - 5) + '">' + dates[0].slice(5) + "</text>"
    + '<text class="clabel" x="' + (w - pad.r) + '" y="' + (h - 5) + '" text-anchor="end">' + dates[n - 1].slice(5) + "</text>";
  return '<div class="chart"><svg viewBox="0 0 ' + w + " " + h + '" role="img" aria-label="' + title + '">'
    + '<text class="ctitle" x="' + pad.l + '" y="13">' + title + "</text>"
    + grid + xlab + draw(us, "var(--s1)") + draw(them, "var(--s2)") + "</svg></div>";
}

function showTrajectory(team, gen) {
  // Build the team's series from the ratings already embedded in the payload — no extra data.
  const dates = DATA.rating_dates.filter(d => (DATA.ratings[d] || []).some(t => t.team === team));
  const ser = (side, m) => dates.map(d => {
    const t = (DATA.ratings[d] || []).find(x => x.team === team); return t && t[side] ? t[side][m] : null;
  });
  const cmpName = gen === "W" ? "Torvik" : "KenPom";
  const charts = [["AdjEM", "em"], ["Off (AdjOE)", "oe"], ["Def (AdjDE)", "de"], ["Tempo", "tempo"]]
    .map(([lbl, m]) => lineChart(lbl, dates, ser("our", m), ser("kp", m))).join("");
  $("detailCard").innerHTML = '<button class="close" onclick="closeDetail()" title="close">\\u00d7</button>'
    + "<h2>" + team + "</h2>"
    + '<div class="snapnote">Rating trajectory across the season \\u2014 ours vs ' + cmpName + ".</div>"
    + '<div class="legend2"><span><i style="background:var(--s1)"></i>ours</span>'
    + '<span><i style="background:var(--s2)"></i>' + cmpName + "</span></div>"
    + '<div class="chartgrid">' + charts + "</div>";
  $("detail").classList.remove("hidden");
}
function closeDetail() { $("detail").classList.add("hidden"); }

function setDate(d) { activeDate = d; $("clock").textContent = d || "\\u2014"; $("datePick").value = d; renderSlate(); renderRatings(); }
function setView(v) {
  view = v;
  $("tabSlate").setAttribute("aria-selected", v === "slate");
  $("tabRatings").setAttribute("aria-selected", v === "ratings");
  $("tabMetrics").setAttribute("aria-selected", v === "metrics");
  $("slateView").classList.toggle("hidden", v !== "slate");
  $("ratingsView").classList.toggle("hidden", v !== "ratings");
  $("metricsView").classList.toggle("hidden", v !== "metrics");
}

function init() {
  $("meta").textContent = "Our model vs KenPom FanMatch across the season \\u00b7 "
    + DATA.meta.n_games.toLocaleString() + " games \\u00b7 " + DATA.meta.n_snapshots + " rating snapshots"
    + (DATA.meta.generated ? " \\u00b7 built " + DATA.meta.generated : "");
  $("datePick").innerHTML = DATA.slate_dates.map(d => '<option value="' + d + '">' + d + "</option>").join("");
  $("prevDay").onclick = () => stepDay(-1); $("nextDay").onclick = () => stepDay(1);
  $("prevWk").onclick = () => stepWeek(-1); $("nextWk").onclick = () => stepWeek(1);
  $("datePick").onchange = (e) => setDate(e.target.value);
  $("gender").onchange = (e) => { gender = e.target.value; renderSlate(); renderRatings(); };
  $("tabSlate").onclick = () => setView("slate"); $("tabRatings").onclick = () => setView("ratings");
  $("tabMetrics").onclick = () => setView("metrics");
  $("detail").onclick = (e) => { if (e.target.id === "detail") closeDetail(); };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDetail(); });
  $("ratingsView").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-team]");
    if (tr) showTrajectory(tr.dataset.team, tr.dataset.gen);
  });
  renderMetrics();
  setView("slate");
  if (activeDate) setDate(activeDate); else $("slateView").innerHTML = '<p class="empty">No data.</p>';
}
init();
</script>
</body>
</html>
"""


def render_html(payload: dict[str, Any]) -> str:
    """Render the payload from :func:`cbb.dashboard.payload.build_payload` into standalone HTML."""
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    # Guard the closing-tag sequence so an embedded string can never break out of the <script>.
    data = data.replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", data)
