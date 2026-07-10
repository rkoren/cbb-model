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
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0b1120; --panel:#0f172a; --fg:#e5e9f0; --muted:#94a3b8; --faint:#64748b;
      --line:#1e293b; --head:#131c2e; --row-hover:#131c2e; --accent:#818cf8; --accent-soft:#1e1b4b; }
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
  table { border-collapse: collapse; width: 100%; font-size: .875rem; font-variant-numeric: tabular-nums; }
  thead th { background: var(--head); color: var(--muted); font-weight: 600; font-size: .7rem;
    text-transform: uppercase; letter-spacing: .04em; text-align: right; padding: .55rem .7rem;
    position: sticky; top: 0; border-bottom: 1px solid var(--line); }
  thead th .sub { display: block; font-size: .62rem; font-weight: 500; text-transform: none;
    letter-spacing: 0; color: var(--faint); }
  tbody td { text-align: right; padding: .5rem .7rem; border-bottom: 1px solid var(--line); white-space: nowrap; }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: var(--row-hover); }
  th.l, td.l { text-align: left; }
  td.matchup { font-weight: 550; }
  .score b { font-weight: 700; }
  .wp { color: var(--muted); font-size: .78rem; margin-left: .15rem; }
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
    </span>
  </div>
  <div id="slateView"></div>
  <div id="ratingsView" class="hidden"></div>
  </div>

<script>
const DATA = __DATA__;
let activeDate = DATA.slate_dates.length ? DATA.slate_dates[DATA.slate_dates.length - 1] : null;
let gender = "all";
let view = "slate";

const $ = (id) => document.getElementById(id);
const DASH = "\\u2013";
const num = (v, d) => (v === null || v === undefined) ? null : (typeof v === "number" ? v.toFixed(d) : v);
const em = (v, d) => { const s = num(v, d); return s === null ? '<span class="muted">\\u2014</span>' : s; };
const signed = (v, d) => {
  if (v === null || v === undefined) return '<span class="muted">\\u2014</span>';
  const c = v > 0 ? "pos" : (v < 0 ? "neg" : "");
  return '<span class="' + c + '">' + (v > 0 ? "+" : "") + v.toFixed(d) + "</span>";
};
// "78–70 · 62%" with the winning score bolded; win prob optional.
function scoreCell(p, withProb) {
  if (!p || p.a === null || p.a === undefined) return '<span class="muted">\\u2014</span>';
  const a = p.a >= p.b ? "<b>" + p.a + "</b>" : "" + p.a;
  const b = p.b > p.a ? "<b>" + p.b + "</b>" : "" + p.b;
  let s = '<span class="score">' + a + DASH + b + "</span>";
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
    <td>${scoreCell(g.kp, true)}</td>
    <td>${scoreCell(g.our, true)}</td>
    <td>${scoreCell(g.actual, false)}</td>
    <td>${signed(g.gap_margin, 1)}</td></tr>`).join("");
  $("slateView").innerHTML = `<div class="scroll"><table>
    <thead><tr>
      <th class="l">Matchup</th>
      <th>KenPom <span class="sub">score · win%</span></th>
      <th>Our model <span class="sub">score · win%</span></th>
      <th>Final <span class="sub">score</span></th>
      <th>Edge <span class="sub">us \\u2212 KP margin</span></th>
    </tr></thead><tbody>${rows}</tbody></table></div>
    ${daySummary(games)}
    <p class="legend"><b>Bold</b> = predicted/actual winner \\u00b7 <b>Edge</b> is how many points more
    our model favors team A than KenPom does; rows are sorted by the biggest disagreement.
    In the day summary, <span class="good">green</span> marks the model closer to the final (smaller avg miss).
    KenPom FanMatch covers men's games only \\u2014 women show \\u2014.</p>`;
}

function renderRatings() {
  const rd = latestRatingOnOrBefore(activeDate);
  const all = rd ? DATA.ratings[rd] : null;
  const teams = all ? all.filter(t => gender === "all" || t.gender === gender) : null;
  if (!teams || !teams.length) { $("ratingsView").innerHTML = '<p class="empty">No ratings snapshot on or before this date.</p>'; return; }
  const rows = teams.map(t => `<tr>
    <td>${em(t.our.rank, 0)}</td><td><span class="badge">${t.gender}</span></td><td class="l matchup">${t.team}</td>
    <td>${em(t.our.em, 1)}</td><td>${em(t.our.oe, 1)}</td><td>${em(t.our.de, 1)}</td><td>${em(t.our.tempo, 1)}</td>
    <td>${em(t.kp.em, 1)}</td><td>${em(t.kp.rank, 0)}</td>
    <td>${signed(t.d_em, 1)}</td><td>${signed(t.d_rank, 0)}</td></tr>`).join("");
  $("ratingsView").innerHTML = `<p class="snapnote">Ratings as of ${rd} (latest snapshot \\u2264 ${activeDate}) \\u2014 our efficiency vs KenPom (men) / Torvik (women), ranked within gender.</p>
    <div class="scroll"><table><thead><tr>
      <th>Rk</th><th></th><th class="l">Team</th>
      <th>Our AdjEM</th><th>Off <span class="sub">AdjOE</span></th><th>Def <span class="sub">AdjDE</span></th><th>Tempo</th>
      <th>Their AdjEM <span class="sub">KP / Torvik</span></th><th>Their Rk</th>
      <th>\\u0394 EM <span class="sub">us \\u2212 them</span></th><th>\\u0394 Rk</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function setDate(d) { activeDate = d; $("clock").textContent = d || "\\u2014"; $("datePick").value = d; renderSlate(); renderRatings(); }
function setView(v) {
  view = v;
  $("tabSlate").setAttribute("aria-selected", v === "slate");
  $("tabRatings").setAttribute("aria-selected", v === "ratings");
  $("slateView").classList.toggle("hidden", v !== "slate");
  $("ratingsView").classList.toggle("hidden", v !== "ratings");
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
