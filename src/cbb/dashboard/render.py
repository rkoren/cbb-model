"""Render the DASH-007 dashboard payload into one self-contained HTML string.

Pure function: ``render_html(payload) -> str``. No I/O, no file writes — the caller (the future
wiring script) decides where to write it. The output embeds the payload as a single JSON blob and
inlines all CSS/JS (no CDN), so the file works offline and is shareable like ``monitoring/
drift.html`` (the DASH-007 decision).

Interaction shape (DASH-004): one ``activeDate`` clock drives two linked table views — the
FanMatch slate (DASH-002) shows that day's games; the ratings view (DASH-003) shows the latest
weekly snapshot on-or-before the clock. Both re-render from one ``setDate`` call. ISO date strings
sort chronologically, so the "latest ≤ clock" lookup is a plain string comparison.
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
  :root { --fg:#111827; --muted:#6b7280; --line:#e5e7eb; --head:#f3f4f6; --pos:#16a34a; --neg:#dc2626; --accent:#2563eb; }
  body { font-family: system-ui, -apple-system, sans-serif; margin: 1.5rem; color: var(--fg); }
  h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
  #meta { color: var(--muted); font-size: .85rem; margin-bottom: 1rem; }
  #controls { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin-bottom: 1rem; }
  button, select { font: inherit; padding: .3rem .6rem; border: 1px solid var(--line); border-radius: 6px; background: #fff; cursor: pointer; }
  button.tab[aria-selected="true"] { background: var(--accent); color: #fff; border-color: var(--accent); }
  #clock { font-weight: 600; min-width: 8.5rem; text-align: center; }
  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: .85rem; font-variant-numeric: tabular-nums; }
  th, td { padding: .3rem .6rem; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
  th { background: var(--head); text-align: right; position: sticky; top: 0; cursor: pointer; }
  th:first-child, td:first-child, td.l, th.l { text-align: left; }
  .pos { color: var(--pos); } .neg { color: var(--neg); }
  .empty { color: var(--muted); padding: 1rem 0; }
  .hidden { display: none; }
</style>
</head>
<body>
  <h1>CBB Handicapper — us vs KenPom</h1>
  <div id="meta"></div>
  <div id="controls">
    <button id="prevWk" title="−1 week">« wk</button>
    <button id="prevDay" title="−1 day">‹ day</button>
    <span id="clock">—</span>
    <button id="nextDay" title="+1 day">day ›</button>
    <button id="nextWk" title="+1 week">wk »</button>
    <select id="datePick"></select>
    <select id="gender"><option value="all">All</option><option value="M">Men</option><option value="W">Women</option></select>
    <span style="flex:1"></span>
    <button class="tab" id="tabSlate" aria-selected="true">FanMatch</button>
    <button class="tab" id="tabRatings" aria-selected="false">Ratings</button>
  </div>
  <div id="slateView"></div>
  <div id="ratingsView" class="hidden"></div>

<script>
const DATA = __DATA__;
let activeDate = DATA.slate_dates.length ? DATA.slate_dates[DATA.slate_dates.length - 1] : null;
let gender = "all";
let view = "slate";

const $ = (id) => document.getElementById(id);
const fmt = (v, d) => (v === null || v === undefined) ? "\\u2014" : (typeof v === "number" ? v.toFixed(d) : v);
const signed = (v, d) => { const s = fmt(v, d); if (v === null || v === undefined) return s; const c = v > 0 ? "pos" : (v < 0 ? "neg" : ""); return '<span class="' + c + '">' + (v > 0 ? "+" : "") + s + "</span>"; };

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

function renderSlate() {
  const games = (DATA.slate[activeDate] || []).filter(g => gender === "all" || g.gender === gender);
  if (!games.length) { $("slateView").innerHTML = '<p class="empty">No games on this date.</p>'; return; }
  const rows = games.map(g => `<tr>
    <td class="l">${g.a} vs ${g.b}</td><td>${g.gender}</td>
    <td>${signed(g.our.margin,1)}</td><td>${fmt(g.our.total,1)}</td><td>${fmt(g.our.prob,3)}</td>
    <td>${g.kp ? signed(g.kp.margin,1) : "\\u2014"}</td><td>${g.kp ? fmt(g.kp.total,1) : "\\u2014"}</td><td>${g.kp ? fmt(g.kp.prob,3) : "\\u2014"}</td>
    <td>${signed(g.actual.margin,0)}</td><td>${fmt(g.actual.total,0)}</td>
    <td>${signed(g.gap_margin,1)}</td></tr>`).join("");
  $("slateView").innerHTML = `<div class="scroll"><table>
    <thead><tr><th class="l">Matchup</th><th>G</th><th>us mrg</th><th>us tot</th><th>us wp</th>
    <th>KP mrg</th><th>KP tot</th><th>KP wp</th><th>act mrg</th><th>act tot</th><th>gap</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function renderRatings() {
  const rd = latestRatingOnOrBefore(activeDate);
  const teams = rd ? DATA.ratings[rd] : null;
  if (!teams || !teams.length) { $("ratingsView").innerHTML = '<p class="empty">No ratings snapshot on or before this date.</p>'; return; }
  const rows = teams.map(t => `<tr>
    <td>${fmt(t.our.rank,0)}</td><td class="l">${t.team}</td>
    <td>${fmt(t.our.em,1)}</td><td>${fmt(t.our.oe,1)}</td><td>${fmt(t.our.de,1)}</td><td>${fmt(t.our.tempo,1)}</td>
    <td>${fmt(t.kp.em,1)}</td><td>${fmt(t.kp.rank,0)}</td>
    <td>${signed(t.d_em,1)}</td><td>${signed(t.d_rank,0)}</td></tr>`).join("");
  $("ratingsView").innerHTML = `<p class="empty">Snapshot as of ${rd} (latest \\u2264 ${activeDate})</p><div class="scroll"><table>
    <thead><tr><th>#</th><th class="l">Team</th><th>our EM</th><th>OE</th><th>DE</th><th>Tempo</th>
    <th>KP EM</th><th>KP #</th><th>d EM</th><th>d #</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
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
  $("meta").textContent = DATA.meta.n_games + " games \\u00b7 " + DATA.meta.n_snapshots + " rating snapshots"
    + (DATA.meta.generated ? " \\u00b7 generated " + DATA.meta.generated : "");
  $("datePick").innerHTML = DATA.slate_dates.map(d => '<option value="' + d + '">' + d + "</option>").join("");
  $("prevDay").onclick = () => stepDay(-1); $("nextDay").onclick = () => stepDay(1);
  $("prevWk").onclick = () => stepWeek(-1); $("nextWk").onclick = () => stepWeek(1);
  $("datePick").onchange = (e) => setDate(e.target.value);
  $("gender").onchange = (e) => { gender = e.target.value; renderSlate(); };
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
