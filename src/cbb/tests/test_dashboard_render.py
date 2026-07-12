"""Tests for the dashboard HTML renderer (DASH-007).

Load-bearing: the output is a single self-contained HTML doc that embeds the payload, wires the
shared-clock controls, and can't be broken out of the <script> by hostile data.
"""

from html.parser import HTMLParser

import pandas as pd

from cbb.dashboard.payload import build_payload
from cbb.dashboard.render import render_html

NAMES = {1101: "Duke", 1102: "Kansas"}


def _payload(**over):
    preds = pd.DataFrame([dict(
        game_date="2026-03-01", A_TeamID=1101, B_TeamID=1102, men_women=0,
        pred_margin=6.0, pred_total=148.0, pred_prob=0.70, Margin=4.0, Total=150.0, Outcome=1,
        cmp_margin=3.0, cmp_total=151.0, cmp_prob=0.62)])
    rates = pd.DataFrame([dict(
        Season=2026, ArchiveDate=pd.Timestamp("2026-03-01"), TeamID=1101,
        our_AdjEM=20.0, our_AdjOE=115.0, our_AdjDE=95.0, our_AdjTempo=68.0,
        cmp_AdjEM=18.0, cmp_AdjOE=113.0, cmp_AdjDE=95.0, cmp_AdjTempo=69.0,
        our_rank=1.0, cmp_rank=2.0, d_AdjEM=2.0, d_rank=-1.0)])
    return build_payload(preds, rates, {**NAMES, **over.get("names", {})}, generated="2026-07-09")


def test_render_is_parseable_standalone_html():
    html = render_html(_payload())
    assert html.lstrip().startswith("<!DOCTYPE html")
    assert "__DATA__" not in html                 # placeholder was substituted
    HTMLParser().feed(html)                        # no parse exception


def test_render_embeds_payload_and_wires_controls():
    html = render_html(_payload())
    assert "const DATA =" in html and "Duke" in html and "Kansas" in html
    # Shared-clock (DASH-004) + two linked views (DASH-002/003) present by id.
    for marker in ('id="datePick"', 'id="clock"', 'id="tabSlate"', 'id="tabRatings"',
                   'id="slateView"', 'id="ratingsView"', "function setDate", "latestRatingOnOrBefore"):
        assert marker in html, marker


def test_render_has_no_nan_and_single_script_close():
    html = render_html(_payload())
    assert "NaN" not in html                       # allow_nan=False in the dump
    # Exactly one real </script> — the closing tag; embedded data can't introduce another.
    assert html.count("</script>") == 1


def test_hostile_team_name_cannot_break_out_of_script():
    html = render_html(_payload(names={1102: "</script><script>x"}))
    assert html.count("</script>") == 1            # the guard escaped the injected close tag


def test_render_wires_trajectory_drilldown():
    html = render_html(_payload())
    # DASH-006: clickable ratings rows (data-attr, not inline onclick) + the chart/detail machinery.
    for marker in ("function lineChart", "function showTrajectory", "data-team=",
                   'id="detail"', "#detail.hidden"):   # the specificity override that keeps it hidden
        assert marker in html, marker
