"""Handicapper dashboard (DASH-007) — build the payload from the two logs, render standalone HTML."""

from .payload import build_payload, build_ratings, build_slate
from .render import render_html

__all__ = ["build_payload", "build_slate", "build_ratings", "render_html"]
