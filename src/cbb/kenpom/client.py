"""KenPom API client.

All endpoints documented in docs/KenpomAPI.pdf.
Requires KENPOM_API_KEY environment variable (Bearer token).
"""

from typing import Any

import httpx
import pandas as pd

_BASE = "https://kenpom.com/api.php"


class KenPomClient:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        if api_key is None:
            from cbb.secrets import kenpom_api_key
            api_key = kenpom_api_key()
        key = api_key
        if not key:
            raise ValueError("KENPOM_API_KEY not set")
        self._headers = {"Authorization": f"Bearer {key}"}
        self._timeout = timeout

    def _get(self, params: dict[str, Any]) -> list[dict]:
        with httpx.Client(headers=self._headers, timeout=self._timeout) as client:
            r = client.get(_BASE, params=params)
            r.raise_for_status()
            data = r.json()
        # API returns either a list or {"error": "..."} dict
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"KenPom API error: {data['error']}")
        return data if isinstance(data, list) else [data]

    def _df(self, params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(self._get(params))

    # ------------------------------------------------------------------
    # Ratings
    # ------------------------------------------------------------------

    def ratings(
        self,
        year: int | None = None,
        team_id: int | None = None,
        conference: str | None = None,
    ) -> pd.DataFrame:
        """Team ratings, SOS, tempo, possession length. year or team_id required."""
        if year is None and team_id is None:
            raise ValueError("year or team_id required")
        params: dict[str, Any] = {"endpoint": "ratings"}
        if year:
            params["y"] = year
        if team_id:
            params["team_id"] = team_id
        if conference:
            params["c"] = conference
        return self._df(params)

    def ratings_archive(
        self,
        date: str | None = None,
        year: int | None = None,
        preseason: bool = False,
        team_id: int | None = None,
        conference: str | None = None,
    ) -> pd.DataFrame:
        """Historical ratings at a specific date (YYYY-MM-DD) or preseason snapshot.

        Use date=<Selection Sunday> to get pre-tournament ratings without leakage.
        """
        params: dict[str, Any] = {"endpoint": "archive"}
        if preseason:
            if year is None:
                raise ValueError("year required with preseason=True")
            params["preseason"] = "true"
            params["y"] = year
        elif date:
            params["d"] = date
        else:
            raise ValueError("date or preseason=True required")
        if team_id:
            params["team_id"] = team_id
        if conference:
            params["c"] = conference
        return self._df(params)

    # ------------------------------------------------------------------
    # Advanced stats
    # ------------------------------------------------------------------

    def four_factors(
        self,
        year: int | None = None,
        team_id: int | None = None,
        conference: str | None = None,
        conf_only: bool = False,
    ) -> pd.DataFrame:
        """Four Factors (eFG%, TO%, OR%, FT rate) for offense and defense."""
        if year is None and team_id is None:
            raise ValueError("year or team_id required")
        params: dict[str, Any] = {"endpoint": "four-factors"}
        if year:
            params["y"] = year
        if team_id:
            params["team_id"] = team_id
        if conference:
            params["c"] = conference
        if conf_only:
            params["conf_only"] = "true"
        return self._df(params)

    def misc_stats(
        self,
        year: int | None = None,
        team_id: int | None = None,
        conference: str | None = None,
        conf_only: bool = False,
    ) -> pd.DataFrame:
        """Shooting pcts, block%, steal rates, assist rates, 3PA rate."""
        if year is None and team_id is None:
            raise ValueError("year or team_id required")
        params: dict[str, Any] = {"endpoint": "misc-stats"}
        if year:
            params["y"] = year
        if team_id:
            params["team_id"] = team_id
        if conference:
            params["c"] = conference
        if conf_only:
            params["conf_only"] = "true"
        return self._df(params)

    def height(
        self,
        year: int | None = None,
        team_id: int | None = None,
        conference: str | None = None,
    ) -> pd.DataFrame:
        """Team height, experience, bench strength, continuity."""
        if year is None and team_id is None:
            raise ValueError("year or team_id required")
        params: dict[str, Any] = {"endpoint": "height"}
        if year:
            params["y"] = year
        if team_id:
            params["team_id"] = team_id
        if conference:
            params["c"] = conference
        return self._df(params)

    def point_distribution(
        self,
        year: int | None = None,
        team_id: int | None = None,
        conference: str | None = None,
        conf_only: bool = False,
    ) -> pd.DataFrame:
        """% of points from FT, 2FG, 3FG for offense and defense."""
        if year is None and team_id is None:
            raise ValueError("year or team_id required")
        params: dict[str, Any] = {"endpoint": "pointdist"}
        if year:
            params["y"] = year
        if team_id:
            params["team_id"] = team_id
        if conference:
            params["c"] = conference
        if conf_only:
            params["conf_only"] = "true"
        return self._df(params)

    # ------------------------------------------------------------------
    # Game predictions
    # ------------------------------------------------------------------

    def fanmatch(self, date: str) -> pd.DataFrame:
        """KenPom game predictions for a past date (YYYY-MM-DD).

        Returns predicted scores, win probabilities, tempo, thrill scores.
        Useful as a baseline and as a feature (market-consensus strength signal).
        """
        return self._df({"endpoint": "fanmatch", "d": date})

    # ------------------------------------------------------------------
    # Reference data
    # ------------------------------------------------------------------

    def teams(self, year: int, conference: str | None = None) -> pd.DataFrame:
        """Team list with TeamID, conference, coach, arena for a given season."""
        params: dict[str, Any] = {"endpoint": "teams", "y": year}
        if conference:
            params["c"] = conference
        return self._df(params)

    def conferences(self, year: int) -> pd.DataFrame:
        """Conference list for a given season."""
        return self._df({"endpoint": "conferences", "y": year})

    def conference_ratings(
        self, year: int | None = None, conference: str | None = None
    ) -> pd.DataFrame:
        """Conference-level net ratings. year or conference required."""
        if year is None and conference is None:
            raise ValueError("year or conference required")
        params: dict[str, Any] = {"endpoint": "conf-ratings"}
        if year:
            params["y"] = year
        if conference:
            params["c"] = conference
        return self._df(params)
