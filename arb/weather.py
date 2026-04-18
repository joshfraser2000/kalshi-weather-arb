"""
Weather forecast client — pulls ensemble forecasts from multiple sources
to build high-confidence probability distributions over daily high temps.

Sources (in order of priority):
  1. Open-Meteo Ensemble API  — GFS (31 members) + ECMWF IFS (51 members) + ICON-EPS (40 members)
  2. NWS API                  — Official NOAA point forecast (deterministic, used as sanity check)
  3. Open-Meteo Historical    — Used for bias correction against observed actuals

The ensemble members give us a natural empirical distribution; we fit a
normal distribution to those members as our temperature model.
"""

from __future__ import annotations

import asyncio
import statistics
from datetime import date, datetime, timedelta
from typing import Optional

import httpx

from arb.logger import get_logger

log = get_logger("weather")

# Limit concurrent Open-Meteo requests to avoid 429 rate limiting
_OPEN_METEO_SEM = asyncio.Semaphore(3)

ENSEMBLE_URL  = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"
NWS_BASE      = "https://api.weather.gov"

# How many days of history to pull for bias correction
BIAS_HISTORY_DAYS = 30


class ForecastResult:
    """
    Ensemble-backed temperature + precipitation forecast for a single city on a single date.

    Attributes
    ----------
    city_key         : str   — e.g. "NYC"
    target_date      : date
    members          : list[float]  — raw ensemble high-temp values (°F)
    mean             : float — ensemble mean (°F)
    std              : float — ensemble std-dev (°F)
    bias_correction  : float — mean observed error from recent history (°F)
    corrected_mean   : float — bias-adjusted mean (°F)
    nws_high         : float | None — NWS deterministic high (°F)
    precip_members   : list[float]  — ensemble precip totals (inches/day)
    precip_mean      : float — ensemble mean precipitation (inches)
    precip_std       : float — ensemble std-dev precipitation (inches)
    nws_precip_prob  : float | None — NWS probability of precipitation (0–1)
    """

    def __init__(
        self,
        city_key: str,
        target_date: date,
        members: list[float],
        bias_correction: float = 0.0,
        nws_high: Optional[float] = None,
        precip_members: Optional[list[float]] = None,
        nws_precip_prob: Optional[float] = None,
    ):
        self.city_key        = city_key
        self.target_date     = target_date
        self.members         = [m - bias_correction for m in members]
        self.mean            = statistics.mean(self.members)
        self.std             = statistics.stdev(self.members) if len(self.members) > 1 else 3.0
        self.bias_correction = bias_correction
        self.corrected_mean  = self.mean
        self.nws_high        = nws_high

        self.precip_members  = precip_members or []
        self.precip_mean     = statistics.mean(self.precip_members) if self.precip_members else 0.0
        self.precip_std      = (statistics.stdev(self.precip_members)
                                if len(self.precip_members) > 1 else 0.1)
        self.nws_precip_prob = nws_precip_prob

    def prob_in_range(self, low: float, high: float) -> float:
        """Fraction of ensemble members whose high temp falls in [low, high)."""
        in_range = sum(1 for t in self.members if low <= t < high)
        return in_range / len(self.members)

    def prob_above(self, threshold: float) -> float:
        """P(high >= threshold) — for 'above X°F' style markets."""
        above = sum(1 for t in self.members if t >= threshold)
        return above / len(self.members)

    def prob_below(self, threshold: float) -> float:
        """P(high < threshold) — for 'below X°F' style markets."""
        return 1.0 - self.prob_above(threshold)

    def prob_precip_above(self, threshold_inches: float) -> float:
        """P(daily precip >= threshold) — for rain/precipitation markets."""
        if not self.precip_members:
            return self.nws_precip_prob or 0.0
        above = sum(1 for p in self.precip_members if p >= threshold_inches)
        return above / len(self.precip_members)

    def prob_any_rain(self) -> float:
        """P(any measurable precipitation) — threshold = 0.01 inches."""
        return self.prob_precip_above(0.01)

    def __repr__(self) -> str:
        precip_str = f" precip_μ={self.precip_mean:.2f}\"" if self.precip_members else ""
        return (
            f"ForecastResult({self.city_key} {self.target_date} "
            f"μ={self.corrected_mean:.1f}°F σ={self.std:.1f}°F "
            f"n={len(self.members)} bias={self.bias_correction:+.1f}°F{precip_str})"
        )


async def _fetch_ensemble(
    lat: float,
    lon: float,
    target_date: date,
    client: httpx.AsyncClient,
) -> tuple[list[float], list[float]]:
    """
    Pull GFS + ECMWF + ICON ensemble members from Open-Meteo.
    Returns (temp_members_°F, precip_members_inches).
    """
    async with _OPEN_METEO_SEM:
        return await _fetch_ensemble_inner(lat, lon, target_date, client)


def _extract_ensemble_field(data: dict, field_prefix: str, idx: int) -> list[float]:
    """Extract all member values for a field prefix at the given date index."""
    values = []
    for key, vals in data["daily"].items():
        if key.startswith(field_prefix) and key != field_prefix:
            v = vals[idx]
            if v is not None:
                values.append(float(v))
    return values


async def _fetch_ensemble_inner(
    lat: float,
    lon: float,
    target_date: date,
    client: httpx.AsyncClient,
) -> tuple[list[float], list[float]]:
    """Returns (temp_members, precip_members) both as flat lists."""
    temp_members:   list[float] = []
    precip_members: list[float] = []

    for model, label in [
        ("gfs_seamless",  "GFS"),
        ("ecmwf_ifs04",   "ECMWF"),
        ("icon_seamless",  "ICON"),
    ]:
        try:
            r = await client.get(
                ENSEMBLE_URL,
                params={
                    "latitude":         lat,
                    "longitude":        lon,
                    "daily":            ["temperature_2m_max", "precipitation_sum"],
                    "temperature_unit": "fahrenheit",
                    "precipitation_unit": "inch",
                    "forecast_days":    7,
                    "models":           model,
                },
                timeout=20,
            )
            r.raise_for_status()
            data  = r.json()
            dates = data["daily"]["time"]
            if str(target_date) in dates:
                idx = dates.index(str(target_date))
            elif dates:
                idx = 0
            else:
                continue
            temp_members   += _extract_ensemble_field(data, "temperature_2m_max",  idx)
            precip_members += _extract_ensemble_field(data, "precipitation_sum",    idx)
            log.debug(f"{label}: temp={len(temp_members)} precip={len(precip_members)} members")
        except Exception as e:
            log.warning(f"{label} ensemble fetch failed: {e}")

    return temp_members, precip_members


async def _fetch_nws_high(
    nws_office: str,
    nws_grid: tuple[int, int],
    target_date: date,
    client: httpx.AsyncClient,
) -> tuple[Optional[float], Optional[float]]:
    """
    Pull official NWS point forecast.
    Returns (high_temp_°F, precip_probability_0_to_1) for target_date.
    """
    try:
        x, y = nws_grid
        url = f"{NWS_BASE}/gridpoints/{nws_office}/{x},{y}/forecast"
        r = await client.get(url, timeout=15, headers={"User-Agent": "kalshi-weather-arb/1.0"})
        r.raise_for_status()
        periods = r.json()["properties"]["periods"]
        for period in periods:
            start = datetime.fromisoformat(period["startTime"]).date()
            if start == target_date and period["isDaytime"]:
                high = float(period["temperature"])
                pop  = period.get("probabilityOfPrecipitation", {})
                pop_val = pop.get("value") if isinstance(pop, dict) else None
                precip_prob = float(pop_val) / 100.0 if pop_val is not None else None
                return high, precip_prob
    except Exception as e:
        log.warning(f"NWS forecast fetch failed for {nws_office}: {e}")
    return None, None


async def _compute_bias_correction(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> float:
    """
    Compare Open-Meteo historical forecast vs observed actuals over the last
    BIAS_HISTORY_DAYS days to estimate systematic forecast bias.

    Returns the mean error (forecast − observed) in °F.
    Subtract this from the forecast to get a corrected estimate.
    """
    end   = date.today() - timedelta(days=2)   # need at least 2 days for obs to be finalized
    start = end - timedelta(days=BIAS_HISTORY_DAYS)

    try:
        r = await client.get(
            ARCHIVE_URL,
            params={
                "latitude":         lat,
                "longitude":        lon,
                "start_date":       str(start),
                "end_date":         str(end),
                "daily":            "temperature_2m_max",
                "temperature_unit": "fahrenheit",
            },
            timeout=20,
        )
        r.raise_for_status()
        obs_data = r.json()["daily"]

        # Now fetch the deterministic forecast for the same historical period
        r2 = await client.get(
            FORECAST_URL,
            params={
                "latitude":         lat,
                "longitude":        lon,
                "start_date":       str(start),
                "end_date":         str(end),
                "daily":            "temperature_2m_max",
                "temperature_unit": "fahrenheit",
            },
            timeout=20,
        )
        r2.raise_for_status()
        fcst_data = r2.json()["daily"]

        errors = []
        for obs_val, fcst_val in zip(obs_data["temperature_2m_max"], fcst_data["temperature_2m_max"]):
            if obs_val is not None and fcst_val is not None:
                errors.append(fcst_val - obs_val)

        if errors:
            bias = statistics.mean(errors)
            log.debug(f"Bias correction: {bias:+.2f}°F ({len(errors)} days)")
            return bias

    except Exception as e:
        log.warning(f"Bias correction failed: {e}")

    return 0.0


async def get_forecast(
    city_key: str,
    city: dict,
    target_date: date,
    client: httpx.AsyncClient,
) -> ForecastResult:
    """
    Full pipeline for a single city:
      1. Fetch multi-model ensemble members
      2. Fetch NWS deterministic high as a sanity check
      3. Compute bias correction from historical data
      4. Return a ForecastResult with all data attached
    """
    lat, lon = city["lat"], city["lon"]

    (members, precip_members), (nws_high, nws_precip_prob), bias = await asyncio.gather(
        _fetch_ensemble(lat, lon, target_date, client),
        _fetch_nws_high(city["nws_office"], city["nws_grid"], target_date, client),
        _compute_bias_correction(lat, lon, client),
    )

    if not members:
        raise RuntimeError(f"No ensemble members fetched for {city_key}")

    result = ForecastResult(
        city_key=city_key,
        target_date=target_date,
        members=members,
        bias_correction=bias,
        nws_high=nws_high,
        precip_members=precip_members,
        nws_precip_prob=nws_precip_prob,
    )
    log.info(
        f"{city_key} {target_date}: {result} | NWS={nws_high}°F | "
        f"{len(members)} temp / {len(precip_members)} precip ensemble members"
    )
    return result


async def get_all_forecasts(
    cities: dict,
    target_date: date,
) -> dict[str, ForecastResult]:
    """
    Fetch forecasts for all cities concurrently.
    Returns {city_key: ForecastResult}.
    """
    async with httpx.AsyncClient() as client:
        tasks = {
            key: get_forecast(key, city, target_date, client)
            for key, city in cities.items()
        }
        results: dict[str, ForecastResult] = {}
        for city_key, coro in tasks.items():
            try:
                results[city_key] = await coro
            except Exception as e:
                log.error(f"{city_key}: forecast failed — {e}")
        return results
