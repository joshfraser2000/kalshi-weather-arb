"""
Discover Kalshi weather series — finds all available rain/precipitation series
and prints the series tickers so you can fill them into cities.py.

Usage:
  python3 discover_series.py
"""

from __future__ import annotations

import re
import sys
from dotenv import load_dotenv

load_dotenv()

from arb.kalshi import KalshiClient
from arb.cities import CITIES
from arb.logger import get_logger

log = get_logger("discover")

PRECIP_PREFIXES = ("KXRAIN", "KXPRECIP", "KXWET", "KXSNOW", "KXSTORM")


def discover():
    kalshi = KalshiClient()
    kalshi.login()

    print("\n=== Discovering precipitation series on Kalshi ===\n")

    # Fetch all series from Kalshi
    try:
        r = kalshi._get("/series", params={"limit": 200})
        r.raise_for_status()
        all_series = r.json().get("series", [])
    except Exception as e:
        print(f"ERROR fetching series list: {e}")
        print("Falling back to per-city market scan...")
        all_series = []

    precip_series = []
    if all_series:
        for s in all_series:
            ticker = (s.get("ticker") or "").upper()
            if any(ticker.startswith(p) for p in PRECIP_PREFIXES):
                precip_series.append(s)
                print(f"  FOUND: {ticker:30s}  {s.get('title', '')}")

    if not precip_series and not all_series:
        # Fallback: probe each city's likely series names
        print("Probing each city with common naming patterns...\n")
        city_codes = {
            "NYC": ["KXRAINNY",   "KXPRECIPNY"],
            "LAX": ["KXRAINLAX",  "KXPRECIPLAX"],
            "CHI": ["KXRAINCHI",  "KXPRECIPCHI"],
            "HOU": ["KXRAINOU",   "KXPRECIPHOU"],
            "PHX": ["KXRAINPHX",  "KXPRECIPPHX"],
            "DAL": ["KXRAINDAL",  "KXPRECIPDAL"],
            "ATX": ["KXRAINAUS",  "KXPRECIPAUS"],
            "MIA": ["KXRAINMIA",  "KXPRECIPMIA"],
            "ATL": ["KXRAINATL",  "KXPRECIPATL"],
            "SEA": ["KXRAINSEA",  "KXPRECIPSEA"],
            "DEN": ["KXRAINDEN",  "KXPRECIPDEN"],
            "BOS": ["KXRAINBOS",  "KXPRECIPBOS"],
            "SFO": ["KXRAINSFO",  "KXPRECIPSFO"],
        }
        found = {}
        for city_key, candidates in city_codes.items():
            for series in candidates:
                try:
                    markets = kalshi.get_markets_for_series(series, status="open")
                    if markets:
                        print(f"  FOUND: {series:30s}  ({len(markets)} open markets)  [{city_key}]")
                        found[city_key] = series
                        break
                except Exception:
                    pass
        if found:
            print("\n=== cities.py snippet — paste these into your city entries ===\n")
            for city_key, series in sorted(found.items()):
                print(f'    "{city_key}": ... "kalshi_precip_series": "{series}",')
        else:
            print("\nNo precipitation series found. Kalshi may not offer them yet.")

    elif precip_series:
        # Try to match series to cities
        print(f"\nFound {len(precip_series)} precipitation series total.\n")
        print("=== Suggested cities.py updates ===\n")

        # Build a reverse map: city code hint → city key
        city_hints = {
            "NY": "NYC", "LAX": "LAX", "CHI": "CHI", "HOU": "HOU",
            "PHX": "PHX", "PHIL": "PHL", "SATX": "SAN", "DAL": "DAL",
            "AUS": "ATX", "SFO": "SFO", "SEA": "SEA", "DEN": "DEN",
            "OKC": "OKC", "DC": "DCA", "LV": "LAS", "MIA": "MIA",
            "ATL": "ATL", "BOS": "BOS", "MIN": "MSP", "NOLA": "MSY",
            "PDX": "PDX", "DET": "DTW", "CLT": "CLT", "TPA": "TPA",
            "KC": "MCI",
        }
        matched = {}
        for s in precip_series:
            ticker = (s.get("ticker") or "").upper()
            for hint, city_key in city_hints.items():
                if hint in ticker:
                    matched[city_key] = ticker
                    break

        if matched:
            for city_key in sorted(matched):
                series_ticker = matched[city_key]
                print(f'    # {city_key}: change kalshi_precip_series to:')
                print(f'    "kalshi_precip_series": "{series_ticker}",')
        else:
            print("Could not auto-match series to cities.")
            print("Here are all found series — update cities.py manually:")
            for s in precip_series:
                print(f"  {s.get('ticker'):30s}  {s.get('title', '')}")

    kalshi.close()
    print("\nDone.")


if __name__ == "__main__":
    discover()
