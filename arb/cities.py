"""
US cities with confirmed live Kalshi weather series tickers.
Verified against api.elections.kalshi.com /series endpoint.

kalshi_precip_series: Kalshi precipitation series ticker, or "" if not yet confirmed.
  The bot skips cities with an empty string — no markets, no trades.
  To discover available series: query GET /series on the Kalshi API and search
  for KXRAIN*, KXPRECIP*, or KXWET* prefixes.
"""

CITIES: dict[str, dict] = {
    "NYC": {
        "name": "New York City",
        "lat": 40.7128, "lon": -74.0060,
        "kalshi_series": "KXHIGHNY", "kalshi_precip_series": "KXRAINNYC",
        "nws_office": "OKX", "nws_grid": (33, 37),
        "tz": "America/New_York",
    },
    "LAX": {
        "name": "Los Angeles",
        "lat": 34.0522, "lon": -118.2437,
        "kalshi_series": "KXHIGHLAX", "kalshi_precip_series": "",
        "nws_office": "LOX", "nws_grid": (149, 48),
        "tz": "America/Los_Angeles",
    },
    "CHI": {
        "name": "Chicago",
        "lat": 41.8781, "lon": -87.6298,
        "kalshi_series": "KXHIGHCHI", "kalshi_precip_series": "",
        "nws_office": "LOT", "nws_grid": (74, 73),
        "tz": "America/Chicago",
    },
    "HOU": {
        "name": "Houston",
        "lat": 29.7604, "lon": -95.3698,
        "kalshi_series": "KXHIGHTHOU", "kalshi_precip_series": "KXRAINHOU",
        "nws_office": "HGX", "nws_grid": (66, 97),
        "tz": "America/Chicago",
    },
    "PHX": {
        "name": "Phoenix",
        "lat": 33.4484, "lon": -112.0740,
        "kalshi_series": "KXHIGHTPHX", "kalshi_precip_series": "",
        "nws_office": "PSR", "nws_grid": (155, 57),
        "tz": "America/Phoenix",
    },
    "PHL": {
        "name": "Philadelphia",
        "lat": 39.9526, "lon": -75.1652,
        "kalshi_series": "KXHIGHPHIL", "kalshi_precip_series": "",
        "nws_office": "PHI", "nws_grid": (49, 68),
        "tz": "America/New_York",
    },
    "SAN": {
        "name": "San Antonio",
        "lat": 29.4241, "lon": -98.4936,
        "kalshi_series": "KXHIGHTSATX", "kalshi_precip_series": "",
        "nws_office": "EWX", "nws_grid": (156, 90),
        "tz": "America/Chicago",
    },
    "DAL": {
        "name": "Dallas",
        "lat": 32.7767, "lon": -96.7970,
        "kalshi_series": "KXHIGHTDAL", "kalshi_precip_series": "",
        "nws_office": "FWD", "nws_grid": (93, 84),
        "tz": "America/Chicago",
    },
    "ATX": {
        "name": "Austin",
        "lat": 30.2672, "lon": -97.7431,
        "kalshi_series": "KXHIGHAUS", "kalshi_precip_series": "",
        "nws_office": "EWX", "nws_grid": (156, 100),
        "tz": "America/Chicago",
    },
    "SFO": {
        "name": "San Francisco",
        "lat": 37.7749, "lon": -122.4194,
        "kalshi_series": "KXHIGHTSFO", "kalshi_precip_series": "",
        "nws_office": "MTR", "nws_grid": (84, 105),
        "tz": "America/Los_Angeles",
    },
    "SEA": {
        "name": "Seattle",
        "lat": 47.6062, "lon": -122.3321,
        "kalshi_series": "KXHIGHTSEA", "kalshi_precip_series": "KXRAINSEA",
        "nws_office": "SEW", "nws_grid": (124, 67),
        "tz": "America/Los_Angeles",
    },
    "DEN": {
        "name": "Denver",
        "lat": 39.7392, "lon": -104.9903,
        "kalshi_series": "KXHIGHDEN", "kalshi_precip_series": "",
        "nws_office": "BOU", "nws_grid": (57, 62),
        "tz": "America/Denver",
    },
    "OKC": {
        "name": "Oklahoma City",
        "lat": 35.4676, "lon": -97.5164,
        "kalshi_series": "KXHIGHTOKC", "kalshi_precip_series": "",
        "nws_office": "OUN", "nws_grid": (74, 63),
        "tz": "America/Chicago",
    },
    "DCA": {
        "name": "Washington DC",
        "lat": 38.9072, "lon": -77.0369,
        "kalshi_series": "KXHIGHTDC", "kalshi_precip_series": "",
        "nws_office": "LWX", "nws_grid": (95, 72),
        "tz": "America/New_York",
    },
    "LAS": {
        "name": "Las Vegas",
        "lat": 36.1699, "lon": -115.1398,
        "kalshi_series": "KXHIGHTLV", "kalshi_precip_series": "",
        "nws_office": "VEF", "nws_grid": (123, 98),
        "tz": "America/Los_Angeles",
    },
    "MIA": {
        "name": "Miami",
        "lat": 25.7617, "lon": -80.1918,
        "kalshi_series": "KXHIGHMIA", "kalshi_precip_series": "KXRAINMIA",
        "nws_office": "MFL", "nws_grid": (110, 50),
        "tz": "America/New_York",
    },
    "ATL": {
        "name": "Atlanta",
        "lat": 33.7490, "lon": -84.3880,
        "kalshi_series": "KXHIGHTATL", "kalshi_precip_series": "",
        "nws_office": "FFC", "nws_grid": (51, 88),
        "tz": "America/New_York",
    },
    "BOS": {
        "name": "Boston",
        "lat": 42.3601, "lon": -71.0589,
        "kalshi_series": "KXHIGHTBOS", "kalshi_precip_series": "",
        "nws_office": "BOX", "nws_grid": (64, 34),
        "tz": "America/New_York",
    },
    "MSP": {
        "name": "Minneapolis",
        "lat": 44.9778, "lon": -93.2650,
        "kalshi_series": "KXHIGHTMIN", "kalshi_precip_series": "",
        "nws_office": "MPX", "nws_grid": (107, 70),
        "tz": "America/Chicago",
    },
    "MSY": {
        "name": "New Orleans",
        "lat": 29.9511, "lon": -90.0715,
        "kalshi_series": "KXHIGHTNOLA", "kalshi_precip_series": "KXRAINNO",
        "nws_office": "LIX", "nws_grid": (66, 52),
        "tz": "America/Chicago",
    },
    "PDX": {
        "name": "Portland",
        "lat": 45.5051, "lon": -122.6750,
        "kalshi_series": "KXHIGHTPDX", "kalshi_precip_series": "",
        "nws_office": "PQR", "nws_grid": (113, 103),
        "tz": "America/Los_Angeles",
    },
    "DTW": {
        "name": "Detroit",
        "lat": 42.3314, "lon": -83.0458,
        "kalshi_series": "KXHIGHTDET", "kalshi_precip_series": "",
        "nws_office": "DTX", "nws_grid": (66, 34),
        "tz": "America/Detroit",
    },
    "CLT": {
        "name": "Charlotte",
        "lat": 35.2271, "lon": -80.8431,
        "kalshi_series": "KXHIGHTCLT", "kalshi_precip_series": "",
        "nws_office": "GSP", "nws_grid": (119, 65),
        "tz": "America/New_York",
    },
    "TPA": {
        "name": "Tampa",
        "lat": 27.9506, "lon": -82.4572,
        "kalshi_series": "KXHIGHTTPA", "kalshi_precip_series": "",
        "nws_office": "TBW", "nws_grid": (71, 98),
        "tz": "America/New_York",
    },
    "MCI": {
        "name": "Kansas City",
        "lat": 39.0997, "lon": -94.5786,
        "kalshi_series": "KXHIGHTKC", "kalshi_precip_series": "",
        "nws_office": "EAX", "nws_grid": (44, 51),
        "tz": "America/Chicago",
    },
}
