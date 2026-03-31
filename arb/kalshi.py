"""
Kalshi API v2 client.

Handles authentication, market discovery, order book polling,
and order placement. All prices are in cents (0–100).
"""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

import httpx

from arb.logger import get_logger

log = get_logger("kalshi")

KALSHI_BASE  = "https://trading.kalshi.com/trade-api/v2"
KALSHI_DEMO  = "https://demo.kalshi.co/trade-api/v2"


class KalshiClient:
    """
    Thin wrapper around the Kalshi REST API.

    Set environment variables:
      KALSHI_EMAIL     — your Kalshi account email
      KALSHI_PASSWORD  — your Kalshi account password
      KALSHI_DEMO      — "true" to use the demo environment (default: false)
    """

    def __init__(self):
        self.email    = os.environ["KALSHI_EMAIL"]
        self.password = os.environ["KALSHI_PASSWORD"]
        demo          = os.getenv("KALSHI_DEMO", "false").lower() == "true"
        self.base_url = KALSHI_DEMO if demo else KALSHI_BASE
        self.token: Optional[str] = None
        self._client  = httpx.Client(base_url=self.base_url, timeout=15)
        log.info(f"KalshiClient init — {'DEMO' if demo else 'LIVE'} @ {self.base_url}")

    # ── Auth ─────────────────────────────────────────────────────────────────

    def login(self) -> None:
        r = self._client.post("/login", json={"email": self.email, "password": self.password})
        r.raise_for_status()
        self.token = r.json()["token"]
        self._client.headers["Authorization"] = f"Bearer {self.token}"
        log.info("Kalshi login successful")

    def _ensure_auth(self) -> None:
        if not self.token:
            self.login()

    # ── Market discovery ──────────────────────────────────────────────────────

    def get_markets_for_series(self, series_ticker: str, status: str = "open") -> list[dict]:
        """
        Fetch all open markets for a given series ticker (e.g. 'KXHIGHNY').
        Returns the raw market objects from the API.
        """
        self._ensure_auth()
        markets = []
        cursor  = None
        while True:
            params: dict = {"series_ticker": series_ticker, "status": status, "limit": 100}
            if cursor:
                params["cursor"] = cursor
            r = self._client.get("/markets", params=params)
            r.raise_for_status()
            body = r.json()
            markets.extend(body.get("markets", []))
            cursor = body.get("cursor")
            if not cursor:
                break
        log.debug(f"{series_ticker}: {len(markets)} markets found")
        return markets

    def get_orderbook(self, ticker: str) -> dict:
        """
        Fetch the order book for a market ticker.
        Returns {yes_bids: [...], yes_asks: [...], no_bids: [...], no_asks: [...]}
        where prices are integers 0–100 (cents).
        """
        self._ensure_auth()
        r = self._client.get(f"/markets/{ticker}/orderbook")
        r.raise_for_status()
        return r.json()["orderbook"]

    def get_market(self, ticker: str) -> dict:
        """
        Fetch a single market's metadata (result, status, yes_ask, yes_bid, etc).
        """
        self._ensure_auth()
        r = self._client.get(f"/markets/{ticker}")
        r.raise_for_status()
        return r.json()["market"]

    # ── Order placement ───────────────────────────────────────────────────────

    def place_order(
        self,
        ticker:    str,
        side:      str,       # "yes" or "no"
        action:    str,       # "buy" or "sell"
        count:     int,       # number of contracts (each = $1 face value)
        price:     int,       # limit price in cents (1–99)
        order_type: str = "limit",
    ) -> dict:
        """
        Place an order on Kalshi.

        Parameters
        ----------
        ticker      : market ticker, e.g. 'KXHIGHNY-25MAR30-B70'
        side        : 'yes' or 'no'
        action      : 'buy' or 'sell'
        count       : number of contracts
        price       : limit price in cents (e.g. 45 = 45¢ per contract)
        order_type  : 'limit' (default) or 'market'
        """
        self._ensure_auth()
        payload = {
            "ticker":     ticker,
            "action":     action,
            "side":       side,
            "type":       order_type,
            "count":      count,
            "yes_price":  price if side == "yes" else (100 - price),
        }
        r = self._client.post("/portfolio/orders", json=payload)
        r.raise_for_status()
        order = r.json()["order"]
        log.info(
            f"ORDER placed: {action.upper()} {count}x {ticker} {side.upper()} @ {price}¢ "
            f"→ order_id={order.get('order_id')}"
        )
        return order

    # ── Portfolio ─────────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Returns available balance in dollars."""
        self._ensure_auth()
        r = self._client.get("/portfolio/balance")
        r.raise_for_status()
        return r.json()["balance"] / 100   # API returns cents

    def get_positions(self) -> list[dict]:
        """Returns all open positions."""
        self._ensure_auth()
        r = self._client.get("/portfolio/positions")
        r.raise_for_status()
        return r.json().get("positions", [])

    def get_fills(self) -> list[dict]:
        """Returns recent order fills."""
        self._ensure_auth()
        r = self._client.get("/portfolio/fills")
        r.raise_for_status()
        return r.json().get("fills", [])

    def close(self) -> None:
        self._client.close()


# ── Market parsing helpers ────────────────────────────────────────────────────

def parse_bin_market(market: dict) -> Optional[dict]:
    """
    Parse a temperature-bin market and extract its temperature range.

    Kalshi bin market titles look like:
      "High temperature between 70°F and 75°F"
      "High temperature of 70°F to 74°F"
      "High of at least 70°F but less than 75°F"

    Returns {low: float, high: float, ticker: str, yes_ask: int, yes_bid: int}
    or None if we can't parse it.
    """
    import re
    title  = market.get("title", "")
    ticker = market.get("ticker", "")

    # Pattern 1: "between Xf and Yf" or "X to Y"
    m = re.search(r"(\d+)[°\s]*F.*?(\d+)[°\s]*F", title, re.IGNORECASE)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return {
            "ticker":   ticker,
            "low":      min(low, high),
            "high":     max(low, high),
            "yes_ask":  market.get("yes_ask", 50),
            "yes_bid":  market.get("yes_bid", 50),
            "title":    title,
        }

    # Pattern 2: ticker suffix like -B70 (above 70) or -BT70 (between 70-75)
    m2 = re.search(r"-B(\d+)$", ticker)
    if m2:
        threshold = float(m2.group(1))
        return {
            "ticker":    ticker,
            "low":       threshold,
            "high":      threshold + 5,   # assume 5°F bins
            "yes_ask":   market.get("yes_ask", 50),
            "yes_bid":   market.get("yes_bid", 50),
            "title":     title,
        }

    return None


def find_adjacent_bins(markets: list[dict]) -> list[tuple[dict, dict]]:
    """
    Given a list of parsed bin markets, return pairs of adjacent bins
    (i.e., where one bin's high == next bin's low).

    Sorted by temperature ascending.
    """
    parsed = [p for m in markets if (p := parse_bin_market(m)) is not None]
    parsed.sort(key=lambda x: x["low"])

    pairs = []
    for i in range(len(parsed) - 1):
        a, b = parsed[i], parsed[i + 1]
        # Adjacent if b.low == a.high (or within 1°F rounding)
        if abs(b["low"] - a["high"]) <= 1.0:
            pairs.append((a, b))
    return pairs
