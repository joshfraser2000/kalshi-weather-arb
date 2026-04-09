"""
Kalshi API v2 client.

Handles authentication, market discovery, order book polling,
and order placement. All prices are in cents (0–100).
"""

from __future__ import annotations

import base64
import os
import time
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from arb.logger import get_logger

log = get_logger("kalshi")

KALSHI_BASE  = "https://api.elections.kalshi.com/trade-api/v2"
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
        self.email    = os.getenv("KALSHI_EMAIL", "")
        self.password = os.getenv("KALSHI_PASSWORD", "")
        self.key_id   = os.getenv("KALSHI_KEY_ID", "")
        self.key_file = os.getenv("KALSHI_PRIVATE_KEY_FILE", "")
        demo          = os.getenv("KALSHI_DEMO", "false").lower() == "true"
        self.base_url = KALSHI_DEMO if demo else KALSHI_BASE
        self.token: Optional[str] = None
        self._private_key = None
        self._client  = httpx.Client(base_url=self.base_url, timeout=15)
        log.info(f"KalshiClient init — {'DEMO' if demo else 'LIVE'} @ {self.base_url}")

        # Load RSA private key — from env var (Render) or file path (local)
        pem_env = os.getenv("KALSHI_PRIVATE_KEY", "")
        if pem_env:
            self._private_key = serialization.load_pem_private_key(pem_env.encode(), password=None)
            log.info("RSA private key loaded from KALSHI_PRIVATE_KEY env var")
        elif self.key_file:
            pem_path = Path(self.key_file)
            if not pem_path.is_absolute():
                pem_path = Path(__file__).parent.parent / self.key_file
            if pem_path.exists():
                pem_data = pem_path.read_bytes()
                self._private_key = serialization.load_pem_private_key(pem_data, password=None)
                log.info(f"RSA private key loaded from {pem_path}")

    # ── Auth ─────────────────────────────────────────────────────────────────

    def _sign(self, method: str, path: str) -> dict:
        """Generate RSA-signed auth headers for Kalshi API key auth."""
        ts = str(int(time.time() * 1000))
        # Full path must include /trade-api/v2 prefix, strip query string before signing
        clean_path = path.split("?")[0]
        full_path  = "/trade-api/v2" + clean_path if not clean_path.startswith("/trade-api") else clean_path
        msg = (ts + method.upper() + full_path).encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY":       self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    def login(self) -> None:
        if self._private_key and self.key_id:
            # RSA key-based auth — no login needed, headers signed per-request
            self.token = "api_key"   # sentinel so _ensure_auth() passes
            log.info("Kalshi auth: using RSA API key signing")
            return
        # Fallback: email/password
        base = self.base_url.replace("/trade-api/v2", "")
        r = self._client.post(f"{base}/v1/log_in", json={"email": self.email, "password": self.password})
        r.raise_for_status()
        body = r.json()
        self.token = body.get("token") or body.get("access_token")
        if not self.token:
            raise RuntimeError(f"No token in login response: {list(body.keys())}")
        self._client.headers["Authorization"] = f"Bearer {self.token}"
        log.info("Kalshi login successful (email/password)")

    def _ensure_auth(self) -> None:
        if not self.token:
            self.login()

    def _get(self, path: str, **kwargs):
        self._ensure_auth()
        headers = self._sign("GET", path) if self._private_key else {}
        return self._client.get(path, headers=headers, **kwargs)

    def _post(self, path: str, **kwargs):
        self._ensure_auth()
        headers = self._sign("POST", path) if self._private_key else {}
        return self._client.post(path, headers=headers, **kwargs)

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
            r = self._get("/markets", params=params)
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
        Returns normalized orderbook: {yes_bid, yes_ask, no_bid, no_ask} in cents (int).
        Kalshi returns orderbook_fp with yes_dollars/no_dollars as bid lists.
        yes_dollars = YES bids; best YES ask = 100 - best_no_bid
        no_dollars  = NO bids;  best NO ask  = 100 - best_yes_bid
        """
        r = self._get(f"/markets/{ticker}/orderbook")
        r.raise_for_status()
        fp = r.json().get("orderbook_fp", {})
        yes_bids = fp.get("yes_dollars", [])
        no_bids  = fp.get("no_dollars", [])
        best_yes_bid = round(float(yes_bids[0][0]) * 100) if yes_bids else None
        best_no_bid  = round(float(no_bids[0][0])  * 100) if no_bids  else None
        return {
            "yes_bid": best_yes_bid,
            "yes_ask": (100 - best_no_bid)  if best_no_bid  is not None else None,
            "no_bid":  best_no_bid,
            "no_ask":  (100 - best_yes_bid) if best_yes_bid is not None else None,
        }

    def get_market(self, ticker: str) -> dict:
        r = self._get(f"/markets/{ticker}")
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
        payload = {
            "ticker":     ticker,
            "action":     action,
            "side":       side,
            "type":       order_type,
            "count":      count,
            "yes_price":  price if side == "yes" else (100 - price),
        }
        r = self._post("/portfolio/orders", json=payload)
        r.raise_for_status()
        order = r.json()["order"]
        log.info(
            f"ORDER placed: {action.upper()} {count}x {ticker} {side.upper()} @ {price}¢ "
            f"→ order_id={order.get('order_id')}"
        )
        return order

    # ── Portfolio ─────────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        r = self._get("/portfolio/balance")
        r.raise_for_status()
        return r.json()["balance"] / 100

    def get_positions(self) -> list[dict]:
        r = self._get("/portfolio/positions")
        r.raise_for_status()
        return r.json().get("positions", [])

    def get_fills(self) -> list[dict]:
        r = self._get("/portfolio/fills")
        r.raise_for_status()
        return r.json().get("fills", [])

    def get_orders(self, status: str = "resting") -> list[dict]:
        r = self._get(f"/portfolio/orders?status={status}")
        r.raise_for_status()
        return r.json().get("orders", [])

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

    # Primary: ticker suffix -B{center} e.g. -B74.5 = range 74–75°F (1°F bin)
    m_bin = re.search(r"-B([\d.]+)$", ticker)
    if m_bin:
        center = float(m_bin.group(1))
        low    = round(center - 0.5, 1)
        high   = round(center + 0.5, 1)
        ask    = market.get("yes_ask")
        bid    = market.get("yes_bid")
        return {
            "ticker":  ticker,
            "low":     low,
            "high":    high,
            "yes_ask": ask if ask is not None else 50,
            "yes_bid": bid if bid is not None else 48,
            "title":   title,
            "type":    "bin",
        }

    # Fallback: parse title "be 74-75°" or "be 74-75°F"
    m_range = re.search(r"be\s+([\d.]+)[–\-]([\d.]+)[°]", title)
    if m_range:
        low  = float(m_range.group(1))
        high = float(m_range.group(2))
        ask  = market.get("yes_ask")
        bid  = market.get("yes_bid")
        return {
            "ticker":  ticker,
            "low":     low,
            "high":    high,
            "yes_ask": ask if ask is not None else 50,
            "yes_bid": bid if bid is not None else 48,
            "title":   title,
            "type":    "bin",
        }

    return None


def parse_threshold_market(market: dict) -> Optional[dict]:
    """
    Parse a threshold market (above/below a single temperature).

    Kalshi threshold tickers look like:
      KXHIGHNY-25MAR30-T60   → "NYC high at least 60°F" (YES = above 60)
      KXHIGHTHOU-25MAR30-T95 → "HOU high at least 95°F"

    Returns {threshold, direction, ticker, yes_ask, yes_bid, type}
    or None if not parseable.
    """
    import re
    ticker = market.get("ticker", "")
    m_t = re.search(r"-T([\d.]+)$", ticker)
    if not m_t:
        return None
    threshold = float(m_t.group(1))
    ask = market.get("yes_ask")
    bid = market.get("yes_bid")
    return {
        "ticker":    ticker,
        "threshold": threshold,
        "direction": "above",   # YES = high is AT LEAST this temp
        "yes_ask":   ask if ask is not None else 50,
        "yes_bid":   bid if bid is not None else 48,
        "title":     market.get("title", ""),
        "type":      "threshold",
    }


def enrich_with_orderbook_prices(
    parsed_markets: list[dict],
    client: "KalshiClient",
    max_spread: int = 20,
) -> list[dict]:
    """
    Fetch real orderbook prices for markets and filter illiquid ones.

    max_spread : maximum yes_ask - yes_bid (in cents) to consider tradeable.
                 Wide-spread markets (e.g., 1¢ bid / 99¢ ask) are excluded.
    """
    enriched = []
    for m in parsed_markets:
        needs_ob = m.get("yes_ask") in (None, 50) or m.get("yes_bid") in (None, 48)
        if needs_ob:
            try:
                ob = client.get_orderbook(m["ticker"])
                if ob.get("yes_ask") is not None:
                    m["yes_ask"] = ob["yes_ask"]
                if ob.get("yes_bid") is not None:
                    m["yes_bid"] = ob["yes_bid"]
                if ob.get("no_ask") is not None:
                    m["no_ask"] = ob["no_ask"]
                if ob.get("no_bid") is not None:
                    m["no_bid"] = ob["no_bid"]
            except Exception:
                pass

        # Skip markets where the bid-ask spread is too wide (illiquid)
        ask = m.get("yes_ask")
        bid = m.get("yes_bid")
        if ask is not None and bid is not None and (ask - bid) > max_spread:
            continue
        enriched.append(m)
    return enriched


def find_adjacent_bins(markets: list[dict]) -> list[tuple[dict, dict]]:
    """
    Return pairs of adjacent 1°F bins sorted by temperature.
    Bins are adjacent when b.low == a.high (within 0.01°F tolerance).
    """
    bins = [m for m in markets if m.get("type") == "bin"]
    bins.sort(key=lambda x: x["low"])

    pairs = []
    for i in range(len(bins) - 1):
        a, b = bins[i], bins[i + 1]
        if abs(b["low"] - a["high"]) < 0.01:
            pairs.append((a, b))
    return pairs
