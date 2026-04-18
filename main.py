"""
Kalshi Weather Arbitrage Bot
════════════════════════════

Finds and trades mispriced temperature-bin markets on Kalshi using
multi-model ensemble weather forecasts as a probability edge.

Usage
─────
  python main.py                  # scan today's markets, DRY RUN (no orders)
  python main.py --date 2026-04-01  # scan a specific date
  python main.py --live           # actually place orders (requires Kalshi creds)
  python main.py --cities NYC,CHI,LAX  # limit to specific cities
  python main.py --report         # show current positions & P&L

Environment variables
─────────────────────
  KALSHI_EMAIL       Kalshi account email
  KALSHI_PASSWORD    Kalshi account password
  KALSHI_DEMO        "true" to use Kalshi demo environment
  MIN_EDGE           Minimum edge to trade (default 0.06 = 6%)
  STRONG_EDGE        Strong-signal threshold (default 0.12 = 12%)
  MIN_PROB           Min win probability to trade (default 0.55)
  KELLY_FRACTION     Fractional Kelly (default 0.25 = quarter-Kelly)
  MAX_PCT_BANKROLL   Max % bankroll per trade (default 0.05 = 5%)
  MAX_TOTAL_DEPLOY   Max % bankroll deployed at once (default 0.40 = 40%)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date, datetime

from dotenv import load_dotenv

load_dotenv()

from arb.cities   import CITIES
from arb.weather  import get_all_forecasts
from arb.kalshi   import KalshiClient, find_adjacent_bins
from arb.strategy import find_opportunities, find_precip_opportunities, summarize_opportunities, TradeOpportunity
from arb.sizing   import allocate
from arb.logger   import get_logger

log = get_logger("main")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kalshi Weather Arbitrage Bot")
    p.add_argument("--date",    default=str(date.today()), help="Target date (YYYY-MM-DD)")
    p.add_argument("--cities",  default=None, help="Comma-separated city keys (default: all)")
    p.add_argument("--live",    action="store_true", help="Place real orders (default: dry run)")
    p.add_argument("--report",  action="store_true", help="Show positions & P&L then exit")
    p.add_argument("--loop",    action="store_true", help="Run continuously (re-scan every hour)")
    return p.parse_args()


def get_target_cities(city_filter: str | None) -> dict:
    if not city_filter:
        return CITIES
    keys = [k.strip().upper() for k in city_filter.split(",")]
    missing = [k for k in keys if k not in CITIES]
    if missing:
        log.warning(f"Unknown city keys ignored: {missing}")
    return {k: CITIES[k] for k in keys if k in CITIES}


# ── Core scan ─────────────────────────────────────────────────────────────────

async def scan(target_date: date, cities: dict) -> list[TradeOpportunity]:
    """
    Fetch ensemble forecasts for all cities concurrently, then
    query Kalshi markets and find all edge opportunities.
    """
    log.info(f"Scanning {len(cities)} cities for {target_date}...")

    # Step 1: Fetch weather forecasts (all cities in parallel)
    forecasts = await get_all_forecasts(cities, target_date)
    log.info(f"Forecasts received: {len(forecasts)}/{len(cities)} cities")

    # Step 2: Query Kalshi markets for each city with a forecast
    kalshi = KalshiClient()
    all_opportunities: list[TradeOpportunity] = []

    for city_key, forecast in forecasts.items():
        city    = cities[city_key]
        series  = city["kalshi_series"]
        log.info(f"Querying Kalshi markets for {city_key} ({series})...")

        try:
            raw_markets = kalshi.get_markets_for_series(series, status="open")
            if not raw_markets:
                log.warning(f"{city_key}: No open markets found for series {series}")
            else:
                from arb.kalshi import parse_bin_market
                parsed = [p for m in raw_markets if (p := parse_bin_market(m)) is not None]
                log.info(f"{city_key}: {len(raw_markets)} markets → {len(parsed)} parsed bins")
                all_opportunities.extend(find_opportunities(forecast, parsed))

        except Exception as e:
            log.error(f"{city_key}: temperature market query failed — {e}")

        # Precipitation markets (skipped if series not configured)
        precip_series = city.get("kalshi_precip_series", "")
        if precip_series:
            try:
                from arb.kalshi import parse_precip_market
                raw_precip = kalshi.get_markets_for_series(precip_series, status="open")
                if raw_precip:
                    parsed_precip = [p for m in raw_precip if (p := parse_precip_market(m)) is not None]
                    log.info(f"{city_key}: {len(raw_precip)} precip markets → {len(parsed_precip)} parsed")
                    all_opportunities.extend(find_precip_opportunities(forecast, parsed_precip))
            except Exception as e:
                log.error(f"{city_key}: precip market query failed — {e}")

    kalshi.close()
    return all_opportunities


def execute_trades(
    opportunities: list[TradeOpportunity],
    live: bool,
) -> None:
    """
    Size and execute all opportunity trades.
    In dry-run mode, logs what would have been done.
    """
    kalshi  = KalshiClient()
    balance = kalshi.get_balance()
    log.info(f"Kalshi balance: ${balance:,.2f}")

    allocations = allocate(opportunities, balance)
    if not allocations:
        log.info("No trades to execute after allocation.")
        kalshi.close()
        return

    for opp, n_contracts in allocations:
        if not live:
            cost = n_contracts * opp.cost_cents / 100
            log.info(
                f"[DRY RUN] Would BUY {n_contracts}x {opp.ticker_a} YES @ {opp.price_a}¢"
                + (f" + {n_contracts}x {opp.ticker_b} YES @ {opp.price_b}¢" if opp.ticker_b else "")
                + f"  (total cost ≈ ${cost:.2f} | edge={opp.edge:+.1%})"
            )
            continue

        try:
            # Leg A
            kalshi.place_order(
                ticker=opp.ticker_a,
                side="yes",
                action="buy",
                count=n_contracts,
                price=opp.price_a,
            )
            # Leg B (if spread)
            if opp.ticker_b and opp.price_b:
                kalshi.place_order(
                    ticker=opp.ticker_b,
                    side="yes",
                    action="buy",
                    count=n_contracts,
                    price=opp.price_b,
                )
        except Exception as e:
            log.error(f"Order failed for {opp.ticker_a}: {e}")

    kalshi.close()


def show_report() -> None:
    """Print current open positions and recent fills."""
    kalshi = KalshiClient()
    log.info(f"Balance: ${kalshi.get_balance():,.2f}")

    positions = kalshi.get_positions()
    if positions:
        log.info(f"\nOpen Positions ({len(positions)}):")
        for p in positions:
            log.info(f"  {p.get('ticker','?')} | {p.get('position','?')} contracts | value={p.get('value','?')}")
    else:
        log.info("No open positions.")

    fills = kalshi.get_fills()[-20:]   # last 20 fills
    if fills:
        log.info(f"\nRecent Fills ({len(fills)}):")
        for f in fills:
            log.info(
                f"  {f.get('ticker','?')} {f.get('action','?')} {f.get('side','?')} "
                f"{f.get('count','?')} @ {f.get('yes_price','?')}¢ "
                f"ts={f.get('created_time','?')}"
            )
    kalshi.close()


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_once(args: argparse.Namespace) -> None:
    target_date = date.fromisoformat(args.date)
    cities      = get_target_cities(args.cities)

    log.info("=" * 60)
    log.info(f"Kalshi Weather Arb Bot — {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"Target date : {target_date}")
    log.info(f"Cities      : {len(cities)}")
    log.info(f"Mode        : {'LIVE TRADING' if args.live else 'DRY RUN'}")
    log.info("=" * 60)

    if args.report:
        show_report()
        return

    opportunities = await scan(target_date, cities)
    summarize_opportunities(opportunities)

    if not opportunities:
        log.info("Nothing to trade today.")
        return

    execute_trades(opportunities, live=args.live)


async def main() -> None:
    args = parse_args()

    if args.loop:
        import time
        while True:
            try:
                await run_once(args)
            except Exception as e:
                log.error(f"Loop iteration failed: {e}", exc_info=True)
            log.info("Sleeping 3600s (1 hour) until next scan...")
            time.sleep(3600)
    else:
        await run_once(args)


if __name__ == "__main__":
    asyncio.run(main())
