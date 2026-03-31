"""
Kalshi Weather Arb — Backtester
════════════════════════════════

Simulates the adjacent-bin spread strategy against historical data.

Methodology
───────────
Since historical Kalshi market prices are not available via API, we model
them using climatological probabilities derived from 10 years of observed
temperature data — the same information a rational market maker would use.

For each historical day:
  1. Pull ACTUAL observed high temp from Open-Meteo archive  (ground truth)
  2. Pull 10-year historical highs for that city/month       (climatology)
  3. Compute empirical bin probabilities from climatology    (simulated market price)
  4. Add synthetic market inefficiency (-2 to -8% per bin)  (the arb opportunity)
  5. Simulate our ensemble forecast using observed + noise   (what we'd have predicted)
  6. Find trades where our model edges the market
  7. Resolve each trade using the actual observed high
  8. Track P&L

Usage
─────
  python3 backtest.py                        # backtest last 90 days, all cities
  python3 backtest.py --days 180             # last 180 days
  python3 backtest.py --cities NYC,CHI,LAX   # specific cities
  python3 backtest.py --start 2025-10-01 --end 2026-01-01
  python3 backtest.py --bankroll 5000        # starting capital
  python3 backtest.py --output results.json  # save results to file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from arb.cities   import CITIES
from arb.weather  import ForecastResult
from arb.strategy import find_opportunities, MIN_EDGE, MIN_PROB
from arb.sizing   import allocate, KELLY_FRACTION, MAX_PCT_BANKROLL, MAX_TOTAL_DEPLOY
from arb.logger   import get_logger

log = get_logger("backtest")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# How many ensemble members to synthesize from historical data
SYNTH_MEMBERS = 51
# Std-dev of noise added to actual temp to simulate ensemble spread
ENSEMBLE_NOISE_STD = 3.5   # °F — realistic NWP spread for day-1 forecasts
# Simulated market inefficiency: market prices bins X% cheaper than climatology
MARKET_DISCOUNT_MEAN = 0.045   # 4.5% average discount
MARKET_DISCOUNT_STD  = 0.020   # ± 2% variation


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DayResult:
    date:       str
    city:       str
    n_trades:   int
    gross_pnl:  float
    net_pnl:    float
    fees:       float
    deployed:   float
    win_trades: int
    lose_trades: int


@dataclass
class BacktestResult:
    start_date:     str
    end_date:       str
    cities:         list[str]
    starting_bank:  float
    ending_bank:    float
    total_days:     int
    trading_days:   int
    win_days:       int
    loss_days:      int
    total_gross:    float
    total_fees:     float
    total_net:      float
    avg_daily_net:  float
    win_rate_days:  float
    win_rate_trades: float
    sharpe_ratio:   float
    max_drawdown:   float
    best_day:       float
    worst_day:      float
    total_trades:   int
    daily:          list[dict] = field(default_factory=list)
    city_stats:     dict       = field(default_factory=dict)


# ── Historical data fetching ──────────────────────────────────────────────────

async def fetch_historical_highs(
    lat: float,
    lon: float,
    start: date,
    end: date,
    client: httpx.AsyncClient,
) -> dict[str, float]:
    """
    Fetch observed daily high temperatures from Open-Meteo archive.
    Returns {date_str: high_temp_F}.
    """
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
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()["daily"]
    return {d: t for d, t in zip(data["time"], data["temperature_2m_max"]) if t is not None}


async def fetch_climatology(
    lat: float,
    lon: float,
    month: int,
    client: httpx.AsyncClient,
    years: int = 10,
) -> list[float]:
    """
    Fetch 10 years of historical highs for the same month to build
    a climatological distribution.
    """
    today    = date.today()
    all_highs: list[float] = []

    for yr in range(today.year - years, today.year):
        try:
            start = date(yr, month, 1)
            # Last day of month
            if month == 12:
                end = date(yr, 12, 31)
            else:
                end = date(yr, month + 1, 1) - timedelta(days=1)
            obs = await fetch_historical_highs(lat, lon, start, end, client)
            all_highs.extend(obs.values())
        except Exception:
            pass

    return all_highs


# ── Simulation helpers ────────────────────────────────────────────────────────

def synthesize_forecast(actual_high: float, noise_std: float = ENSEMBLE_NOISE_STD) -> ForecastResult:
    """
    Simulate what an ensemble forecast would have looked like for a day
    where the actual high was `actual_high`. We add Gaussian noise to
    represent the uncertainty in a real ensemble.
    """
    rng = random.gauss
    members = [actual_high + rng(0, noise_std) for _ in range(SYNTH_MEMBERS)]
    return ForecastResult(
        city_key    = "SIM",
        target_date = date.today(),
        members     = members,
        bias_correction = rng(0, 0.5),   # small realistic bias
        nws_high    = actual_high + rng(0, 1.0),
    )


def synthetic_market_price(climatological_prob: float) -> int:
    """
    Model the market price for a bin with true climatological probability p.
    Markets are systematically underpriced by MARKET_DISCOUNT% (the arb).
    Returns price in cents (1–99).
    """
    discount = max(0.01, min(0.15, random.gauss(MARKET_DISCOUNT_MEAN, MARKET_DISCOUNT_STD)))
    market_prob = max(0.01, min(0.99, climatological_prob - discount))
    return max(1, min(99, round(market_prob * 100)))


def build_synthetic_markets(
    clim_highs: list[float],
    forecast:   ForecastResult,
    bin_width:  float = 5.0,
) -> list[dict]:
    """
    Build synthetic Kalshi-style bin markets from climatological data.
    Only create bins where the climatological probability is meaningful (>2%).
    """
    if not clim_highs:
        return []

    low_t  = math.floor(min(clim_highs) / bin_width) * bin_width
    high_t = math.ceil(max(clim_highs) / bin_width) * bin_width
    n      = len(clim_highs)

    markets = []
    t = low_t
    while t < high_t:
        in_bin   = sum(1 for h in clim_highs if t <= h < t + bin_width)
        clim_prob = in_bin / n
        if clim_prob < 0.02:
            t += bin_width
            continue
        markets.append({
            "ticker":   f"SIM-B{int(t)}",
            "low":      t,
            "high":     t + bin_width,
            "yes_ask":  synthetic_market_price(clim_prob),
            "yes_bid":  synthetic_market_price(clim_prob) - random.randint(1, 3),
            "title":    f"High between {t:.0f}°F and {t+bin_width:.0f}°F",
        })
        t += bin_width

    return markets


def resolve_trade(opp, actual_high: float) -> tuple[bool, float]:
    """
    Determine if a trade won, and return gross P&L per contract in dollars.

    Adjacent spread: win if actual_high falls in [low, high)
    Single bin:      win if actual_high falls in [low, high)
    """
    won = opp.low_temp <= actual_high < opp.high_temp
    if won:
        gross_per_contract = (100 - opp.cost_cents) / 100.0   # payout minus cost
    else:
        gross_per_contract = -(opp.cost_cents / 100.0)         # lose cost
    return won, gross_per_contract


# ── Main backtest loop ────────────────────────────────────────────────────────

async def backtest_city(
    city_key:   str,
    city:       dict,
    start:      date,
    end:        date,
    client:     httpx.AsyncClient,
) -> list[DayResult]:
    """Run the strategy simulation for one city over the date range."""
    log.info(f"{city_key}: fetching historical data {start} → {end}")

    try:
        actual_highs = await fetch_historical_highs(city["lat"], city["lon"], start, end, client)
    except Exception as e:
        log.error(f"{city_key}: failed to fetch historical data — {e}")
        return []

    # Pre-fetch climatology by month (cached within this run)
    clim_cache: dict[int, list[float]] = {}

    results: list[DayResult] = []
    for day_str, actual_high in sorted(actual_highs.items()):
        day = date.fromisoformat(day_str)
        m   = day.month

        if m not in clim_cache:
            try:
                clim_cache[m] = await fetch_climatology(city["lat"], city["lon"], m, client)
            except Exception as e:
                log.warning(f"{city_key} {day}: climatology fetch failed — {e}")
                clim_cache[m] = []

        clim = clim_cache.get(m, [])
        if not clim:
            continue

        # Synthesize what our forecast would have been
        forecast          = synthesize_forecast(actual_high)
        forecast.city_key = city_key

        # Build synthetic market
        markets = build_synthetic_markets(clim, forecast)
        if not markets:
            continue

        # Find opportunities (same logic as live trading)
        opps = find_opportunities(forecast, markets)
        if not opps:
            results.append(DayResult(day_str, city_key, 0, 0, 0, 0, 0, 0, 0))
            continue

        # Size positions (use unit bankroll — we scale later)
        allocs = allocate(opps, 1000.0)

        gross, fees, deployed = 0.0, 0.0, 0.0
        win_t, lose_t = 0, 0
        for opp, n in allocs:
            won, pnl_per = resolve_trade(opp, actual_high)
            cost = n * opp.cost_cents / 100
            fee  = cost * 0.001   # 0.1% Kalshi fee
            gross    += n * pnl_per
            fees     += fee
            deployed += cost
            if won: win_t += 1
            else:   lose_t += 1

        results.append(DayResult(
            date        = day_str,
            city        = city_key,
            n_trades    = len(allocs),
            gross_pnl   = round(gross, 4),
            net_pnl     = round(gross - fees, 4),
            fees        = round(fees, 4),
            deployed    = round(deployed, 4),
            win_trades  = win_t,
            lose_trades = lose_t,
        ))

    return results


async def run_backtest(
    cities:    dict,
    start:     date,
    end:       date,
    bankroll:  float,
) -> BacktestResult:
    """Run the full multi-city backtest."""
    log.info("=" * 60)
    log.info(f"Backtest: {start} → {end} | {len(cities)} cities | ${bankroll:,.0f} bankroll")
    log.info("=" * 60)

    async with httpx.AsyncClient() as client:
        tasks = [
            backtest_city(k, c, start, end, client)
            for k, c in cities.items()
        ]
        city_results = await asyncio.gather(*tasks)

    # Aggregate by date across all cities (one portfolio per day)
    by_date: dict[str, DayResult] = {}
    city_stats: dict[str, dict] = {k: {"net": 0, "trades": 0, "wins": 0} for k in cities}

    for city_key, results in zip(cities.keys(), city_results):
        for r in results:
            if r.date not in by_date:
                by_date[r.date] = DayResult(r.date, "portfolio", 0, 0, 0, 0, 0, 0, 0)
            d = by_date[r.date]
            # Scale P&L from unit ($1000) bankroll to actual bankroll
            scale = bankroll / 1000.0
            d.n_trades   += r.n_trades
            d.gross_pnl  += r.gross_pnl * scale
            d.net_pnl    += r.net_pnl   * scale
            d.fees       += r.fees      * scale
            d.deployed   += r.deployed  * scale
            d.win_trades += r.win_trades
            d.lose_trades += r.lose_trades

            city_stats[city_key]["net"]    += r.net_pnl * scale
            city_stats[city_key]["trades"] += r.n_trades
            city_stats[city_key]["wins"]   += r.win_trades

    daily = sorted(by_date.values(), key=lambda x: x.date)
    daily_nets = [d.net_pnl for d in daily if d.n_trades > 0]

    if not daily_nets:
        log.error("No trading days found — check date range and city data")
        raise ValueError("Empty backtest")

    win_days  = sum(1 for n in daily_nets if n > 0)
    loss_days = len(daily_nets) - win_days

    total_gross = sum(d.gross_pnl for d in daily)
    total_fees  = sum(d.fees      for d in daily)
    total_net   = sum(d.net_pnl   for d in daily)

    avg_daily = statistics.mean(daily_nets)
    std_daily = statistics.stdev(daily_nets) if len(daily_nets) > 1 else 0

    sharpe = (avg_daily / std_daily * math.sqrt(252)) if std_daily > 0 else 0

    # Max drawdown
    peak, max_dd = 0.0, 0.0
    cumulative = 0.0
    for n in daily_nets:
        cumulative += n
        peak = max(peak, cumulative)
        dd   = peak - cumulative
        max_dd = max(max_dd, dd)

    total_trades = sum(d.n_trades   for d in daily)
    total_wins   = sum(d.win_trades for d in daily)
    total_loses  = sum(d.lose_trades for d in daily)

    result = BacktestResult(
        start_date      = str(start),
        end_date        = str(end),
        cities          = list(cities.keys()),
        starting_bank   = bankroll,
        ending_bank     = round(bankroll + total_net, 2),
        total_days      = (end - start).days,
        trading_days    = len(daily_nets),
        win_days        = win_days,
        loss_days       = loss_days,
        total_gross     = round(total_gross, 2),
        total_fees      = round(total_fees, 2),
        total_net       = round(total_net, 2),
        avg_daily_net   = round(avg_daily, 2),
        win_rate_days   = round(win_days / len(daily_nets) * 100, 1),
        win_rate_trades = round(total_wins / max(total_trades, 1) * 100, 1),
        sharpe_ratio    = round(sharpe, 2),
        max_drawdown    = round(max_dd, 2),
        best_day        = round(max(daily_nets), 2),
        worst_day       = round(min(daily_nets), 2),
        total_trades    = total_trades,
        daily           = [asdict(d) for d in daily],
        city_stats      = {k: {**v, "net": round(v["net"], 2)} for k, v in city_stats.items()},
    )
    return result


def print_report(r: BacktestResult) -> None:
    print()
    print("═" * 60)
    print("  KALSHI WEATHER ARB — BACKTEST RESULTS")
    print("═" * 60)
    print(f"  Period:          {r.start_date} → {r.end_date} ({r.total_days} days)")
    print(f"  Cities:          {len(r.cities)}")
    print(f"  Starting bank:   ${r.starting_bank:>10,.2f}")
    print(f"  Ending bank:     ${r.ending_bank:>10,.2f}")
    print(f"  Total return:    {(r.ending_bank/r.starting_bank - 1)*100:>+9.1f}%")
    print("─" * 60)
    print(f"  Net P&L:         ${r.total_net:>+10,.2f}")
    print(f"  Gross P&L:       ${r.total_gross:>+10,.2f}")
    print(f"  Total fees:      ${r.total_fees:>10,.2f}")
    print(f"  Avg daily net:   ${r.avg_daily_net:>+10,.2f}")
    print("─" * 60)
    print(f"  Trading days:    {r.trading_days}")
    print(f"  Win days:        {r.win_days}  ({r.win_rate_days:.1f}%)")
    print(f"  Loss days:       {r.loss_days}")
    print(f"  Best day:        ${r.best_day:>+10,.2f}")
    print(f"  Worst day:       ${r.worst_day:>+10,.2f}")
    print("─" * 60)
    print(f"  Total trades:    {r.total_trades}")
    print(f"  Trade win rate:  {r.win_rate_trades:.1f}%")
    print(f"  Sharpe ratio:    {r.sharpe_ratio:.2f}  (annualized)")
    print(f"  Max drawdown:    ${r.max_drawdown:>10,.2f}")
    print("─" * 60)
    print("  TOP CITIES BY NET P&L:")
    top_cities = sorted(r.city_stats.items(), key=lambda x: x[1]["net"], reverse=True)[:10]
    for city, stats in top_cities:
        wr = stats["wins"] / max(stats["trades"], 1) * 100
        print(f"    {city:<10}  ${stats['net']:>+8,.2f}   {stats['trades']:>4} trades  {wr:.0f}% win")
    print("═" * 60)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Kalshi Weather Arb Backtester")
    p.add_argument("--days",     type=int,   default=90,           help="Number of days to backtest (default 90)")
    p.add_argument("--start",    default=None,                     help="Start date YYYY-MM-DD (overrides --days)")
    p.add_argument("--end",      default=str(date.today() - timedelta(days=2)), help="End date YYYY-MM-DD")
    p.add_argument("--cities",   default=None,                     help="Comma-separated city keys (default: all)")
    p.add_argument("--bankroll", type=float, default=1000.0,       help="Starting bankroll in $ (default 1000)")
    p.add_argument("--output",   default=None,                     help="Save results JSON to file")
    p.add_argument("--seed",     type=int,   default=42,           help="Random seed for reproducibility")
    return p.parse_args()


async def main():
    args = parse_args()
    random.seed(args.seed)

    end_date   = date.fromisoformat(args.end)
    start_date = date.fromisoformat(args.start) if args.start else end_date - timedelta(days=args.days)

    if args.cities:
        keys   = [k.strip().upper() for k in args.cities.split(",")]
        cities = {k: CITIES[k] for k in keys if k in CITIES}
    else:
        cities = CITIES

    result = await run_backtest(cities, start_date, end_date, args.bankroll)
    print_report(result)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(asdict(result), indent=2))
        log.info(f"Results saved to {out}")
    else:
        default_out = Path("backtest_results") / f"backtest_{start_date}_{end_date}.json"
        default_out.write_text(json.dumps(asdict(result), indent=2))
        log.info(f"Results saved to {default_out}")


if __name__ == "__main__":
    asyncio.run(main())
