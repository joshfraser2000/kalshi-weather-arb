"""
Edge finding and trade decision engine.

Strategy: Adjacent-Bin Temperature Spread Arbitrage
─────────────────────────────────────────────────────
Kalshi weather markets price individual temperature bins (e.g., 70-75°F).
Because the market prices each bin independently, the sum of adjacent
bin prices is often LESS than the true combined probability — creating
a structural pricing inefficiency.

We:
  1. Fetch our ensemble-based probability for each bin
  2. Compare to market's implied probability (mid-price of yes_ask/yes_bid)
  3. If (our_prob − market_implied) > MIN_EDGE, flag as a trade opportunity
  4. For adjacent-bin spreads: buy YES on both adjacent bins simultaneously
     — we collect $1 if either resolves, paying only the sum of both prices.

Single-bin trades are also supported when the edge is very large (≥ STRONG_EDGE).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from arb.logger  import get_logger
from arb.weather import ForecastResult

log = get_logger("strategy")

# ── Config (tunable via env) ──────────────────────────────────────────────────

MIN_EDGE    = float(os.getenv("MIN_EDGE",    "0.06"))   # 6% minimum edge to trade
STRONG_EDGE = float(os.getenv("STRONG_EDGE", "0.12"))   # 12% = strong confidence
MIN_PROB    = float(os.getenv("MIN_PROB",    "0.55"))   # only trade bins with ≥55% win probability
MAX_SPREAD  = int(os.getenv("MAX_SPREAD",    "5"))      # max positions per city per day

TradeType = Literal["single_bin", "adjacent_spread"]


@dataclass
class TradeOpportunity:
    """
    A single trade opportunity identified by the strategy.

    For adjacent_spread: buy YES on both bin_a and bin_b.
      - We win if the actual high falls in [bin_a.low, bin_b.high).
      - Cost = price_a + price_b (cents)
      - Payout = 100¢ ($1.00)
      - Edge = our_prob_combined − ((price_a + price_b) / 100)

    For single_bin: buy YES on one bin.
      - Edge = our_prob_bin − (price / 100)
    """
    city_key:       str
    trade_type:     TradeType
    ticker_a:       str
    price_a:        int        # yes_ask in cents
    ticker_b:       str | None = None
    price_b:        int | None = None

    our_prob:       float = 0.0    # our model's P(win)
    market_implied: float = 0.0    # market's implied P(win), = cost/100
    edge:           float = 0.0    # our_prob − market_implied

    low_temp:  float = 0.0         # lower bound of covered range (°F)
    high_temp: float = 0.0         # upper bound of covered range (°F)

    forecast_mean: float = 0.0
    forecast_std:  float = 0.0
    ensemble_n:    int   = 0

    notes: list[str] = field(default_factory=list)

    @property
    def cost_cents(self) -> int:
        """Total cost per spread/single in cents (0–100)."""
        return self.price_a + (self.price_b or 0)

    @property
    def is_strong(self) -> bool:
        return self.edge >= STRONG_EDGE

    def __repr__(self) -> str:
        bins = f"{self.ticker_a}"
        if self.ticker_b:
            bins += f" + {self.ticker_b}"
        return (
            f"TradeOpportunity({self.city_key} {self.trade_type} "
            f"{self.low_temp:.0f}-{self.high_temp:.0f}°F "
            f"P={self.our_prob:.1%} mkt={self.market_implied:.1%} "
            f"edge={self.edge:+.1%} cost={self.cost_cents}¢)"
        )


def find_opportunities(
    forecast: ForecastResult,
    markets:  list[dict],      # parsed bin markets for this city (from kalshi.find_adjacent_bins)
) -> list[TradeOpportunity]:
    """
    Given a city forecast and its open Kalshi bin markets, find all
    trades where our model edge exceeds MIN_EDGE.

    Parameters
    ----------
    forecast : ForecastResult from weather.py
    markets  : list of parsed market dicts {ticker, low, high, yes_ask, yes_bid}
    """
    from arb.kalshi import find_adjacent_bins

    opportunities: list[TradeOpportunity] = []

    # ── Adjacent-bin spreads ─────────────────────────────────────────────────
    pairs = find_adjacent_bins(markets)
    for bin_a, bin_b in pairs:
        low  = bin_a["low"]
        high = bin_b["high"]

        # Our combined probability for both bins
        our_prob = forecast.prob_in_range(low, high)

        # Market's implied probability: sum of the ask prices (cost to enter)
        # We buy YES on both at the ask price
        price_a = bin_a["yes_ask"]
        price_b = bin_b["yes_ask"]
        market_implied = (price_a + price_b) / 100.0

        edge = our_prob - market_implied

        if edge < MIN_EDGE:
            continue
        if our_prob < MIN_PROB:
            continue

        opp = TradeOpportunity(
            city_key       = forecast.city_key,
            trade_type     = "adjacent_spread",
            ticker_a       = bin_a["ticker"],
            price_a        = price_a,
            ticker_b       = bin_b["ticker"],
            price_b        = price_b,
            our_prob       = our_prob,
            market_implied = market_implied,
            edge           = edge,
            low_temp       = low,
            high_temp      = high,
            forecast_mean  = forecast.corrected_mean,
            forecast_std   = forecast.std,
            ensemble_n     = len(forecast.members),
        )

        notes = [
            f"ensemble μ={forecast.corrected_mean:.1f}°F σ={forecast.std:.1f}°F",
            f"covers {high-low:.0f}°F range centered on forecast",
        ]
        if forecast.nws_high:
            nws_in_range = low <= forecast.nws_high < high
            notes.append(f"NWS high={forecast.nws_high:.0f}°F {'✓ in range' if nws_in_range else '✗ outside range'}")
        opp.notes = notes

        log.info(f"  SPREAD found: {opp}")
        opportunities.append(opp)

    # ── Single-bin opportunities (strong edge only) ───────────────────────────
    for market in markets:
        our_prob = forecast.prob_in_range(market["low"], market["high"])
        price    = market["yes_ask"]
        market_implied = price / 100.0
        edge = our_prob - market_implied

        if edge < STRONG_EDGE:   # higher bar for single-bin
            continue
        if our_prob < MIN_PROB:
            continue

        opp = TradeOpportunity(
            city_key       = forecast.city_key,
            trade_type     = "single_bin",
            ticker_a       = market["ticker"],
            price_a        = price,
            our_prob       = our_prob,
            market_implied = market_implied,
            edge           = edge,
            low_temp       = market["low"],
            high_temp      = market["high"],
            forecast_mean  = forecast.corrected_mean,
            forecast_std   = forecast.std,
            ensemble_n     = len(forecast.members),
        )
        log.info(f"  SINGLE-BIN found: {opp}")
        opportunities.append(opp)

    # Sort best edge first, cap per city
    opportunities.sort(key=lambda o: o.edge, reverse=True)
    return opportunities[:MAX_SPREAD]


def summarize_opportunities(opps: list[TradeOpportunity]) -> None:
    """Pretty-print a ranked table of all opportunities."""
    if not opps:
        log.info("No opportunities found meeting edge criteria.")
        return

    log.info("=" * 70)
    log.info(f"{'CITY':<8} {'TYPE':<18} {'RANGE':<12} {'P(win)':<8} {'MKT':<8} {'EDGE':<8} {'COST'}")
    log.info("-" * 70)
    for o in sorted(opps, key=lambda x: x.edge, reverse=True):
        rng = f"{o.low_temp:.0f}-{o.high_temp:.0f}°F"
        log.info(
            f"{o.city_key:<8} {o.trade_type:<18} {rng:<12} "
            f"{o.our_prob:.1%}    {o.market_implied:.1%}   "
            f"{o.edge:+.1%}   {o.cost_cents}¢"
        )
    log.info("=" * 70)
    log.info(f"Total opportunities: {len(opps)} | "
             f"Strong ({STRONG_EDGE:.0%}+ edge): {sum(1 for o in opps if o.is_strong)}")
