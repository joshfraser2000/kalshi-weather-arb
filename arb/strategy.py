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

# Kalshi charges a fee on each winning contract. We deduct this from the
# expected payout when computing edge so that thin trades are filtered out
# before they're placed. Set KALSHI_FEE_RATE=0 to disable fee adjustment.
KALSHI_FEE_RATE    = float(os.getenv("KALSHI_FEE_RATE",    "0.07"))  # 7% of payout
# Minimum expected gross profit in cents per contract after fees.
# Trades earning less than this are skipped even if edge% passes.
MIN_PROFIT_CENTS   = float(os.getenv("MIN_PROFIT_CENTS",   "3"))     # at least 3¢ per contract

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
    side_a:         str = "yes"     # "yes" or "no" — which side to buy for ticker_a
    side_b:         str = "yes"     # always "yes" for adjacent spreads

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

        # Fee-adjusted edge: expected payout is reduced by Kalshi's fee on
        # the winning contract. Without this, thin trades look profitable but
        # lose money once the fee is deducted at settlement.
        fee_adj_payout = 1.0 - KALSHI_FEE_RATE
        edge = our_prob * fee_adj_payout - market_implied

        # Also require a minimum absolute profit (in cents) to filter out
        # trades where the gross profit is so small fees make it a loss.
        expected_profit_cents = (our_prob * fee_adj_payout - market_implied) * 100

        if edge < MIN_EDGE:
            continue
        if expected_profit_cents < MIN_PROFIT_CENTS:
            log.info(f"  SKIP spread {bin_a['ticker']}+{bin_b['ticker']}: "
                     f"profit {expected_profit_cents:.1f}¢ < {MIN_PROFIT_CENTS:.0f}¢ min")
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
        if market.get("type") != "bin":
            continue
        our_prob = forecast.prob_in_range(market["low"], market["high"])
        price    = market["yes_ask"]
        market_implied = price / 100.0
        fee_adj_payout = 1.0 - KALSHI_FEE_RATE
        edge = our_prob * fee_adj_payout - market_implied
        expected_profit_cents = edge * 100

        if edge < STRONG_EDGE:   # higher bar for single-bin
            continue
        if expected_profit_cents < MIN_PROFIT_CENTS:
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

    # ── Threshold markets (above/below X°F) — check both YES and NO sides ─────
    threshold_markets = [m for m in markets if m.get("type") == "threshold"]
    for mkt in threshold_markets:
        threshold = mkt["threshold"]
        yes_ask = mkt["yes_ask"]
        # NO ask = 100 - yes_bid (what it costs to buy NO / be short)
        yes_bid = mkt.get("yes_bid", 100 - yes_ask)
        no_ask  = mkt.get("no_ask", 100 - yes_bid)

        prob_above = forecast.prob_above(threshold)
        prob_below = 1.0 - prob_above

        # YES direction: buy YES at yes_ask, win if high ≥ threshold
        fee_adj = 1.0 - KALSHI_FEE_RATE
        edge_yes = prob_above * fee_adj - yes_ask / 100.0
        # NO direction: buy NO at no_ask, win if high < threshold
        edge_no  = prob_below * fee_adj - no_ask / 100.0

        # Take whichever direction has better edge
        if edge_yes >= edge_no:
            edge, side, price, our_prob = edge_yes, "yes", yes_ask, prob_above
            low_t, high_t = threshold, 999.0
            dir_label = f"≥{threshold:.0f}°F"
        else:
            edge, side, price, our_prob = edge_no, "no", no_ask, prob_below
            low_t, high_t = 0.0, threshold
            dir_label = f"<{threshold:.0f}°F"

        if edge < MIN_EDGE:
            continue
        if our_prob < MIN_PROB:
            continue

        # NWS sanity check: don't trade against the official forecast
        if forecast.nws_high is not None:
            nws_above = forecast.nws_high >= threshold
            model_above = side == "yes"
            if nws_above != model_above:
                log.info(
                    f"  VETO {forecast.city_key} threshold {threshold:.0f}°F: "
                    f"NWS says {forecast.nws_high:.0f}°F but model wants {side.upper()}"
                )
                continue

        opp = TradeOpportunity(
            city_key       = forecast.city_key,
            trade_type     = "single_bin",
            ticker_a       = mkt["ticker"],
            price_a        = price,
            side_a         = side,
            our_prob       = our_prob,
            market_implied = price / 100.0,
            edge           = edge,
            low_temp       = low_t,
            high_temp      = high_t,
            forecast_mean  = forecast.corrected_mean,
            forecast_std   = forecast.std,
            ensemble_n     = len(forecast.members),
        )
        opp.notes = [
            f"threshold: high {dir_label} (buy {side.upper()})",
            f"ensemble μ={forecast.corrected_mean:.1f}°F σ={forecast.std:.1f}°F",
        ]
        log.info(f"  THRESHOLD found: {opp}")
        opportunities.append(opp)

    # Sort best edge first, cap per city
    opportunities.sort(key=lambda o: o.edge, reverse=True)
    return opportunities[:MAX_SPREAD]


def find_precip_opportunities(
    forecast: ForecastResult,
    markets:  list[dict],   # parsed precipitation markets for this city
) -> list[TradeOpportunity]:
    """
    Find edge in Kalshi precipitation threshold markets.

    Kalshi rain markets are threshold-style: "Will it rain at least X inches?"
    YES = precip >= threshold, NO = precip < threshold.

    We compare our ensemble-derived probability to the market's implied price
    using the same edge formula as temperature threshold markets.
    """
    opportunities: list[TradeOpportunity] = []

    for mkt in markets:
        if mkt.get("type") != "precip_threshold":
            continue

        threshold = mkt["threshold"]   # inches
        yes_ask   = mkt["yes_ask"]
        yes_bid   = mkt.get("yes_bid", 100 - yes_ask)
        no_ask    = mkt.get("no_ask", 100 - yes_bid)

        prob_rain    = forecast.prob_precip_above(threshold)
        prob_no_rain = 1.0 - prob_rain

        fee_adj   = 1.0 - KALSHI_FEE_RATE
        edge_yes  = prob_rain    * fee_adj - yes_ask / 100.0
        edge_no   = prob_no_rain * fee_adj - no_ask  / 100.0

        # NWS sanity check: if NWS precip probability available, veto if contradictory
        if forecast.nws_precip_prob is not None:
            nws_says_rain = forecast.nws_precip_prob >= 0.5
            if edge_yes >= edge_no and not nws_says_rain:
                log.info(
                    f"  VETO {forecast.city_key} rain ≥{threshold}\": "
                    f"model wants YES but NWS POP={forecast.nws_precip_prob:.0%}"
                )
                continue
            if edge_no > edge_yes and nws_says_rain:
                log.info(
                    f"  VETO {forecast.city_key} rain ≥{threshold}\": "
                    f"model wants NO but NWS POP={forecast.nws_precip_prob:.0%}"
                )
                continue

        if edge_yes >= edge_no:
            edge, side, price, our_prob = edge_yes, "yes", yes_ask, prob_rain
            dir_label = f"≥{threshold}\" rain"
        else:
            edge, side, price, our_prob = edge_no, "no", no_ask, prob_no_rain
            dir_label = f"<{threshold}\" rain"

        if edge < MIN_EDGE:
            continue
        if our_prob < MIN_PROB:
            continue
        if (edge * 100) < MIN_PROFIT_CENTS:
            continue

        opp = TradeOpportunity(
            city_key       = forecast.city_key,
            trade_type     = "single_bin",
            ticker_a       = mkt["ticker"],
            price_a        = price,
            side_a         = side,
            our_prob       = our_prob,
            market_implied = price / 100.0,
            edge           = edge,
            low_temp       = 0.0,
            high_temp      = threshold,
            forecast_mean  = forecast.precip_mean,
            forecast_std   = forecast.precip_std,
            ensemble_n     = len(forecast.precip_members),
        )
        opp.notes = [
            f"rain threshold: {dir_label} (buy {side.upper()})",
            f"ensemble precip μ={forecast.precip_mean:.2f}\" σ={forecast.precip_std:.2f}\"",
        ]
        if forecast.nws_precip_prob is not None:
            opp.notes.append(f"NWS POP={forecast.nws_precip_prob:.0%}")
        log.info(f"  PRECIP found: {opp}")
        opportunities.append(opp)

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
