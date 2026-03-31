"""
Position sizing via fractional Kelly criterion.

Kelly formula (for binary bets):
  f* = (p * b - q) / b
where:
  p = probability of winning
  q = 1 - p
  b = net odds (payout / stake - 1)

For a Kalshi adjacent-bin spread:
  stake  = cost_cents / 100  (fraction of $1 notional)
  payout = $1.00 per contract
  b = (1.00 - stake) / stake = (100 - cost_cents) / cost_cents

We apply a KELLY_FRACTION (default 0.25 = quarter-Kelly) for safety,
then cap at MAX_PCT_BANKROLL per trade and MAX_TOTAL_DEPLOYED overall.
"""

from __future__ import annotations

import os

from arb.logger   import get_logger
from arb.strategy import TradeOpportunity

log = get_logger("sizing")

KELLY_FRACTION    = float(os.getenv("KELLY_FRACTION",    "0.25"))  # fractional Kelly (safety)
MAX_PCT_BANKROLL  = float(os.getenv("MAX_PCT_BANKROLL",  "0.05"))  # max 5% per trade
MAX_TOTAL_DEPLOY  = float(os.getenv("MAX_TOTAL_DEPLOY",  "0.40"))  # max 40% of bankroll at risk
MIN_CONTRACTS     = int(os.getenv("MIN_CONTRACTS",       "1"))
MAX_CONTRACTS     = int(os.getenv("MAX_CONTRACTS",       "500"))


def kelly_contracts(
    opp:      TradeOpportunity,
    bankroll: float,           # available capital in $
) -> int:
    """
    Return the number of contracts to buy for a given opportunity.
    Each Kalshi contract has a face value of $1.

    For a spread: you buy 'n' contracts on EACH leg, total cost = n * cost_cents / 100.
    Returns number of contracts (integer, ≥ 0).
    """
    p    = opp.our_prob
    q    = 1.0 - p
    cost = opp.cost_cents / 100.0     # fraction of $1 per contract
    net_odds = (1.0 - cost) / cost    # b in the Kelly formula

    if net_odds <= 0 or p <= 0:
        return 0

    raw_kelly = (p * net_odds - q) / net_odds   # Kelly fraction of bankroll
    if raw_kelly <= 0:
        return 0

    # Apply fractional Kelly and hard cap
    target_dollar = bankroll * min(raw_kelly * KELLY_FRACTION, MAX_PCT_BANKROLL)
    contracts = int(target_dollar / cost)

    contracts = max(MIN_CONTRACTS, min(MAX_CONTRACTS, contracts))
    log.debug(
        f"{opp.city_key} kelly: p={p:.2%} odds={net_odds:.2f} "
        f"raw_f={raw_kelly:.3f} → {contracts} contracts (${contracts*cost:.2f} cost)"
    )
    return contracts


def allocate(
    opportunities: list[TradeOpportunity],
    bankroll:      float,
) -> list[tuple[TradeOpportunity, int]]:
    """
    Allocate contracts across all opportunities, respecting MAX_TOTAL_DEPLOY.

    Returns list of (opportunity, n_contracts) sorted by edge descending.
    """
    max_deploy   = bankroll * MAX_TOTAL_DEPLOY
    deployed     = 0.0
    allocations  = []

    # Best edge first
    for opp in sorted(opportunities, key=lambda o: o.edge, reverse=True):
        remaining = max_deploy - deployed
        if remaining <= 0:
            break

        n = kelly_contracts(opp, bankroll)
        if n <= 0:
            continue

        cost_per_contract = opp.cost_cents / 100.0
        actual_cost = n * cost_per_contract
        if actual_cost > remaining:
            n = int(remaining / cost_per_contract)
        if n <= 0:
            continue

        deployed += n * cost_per_contract
        allocations.append((opp, n))
        log.info(
            f"ALLOCATE {opp.city_key} {opp.trade_type}: {n} contracts "
            f"× {opp.cost_cents}¢ = ${n * cost_per_contract:.2f} | "
            f"edge={opp.edge:+.1%} | deployed={deployed:.2f}/{max_deploy:.2f}"
        )

    log.info(
        f"Total allocated: {len(allocations)} trades | "
        f"${deployed:.2f} deployed ({deployed/bankroll:.1%} of bankroll)"
    )
    return allocations
