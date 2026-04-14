"""
Kalshi Weather Arb — Web Dashboard + Auto-Scanner

Run with:
  python3 dashboard/app.py

Then open http://localhost:5050

The background scanner runs every SCAN_INTERVAL_MINS during trading hours
(SCAN_START_ET – SCAN_END_ET) and auto-executes if AUTO_EXECUTE=true and
today's profit is below DAILY_PROFIT_GOAL.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

load_dotenv()

from arb.cities   import CITIES
from arb.weather  import get_all_forecasts
from arb.kalshi   import KalshiClient
from arb.strategy import find_opportunities, TradeOpportunity
from arb.sizing   import allocate
from arb.logger   import get_logger

log = get_logger("dashboard")
app = Flask(__name__)

KALSHI_CONFIGURED   = bool(os.getenv("KALSHI_KEY_ID") or (os.getenv("KALSHI_EMAIL") and os.getenv("KALSHI_PASSWORD")))
AUTO_EXECUTE        = os.getenv("AUTO_EXECUTE",       "false").lower() == "true"
DAILY_PROFIT_GOAL   = float(os.getenv("DAILY_PROFIT_GOAL", "100"))   # stop trading once daily P&L hits this
SCAN_INTERVAL_MINS  = int(os.getenv("SCAN_INTERVAL_MINS",  "30"))    # how often to scan (minutes)
SCAN_START_ET       = int(os.getenv("SCAN_START_ET",       "9"))     # 9 AM ET — markets open
SCAN_END_ET         = int(os.getenv("SCAN_END_ET",         "15"))    # 3 PM ET — markets thin out
MAX_BID_ASK_SPREAD  = int(os.getenv("MAX_BID_ASK_SPREAD",  "55"))    # max bid-ask spread in cents to consider liquid
KALSHI_FUNDED       = float(os.getenv("KALSHI_FUNDED",     "0"))     # amount deposited — used to compute total net P&L
CACHE_TTL_SECONDS   = 300

ET = ZoneInfo("America/New_York")

# ── Ticker date filter ────────────────────────────────────────────────────────

_MONTH_NUM = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
              "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

def _ticker_date_ok(ticker: str) -> bool:
    """Return True if the market date encoded in the ticker is today or future.

    Kalshi tickers look like KXHIGHTHOU-26APR13-T89 where 26APR13 = 2026-04-13.
    Returns True (keep market) when the date can't be parsed.
    """
    import re
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker)
    if not m:
        return True
    month = _MONTH_NUM.get(m.group(2))
    if not month:
        return True
    try:
        market_date = date(2000 + int(m.group(1)), month, int(m.group(3)))
        return market_date >= date.today()
    except ValueError:
        return True

# ── Singletons ────────────────────────────────────────────────────────────────

_kalshi_client: KalshiClient | None = None
_cache: dict = {}

# Scanner state (thread-safe via GIL for simple reads/writes)
_scanner_state: dict = {
    "last_scan":      None,   # datetime of last scan
    "next_scan":      None,   # datetime of next scheduled scan
    "last_result":    None,   # brief string describing last scan outcome
    "trades_today":   0,
    "daily_pnl":      0.0,
    "goal_hit":       False,
}


def get_kalshi() -> KalshiClient:
    global _kalshi_client
    if _kalshi_client is None:
        _kalshi_client = KalshiClient()
        _kalshi_client.login()
    return _kalshi_client


@app.errorhandler(Exception)
def handle_error(e):
    log.error(f"Unhandled error: {e}", exc_info=True)
    return jsonify({"error": str(e)}), 500


def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and (datetime.now() - entry["ts"]).total_seconds() < CACHE_TTL_SECONDS:
        return entry["data"]
    return None


def _cache_set(key: str, data):
    _cache[key] = {"data": data, "ts": datetime.now()}


# ── Daily P&L ─────────────────────────────────────────────────────────────────

def _get_daily_pnl() -> float:
    """Return today's net P&L in dollars from Kalshi fills."""
    try:
        kalshi   = get_kalshi()
        fills    = kalshi.get_fills()
        pnl_data = _compute_trade_pnl(fills)
        return pnl_data["daily"].get(str(date.today()), 0.0)
    except Exception:
        return 0.0


# ── Core scan + execute ───────────────────────────────────────────────────────

def _run_scan(execute: bool = False) -> dict:
    """
    Full pipeline: fetch weather, fetch markets, find opportunities.
    If execute=True and daily goal not yet hit, place orders.
    Returns a result dict (same shape as /api/opportunities).
    """
    from arb.kalshi import parse_bin_market, parse_threshold_market, enrich_with_orderbook_prices

    target    = str(date.today())
    cache_key = f"opps:{target}"

    # Invalidate cache so we get fresh data
    _cache.pop(cache_key, None)

    try:
        forecasts = asyncio.run(get_all_forecasts(CITIES, date.fromisoformat(target)))
    except Exception as e:
        return {"error": f"Weather fetch failed: {e}", "opportunities": []}

    if not KALSHI_CONFIGURED:
        return {"opportunities": [], "kalshi_status": "not_configured"}

    try:
        kalshi = get_kalshi()
    except Exception as e:
        return {"error": f"Kalshi login failed: {e}", "opportunities": []}

    all_opps   = []
    total_raw  = 0
    total_liquid = 0
    for city_key, forecast in forecasts.items():
        series = CITIES[city_key]["kalshi_series"]
        try:
            raw        = kalshi.get_markets_for_series(series, status="open")
            raw        = [m for m in raw if _ticker_date_ok(m.get("ticker", ""))]
            parsed_raw = [p for m in raw if (p := parse_bin_market(m))]
            parsed_raw += [p for m in raw if (p := parse_threshold_market(m))]
            parsed     = enrich_with_orderbook_prices(parsed_raw, kalshi, max_spread=MAX_BID_ASK_SPREAD)
            total_raw    += len(parsed_raw)
            total_liquid += len(parsed)
            opps = find_opportunities(forecast, parsed)
            for o in opps:
                all_opps.append({
                    "city":          o.city_key,
                    "city_name":     CITIES[o.city_key]["name"],
                    "type":          o.trade_type,
                    "range":         (f"≥{o.low_temp:.0f}F" if o.high_temp > 900
                                     else f"{o.low_temp:.1f}-{o.high_temp:.1f}F"),
                    "our_prob":      round(o.our_prob * 100, 1),
                    "market_impl":   round(o.market_implied * 100, 1),
                    "edge":          round(o.edge * 100, 1),
                    "cost_cents":    o.cost_cents,
                    "ticker_a":      o.ticker_a,
                    "ticker_b":      o.ticker_b,
                    "forecast_mean": round(o.forecast_mean, 1),
                    "forecast_std":  round(o.forecast_std, 1),
                    "ensemble_n":    o.ensemble_n,
                    "strong":        o.is_strong,
                    "side_a":        o.side_a,
                    "notes":         o.notes,
                })
        except Exception as e:
            log.error(f"{city_key}: {e}")

    all_opps.sort(key=lambda x: x["edge"], reverse=True)
    illiquid = total_raw > 0 and total_liquid == 0

    result = {
        "opportunities": all_opps,
        "generated_at":  datetime.now().isoformat(),
        "date":          target,
        "auto_execute":  AUTO_EXECUTE,
        "market_status": "illiquid" if illiquid else "ok",
        "message":       ("Markets are illiquid right now (wide bid-ask spreads). "
                          "Best results 9 AM – 2 PM Eastern when market makers are active."
                          if illiquid else None),
    }
    _cache_set(cache_key, result)

    if execute and all_opps and not illiquid:
        daily_pnl    = _get_daily_pnl()
        conservative = daily_pnl >= DAILY_PROFIT_GOAL
        if conservative:
            # Goal hit — switch to conservative mode: only strong-edge trades, half sizing
            strong_opps = [o for o in all_opps if o.get("strong")]
            log.info(
                f"Daily goal ${DAILY_PROFIT_GOAL:.0f} reached (${daily_pnl:.2f}). "
                f"Conservative mode — {len(strong_opps)} strong opp(s) only."
            )
            _scanner_state["goal_hit"] = True
            if strong_opps:
                results = _execute_opportunities(strong_opps, kalshi, conservative=True)
                _scanner_state["trades_today"] += sum(1 for r in results if "error" not in r.get("status", ""))
        else:
            log.info(f"Daily P&L: ${daily_pnl:.2f} / ${DAILY_PROFIT_GOAL:.0f} goal. Executing {len(all_opps)} opp(s).")
            results = _execute_opportunities(all_opps, kalshi)
            _scanner_state["trades_today"] += sum(1 for r in results if "error" not in r.get("status", ""))
            _scanner_state["daily_pnl"]     = daily_pnl

    return result


# ── Background scanner ────────────────────────────────────────────────────────

def _scanner_loop() -> None:
    """
    Background thread: wakes every minute, checks if it's time for a scan,
    and runs one during ET trading hours (SCAN_START_ET – SCAN_END_ET).
    Auto-executes if AUTO_EXECUTE=true.
    """
    log.info(
        f"Scanner started — interval={SCAN_INTERVAL_MINS}m, "
        f"window={SCAN_START_ET}–{SCAN_END_ET} ET, "
        f"auto_execute={AUTO_EXECUTE}, goal=${DAILY_PROFIT_GOAL:.0f}"
    )
    # Delay first scan to avoid hammering APIs immediately on (re)deploy
    time.sleep(90)
    last_run: datetime | None = None

    while True:
        now_et = datetime.now(ET)
        hour_et = now_et.hour + now_et.minute / 60

        in_window = SCAN_START_ET <= hour_et < SCAN_END_ET

        if in_window:
            due = (last_run is None or
                   (datetime.now() - last_run).total_seconds() >= SCAN_INTERVAL_MINS * 60)
            if due:
                log.info("Scanner: starting scheduled scan…")
                _scanner_state["last_scan"] = datetime.now()
                try:
                    result = _run_scan(execute=AUTO_EXECUTE)
                    n = len(result.get("opportunities", []))
                    status = result.get("market_status", "ok")
                    _scanner_state["last_result"] = f"{n} opp(s), market={status}"
                    log.info(f"Scanner: scan complete — {_scanner_state['last_result']}")
                except Exception as e:
                    _scanner_state["last_result"] = f"error: {e}"
                    log.error(f"Scanner error: {e}", exc_info=True)
                last_run = datetime.now()
                # Schedule next scan
                _scanner_state["next_scan"] = last_run + timedelta(minutes=SCAN_INTERVAL_MINS)

            # Reset daily goal flag at midnight
            if now_et.hour == 0 and now_et.minute < 2:
                if _scanner_state["goal_hit"]:
                    _scanner_state["goal_hit"]     = False
                    _scanner_state["trades_today"]  = 0
                    _scanner_state["daily_pnl"]     = 0.0
                    log.info("Scanner: daily goal flag reset at midnight")

        time.sleep(60)   # check once per minute


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/opportunities")
def api_opportunities():
    default_date = str(date.today() + timedelta(days=1))
    target = request.args.get("date", default_date)
    cache_key = f"opps:{target}"
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached)

    try:
        forecasts = asyncio.run(get_all_forecasts(CITIES, date.fromisoformat(target)))
    except Exception as e:
        return jsonify({"error": f"Weather fetch failed: {e}", "opportunities": []}), 500

    if not KALSHI_CONFIGURED:
        forecast_list = [
            {
                "city":          k,
                "city_name":     CITIES[k]["name"],
                "forecast_mean": round(fc.corrected_mean, 1),
                "forecast_std":  round(fc.std, 1),
                "ensemble_n":    len(fc.members),
                "nws_high":      fc.nws_high,
            }
            for k, fc in forecasts.items()
        ]
        return jsonify({
            "opportunities": [],
            "forecasts":     forecast_list,
            "kalshi_status": "not_configured",
            "message":       "Add KALSHI_EMAIL and KALSHI_PASSWORD to .env to see live market edges.",
            "generated_at":  datetime.now().isoformat(),
            "date":          target,
        })

    try:
        kalshi = get_kalshi()
    except Exception as e:
        return jsonify({"error": f"Kalshi login failed: {e}", "opportunities": []}), 500

    from arb.kalshi import parse_bin_market, parse_threshold_market, enrich_with_orderbook_prices

    all_opps     = []
    total_raw    = 0
    total_liquid = 0
    for city_key, forecast in forecasts.items():
        series = CITIES[city_key]["kalshi_series"]
        try:
            raw        = kalshi.get_markets_for_series(series, status="open")
            raw        = [m for m in raw if _ticker_date_ok(m.get("ticker", ""))]
            parsed_raw = [p for m in raw if (p := parse_bin_market(m))]
            parsed_raw += [p for m in raw if (p := parse_threshold_market(m))]
            parsed     = enrich_with_orderbook_prices(parsed_raw, kalshi, max_spread=MAX_BID_ASK_SPREAD)
            total_raw    += len(parsed_raw)
            total_liquid += len(parsed)
            if raw:
                n_bin = sum(1 for m in parsed if m.get("type") == "bin")
                n_thr = sum(1 for m in parsed if m.get("type") == "threshold")
                log.info(f"{city_key}: {len(raw)} raw → {n_bin} bins + {n_thr} thresholds (liquid)")
            opps = find_opportunities(forecast, parsed)
            for o in opps:
                all_opps.append({
                    "city":          o.city_key,
                    "city_name":     CITIES[o.city_key]["name"],
                    "type":          o.trade_type,
                    "range":         (f"≥{o.low_temp:.0f}F" if o.high_temp > 900
                                     else f"{o.low_temp:.1f}-{o.high_temp:.1f}F"),
                    "our_prob":      round(o.our_prob * 100, 1),
                    "market_impl":   round(o.market_implied * 100, 1),
                    "edge":          round(o.edge * 100, 1),
                    "cost_cents":    o.cost_cents,
                    "ticker_a":      o.ticker_a,
                    "ticker_b":      o.ticker_b,
                    "forecast_mean": round(o.forecast_mean, 1),
                    "forecast_std":  round(o.forecast_std, 1),
                    "ensemble_n":    o.ensemble_n,
                    "strong":        o.is_strong,
                    "side_a":        o.side_a,
                    "notes":         o.notes,
                })
        except Exception as e:
            log.error(f"{city_key}: {e}")

    all_opps.sort(key=lambda x: x["edge"], reverse=True)
    illiquid = total_raw > 0 and total_liquid == 0
    result = {
        "opportunities": all_opps,
        "generated_at":  datetime.now().isoformat(),
        "date":          target,
        "auto_execute":  AUTO_EXECUTE,
        "market_status": "illiquid" if illiquid else "ok",
        "message":       ("Markets are illiquid right now (wide bid-ask spreads). "
                          "Best results 9 AM – 2 PM Eastern when market makers are active."
                          if illiquid else None),
    }
    _cache_set(cache_key, result)

    if AUTO_EXECUTE and all_opps and not illiquid:
        daily_pnl = _get_daily_pnl()
        if daily_pnl >= DAILY_PROFIT_GOAL:
            log.info(f"Daily goal ${DAILY_PROFIT_GOAL:.0f} reached (${daily_pnl:.2f}). Not executing.")
        else:
            _execute_opportunities(all_opps, kalshi)

    return jsonify(result)


def _execute_opportunities(opps: list[dict], kalshi, conservative: bool = False) -> list[dict]:
    """Place live orders for a list of opportunity dicts. Returns execution log."""
    balance  = kalshi.get_balance()
    opp_objs = []
    for o in opps:
        obj = TradeOpportunity(
            city_key       = o["city"],
            trade_type     = o["type"],
            ticker_a       = o["ticker_a"],
            price_a        = o["cost_cents"] if not o["ticker_b"] else o["cost_cents"] // 2,
            ticker_b       = o.get("ticker_b"),
            price_b        = o["cost_cents"] // 2 if o.get("ticker_b") else None,
            our_prob       = o["our_prob"] / 100,
            market_implied = o["market_impl"] / 100,
            edge           = o["edge"] / 100,
            side_a         = o.get("side_a", "yes"),
            low_temp       = float(o["range"].lstrip("≥").rstrip("F").split("-")[0]),
            high_temp      = (999.0 if o["range"].startswith("≥")
                              else float(o["range"].split("-")[1].rstrip("F"))),
        )
        opp_objs.append(obj)
    allocs  = allocate(opp_objs, balance, conservative=conservative)
    results = []
    for opp_obj, n in allocs:
        try:
            kalshi.place_order(opp_obj.ticker_a, opp_obj.side_a, "buy", n, opp_obj.price_a)
            if opp_obj.ticker_b and opp_obj.price_b:
                kalshi.place_order(opp_obj.ticker_b, opp_obj.side_b, "buy", n, opp_obj.price_b)
            results.append({"ticker": opp_obj.ticker_a, "contracts": n, "status": "filled"})
        except Exception as e:
            results.append({"ticker": opp_obj.ticker_a, "contracts": n, "status": f"error: {e}"})
    return results


@app.route("/api/execute", methods=["POST"])
def api_execute():
    """Manually trigger live trade execution for cached opportunities."""
    if not KALSHI_CONFIGURED:
        return jsonify({"error": "Kalshi not configured"}), 400
    try:
        kalshi    = get_kalshi()
        target    = str(date.today())
        cache_key = f"opps:{target}"
        cached    = _cache_get(cache_key)
        opps      = cached["opportunities"] if cached else []
        if not opps:
            return jsonify({"error": "No opportunities cached. Refresh first."}), 400
        daily_pnl = _get_daily_pnl()
        if daily_pnl >= DAILY_PROFIT_GOAL:
            return jsonify({
                "error": f"Daily goal ${DAILY_PROFIT_GOAL:.0f} already reached (${daily_pnl:.2f}). No trades placed."
            }), 400
        # Filter to selected tickers if provided
        body = request.get_json(silent=True) or {}
        selected = body.get("tickers")  # list of ticker_a values, or None = all
        if selected is not None:
            opps = [o for o in opps if o.get("ticker_a") in selected]
        if not opps:
            return jsonify({"error": "No matching opportunities found for selected tickers."}), 400
        results = _execute_opportunities(opps, kalshi)
        return jsonify({"executed": len(results), "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scanner")
def api_scanner():
    """Return current scanner state (last scan, next scan, daily P&L)."""
    state = dict(_scanner_state)
    state["last_scan"]  = state["last_scan"].isoformat()  if state["last_scan"]  else None
    state["next_scan"]  = state["next_scan"].isoformat()  if state["next_scan"]  else None
    state["daily_pnl"]  = round(_get_daily_pnl(), 2)
    state["goal"]       = DAILY_PROFIT_GOAL
    state["goal_hit"]   = state["daily_pnl"] >= DAILY_PROFIT_GOAL
    state["auto_execute"] = AUTO_EXECUTE
    now_et = datetime.now(ET)
    state["in_trading_window"] = SCAN_START_ET <= (now_et.hour + now_et.minute/60) < SCAN_END_ET
    return jsonify(state)


@app.route("/api/close", methods=["POST"])
def api_close():
    """Sell (close) an open position by ticker at current market price."""
    data   = request.get_json(force=True)
    ticker = data.get("ticker")
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    try:
        kalshi    = get_kalshi()
        positions = kalshi.get_positions()
        pos = next((p for p in positions if p.get("ticker") == ticker), None)
        if not pos:
            return jsonify({"error": f"No open position for {ticker}"}), 404

        contracts = int(float(pos.get("position_fp") or 0))
        if contracts == 0:
            return jsonify({"error": "Position is already flat"}), 400

        # Fetch current market to get live yes_bid / yes_ask for pricing
        market     = kalshi.get_market(ticker)
        yes_bid    = market.get("yes_bid") or 50
        yes_ask    = market.get("yes_ask") or 50

        if contracts > 0:
            # Long YES → sell YES at yes_bid (what buyers will pay)
            side  = "yes"
            price = max(1, yes_bid)
            count = contracts
        else:
            # Long NO → sell NO; NO bid = 100 - YES ask
            side  = "no"
            price = max(1, 100 - yes_ask)
            count = abs(contracts)

        order = kalshi.place_order(ticker, side, "sell", count, price)
        log.info(f"Closed position: {ticker} {side.upper()} x{count} @ {price}¢")
        return jsonify({"ok": True, "ticker": ticker, "side": side, "count": count, "price": price, "order": order})
    except Exception as e:
        log.error(f"/api/close error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500



@app.route("/api/positions")
def api_positions():
    try:
        kalshi  = get_kalshi()
        all_pos = kalshi.get_positions()

        # Real field name is position_fp (fixed-point contract count); non-zero = open
        positions = [p for p in all_pos if float(p.get("position_fp") or 0) != 0]
        orders    = kalshi.get_orders(status="resting")
        balance   = kalshi.get_balance()
        fills     = kalshi.get_fills()[-50:]
        return jsonify({"positions": positions, "orders": orders, "balance": balance, "fills": fills})
    except Exception as e:
        return jsonify({"error": str(e), "positions": [], "orders": [], "balance": 0, "fills": []})


def _compute_trade_pnl(fills: list[dict]) -> dict:
    """
    Compute true P&L by grouping fills by ticker.

    Kalshi API v2 fill fields (discovered via /api/debug_fills):
      action          : "buy" (entry) or "sell" (settlement payout)
      side            : "yes" or "no" — which side was transacted
      count_fp        : contract count as a decimal string (e.g. "16.00")
      yes_price_dollars / no_price_dollars : price per contract in dollars
      fee_cost        : fee in dollars
      created_time    : ISO timestamp

    Buy fills  → accumulate cost  (action == "buy")
    Sell fills → settlement payout (action == "sell"); losing contracts
                 expire worthless with no sell fill at all.

    Returns dict with keys: by_ticker, total_net, total_fees, daily.
    """
    by_ticker: dict[str, dict] = {}

    # Pass 1 — initialise entries; record bought_side from buy fills so we
    # know which price column to use when we see the settlement sell fill.
    for f in fills:
        ticker = f.get("ticker") or "unknown"
        if ticker not in by_ticker:
            by_ticker[ticker] = {
                "date":        (f.get("created_time") or "")[:10],
                "cost":        0.0,
                "payout":      0.0,
                "fees":        0.0,
                "bought_side": None,   # "yes" or "no"
            }
        if (f.get("action") or "").lower() == "buy" and by_ticker[ticker]["bought_side"] is None:
            by_ticker[ticker]["bought_side"] = (f.get("side") or "no").lower()

    # Pass 2 — accumulate costs and payouts
    for f in fills:
        ticker = f.get("ticker") or "unknown"
        entry  = by_ticker[ticker]
        action = (f.get("action") or "").lower()
        side   = (f.get("side")   or "no").lower()
        count  = float(f.get("count_fp") or f.get("count") or 0)
        fee    = float(f.get("fee_cost") or f.get("fees") or 0)
        yes_px = float(f.get("yes_price_dollars") or f.get("yes_price") or 0)
        no_px  = float(f.get("no_price_dollars")  or f.get("no_price")  or 0)

        entry["fees"] += fee

        if action == "buy":
            price = no_px if side == "no" else yes_px
            entry["cost"] += price * count
            buy_date = (f.get("created_time") or "")[:10]
            if buy_date:
                entry["date"] = buy_date
        elif action == "sell":
            # Settlement fill — payout price matches the side the user held
            bought = entry.get("bought_side") or side
            entry["payout"] += (no_px if bought == "no" else yes_px) * count

    daily: dict[str, float] = {}
    total_net  = 0.0
    total_fees = 0.0

    for entry in by_ticker.values():
        net = entry["payout"] - entry["cost"] - entry["fees"]
        total_net  += net
        total_fees += entry["fees"]
        d = entry["date"]
        if d:
            daily[d] = daily.get(d, 0.0) + net

    return {
        "by_ticker":  by_ticker,
        "total_net":  total_net,
        "total_fees": total_fees,
        "daily":      daily,
    }


@app.route("/api/stats")
def api_stats():
    """Aggregate P&L stats from all fills."""
    try:
        kalshi  = get_kalshi()
        fills   = kalshi.get_fills()
        balance = kalshi.get_balance()   # cash balance only

        # Add mark-to-market value of open positions to get true portfolio value.
        # market_exposure_dollars = max payout; use as proxy for current value.
        portfolio_value = balance
        try:
            positions = kalshi.get_positions()
            for p in positions:
                contracts = float(p.get("position_fp") or 0)
                if contracts != 0:
                    # market_exposure_dollars is the max payout in dollars
                    portfolio_value += float(p.get("market_exposure_dollars") or 0)
        except Exception:
            pass  # fall back to cash-only if positions unavailable

        pnl_data    = _compute_trade_pnl(fills)
        total_net   = pnl_data["total_net"]
        total_fees  = pnl_data["total_fees"]
        daily       = pnl_data["daily"]
        by_ticker   = pnl_data["by_ticker"]

        today_pnl   = daily.get(str(date.today()), 0.0)
        days_traded = len(daily)
        win_days    = sum(1 for v in daily.values() if v > 0)
        daily_list  = [{"date": d, "pnl": round(v, 2)} for d, v in sorted(daily.items())]

        # Per-ticker win/loss: won if net_pnl > 0, lost if net_pnl <= 0 and cost > 0
        win_trades  = sum(1 for e in by_ticker.values()
                         if (e["payout"] - e["cost"] - e["fees"]) > 0)
        loss_trades = sum(1 for e in by_ticker.values()
                         if e["cost"] > 0 and (e["payout"] - e["cost"] - e["fees"]) <= 0)

        # Use funded amount for true net P&L if configured — more reliable than
        # parsing fills (which can have varying field names across API versions)
        true_net = round(portfolio_value - KALSHI_FUNDED, 2) if KALSHI_FUNDED > 0 else round(total_net, 2)

        return jsonify({
            "balance":        round(portfolio_value, 2),   # total: cash + open positions
            "cash_balance":   round(balance, 2),
            "funded_amount":  KALSHI_FUNDED,
            "total_gross":    round(total_net + total_fees, 2),
            "total_fees":     round(total_fees, 2),
            "total_net":      true_net,
            "today_pnl":      round(today_pnl, 2),
            "daily_goal":     DAILY_PROFIT_GOAL,
            "goal_progress":  round(min(today_pnl / DAILY_PROFIT_GOAL * 100, 100), 1) if DAILY_PROFIT_GOAL else 0,
            "days_traded":    days_traded,
            "win_days":       win_days,
            "loss_days":      days_traded - win_days,
            "win_rate":       round(win_days / days_traded * 100, 1) if days_traded else 0,
            "avg_daily":      round(total_net / days_traded, 2) if days_traded else 0,
            "win_trades":     win_trades,
            "loss_trades":    loss_trades,
            "trade_win_rate": round(win_trades / (win_trades + loss_trades) * 100, 1) if (win_trades + loss_trades) else 0,
            "daily":          daily_list,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/forecast/<city_key>")
def api_forecast(city_key: str):
    """Deep forecast data for a single city."""
    city_key = city_key.upper()
    if city_key not in CITIES:
        return jsonify({"error": "Unknown city"}), 404

    target    = request.args.get("date", str(date.today()))
    cache_key = f"fcst:{city_key}:{target}"
    cached    = _cache_get(cache_key)
    if cached:
        return jsonify(cached)

    forecasts = asyncio.run(get_all_forecasts({city_key: CITIES[city_key]}, date.fromisoformat(target)))
    fc = forecasts.get(city_key)
    if not fc:
        return jsonify({"error": "Forecast failed"}), 500

    result = {
        "city":            city_key,
        "city_name":       CITIES[city_key]["name"],
        "date":            target,
        "mean":            round(fc.mean, 1),
        "corrected_mean":  round(fc.corrected_mean, 1),
        "std":             round(fc.std, 1),
        "bias_correction": round(fc.bias_correction, 2),
        "nws_high":        fc.nws_high,
        "ensemble_n":      len(fc.members),
        "members":         [round(m, 1) for m in sorted(fc.members)],
        "bins_5f": [
            {"low": t, "high": t + 5, "prob": round(fc.prob_in_range(t, t + 5) * 100, 1)}
            for t in range(20, 115, 5)
            if fc.prob_in_range(t, t + 5) > 0.01
        ],
    }
    _cache_set(cache_key, result)
    return jsonify(result)


@app.route("/api/trades")
def api_trades():
    """Trade history from Kalshi fills — consumed by the main trading dashboard.

    Kalshi only creates settlement fills for WINNING contracts; losing contracts
    expire worthless with no record.  We must compute true P&L by grouping each
    ticker's buy-fills (cost) against any settlement payout (profit):
        net = settlement_payout - amount_paid - fees
    A position with no settlement fill is a total loss (net = -amount_paid).
    """
    try:
        kalshi  = get_kalshi()
        fills   = kalshi.get_fills()
        pnl     = _compute_trade_pnl(fills)

        # Build description from market_ticker (friendlier than raw ticker)
        desc_map = {}
        for f in fills:
            t = f.get("ticker") or "unknown"
            if t not in desc_map:
                desc_map[t] = f.get("market_ticker") or f.get("market_title") or t

        trades = []
        for ticker, entry in pnl["by_ticker"].items():
            net_pnl = entry["payout"] - entry["cost"] - entry["fees"]
            trades.append({
                "date":        entry["date"],
                "description": desc_map.get(ticker, ticker),
                "pnl":         round(net_pnl, 2),
            })

        trades.sort(key=lambda x: x["date"], reverse=True)
        return jsonify(trades)
    except Exception as e:
        log.error(f"/api/trades error: {e}", exc_info=True)
        return jsonify([])



# ── Main page ─────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/")
def index():
    today = str(date.today())
    return render_template("index.html", today=today, cities=list(CITIES.keys()))


# ── Start background scanner (runs under both gunicorn and direct python) ─────
_scanner_thread = threading.Thread(target=_scanner_loop, daemon=True, name="scanner")
_scanner_thread.start()


# ── Entry point (direct python only) ─────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "5050"))
    log.info(f"Dashboard running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
