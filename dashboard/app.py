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
CACHE_TTL_SECONDS   = 300

ET = ZoneInfo("America/New_York")

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
        kalshi = get_kalshi()
        fills  = kalshi.get_fills()
        today  = str(date.today())
        total  = 0.0
        for f in fills:
            if f.get("created_time", "")[:10] == today:
                pnl = (f.get("profit", 0) or 0) / 100
                fee = (f.get("fees",   0) or 0) / 100
                total += pnl - fee
        return total
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

    target    = str(date.today() + timedelta(days=1))
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
            parsed_raw = [p for m in raw if (p := parse_bin_market(m))]
            parsed_raw += [p for m in raw if (p := parse_threshold_market(m))]
            parsed     = enrich_with_orderbook_prices(parsed_raw, kalshi)
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
            parsed_raw = [p for m in raw if (p := parse_bin_market(m))]
            parsed_raw += [p for m in raw if (p := parse_threshold_market(m))]
            parsed     = enrich_with_orderbook_prices(parsed_raw, kalshi)
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
        target    = str(date.today() + timedelta(days=1))
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


@app.route("/api/positions")
def api_positions():
    try:
        kalshi    = get_kalshi()
        positions = kalshi.get_positions()
        orders    = kalshi.get_orders(status="resting")
        balance   = kalshi.get_balance()
        fills     = kalshi.get_fills()[-50:]
        return jsonify({"positions": positions, "orders": orders, "balance": balance, "fills": fills})
    except Exception as e:
        return jsonify({"error": str(e), "positions": [], "orders": [], "balance": 0, "fills": []})


@app.route("/api/stats")
def api_stats():
    """Aggregate P&L stats from all fills."""
    try:
        kalshi  = get_kalshi()
        fills   = kalshi.get_fills()
        balance = kalshi.get_balance()

        total_gross = 0.0
        total_fees  = 0.0
        daily: dict[str, float] = {}

        for f in fills:
            ts  = f.get("created_time", "")[:10]
            pnl = (f.get("profit", 0) or 0) / 100
            fee = (f.get("fees",   0) or 0) / 100
            total_gross += pnl
            total_fees  += fee
            daily[ts]    = daily.get(ts, 0) + pnl - fee

        today_pnl   = daily.get(str(date.today()), 0.0)
        days_traded = len(daily)
        win_days    = sum(1 for v in daily.values() if v > 0)
        daily_list  = [{"date": d, "pnl": round(v, 2)} for d, v in sorted(daily.items())]

        # Per-trade win/loss (only settled fills with non-zero profit)
        settled = [f for f in fills if (f.get("profit") or 0) != 0]
        win_trades  = sum(1 for f in settled if (f.get("profit") or 0) > 0)
        loss_trades = sum(1 for f in settled if (f.get("profit") or 0) < 0)

        return jsonify({
            "balance":       round(balance, 2),
            "total_gross":   round(total_gross, 2),
            "total_fees":    round(total_fees, 2),
            "total_net":     round(total_gross - total_fees, 2),
            "today_pnl":     round(today_pnl, 2),
            "daily_goal":    DAILY_PROFIT_GOAL,
            "goal_progress": round(min(today_pnl / DAILY_PROFIT_GOAL * 100, 100), 1) if DAILY_PROFIT_GOAL else 0,
            "days_traded":   days_traded,
            "win_days":      win_days,
            "loss_days":     days_traded - win_days,
            "win_rate":      round(win_days / days_traded * 100, 1) if days_traded else 0,
            "avg_daily":     round((total_gross - total_fees) / days_traded, 2) if days_traded else 0,
            "win_trades":    win_trades,
            "loss_trades":   loss_trades,
            "trade_win_rate": round(win_trades / (win_trades + loss_trades) * 100, 1) if (win_trades + loss_trades) else 0,
            "daily":         daily_list,
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


# ── Main page ─────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/")
def index():
    today = str(date.today() + timedelta(days=1))
    return render_template("index.html", today=today, cities=list(CITIES.keys()))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "5050"))

    # Start background scanner thread
    t = threading.Thread(target=_scanner_loop, daemon=True, name="scanner")
    t.start()

    log.info(f"Dashboard running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
