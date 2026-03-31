"""
Kalshi Weather Arb — Web Dashboard

Run with:
  python3 dashboard/app.py

Then open http://localhost:5050
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

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

CACHE_TTL_SECONDS = 300   # re-fetch weather every 5 minutes
_cache: dict = {}


def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and (datetime.now() - entry["ts"]).seconds < CACHE_TTL_SECONDS:
        return entry["data"]
    return None


def _cache_set(key: str, data):
    _cache[key] = {"data": data, "ts": datetime.now()}


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/opportunities")
def api_opportunities():
    target = request.args.get("date", str(date.today()))
    cache_key = f"opps:{target}"
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached)

    forecasts = asyncio.run(get_all_forecasts(CITIES, date.fromisoformat(target)))

    kalshi = KalshiClient()
    all_opps = []
    for city_key, forecast in forecasts.items():
        series = CITIES[city_key]["kalshi_series"]
        try:
            raw = kalshi.get_markets_for_series(series, status="open")
            from arb.kalshi import parse_bin_market
            parsed = [p for m in raw if (p := parse_bin_market(m))]
            opps = find_opportunities(forecast, parsed)
            for o in opps:
                all_opps.append({
                    "city":          o.city_key,
                    "city_name":     CITIES[o.city_key]["name"],
                    "type":          o.trade_type,
                    "range":         f"{o.low_temp:.0f}–{o.high_temp:.0f}°F",
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
                    "notes":         o.notes,
                })
        except Exception as e:
            log.error(f"{city_key}: {e}")
    kalshi.close()

    all_opps.sort(key=lambda x: x["edge"], reverse=True)
    result = {"opportunities": all_opps, "generated_at": datetime.now().isoformat(), "date": target}
    _cache_set(cache_key, result)
    return jsonify(result)


@app.route("/api/positions")
def api_positions():
    try:
        kalshi = KalshiClient()
        positions = kalshi.get_positions()
        balance   = kalshi.get_balance()
        fills     = kalshi.get_fills()[-50:]
        kalshi.close()
        return jsonify({"positions": positions, "balance": balance, "fills": fills})
    except Exception as e:
        return jsonify({"error": str(e), "positions": [], "balance": 0, "fills": []})


@app.route("/api/stats")
def api_stats():
    """Aggregate P&L stats from recent fills."""
    try:
        kalshi  = KalshiClient()
        fills   = kalshi.get_fills()
        balance = kalshi.get_balance()
        kalshi.close()

        total_gross = 0.0
        total_fees  = 0.0
        daily: dict[str, float] = {}

        for f in fills:
            ts  = f.get("created_time", "")[:10]
            pnl = (f.get("profit", 0) or 0) / 100
            fee = (f.get("fees", 0) or 0) / 100
            total_gross += pnl
            total_fees  += fee
            daily[ts]   = daily.get(ts, 0) + pnl - fee

        days_traded = len(daily)
        win_days    = sum(1 for v in daily.values() if v > 0)
        daily_list  = [{"date": d, "pnl": round(v, 2)} for d, v in sorted(daily.items())]

        return jsonify({
            "balance":      round(balance, 2),
            "total_gross":  round(total_gross, 2),
            "total_fees":   round(total_fees, 2),
            "total_net":    round(total_gross - total_fees, 2),
            "days_traded":  days_traded,
            "win_days":     win_days,
            "loss_days":    days_traded - win_days,
            "win_rate":     round(win_days / days_traded * 100, 1) if days_traded else 0,
            "avg_daily":    round((total_gross - total_fees) / days_traded, 2) if days_traded else 0,
            "daily":        daily_list,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/forecast/<city_key>")
def api_forecast(city_key: str):
    """Deep forecast data for a single city."""
    city_key = city_key.upper()
    if city_key not in CITIES:
        return jsonify({"error": "Unknown city"}), 404

    target = request.args.get("date", str(date.today()))
    cache_key = f"fcst:{city_key}:{target}"
    cached = _cache_get(cache_key)
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
        # Pre-compute bin probabilities for common Kalshi bin widths
        "bins_5f": [
            {"low": t, "high": t + 5, "prob": round(fc.prob_in_range(t, t + 5) * 100, 1)}
            for t in range(20, 115, 5)
            if fc.prob_in_range(t, t + 5) > 0.01
        ],
    }
    _cache_set(cache_key, result)
    return jsonify(result)


# ── Main page ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    today = str(date.today())
    return render_template("index.html", today=today, cities=list(CITIES.keys()))


if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "5050"))
    log.info(f"Dashboard running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
