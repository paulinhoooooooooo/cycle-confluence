"""
Signal TradingView : notation technique agrégée.
================================================================
Utilise l'endpoint public du scanner TradingView
(scanner.tradingview.com) — le même que celui du widget
"Technical Analysis" affiché sur le site. Pas de clé requise.

On récupère Recommend.All (moyenne de ~26 indicateurs : MAs,
oscillateurs) sur 3 horizons (1D, 1W, 1M) et on en tire un score
-100..+100 puis 0..100.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

SCAN_URL = "https://scanner.tradingview.com/{market}/scan"

# Colonnes Recommend.All par horizon (suffixe = timeframe)
_COLUMNS = [
    "Recommend.All",          # 1 jour
    "Recommend.All|1W",       # 1 semaine
    "Recommend.All|1M",       # 1 mois
    "Recommend.MA",
    "Recommend.Other",
    "RSI",
    "close",
]

_LABELS = {
    (-1.0, -0.5): "STRONG SELL",
    (-0.5, -0.1): "SELL",
    (-0.1, 0.1): "NEUTRAL",
    (0.1, 0.5): "BUY",
    (0.5, 1.01): "STRONG BUY",
}


def _label(v: float) -> str:
    for (lo, hi), name in zip(_LABELS.keys(), _LABELS.values()):
        if lo <= v < hi:
            return name
    return "NEUTRAL"


@dataclass
class TradingViewSnapshot:
    symbol: str
    score: float = 50.0                    # 0-100 (50 = neutre)
    rating_1d: Optional[float] = None      # -1..+1
    rating_1w: Optional[float] = None
    rating_1m: Optional[float] = None
    label_1d: str = "N/A"
    rsi: Optional[float] = None
    available: bool = True
    note: str = ""
    raw: Dict = field(default_factory=dict)


def _guess_tv_symbols(ticker: str) -> List[tuple]:
    """Retourne des couples (market, symbol) candidats pour TradingView."""
    t = ticker.upper().lstrip("^")
    candidates = []
    if ticker.endswith("-USD"):  # crypto
        base = ticker.replace("-USD", "USD").upper()
        candidates += [("crypto", f"BINANCE:{base}"), ("crypto", f"COINBASE:{base.replace('USD','') + 'USD'}")]
    if ticker.startswith("^"):
        idx_map = {"GSPC": "SP:SPX", "NDX": "NASDAQ:NDX", "DJI": "DJ:DJI",
                   "SOX": "NASDAQ:SOX", "FCHI": "EURONEXT:PX1", "VIX": "CBOE:VIX",
                   "GDAXI": "XETR:DAX", "IXIC": "NASDAQ:IXIC"}
        if t in idx_map:
            candidates.append(("america", idx_map[t]))
    candidates += [("america", f"NASDAQ:{t}"), ("america", f"NYSE:{t}"), ("america", f"AMEX:{t}")]
    return candidates


def fetch_tv_rating(ticker: str, timeout: int = 10) -> TradingViewSnapshot:
    """Récupère la notation technique TradingView, avec repli gracieux."""
    for market, symbol in _guess_tv_symbols(ticker):
        try:
            payload = {
                "symbols": {"tickers": [symbol], "query": {"types": []}},
                "columns": _COLUMNS,
            }
            r = requests.post(SCAN_URL.format(market=market), json=payload, timeout=timeout)
            if r.status_code != 200:
                continue
            data = r.json().get("data") or []
            if not data:
                continue
            vals = data[0].get("d", [])
            row = dict(zip(_COLUMNS, vals))
            r1d = row.get("Recommend.All")
            if r1d is None:
                continue
            r1w = row.get("Recommend.All|1W")
            r1m = row.get("Recommend.All|1M")

            # Score 0-100 : moyenne pondérée des 3 horizons (1D 50%, 1W 30%, 1M 20%)
            parts, weights = [], []
            for v, w in [(r1d, 0.5), (r1w, 0.3), (r1m, 0.2)]:
                if v is not None:
                    parts.append(v * w)
                    weights.append(w)
            avg = sum(parts) / sum(weights) if weights else 0.0
            score = round(50 + avg * 50, 1)

            return TradingViewSnapshot(
                symbol=symbol, score=score,
                rating_1d=r1d, rating_1w=r1w, rating_1m=r1m,
                label_1d=_label(r1d), rsi=row.get("RSI"), raw=row,
            )
        except Exception:
            continue

    return TradingViewSnapshot(symbol=ticker, score=50.0, available=False,
                               note="Notation TradingView indisponible — neutre (50).")
