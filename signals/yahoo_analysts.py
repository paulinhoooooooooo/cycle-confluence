"""
Signal Yahoo Finance : consensus analystes et positionnement.
================================================================
Via yfinance (déjà utilisé par le moteur original) :
  - recommandations analystes (strongBuy/buy/hold/sell/strongSell)
  - objectif de cours moyen vs prix actuel (upside implicite)
  - short interest (% du flottant vendu à découvert)

Score 0-100. Neutre (50) si l'info n'existe pas (indices, cryptos).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import yfinance as yf


@dataclass
class YahooAnalystSnapshot:
    ticker: str
    score: float = 50.0
    reco_score: Optional[float] = None       # 0-100 dérivé du consensus
    n_analysts: int = 0
    target_upside_pct: Optional[float] = None
    short_pct_float: Optional[float] = None
    available: bool = True
    note: str = ""
    detail: Dict = field(default_factory=dict)


def fetch_yahoo_analysts(ticker: str) -> YahooAnalystSnapshot:
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception as e:
        return YahooAnalystSnapshot(ticker=ticker, available=False,
                                    note=f"Yahoo indisponible : {e}")

    snap = YahooAnalystSnapshot(ticker=ticker)
    components, weights = [], []

    # 1) Consensus analystes -----------------------------------------------
    try:
        recos = tk.recommendations_summary
        if recos is not None and len(recos) > 0:
            row = recos.iloc[0]  # période courante "0m"
            sb, b = int(row.get("strongBuy", 0)), int(row.get("buy", 0))
            h = int(row.get("hold", 0))
            s, ss = int(row.get("sell", 0)), int(row.get("strongSell", 0))
            n = sb + b + h + s + ss
            if n > 0:
                # pondération : SB=100, B=75, H=50, S=25, SS=0
                reco = (sb * 100 + b * 75 + h * 50 + s * 25 + ss * 0) / n
                snap.reco_score = round(reco, 1)
                snap.n_analysts = n
                components.append(reco)
                weights.append(0.45)
                snap.detail["recommendations"] = {"strongBuy": sb, "buy": b,
                                                  "hold": h, "sell": s, "strongSell": ss}
    except Exception:
        pass

    # 2) Objectif de cours vs prix -----------------------------------------
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    target = info.get("targetMeanPrice")
    if price and target:
        upside = (target - price) / price * 100
        snap.target_upside_pct = round(upside, 1)
        # -20% -> 0 ; 0% -> 50 ; +30% -> 100 (borné)
        up_score = min(100.0, max(0.0, 50 + upside * (50 / 30) if upside >= 0
                                  else 50 + upside * (50 / 20)))
        components.append(up_score)
        weights.append(0.35)

    # 3) Short interest ------------------------------------------------------
    short_pct = info.get("shortPercentOfFloat")
    if short_pct is not None:
        short_pct *= 100 if short_pct < 1 else 1
        snap.short_pct_float = round(short_pct, 2)
        # 0% -> 70 (personne ne parie contre), 5% -> 50, 15%+ -> 10
        sh_score = min(100.0, max(0.0, 70 - short_pct * 4))
        components.append(sh_score)
        weights.append(0.20)

    if components:
        snap.score = round(sum(c * w for c, w in zip(components, weights)) / sum(weights), 1)
    else:
        snap.note = "Pas de données analystes (indice/crypto ?) — neutre (50)."

    return snap
