"""
Signal "smart money" : ce que font les plus gros acteurs.
================================================================
Trois sources gratuites et légales :

1. Transactions d'initiés (dirigeants) — via yfinance
   (déclarations SEC Form 4). Achat net d'initiés = signal fort.

2. Détentions institutionnelles — via yfinance (13F agrégés) :
   % détenu par les institutions et top holders.

3. COT (Commitments of Traders) de la CFTC — pour indices/futures :
   positionnement net des "Large Speculators" (hedge funds) et des
   "Commercials". Rapport hebdomadaire officiel, CSV public.

Score 0-100 (50 = neutre / non disponible).
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import requests
import yfinance as yf

# Correspondance ticker -> nom de marché COT CFTC
COT_MARKET_MAP = {
    "^GSPC": "E-MINI S&P 500",
    "SPY": "E-MINI S&P 500",
    "^NDX": "NASDAQ MINI",
    "QQQ": "NASDAQ MINI",
    "^DJI": "DJIA Consolidated",
    "^RUT": "RUSSELL E-MINI",
    "BTC-USD": "BITCOIN",
    "ETH-USD": "ETHER CASH SETTLED",
    "GC=F": "GOLD",
    "CL=F": "CRUDE OIL",
    "^VIX": "VIX FUTURES",
}

COT_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"


@dataclass
class SmartMoneySnapshot:
    ticker: str
    score: float = 50.0
    insider_score: Optional[float] = None
    insider_net_shares_6m: Optional[float] = None
    inst_pct_held: Optional[float] = None
    cot_score: Optional[float] = None
    cot_large_spec_net: Optional[int] = None
    cot_percentile_1y: Optional[float] = None   # percentile du net spec sur 1 an
    available: bool = True
    note: str = ""
    detail: Dict = field(default_factory=dict)


# ── 1. Initiés ────────────────────────────────────────────────────────────

def _insider_component(tk: yf.Ticker) -> tuple:
    """Retourne (score 0-100 ou None, net_shares, détail)."""
    try:
        tx = tk.insider_transactions
        if tx is None or tx.empty:
            return None, None, {}
        tx = tx.copy()
        date_col = next((c for c in tx.columns if "date" in c.lower()), None)
        if date_col:
            tx[date_col] = pd.to_datetime(tx[date_col], errors="coerce")
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=182)
            tx = tx[tx[date_col] >= cutoff]
        if tx.empty:
            return 50.0, 0.0, {"n_transactions_6m": 0}

        text_col = next((c for c in tx.columns
                         if c.lower() in ("transaction", "text", "transaction text")), None)
        shares_col = next((c for c in tx.columns if "share" in c.lower()), None)

        buys = sells = 0.0
        if text_col is not None and shares_col is not None:
            for _, row in tx.iterrows():
                t = str(row.get(text_col, "")).lower()
                sh = float(row.get(shares_col) or 0)
                if "purchase" in t or "buy" in t:
                    buys += sh
                elif "sale" in t or "sell" in t:
                    sells += sh
        net = buys - sells
        total = buys + sells
        if total <= 0:
            return 50.0, 0.0, {"n_transactions_6m": len(tx)}
        ratio = net / total          # -1 (que des ventes) .. +1 (que des achats)
        # Les ventes d'initiés sont banales (compensation), les achats rares
        # et significatifs -> asymétrie : achats récompensés davantage.
        score = 50 + (ratio * 40 if ratio > 0 else ratio * 25)
        return round(min(100, max(0, score)), 1), net, {
            "buys_shares_6m": buys, "sells_shares_6m": sells,
            "n_transactions_6m": len(tx),
        }
    except Exception:
        return None, None, {}


# ── 2. Institutionnels ────────────────────────────────────────────────────

def _institutional_component(tk: yf.Ticker, info: dict) -> tuple:
    try:
        pct = info.get("heldPercentInstitutions")
        if pct is None:
            return None, None
        pct *= 100 if pct <= 1 else 1
        return round(pct, 1), {"held_percent_institutions": round(pct, 1)}
    except Exception:
        return None, None


# ── 3. COT CFTC ───────────────────────────────────────────────────────────

def _fetch_cot_year(year: int, timeout: int = 30) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(COT_URL.format(year=year), timeout=timeout)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        name = zf.namelist()[0]
        df = pd.read_csv(zf.open(name), low_memory=False)
        return df
    except Exception:
        return None


def _cot_component(ticker: str) -> tuple:
    """Retourne (score, net_spec, percentile, détail) ou (None, ...)."""
    market_kw = COT_MARKET_MAP.get(ticker.upper())
    if market_kw is None:
        return None, None, None, {}

    year = datetime.now().year
    df = _fetch_cot_year(year)
    prev = _fetch_cot_year(year - 1)
    if df is None and prev is None:
        return None, None, None, {}
    if df is not None and prev is not None:
        df = pd.concat([prev, df], ignore_index=True)
    elif df is None:
        df = prev

    name_col = next((c for c in df.columns if "market_and_exchange" in c.lower()), None)
    long_col = next((c for c in df.columns if "lev_money_positions_long" in c.lower()), None)
    short_col = next((c for c in df.columns if "lev_money_positions_short" in c.lower()), None)
    date_col = next((c for c in df.columns if "report_date" in c.lower()
                     or "as_of_date_in_form" in c.lower()), None)
    if not all([name_col, long_col, short_col, date_col]):
        return None, None, None, {}

    sub = df[df[name_col].astype(str).str.upper().str.contains(market_kw, na=False)].copy()
    if sub.empty:
        return None, None, None, {}
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.sort_values(date_col).tail(52)
    sub["net_spec"] = pd.to_numeric(sub[long_col], errors="coerce") - \
                      pd.to_numeric(sub[short_col], errors="coerce")
    sub = sub.dropna(subset=["net_spec"])
    if len(sub) < 4:
        return None, None, None, {}

    latest = float(sub["net_spec"].iloc[-1])
    pct_rank = float((sub["net_spec"] <= latest).mean() * 100)
    # Percentile élevé = hedge funds massivement longs = plutôt haussier,
    # mais extrêmes (>95) = positionnement saturé, risque de retournement.
    if pct_rank >= 95:
        score = 60.0
    elif pct_rank <= 5:
        score = 40.0
    else:
        score = 20 + pct_rank * 0.6      # 5% -> 23 ; 50% -> 50 ; 95% -> 77
    return round(score, 1), int(latest), round(pct_rank, 1), {
        "cot_market": market_kw, "n_weeks": len(sub),
        "net_spec_latest": int(latest),
    }


# ── Agrégation ────────────────────────────────────────────────────────────

def fetch_smart_money(ticker: str, include_cot: bool = True) -> SmartMoneySnapshot:
    snap = SmartMoneySnapshot(ticker=ticker)
    components, weights = [], []

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception as e:
        tk, info = None, {}
        snap.note = f"yfinance : {e}"

    if tk is not None:
        ins_score, net, det = _insider_component(tk)
        if ins_score is not None:
            snap.insider_score = ins_score
            snap.insider_net_shares_6m = net
            snap.detail.update(det)
            components.append(ins_score)
            weights.append(0.45)

        inst_pct, det2 = _institutional_component(tk, info)
        if inst_pct is not None:
            snap.inst_pct_held = inst_pct
            snap.detail.update(det2 or {})
            # 30% -> 45 ; 70% -> 60 (faible poids : c'est un stock, pas un flux)
            components.append(min(100, max(0, 30 + inst_pct * 0.45)))
            weights.append(0.15)

    if include_cot:
        cot_score, net_spec, pctile, det3 = _cot_component(ticker)
        if cot_score is not None:
            snap.cot_score = cot_score
            snap.cot_large_spec_net = net_spec
            snap.cot_percentile_1y = pctile
            snap.detail.update(det3)
            components.append(cot_score)
            weights.append(0.40)

    if components:
        snap.score = round(sum(c * w for c, w in zip(components, weights)) / sum(weights), 1)
    else:
        snap.available = False
        snap.note = snap.note or "Aucune donnée smart money — neutre (50)."

    return snap
