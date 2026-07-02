"""
Signal Polymarket : risque macro perçu par les marchés prédictifs.
================================================================
Utilise l'API publique Gamma (pas de clé requise) :
  https://gamma-api.polymarket.com

Idée : les prix Polymarket sont des probabilités agrégées par de
l'argent réel. On interroge des marchés macro (récession, décisions
de la Fed, crises géopolitiques, crash) et on en dérive un score de
risque macro 0-100 (0 = ciel dégagé, 100 = risque extrême).

Ce score sert de MODULATEUR global du signal cyclique : un signal
d'achat de cycle par temps de risque macro élevé est atténué.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

GAMMA_URL = "https://gamma-api.polymarket.com/markets"

# Mots-clés macro à risque et leur poids dans le score
RISK_KEYWORDS: Dict[str, float] = {
    "recession": 1.0,
    "market crash": 1.0,
    "stock market decline": 0.9,
    "fed rate hike": 0.6,
    "rate cut": -0.4,          # une baisse de taux probable RÉDUIT le risque perçu
    "government shutdown": 0.5,
    "default": 0.8,
    "war": 0.7,
    "inflation above": 0.6,
}


@dataclass
class PolymarketSnapshot:
    macro_risk_score: float                 # 0-100
    markets_used: List[dict] = field(default_factory=list)
    available: bool = True
    note: str = ""

    @property
    def risk_multiplier(self) -> float:
        """
        Multiplicateur appliqué au score de confluence final.
        risque 0   -> 1.10 (vent dans le dos)
        risque 50  -> 1.00 (neutre)
        risque 100 -> 0.70 (fort vent de face)
        """
        return round(1.10 - 0.004 * self.macro_risk_score, 3)


def _search_markets(query: str, limit: int = 5, timeout: int = 10) -> List[dict]:
    params = {
        "limit": limit,
        "active": "true",
        "closed": "false",
        "order": "volume24hr",
        "ascending": "false",
    }
    r = requests.get(GAMMA_URL, params=params | {"tag": None}, timeout=timeout)
    r.raise_for_status()
    markets = r.json()
    q = query.lower()
    return [m for m in markets if q in (m.get("question", "") or "").lower()]


def _fetch_all_active(limit: int = 300, timeout: int = 12) -> List[dict]:
    params = {"limit": limit, "active": "true", "closed": "false",
              "order": "volume24hr", "ascending": "false"}
    r = requests.get(GAMMA_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _yes_probability(market: dict) -> Optional[float]:
    """Extrait la probabilité 'Yes' d'un marché Gamma (champ outcomePrices)."""
    try:
        import json as _json
        prices = market.get("outcomePrices")
        if isinstance(prices, str):
            prices = _json.loads(prices)
        outcomes = market.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = _json.loads(outcomes)
        if not prices or not outcomes:
            return None
        for o, p in zip(outcomes, prices):
            if str(o).strip().lower() == "yes":
                return float(p)
        return float(prices[0])
    except Exception:
        return None


def fetch_macro_risk(timeout: int = 12) -> PolymarketSnapshot:
    """
    Interroge Polymarket et calcule le score de risque macro.
    Robuste : en cas d'échec réseau, retourne un snapshot neutre
    (risque 50, multiplicateur 1.0) avec available=False.
    """
    try:
        markets = _fetch_all_active(limit=300, timeout=timeout)
    except Exception as e:
        return PolymarketSnapshot(macro_risk_score=50.0, available=False,
                                  note=f"Polymarket indisponible : {e}")

    contributions: List[float] = []
    used: List[dict] = []

    for m in markets:
        question = (m.get("question") or "").lower()
        prob = _yes_probability(m)
        if prob is None:
            continue
        for kw, weight in RISK_KEYWORDS.items():
            if kw in question:
                # contribution signée : prob * poids (poids négatif = apaisant)
                contributions.append(prob * weight)
                used.append({
                    "question": m.get("question"),
                    "yes_prob": round(prob, 3),
                    "volume24h": m.get("volume24hr"),
                    "keyword": kw,
                })
                break

    if not contributions:
        return PolymarketSnapshot(macro_risk_score=50.0, available=True,
                                  note="Aucun marché macro pertinent trouvé — neutre.")

    # Normalisation : moyenne des contributions ramenée sur 0-100 autour de 50
    raw = float(sum(contributions) / len(contributions))   # ~[-0.4 .. 1.0]
    score = 50 + raw * 50
    score = round(min(100.0, max(0.0, score)), 1)

    return PolymarketSnapshot(macro_risk_score=score, markets_used=used[:10])
