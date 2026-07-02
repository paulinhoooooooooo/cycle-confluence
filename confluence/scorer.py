"""
Score de confluence : combine le signal cyclique original avec les
couches de confirmation pour un signal plus précis et plus fiable.
================================================================
Composantes :
  1. Signal cyclique (moteur original)                    — poids 35%
     ... pondéré par la fiabilité walk-forward (OOS)
  2. Fiabilité hors échantillon (walk-forward)            — poids 20%
  3. Notation technique TradingView (1D/1W/1M)            — poids 15%
  4. Consensus analystes Yahoo (recos, objectifs, short)  — poids 15%
  5. Smart money (initiés, institutions, COT)             — poids 15%

Puis modulateur macro Polymarket (x0.70 à x1.10).

Verdicts :
  >= 75  ACHAT FORT     confluence maximale
  60-74  ACHAT          signal cyclique confirmé
  45-59  NEUTRE         attendre
  30-44  ÉVITER         signaux contradictoires
  < 30   VENTE/SHORT    confluence baissière
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_WEIGHTS = {
    "cycle": 0.35,
    "walk_forward": 0.20,
    "tradingview": 0.15,
    "yahoo": 0.15,
    "smart_money": 0.15,
}


@dataclass
class ConfluenceResult:
    ticker: str
    final_score: float                  # 0-100 après modulateur macro
    raw_score: float                    # avant modulateur
    verdict: str
    components: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    macro_multiplier: float = 1.0
    macro_risk: float = 50.0
    warnings: List[str] = field(default_factory=list)
    detail: Dict = field(default_factory=dict)


def _verdict(score: float) -> str:
    if score >= 75:
        return "ACHAT FORT"
    if score >= 60:
        return "ACHAT"
    if score >= 45:
        return "NEUTRE"
    if score >= 30:
        return "ÉVITER"
    return "VENTE / SHORT"


def cycle_signal_score(phase_states: List[str], in_sample_hit_rate: float) -> float:
    """
    Traduit l'état de phase de la combinaison en score 0-100.
    phase_states : états des cycles de la combinaison retenue.
    """
    n = len(phase_states)
    if n == 0:
        return 50.0
    bull = sum(1 for s in phase_states if s == "bullish")
    bear = sum(1 for s in phase_states if s == "bearish")
    trough = sum(1 for s in phase_states if s == "trough")
    peak = sum(1 for s in phase_states if s == "peak")

    if bull == n:                       # tous haussiers = zone d'achat active
        base = 85.0
    elif bull + trough == n and trough > 0:   # creux + haussiers = entrée imminente
        base = 75.0
    elif bear == n:
        base = 10.0
    elif bear + peak == n and peak > 0:
        base = 22.0
    else:
        base = 50.0

    # Moduler légèrement par le hit rate in-sample (indicatif seulement)
    adj = (in_sample_hit_rate - 60) * 0.2 if in_sample_hit_rate else 0.0
    return round(min(100.0, max(0.0, base + adj)), 1)


def compute_confluence(
    ticker: str,
    cycle_score: float,
    wf_reliability: float,
    tv_score: float = 50.0,
    yahoo_score: float = 50.0,
    smart_money_score: float = 50.0,
    macro_risk: float = 50.0,
    macro_multiplier: float = 1.0,
    weights: Optional[Dict[str, float]] = None,
    unavailable: Optional[List[str]] = None,
) -> ConfluenceResult:
    """
    Combine toutes les composantes. Les sources indisponibles
    (listées dans `unavailable`) voient leur poids redistribué au
    lieu de compter comme neutres — évite de diluer le signal.
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    unavailable = unavailable or []
    comp = {
        "cycle": cycle_score,
        "walk_forward": wf_reliability,
        "tradingview": tv_score,
        "yahoo": yahoo_score,
        "smart_money": smart_money_score,
    }

    for k in unavailable:
        w.pop(k, None)
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}

    raw = sum(comp[k] * w[k] for k in w)
    final = round(min(100.0, max(0.0, raw * macro_multiplier)), 1)

    warnings: List[str] = []
    if wf_reliability < 50:
        warnings.append(
            f"Fiabilité hors échantillon faible ({wf_reliability:.0f}/100) : "
            "les performances in-sample du cycle sont probablement sur-ajustées."
        )
    if macro_risk >= 70:
        warnings.append(
            f"Risque macro Polymarket élevé ({macro_risk:.0f}/100) : "
            "signal atténué, prudence sur la taille de position."
        )
    if cycle_score >= 70 and smart_money_score <= 35:
        warnings.append(
            "Divergence : cycle haussier mais smart money vendeuse (initiés/COT)."
        )
    if cycle_score >= 70 and tv_score <= 35:
        warnings.append(
            "Divergence : cycle haussier mais momentum technique TradingView baissier."
        )

    return ConfluenceResult(
        ticker=ticker,
        final_score=final,
        raw_score=round(raw, 1),
        verdict=_verdict(final),
        components=comp,
        weights={k: round(v, 3) for k, v in w.items()},
        macro_multiplier=macro_multiplier,
        macro_risk=macro_risk,
        warnings=warnings,
    )
