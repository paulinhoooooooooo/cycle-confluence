"""
Validation walk-forward des combinaisons de cycles.
================================================================
Problème corrigé : dans cycle_analyser original, la sinusoïde est
ajustée sur TOUT l'historique, puis les hit rates sont calculés sur
ce même historique (biais de look-ahead). Les performances affichées
sont donc structurellement optimistes.

Ici, à chaque pas de temps t, on ré-ajuste les sinusoïdes uniquement
sur les données [0..t], on lit le signal (haussier/baissier) pour t,
et on le compare au rendement réel t -> t+1. On obtient ainsi un
hit rate HORS ÉCHANTILLON, seul chiffre réellement fiable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.cycle_detector import _detrend_log, _fit_sine


@dataclass
class WalkForwardResult:
    periods: List[int]
    oos_hit_rate: float          # % de barres haussières prédites suivies d'une hausse
    oos_short_hit_rate: float    # % de barres baissières prédites suivies d'une baisse
    oos_avg_return_bull: float   # rendement moyen/barre (%) quand signal haussier
    oos_avg_return_bear: float   # rendement moyen/barre (%) quand signal baissier
    n_bull_signals: int
    n_bear_signals: int
    in_sample_hit_rate: Optional[float] = None   # pour mesurer la dégradation
    degradation_pct: Optional[float] = None      # IS - OOS (points de %)
    reliability_score: float = 0.0               # 0-100

    @property
    def is_reliable(self) -> bool:
        return self.reliability_score >= 55


def _bullish_at_t(prices: np.ndarray, periods: List[int]) -> bool:
    """Signal haussier au dernier point, sinusoïdes ajustées SEULEMENT sur prices."""
    detrended, _ = _detrend_log(prices)
    N = len(prices)
    t = np.arange(N, dtype=float)
    for T in periods:
        A, B, amp = _fit_sine(detrended, float(T))
        if amp < 1e-12:
            return False
        osc_last = A * np.cos(2 * np.pi * (N - 1) / T) + B * np.sin(2 * np.pi * (N - 1) / T)
        osc_prev = A * np.cos(2 * np.pi * (N - 2) / T) + B * np.sin(2 * np.pi * (N - 2) / T)
        if osc_last <= osc_prev:
            return False
    return True


def _bearish_at_t(prices: np.ndarray, periods: List[int]) -> bool:
    detrended, _ = _detrend_log(prices)
    N = len(prices)
    for T in periods:
        A, B, amp = _fit_sine(detrended, float(T))
        if amp < 1e-12:
            return False
        osc_last = A * np.cos(2 * np.pi * (N - 1) / T) + B * np.sin(2 * np.pi * (N - 1) / T)
        osc_prev = A * np.cos(2 * np.pi * (N - 2) / T) + B * np.sin(2 * np.pi * (N - 2) / T)
        if osc_last >= osc_prev:
            return False
    return True


def walk_forward_validate(
    prices: np.ndarray,
    periods: List[int],
    min_train: Optional[int] = None,
    step: int = 1,
    in_sample_hit_rate: Optional[float] = None,
) -> WalkForwardResult:
    """
    Validation hors échantillon d'une combinaison de périodes.

    prices      : série de clôtures
    periods     : périodes de la combinaison (ex: [121, 80])
    min_train   : taille minimale de la fenêtre d'apprentissage
                  (défaut : 3x la plus grande période, min 120 barres)
    step        : pas d'avancement (1 = chaque barre; augmenter pour accélérer)
    """
    N = len(prices)
    if min_train is None:
        min_train = max(120, int(max(periods) * 3))
    min_train = min(min_train, N - 20)

    bull_next: List[float] = []
    bear_next: List[float] = []

    for t in range(min_train, N - 1, step):
        window = prices[: t + 1]
        ret_next = (prices[t + 1] - prices[t]) / prices[t] * 100
        if _bullish_at_t(window, periods):
            bull_next.append(ret_next)
        elif _bearish_at_t(window, periods):
            bear_next.append(ret_next)

    n_bull, n_bear = len(bull_next), len(bear_next)
    hit = (sum(1 for r in bull_next if r > 0) / n_bull * 100) if n_bull else 0.0
    short_hit = (sum(1 for r in bear_next if r < 0) / n_bear * 100) if n_bear else 0.0
    avg_bull = float(np.mean(bull_next)) if n_bull else 0.0
    avg_bear = float(np.mean(bear_next)) if n_bear else 0.0

    degradation = None
    if in_sample_hit_rate is not None:
        degradation = round(in_sample_hit_rate - hit, 1)

    # Score de fiabilité 0-100 :
    #  - hit rate OOS (base)
    #  - pénalité si peu de signaux (< 30 : peu significatif statistiquement)
    #  - pénalité si forte dégradation IS -> OOS (> 20 pts : sur-ajustement)
    if n_bull == 0:
        # INVÉRIFIABLE, pas invalidé : périodes trop longues pour l'historique
        # ou zone haussière jamais atteinte hors échantillon. Score neutre bas.
        score = 30.0
    else:
        score = hit
        if n_bull < 30:
            score *= 0.6 + 0.4 * (n_bull / 30)
        if degradation is not None and degradation > 20:
            score *= max(0.5, 1 - (degradation - 20) / 60)
        score = round(min(100.0, max(0.0, score)), 1)

    return WalkForwardResult(
        periods=list(periods),
        oos_hit_rate=round(hit, 1),
        oos_short_hit_rate=round(short_hit, 1),
        oos_avg_return_bull=round(avg_bull, 3),
        oos_avg_return_bear=round(avg_bear, 3),
        n_bull_signals=n_bull,
        n_bear_signals=n_bear,
        in_sample_hit_rate=in_sample_hit_rate,
        degradation_pct=degradation,
        reliability_score=score,
    )


def cycle_persistence(prices: np.ndarray, period: int, n_checks: int = 4) -> float:
    """
    Vérifie qu'un cycle détecté aujourd'hui existait déjà dans le passé.
    On tronque l'historique à 70%, 80%, 90%, 100% et on mesure si la
    période reste porteuse d'énergie spectrale locale.
    Retourne un score 0-1 (1 = le cycle était détectable à chaque tronçon).
    """
    from core.cycle_detector import detect_cycles

    fractions = np.linspace(0.7, 1.0, n_checks)
    found = 0
    for f in fractions:
        n = int(len(prices) * f)
        if n < period * 2.5:
            continue
        try:
            cycles = detect_cycles(prices[:n], n_cycles=25)
        except Exception:
            continue
        if any(abs(c.period - period) / period < 0.10 for c in cycles):
            found += 1
    return round(found / n_checks, 2)
