from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from core.cycle_detector import CycleInfo, get_bullish_mask, _detrend_log, _fit_sine, _phase_state


@dataclass
class ZoneResult:
    start: int
    end: int
    return_pct: float
    duration: int


@dataclass
class SLZoneResult:
    """Result of trailing stop-loss simulation on one bullish zone."""
    zone: ZoneResult
    sl_return_pct: float           # actual return with SL applied
    sl_hit: bool                   # True if SL was triggered before zone end
    sl_exit_bar: int               # absolute bar index where trade was exited
    sl_path: np.ndarray            # SL price level per bar (zone.start → sl_exit_bar)
    original_return_pct: float     # return without SL (= zone.return_pct)


@dataclass
class CombinationResult:
    cycles: List[CycleInfo]
    periods: List[int]
    # Bullish zones (all cycles rising simultaneously)
    zones: List[ZoneResult]
    total_return_pct: float        # simple sum
    compound_return_pct: float     # compounded (reinvestment)
    hit_rate: float
    avg_return_pct: float
    n_zones: int
    bullish_mask: np.ndarray = field(repr=False)
    # Bearish zones (all cycles falling simultaneously)
    bearish_zones: List[ZoneResult] = field(default_factory=list)
    bearish_total_return_pct: float = 0.0
    bearish_compound_return_pct: float = 0.0
    bearish_hit_rate: float = 0.0       # % of bearish zones where market fell (short profit)
    short_compound_return_pct: float = 0.0  # compounded return if shorting every bearish zone
    bearish_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool), repr=False)
    combo_size: int = 2

    @property
    def label(self) -> str:
        return " + ".join(str(c.period) for c in self.cycles)

    @property
    def periods_str(self) -> str:
        return ", ".join(str(p) for p in self.periods)


def _compute_zones(prices: np.ndarray, mask: np.ndarray) -> List[ZoneResult]:
    """Find contiguous True zones in mask and compute their price returns."""
    zones: List[ZoneResult] = []
    N = len(prices)
    i = 0
    while i < N:
        if mask[i]:
            start = i
            while i < N and mask[i]:
                i += 1
            end = i - 1
            if end > start:
                ret = (prices[end] - prices[start]) / prices[start] * 100
                zones.append(ZoneResult(start=start, end=end,
                                        return_pct=round(ret, 2), duration=end - start + 1))
        else:
            i += 1
    return zones


def _combined_bullish_mask(prices: np.ndarray, cycles: List[CycleInfo]) -> np.ndarray:
    mask = np.ones(len(prices), dtype=bool)
    for c in cycles:
        mask &= get_bullish_mask(prices, c.period)
    return mask


def _combined_bearish_mask(prices: np.ndarray, cycles: List[CycleInfo]) -> np.ndarray:
    mask = np.ones(len(prices), dtype=bool)
    for c in cycles:
        mask &= ~get_bullish_mask(prices, c.period)
    return mask


def _short_compound_return(zones: List[ZoneResult]) -> float:
    """Compounded return when shorting every zone (gain = -price_change)."""
    if not zones:
        return 0.0
    c = 1.0
    for z in zones:
        c *= 1 + (-z.return_pct) / 100
    return round((c - 1) * 100, 2)


def compute_single_cycle_hit_rates(prices: np.ndarray, period: int) -> Tuple[float, float]:
    """Return (hit_rate, short_hit_rate) as percentages for a single cycle."""
    bullish = get_bullish_mask(prices, period)
    N = len(prices)
    bull_hits = bull_total = 0
    bear_hits = bear_total = 0
    i = 0
    while i < N:
        if bullish[i]:
            start = i
            while i < N and bullish[i]:
                i += 1
            last = i - 1
            if last > start:
                ret = (prices[last] - prices[start]) / prices[start]
                bull_total += 1
                if ret > 0:
                    bull_hits += 1
        else:
            start = i
            while i < N and not bullish[i]:
                i += 1
            last = i - 1
            if last > start:
                ret = (prices[last] - prices[start]) / prices[start]
                bear_total += 1
                if ret < 0:
                    bear_hits += 1
    hit_rate = (bull_hits / bull_total * 100) if bull_total > 0 else 0.0
    short_hit_rate = (bear_hits / bear_total * 100) if bear_total > 0 else 0.0
    return hit_rate, short_hit_rate


def _short_hit_rate(zones: List[ZoneResult]) -> float:
    """% of zones where market fell (short was profitable)."""
    if not zones:
        return 0.0
    hits = sum(1 for z in zones if z.return_pct < 0)
    return round(hits / len(zones) * 100, 1)


def _compound_return(zones: List[ZoneResult]) -> float:
    if not zones:
        return 0.0
    c = 1.0
    for z in zones:
        c *= 1 + z.return_pct / 100
    return round((c - 1) * 100, 2)


def _zone_stats(zones: List[ZoneResult]) -> Tuple[float, float, float]:
    if not zones:
        return 0.0, 0.0, 0.0
    total_return = sum(z.return_pct for z in zones)
    hits = sum(1 for z in zones if z.return_pct > 0)
    hit_rate = hits / len(zones) * 100
    avg_return = float(np.mean([z.return_pct for z in zones]))
    return round(total_return, 2), round(hit_rate, 1), round(avg_return, 2)


def _build_combo(prices: np.ndarray, combo: List[CycleInfo]) -> CombinationResult:
    bull_mask = _combined_bullish_mask(prices, combo)
    bear_mask = _combined_bearish_mask(prices, combo)

    # Skip degenerate combos
    bull_pct = bull_mask.mean()
    if bull_pct < 0.05 or bull_pct > 0.95:
        return None

    bull_zones = _compute_zones(prices, bull_mask)
    bear_zones = _compute_zones(prices, bear_mask)

    if not bull_zones:
        return None

    total_ret, hit_rate, avg_ret = _zone_stats(bull_zones)
    bear_total, _, _ = _zone_stats(bear_zones)

    return CombinationResult(
        cycles=combo,
        periods=[c.period for c in combo],
        zones=bull_zones,
        total_return_pct=total_ret,
        compound_return_pct=_compound_return(bull_zones),
        hit_rate=hit_rate,
        avg_return_pct=avg_ret,
        n_zones=len(bull_zones),
        bullish_mask=bull_mask,
        bearish_zones=bear_zones,
        bearish_total_return_pct=bear_total,
        bearish_compound_return_pct=_compound_return(bear_zones),
        bearish_hit_rate=_short_hit_rate(bear_zones),
        short_compound_return_pct=_short_compound_return(bear_zones),
        bearish_mask=bear_mask,
        combo_size=len(combo),
    )


def _return_scan_pool(prices: np.ndarray, top_n: int = 20) -> List[CycleInfo]:
    """
    Brute-force scan: test every integer period from 10 to N//2,
    rank by single-cycle compound return, return top_n as CycleInfo.
    This ensures good cycles that the FFT misses still enter the pool.
    """
    N = len(prices)
    max_p = N // 2
    detrended, _ = _detrend_log(prices)
    candidates: List[Tuple[float, int]] = []  # (compound_return, period)

    for p in range(10, max_p + 1):
        mask = get_bullish_mask(prices, p)
        bull_pct = mask.mean()
        if bull_pct < 0.05 or bull_pct > 0.95:
            continue
        zones = _compute_zones(prices, mask)
        if not zones:
            continue
        cr = _compound_return(zones)
        candidates.append((cr, p))

    candidates.sort(reverse=True)
    result = []
    for cr, p in candidates[:top_n]:
        A, B, amp = _fit_sine(detrended, float(p))
        if amp < 1e-10:
            continue
        state, osc_val, direction = _phase_state(A, B, float(p), N - 1)
        result.append(CycleInfo(
            period=p, period_exact=float(p),
            amplitude=round(amp * prices[-1], 2), strength=0.0, stability=0.0,
            phase_state=state, current_value=osc_val, current_direction=direction,
            oscillator=np.array([]), r_squared=0.0, amplitude_log=amp,
            coeff_a=A, coeff_b=B,
        ))
    return result


def analyze_combinations(
    prices: np.ndarray,
    cycles: List[CycleInfo],
    top_n_per_size: int = 3,
) -> Dict:
    """
    Returns combinations grouped by size, with separate long and short rankings:
      {2: [top N long pairs], 3: [top N long triples],
       "short_2": [top N short pairs], "short_3": [top N short triples]}
    Long  ranked by compound return on bullish zones.
    Short ranked by compound return when shorting bearish zones.
    Both rankings are independent, computed over the full valid combo pool.
    Pool = top 20 FFT cycles + top 20 return-scan cycles (deduplicated).
    """
    # FFT-detected cycles (deduplicated by integer period)
    seen_periods: set = set()
    pool = []
    for c in cycles[:25]:
        if c.period not in seen_periods:
            pool.append(c)
            seen_periods.add(c.period)
        if len(pool) >= 20:
            break

    # Supplement with brute-force return scan so strong periods not caught by FFT
    # (e.g. 86, 26 on MRNA) are always tested
    for c in _return_scan_pool(prices, top_n=20):
        if c.period not in seen_periods:
            pool.append(c)
            seen_periods.add(c.period)

    results: Dict = {2: [], 3: [], "short_2": [], "short_3": []}

    for size in (2, 3):
        all_valid: List[CombinationResult] = []
        for combo in itertools.combinations(pool, size):
            # Skip if any two cycles within the combo are too similar to each other
            periods = [c.period for c in combo]
            if any(
                _periods_too_close(pa, pb)
                for pa, pb in itertools.combinations(periods, 2)
            ):
                continue
            cr = _build_combo(prices, list(combo))
            if cr is not None:
                all_valid.append(cr)

        def _top_n(combos: List[CombinationResult], key, n: int) -> List[CombinationResult]:
            selected: List[CombinationResult] = []
            for r in sorted(combos, key=key, reverse=True):
                if not any(_combos_too_similar(r.periods, s.periods) for s in selected):
                    selected.append(r)
                if len(selected) >= n:
                    break
            return selected

        results[size] = _top_n(all_valid, key=lambda r: r.total_return_pct, n=top_n_per_size)
        results[f"short_{size}"] = _top_n(all_valid, key=lambda r: r.short_compound_return_pct, n=top_n_per_size)

    return results


def _periods_too_close(p1: int, p2: int) -> bool:
    """True if two periods within the same combo are within 10% of each other."""
    return abs(p1 - p2) < max(5, int(0.10 * max(p1, p2)))


def _combos_too_similar(periods_a: List[int], periods_b: List[int]) -> bool:
    """True when all sorted period pairs are within 10% of each other."""
    if len(periods_a) != len(periods_b):
        return False
    for pa, pb in zip(sorted(periods_a), sorted(periods_b)):
        threshold = max(5, int(0.10 * max(pa, pb)))
        if abs(pa - pb) >= threshold:
            return False
    return True


def get_custom_combination(prices: np.ndarray, selected_cycles: List[CycleInfo]) -> CombinationResult:
    """Build a combination result for a user-selected set of cycles."""
    cr = _build_combo(prices, selected_cycles)
    if cr is None:
        # Fallback with no filter
        bull_mask = _combined_bullish_mask(prices, selected_cycles)
        bear_mask = _combined_bearish_mask(prices, selected_cycles)
        bull_zones = _compute_zones(prices, bull_mask)
        bear_zones = _compute_zones(prices, bear_mask)
        total_ret, hit_rate, avg_ret = _zone_stats(bull_zones)
        bear_total, _, _ = _zone_stats(bear_zones)
        return CombinationResult(
            cycles=selected_cycles,
            periods=[c.period for c in selected_cycles],
            zones=bull_zones,
            total_return_pct=total_ret,
            compound_return_pct=_compound_return(bull_zones),
            hit_rate=hit_rate,
            avg_return_pct=avg_ret,
            n_zones=len(bull_zones),
            bullish_mask=bull_mask,
            bearish_zones=bear_zones,
            bearish_total_return_pct=bear_total,
            bearish_compound_return_pct=_compound_return(bear_zones),
            bearish_hit_rate=_short_hit_rate(bear_zones),
            short_compound_return_pct=_short_compound_return(bear_zones),
            bearish_mask=bear_mask,
            combo_size=len(selected_cycles),
        )
    return cr


# ── Stop-Loss simulation ───────────────────────────────────────────────────────

def _simulate_sl_zone(prices: np.ndarray, zone: ZoneResult, sl_pct: float) -> SLZoneResult:
    """
    Simulate a trailing step stop-loss on a single bullish zone.

    Mechanism:
      - Entry at zone.start price; initial SL = entry × (1 - sl_pct/100)
      - Each time price rises sl_pct% from the previous tier, SL steps up by sl_pct%
        → tier 0: SL = entry × (1 - sl_pct%)
        → tier 1 (gain ≥ sl_pct%): SL = entry (break-even)
        → tier n (gain ≥ n × sl_pct%): SL = entry × (1 + (n-1) × sl_pct%)
      - If price closes at or below SL: trade exits at SL price
      - Otherwise trade exits at zone end price
    """
    zone_prices = prices[zone.start: zone.end + 1]
    n = len(zone_prices)
    if n < 2:
        path = zone_prices[:1].copy() if n else np.array([])
        return SLZoneResult(zone=zone, sl_return_pct=0.0, sl_hit=False,
                            sl_exit_bar=zone.end, sl_path=path,
                            original_return_pct=zone.return_pct)

    entry = zone_prices[0]
    sl_frac = sl_pct / 100.0
    sl = entry * (1.0 - sl_frac)
    max_price = entry
    tier = 0

    sl_path: List[float] = [sl]
    sl_hit = False
    exit_bar = zone.end
    exit_ret = (zone_prices[-1] - entry) / entry * 100.0

    for i in range(1, n):
        price = float(zone_prices[i])

        if price <= sl:
            sl_hit = True
            exit_bar = zone.start + i
            exit_ret = (sl - entry) / entry * 100.0
            sl_path.append(sl)
            break

        if price > max_price:
            max_price = price
            new_tier = int((max_price - entry) / entry / sl_frac)
            if new_tier > tier:
                tier = new_tier
                sl = entry * (1.0 + (tier - 1) * sl_frac)

        sl_path.append(sl)

    return SLZoneResult(
        zone=zone,
        sl_return_pct=round(exit_ret, 2),
        sl_hit=sl_hit,
        sl_exit_bar=exit_bar,
        sl_path=np.array(sl_path),
        original_return_pct=zone.return_pct,
    )


def simulate_sl_zones(prices: np.ndarray, zones: List[ZoneResult], sl_pct: float) -> List[SLZoneResult]:
    """Simulate trailing SL on every bullish zone of a combination."""
    return [_simulate_sl_zone(prices, zone, sl_pct) for zone in zones]


def _simulate_sl_zone_fixed(prices: np.ndarray, zone: ZoneResult, sl_pct: float) -> SLZoneResult:
    """
    Fixed stop-loss: SL is set once at entry × (1 - sl_pct/100) and NEVER moves,
    regardless of how much the price rises.
    """
    zone_prices = prices[zone.start: zone.end + 1]
    n = len(zone_prices)
    if n < 2:
        path = zone_prices[:1].copy() if n else np.array([])
        return SLZoneResult(zone=zone, sl_return_pct=0.0, sl_hit=False,
                            sl_exit_bar=zone.end, sl_path=path,
                            original_return_pct=zone.return_pct)

    entry = zone_prices[0]
    sl = entry * (1.0 - sl_pct / 100.0)

    sl_hit = False
    exit_bar = zone.end
    exit_ret = (zone_prices[-1] - entry) / entry * 100.0

    for i in range(1, n):
        price = float(zone_prices[i])
        if price <= sl:
            sl_hit = True
            exit_bar = zone.start + i
            exit_ret = (sl - entry) / entry * 100.0
            break

    sl_path = np.full(min(exit_bar - zone.start + 1, n), sl)

    return SLZoneResult(
        zone=zone,
        sl_return_pct=round(exit_ret, 2),
        sl_hit=sl_hit,
        sl_exit_bar=exit_bar,
        sl_path=sl_path,
        original_return_pct=zone.return_pct,
    )


def simulate_sl_zones_fixed(prices: np.ndarray, zones: List[ZoneResult], sl_pct: float) -> List[SLZoneResult]:
    """Simulate fixed SL on every bullish zone of a combination."""
    return [_simulate_sl_zone_fixed(prices, zone, sl_pct) for zone in zones]
