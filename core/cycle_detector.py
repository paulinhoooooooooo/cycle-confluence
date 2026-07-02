from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar


@dataclass
class CycleInfo:
    period: int
    period_exact: float     # refined non-integer period
    amplitude: float        # in price units (amplitude of oscillation)
    strength: float         # local FFT SNR: sqrt(peak_power / local_noise_power)
    stability: float        # 0-1, rolling window consistency
    phase_state: str        # 'bullish', 'bearish', 'peak', 'trough'
    current_value: float    # oscillator value at last bar (log-price units)
    current_direction: float
    oscillator: np.ndarray  # oscillator in price units, centered on price
    r_squared: float
    amplitude_log: float
    coeff_a: float
    coeff_b: float
    rank: int = 0
    hit_rate: float = 0.0        # % bullish zones where price went up
    short_hit_rate: float = 0.0  # % bearish zones where price went down


def _detrend_log(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = len(prices)
    log_p = np.log(prices)
    t = np.arange(N, dtype=float)
    poly = np.polyfit(t, log_p, 1)
    trend = np.polyval(poly, t)
    return log_p - trend, trend


def _fit_sine(detrended: np.ndarray, period: float) -> Tuple[float, float, float]:
    """Fit A*cos(2π*t/T) + B*sin(2π*t/T) via least squares."""
    N = len(detrended)
    t = np.arange(N, dtype=float)
    cos_c = np.cos(2 * np.pi * t / period)
    sin_c = np.sin(2 * np.pi * t / period)
    X = np.column_stack([cos_c, sin_c])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, detrended, rcond=None)
    except Exception:
        return 0.0, 0.0, 0.0
    A, B = coeffs
    return float(A), float(B), float(np.sqrt(A**2 + B**2))


def _residuals_ss(period: float, detrended: np.ndarray) -> float:
    A, B, _ = _fit_sine(detrended, period)
    N = len(detrended)
    t = np.arange(N, dtype=float)
    fitted = A * np.cos(2 * np.pi * t / period) + B * np.sin(2 * np.pi * t / period)
    return float(np.sum((detrended - fitted) ** 2))


def _refine_period(detrended: np.ndarray, T_init: float, search_frac: float = 0.12) -> float:
    """Refine a period estimate by minimizing residual SS in a local window."""
    lo = max(4.0, T_init * (1 - search_frac))
    hi = T_init * (1 + search_frac)
    result = minimize_scalar(_residuals_ss, bounds=(lo, hi), method="bounded",
                             args=(detrended,),
                             options={"xatol": 0.1, "maxiter": 40})
    return float(result.x) if result.success else T_init


def _compute_r2(detrended: np.ndarray, period: float, A: float, B: float) -> float:
    N = len(detrended)
    t = np.arange(N, dtype=float)
    fitted = A * np.cos(2 * np.pi * t / period) + B * np.sin(2 * np.pi * t / period)
    ss_res = np.sum((detrended - fitted) ** 2)
    ss_tot = np.sum((detrended - np.mean(detrended)) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(np.clip(1 - ss_res / ss_tot, 0, 1))


def _local_snr(power: np.ndarray, idx: int, half_band: int = 6) -> float:
    """
    Compute sqrt(peak_power / local_noise) using bins outside the peak.
    This gives values typically in range 1-10 for real market cycles.
    """
    lo = max(0, idx - half_band)
    hi = min(len(power) - 1, idx + half_band)
    band = np.concatenate([power[lo: max(lo, idx - 1)], power[min(hi, idx + 2): hi + 1]])
    if len(band) == 0 or np.mean(band) <= 0:
        return 1.0
    snr = power[idx] / np.mean(band)
    return float(np.sqrt(max(snr, 1.0)))


def _compute_stability(detrended: np.ndarray, period: float) -> float:
    """
    Phase-coherence stability: split series into segments and check how
    consistent the cycle's phase is across segments.
    R=1 (perfect regularity) → R=0 (random).
    This is amplitude-invariant and robust to overlapping cycles.
    """
    N = len(detrended)
    seg_len = max(int(period * 2.5), 20)
    n_segs = N // seg_len
    if n_segs < 3:
        # Try with smaller segments
        seg_len = max(int(period * 1.5), 15)
        n_segs = N // seg_len
    if n_segs < 2:
        return 0.5

    phases: List[float] = []
    for i in range(n_segs):
        start = i * seg_len
        end = min(start + seg_len, N)
        seg = detrended[start:end]
        A, B, amp = _fit_sine(seg, period)
        if amp < 1e-10:
            continue
        # Global phase at segment start: ψ = 2π*start/T + atan2(B, A)
        phi_global = (2 * np.pi * start / period + np.arctan2(B, A)) % (2 * np.pi)
        phases.append(phi_global)

    if len(phases) < 2:
        return 0.5

    phi_arr = np.array(phases)
    # Circular resultant length R: 1=perfect coherence, 0=random
    R = float(np.sqrt(np.mean(np.cos(phi_arr)) ** 2 + np.mean(np.sin(phi_arr)) ** 2))
    return round(R, 2)


def _phase_state(A: float, B: float, period: float, t_last: int) -> Tuple[str, float, float]:
    osc = A * np.cos(2 * np.pi * t_last / period) + B * np.sin(2 * np.pi * t_last / period)
    direction = (
        -A * np.sin(2 * np.pi * t_last / period) + B * np.cos(2 * np.pi * t_last / period)
    ) * (2 * np.pi / period)
    amp = np.sqrt(A**2 + B**2)
    max_dir = amp * (2 * np.pi / period)
    turning_threshold = 0.22 * max_dir if max_dir > 0 else 0.0

    if abs(direction) <= turning_threshold:
        state = "peak" if osc > 0 else "trough"
    elif direction > 0:
        state = "bullish"
    else:
        state = "bearish"

    return state, float(osc), float(direction)


def detect_cycles(
    prices: np.ndarray,
    min_period: int = 10,
    max_period: Optional[int] = None,
    n_cycles: int = 25,
) -> List[CycleInfo]:
    """
    Detect dominant cycles using FFT + sine-wave refinement.
    Sorted by stability descending.
    """
    N = len(prices)
    if max_period is None:
        max_period = N // 2
    max_period = min(max_period, N // 2)

    detrended, trend = _detrend_log(prices)

    # ── FFT for period detection ───────────────────────────────────────────
    win = np.hanning(N)
    fft = np.fft.rfft(detrended * win)
    freqs = np.fft.rfftfreq(N)[1:]  # skip DC
    power = np.abs(fft[1:]) ** 2

    # Convert to period domain and filter
    with np.errstate(divide="ignore"):
        raw_periods = np.where(freqs > 0, 1.0 / freqs, 0.0)

    valid_mask = (raw_periods >= min_period) & (raw_periods <= max_period)
    if not valid_mask.any():
        return []

    v_idx = np.where(valid_mask)[0]
    v_periods = raw_periods[v_idx]
    v_power = power[v_idx]
    v_freqs = freqs[v_idx]

    # Adaptive noise floor: rolling median of the power spectrum
    # Use a generous window so weak-but-real cycles still appear
    n_v = len(v_power)
    noise_window = max(5, n_v // 8)
    noise_floor = np.array([
        np.median(v_power[max(0, i - noise_window): i + noise_window + 1])
        for i in range(n_v)
    ])

    # Find peaks above noise floor * 1.3 (permissive threshold)
    excess = v_power / (noise_floor + 1e-30)
    peaks, _ = find_peaks(excess, distance=2, height=1.3)

    if len(peaks) == 0:
        peaks = np.argsort(v_power)[::-1][: n_cycles]

    sorted_peaks = peaks[np.argsort(v_power[peaks])[::-1]]

    # ── Build cycle list ───────────────────────────────────────────────────
    cycles: List[CycleInfo] = []
    seen_periods: List[float] = []

    for peak_vi in sorted_peaks:
        T_raw = float(v_periods[peak_vi])

        # Skip if too close to an already accepted period
        if any(abs(T_raw - sp) / sp < 0.07 for sp in seen_periods):
            continue

        # Refine period by minimizing residuals
        T_refined = _refine_period(detrended, T_raw, search_frac=0.12)

        A, B, amp_log = _fit_sine(detrended, T_refined)
        if amp_log < 1e-10:
            continue

        r2 = _compute_r2(detrended, T_refined, A, B)

        # Strength: local FFT SNR at the detected bin
        strength = _local_snr(v_power, peak_vi, half_band=6)

        stability = _compute_stability(detrended, T_refined)

        T_int = max(min_period, round(T_refined))
        state, osc_val, direction = _phase_state(A, B, T_refined, N - 1)

        # Oscillator in price space: additive (oscillates around price trend)
        price_center = np.exp(trend)
        t_arr = np.arange(N, dtype=float)
        osc_log = A * np.cos(2 * np.pi * t_arr / T_refined) + B * np.sin(2 * np.pi * t_arr / T_refined)
        oscillator_price = price_center * (1 + osc_log)

        amp_price = float(amp_log * prices[-1])

        # Also skip if the rounded integer period already exists
        if any(c.period == T_int for c in cycles):
            continue

        seen_periods.append(T_raw)
        cycles.append(
            CycleInfo(
                period=T_int,
                period_exact=round(T_refined, 2),
                amplitude=round(amp_price, 2),
                strength=round(strength, 2),
                stability=stability,
                phase_state=state,
                current_value=osc_val,
                current_direction=direction,
                oscillator=oscillator_price,
                r_squared=round(r2, 4),
                amplitude_log=float(amp_log),
                coeff_a=float(A),
                coeff_b=float(B),
            )
        )

        if len(cycles) >= n_cycles:
            break

    # Sort by stability desc, then by amplitude desc as tiebreaker
    cycles.sort(key=lambda c: (c.stability, c.amplitude), reverse=True)
    for i, c in enumerate(cycles):
        c.rank = i + 1

    return cycles


def get_oscillator_series(prices: np.ndarray, period: int) -> np.ndarray:
    """Normalized oscillator (log-price units, centered on zero)."""
    detrended, _ = _detrend_log(prices)
    A, B, _ = _fit_sine(detrended, float(period))
    N = len(prices)
    t = np.arange(N, dtype=float)
    return A * np.cos(2 * np.pi * t / period) + B * np.sin(2 * np.pi * t / period)


def get_bullish_mask(prices: np.ndarray, period: int) -> np.ndarray:
    """True where the cycle is rising (discrete difference, matches TradingView's sine > sine[1])."""
    detrended, _ = _detrend_log(prices)
    A, B, _ = _fit_sine(detrended, float(period))
    N = len(prices)
    t = np.arange(N, dtype=float)
    osc = A * np.cos(2 * np.pi * t / period) + B * np.sin(2 * np.pi * t / period)
    mask = np.empty(N, dtype=bool)
    mask[1:] = osc[1:] > osc[:-1]
    mask[0] = mask[1]
    return mask


def bars_to_next_turning_point(cycle: "CycleInfo") -> Tuple[int, str]:
    """
    Return (n_bars, direction) until the next cycle peak or trough.
    direction = 'peak' or 'trough'.
    """
    T = cycle.period_exact
    A, B = cycle.coeff_a, cycle.coeff_b
    # Current direction: rising → next event is peak, falling → next event is trough
    if cycle.current_direction > 0:
        target_state = "peak"
        # Time remaining: quarter cycle from now is approximately the peak
        # More precisely: t_peak where cos(2π*t/T + φ) = max → t_peak is in [0, T/4] ahead
        # Solve -A*sin(2π*t/T) + B*cos(2π*t/T) = 0 for next t after N-1
        pass
    else:
        target_state = "trough"

    # Phase at last bar
    N_last = cycle.period  # placeholder — computed in caller
    phi_last = np.arctan2(B, A) + 2 * np.pi * (N_last - 1) / T  # not quite right

    # Simple approximation: quarter-cycle remaining
    amp = np.sqrt(A**2 + B**2)
    if amp < 1e-12:
        return T // 4, "peak"

    # Current normalized value (position in cycle: -1 to 1)
    osc_norm = cycle.current_value / amp
    # Phase in [0, 2π]: osc_norm = cos(phase_effective) → phase_eff = arccos(osc_norm) or 2π-arccos
    osc_norm = float(np.clip(osc_norm, -1, 1))
    phase_eff = float(np.arccos(osc_norm))  # in [0, π]
    if cycle.current_direction < 0:
        phase_eff = 2 * np.pi - phase_eff   # in [π, 2π]

    if cycle.current_direction > 0:
        bars_to_next = max(1, round((np.pi - phase_eff) / (2 * np.pi) * T))
        return bars_to_next, "peak"
    else:
        bars_to_next = max(1, round((2 * np.pi - phase_eff) / (2 * np.pi) * T))
        return bars_to_next, "trough"
