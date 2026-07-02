#!/usr/bin/env python3
"""
Cycle Confluence Analyser
================================================================
Surcouche de fiabilisation du projet cycle_analyser :
  moteur de cycles original (non modifié)
  + validation walk-forward hors échantillon
  + confirmation TradingView / Yahoo analystes / smart money
  + modulateur de risque macro Polymarket
  = score de confluence 0-100 et verdict.

Usage :
  python confluence_analyser.py AAPL
  python confluence_analyser.py ^SOX --period 5y --cycles 121,80
  python confluence_analyser.py BTC-USD --no-external   (cycles + WF seulement)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from core.data_fetcher import fetch_data
from core.cycle_detector import detect_cycles, CycleInfo
from core.combination_analyzer import analyze_combinations, get_custom_combination
from validation.walk_forward import walk_forward_validate, cycle_persistence
from confluence.scorer import compute_confluence, cycle_signal_score

console = Console()

VERDICT_STYLE = {
    "ACHAT FORT": "bold green",
    "ACHAT": "green",
    "NEUTRE": "yellow",
    "ÉVITER": "red",
    "VENTE / SHORT": "bold red",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse de cycles + confluence multi-sources")
    p.add_argument("ticker", help="Symbole Yahoo Finance (AAPL, ^SOX, BTC-USD…)")
    p.add_argument("--period", default="5y", help="Historique (défaut : 5y)")
    p.add_argument("--interval", default="1d", help="Intervalle (défaut : 1d)")
    p.add_argument("--cycles", default=None,
                   help="Périodes imposées, ex : 121,80 (sinon auto)")
    p.add_argument("--wf-step", type=int, default=2,
                   help="Pas du walk-forward (1=précis/lent, 3=rapide)")
    p.add_argument("--no-external", action="store_true",
                   help="Désactive Polymarket/TradingView/Yahoo/smart money")
    p.add_argument("--no-cot", action="store_true", help="Désactive le COT CFTC (lent)")
    p.add_argument("--html", action="store_true", help="Génère le rapport HTML")
    return p.parse_args()


def pick_combo(prices: np.ndarray, cycles_arg: str | None):
    """Sélectionne la combinaison : imposée par --cycles, sinon meilleure auto.

    En mode auto, les périodes sont plafonnées à N/4 : un cycle doit se
    répéter au moins ~4 fois pour être statistiquement crédible ET rester
    validable par walk-forward (fenêtre d'apprentissage = 3x la période).
    """
    max_p = len(prices) // 4 if not cycles_arg else None
    all_cycles = detect_cycles(prices, n_cycles=25, max_period=max_p)
    if not all_cycles:
        console.print("[red]Aucun cycle détecté.[/red]")
        sys.exit(1)

    if cycles_arg:
        wanted = [int(x) for x in cycles_arg.split(",")]
        selected = []
        for w in wanted:
            match = min(all_cycles, key=lambda c: abs(c.period - w))
            if abs(match.period - w) > max(3, w * 0.1):
                # période absente de la détection : on la force
                match = CycleInfo(period=w, period_exact=float(w), amplitude=0,
                                  strength=0, stability=0, phase_state="bullish",
                                  current_value=0, current_direction=0,
                                  oscillator=np.zeros(len(prices)), r_squared=0,
                                  amplitude_log=0, coeff_a=0, coeff_b=0)
            selected.append(match)
        combo = get_custom_combination(prices, selected)
    else:
        combos = analyze_combinations(prices, cycles=all_cycles)
        best = None
        for size in (2, 3):
            for c in combos.get(size, []):
                if best is None or c.compound_return_pct > best.compound_return_pct:
                    best = c
        combo = best
    if combo is None:
        console.print("[red]Aucune combinaison exploitable.[/red]")
        sys.exit(1)
    return combo, all_cycles


def main() -> None:
    args = parse_args()
    console.print(Panel.fit(
        "[bold cyan]Cycle Confluence Analyser[/bold cyan]\n"
        "[dim]Cycles FFT + walk-forward + Polymarket + TradingView + Yahoo + smart money[/dim]",
        border_style="cyan"))

    # 1. Données + cycles (moteur original) ---------------------------------
    with console.status("Téléchargement des données…"):
        data = fetch_data(args.ticker, period=args.period, interval=args.interval)
    prices = data["Close"].to_numpy().ravel()
    console.print(f"[dim]{len(prices)} barres chargées pour {args.ticker}.[/dim]")

    with console.status("Détection des cycles…"):
        combo, all_cycles = pick_combo(prices, args.cycles)
    periods = combo.periods
    console.print(f"Combinaison retenue : [bold]{combo.label}[/bold] "
                  f"(hit rate in-sample : {combo.hit_rate:.0f}%, "
                  f"rendement composé : {combo.compound_return_pct:.1f}%)")

    # 2. Validation walk-forward (hors échantillon) --------------------------
    with console.status("Validation walk-forward (hors échantillon)…"):
        wf = walk_forward_validate(prices, periods, step=args.wf_step,
                                   in_sample_hit_rate=combo.hit_rate)
        persistence = float(np.mean([cycle_persistence(prices, p) for p in periods]))

    t = Table(title="Fiabilité hors échantillon", box=box.ROUNDED, border_style="dim")
    t.add_column("Métrique"); t.add_column("Valeur", justify="right")
    t.add_row("Hit rate in-sample (original)", f"{combo.hit_rate:.1f} %")
    t.add_row("Hit rate OOS (walk-forward)", f"[bold]{wf.oos_hit_rate:.1f} %[/bold]")
    if wf.degradation_pct is not None:
        style = "red" if wf.degradation_pct > 20 else "green"
        t.add_row("Dégradation IS → OOS", f"[{style}]{wf.degradation_pct:+.1f} pts[/{style}]")
    t.add_row("Signaux haussiers OOS", str(wf.n_bull_signals))
    t.add_row("Persistance des cycles (0-1)", f"{persistence:.2f}")
    t.add_row("Score de fiabilité", f"[bold]{wf.reliability_score:.0f} / 100[/bold]")
    console.print(t)

    # 3. Signaux externes -----------------------------------------------------
    tv_score = yahoo_score = sm_score = 50.0
    macro_risk, macro_mult = 50.0, 1.0
    unavailable = []
    ext_detail = {}

    if not args.no_external:
        from signals.tradingview import fetch_tv_rating
        from signals.yahoo_analysts import fetch_yahoo_analysts
        from signals.smart_money import fetch_smart_money
        from signals.polymarket import fetch_macro_risk

        with console.status("TradingView…"):
            tv = fetch_tv_rating(args.ticker)
        tv_score = tv.score
        if not tv.available:
            unavailable.append("tradingview")
        ext_detail["tradingview"] = tv

        with console.status("Consensus analystes Yahoo…"):
            ya = fetch_yahoo_analysts(args.ticker)
        yahoo_score = ya.score
        if not ya.available or (ya.reco_score is None and ya.target_upside_pct is None):
            unavailable.append("yahoo")
        ext_detail["yahoo"] = ya

        with console.status("Smart money (initiés, institutions, COT)…"):
            sm = fetch_smart_money(args.ticker, include_cot=not args.no_cot)
        sm_score = sm.score
        if not sm.available:
            unavailable.append("smart_money")
        ext_detail["smart_money"] = sm

        with console.status("Risque macro Polymarket…"):
            pm = fetch_macro_risk()
        macro_risk = pm.macro_risk_score
        macro_mult = pm.risk_multiplier
        ext_detail["polymarket"] = pm

        t2 = Table(title="Couches de confirmation", box=box.ROUNDED, border_style="dim")
        t2.add_column("Source"); t2.add_column("Score", justify="right"); t2.add_column("Détail")
        t2.add_row("TradingView", f"{tv_score:.0f}",
                   f"{tv.label_1d} (1D {tv.rating_1d if tv.rating_1d is not None else '—'})")
        det_y = []
        if ya.n_analysts:
            det_y.append(f"{ya.n_analysts} analystes")
        if ya.target_upside_pct is not None:
            det_y.append(f"upside cible {ya.target_upside_pct:+.1f}%")
        if ya.short_pct_float is not None:
            det_y.append(f"short {ya.short_pct_float:.1f}%")
        t2.add_row("Yahoo analystes", f"{yahoo_score:.0f}", ", ".join(det_y) or ya.note)
        det_s = []
        if sm.insider_score is not None:
            det_s.append(f"initiés {sm.insider_score:.0f}")
        if sm.cot_percentile_1y is not None:
            det_s.append(f"COT pctile {sm.cot_percentile_1y:.0f}%")
        if sm.inst_pct_held is not None:
            det_s.append(f"instit. {sm.inst_pct_held:.0f}%")
        t2.add_row("Smart money", f"{sm_score:.0f}", ", ".join(det_s) or sm.note)
        t2.add_row("Polymarket (macro)", f"risque {macro_risk:.0f}",
                   f"multiplicateur x{macro_mult:.2f}")
        console.print(t2)
    else:
        unavailable = ["tradingview", "yahoo", "smart_money"]

    # 4. Confluence -----------------------------------------------------------
    phase_states = [c.phase_state for c in combo.cycles]
    c_score = cycle_signal_score(phase_states, combo.hit_rate)

    result = compute_confluence(
        ticker=args.ticker,
        cycle_score=c_score,
        wf_reliability=wf.reliability_score,
        tv_score=tv_score,
        yahoo_score=yahoo_score,
        smart_money_score=sm_score,
        macro_risk=macro_risk,
        macro_multiplier=macro_mult,
        unavailable=unavailable,
    )

    style = VERDICT_STYLE.get(result.verdict, "white")
    console.print(Panel(
        f"[bold]{args.ticker}[/bold] — Phases : {', '.join(phase_states)}\n"
        f"Score de confluence : [bold]{result.final_score:.0f} / 100[/bold] "
        f"(brut {result.raw_score:.0f}, macro x{result.macro_multiplier:.2f})\n"
        f"Verdict : [{style}]{result.verdict}[/{style}]",
        title="CONFLUENCE", border_style=style.split()[-1]))

    if wf.n_bull_signals == 0:
        console.print(
            "[yellow]⚠ 0 signal haussier hors échantillon : la combinaison est "
            "invérifiable (périodes trop longues pour l'historique). "
            "Essayez --period 10y ou des périodes plus courtes.[/yellow]")
    for wmsg in result.warnings:
        console.print(f"[yellow]⚠ {wmsg}[/yellow]")

    if args.html:
        import webbrowser
        from reporting.html_report import generate_report
        path = generate_report(args.ticker, prices, combo, wf, result,
                               persistence, ext_detail)
        console.print(f"[dim]Rapport HTML : {path} (ouverture dans le navigateur…)[/dim]")
        webbrowser.open(Path(path).resolve().as_uri())

    console.print("\n[dim]Outil d'aide à la décision — pas un conseil en investissement.[/dim]")


if __name__ == "__main__":
    main()
