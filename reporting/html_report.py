"""Rapport HTML compact : score de confluence, composantes, avertissements."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

_TPL = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>Confluence — {ticker}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, sans-serif; background:#0f1420; color:#e6e9f0;
        max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
 h1 {{ font-size: 1.5rem; }} .dim {{ color:#8a93a8; font-size:.85rem; }}
 .score {{ font-size: 3rem; font-weight: 700; }}
 .verdict {{ display:inline-block; padding:.3rem 1rem; border-radius: 6px; font-weight:600; }}
 .v-strongbuy {{ background:#0d6b2f; }} .v-buy {{ background:#1d7a3f; }}
 .v-neutral {{ background:#8a6d1a; }} .v-avoid {{ background:#8a3a1a; }} .v-sell {{ background:#8a1a1a; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
 th, td {{ text-align:left; padding: .45rem .7rem; border-bottom: 1px solid #232a3d; }}
 .bar {{ height: 10px; background:#232a3d; border-radius:5px; overflow:hidden; }}
 .bar>div {{ height:100%; background: linear-gradient(90deg,#e05252,#e0c052 50%,#52e07a); }}
 .warn {{ background:#3d2f1a; border-left: 4px solid #e0c052; padding:.6rem .8rem;
          margin:.4rem 0; border-radius: 4px; }}
</style></head><body>
<h1>Cycle Confluence — {ticker}</h1>
<p class="dim">Généré le {date} • Combinaison de cycles : {combo_label} •
Dernier prix : {last_price}</p>

<p><span class="score">{score:.0f}</span> / 100
&nbsp;<span class="verdict {vclass}">{verdict}</span></p>
<div class="bar"><div style="width:{score:.0f}%"></div></div>
<p class="dim">Score brut {raw:.0f} × modulateur macro {mult:.2f}
(risque Polymarket : {macro:.0f}/100)</p>

<h2>Composantes</h2>
<table>
<tr><th>Source</th><th>Score</th><th>Poids</th></tr>
{component_rows}
</table>

<h2>Fiabilité hors échantillon (walk-forward)</h2>
<table>
<tr><td>Hit rate in-sample (méthode originale)</td><td>{is_hit:.1f} %</td></tr>
<tr><td><b>Hit rate hors échantillon</b></td><td><b>{oos_hit:.1f} %</b></td></tr>
<tr><td>Dégradation IS → OOS</td><td>{degrad}</td></tr>
<tr><td>Signaux haussiers testés</td><td>{n_bull}</td></tr>
<tr><td>Persistance des cycles</td><td>{persistence:.2f} / 1</td></tr>
</table>

{warnings_html}

<p class="dim">Outil d'aide à la décision, pas un conseil en investissement.
Moteur de cycles : cycle_analyser (github.com/paulinhoooooooooo).</p>
</body></html>"""

_VCLASS = {"ACHAT FORT": "v-strongbuy", "ACHAT": "v-buy", "NEUTRE": "v-neutral",
           "ÉVITER": "v-avoid", "VENTE / SHORT": "v-sell"}

_NAMES = {"cycle": "Signal cyclique", "walk_forward": "Fiabilité walk-forward",
          "tradingview": "TradingView (technique)", "yahoo": "Yahoo (analystes)",
          "smart_money": "Smart money (initiés/COT)"}


def generate_report(ticker, prices: np.ndarray, combo, wf, result,
                    persistence: float, ext_detail: dict,
                    out_dir: str = "reports") -> str:
    Path(out_dir).mkdir(exist_ok=True)

    rows = ""
    for k, v in result.components.items():
        w = result.weights.get(k)
        w_str = f"{w*100:.0f} %" if w is not None else "<span class='dim'>indisponible</span>"
        rows += f"<tr><td>{_NAMES.get(k, k)}</td><td>{v:.0f}</td><td>{w_str}</td></tr>\n"

    warns = "".join(f"<div class='warn'>⚠ {w}</div>" for w in result.warnings)
    if warns:
        warns = "<h2>Avertissements</h2>" + warns

    degrad = "—"
    if wf.degradation_pct is not None:
        degrad = f"{wf.degradation_pct:+.1f} pts"

    html = _TPL.format(
        ticker=ticker, date=datetime.now().strftime("%d/%m/%Y %H:%M"),
        combo_label=combo.label, last_price=f"{prices[-1]:,.2f}",
        score=result.final_score, verdict=result.verdict,
        vclass=_VCLASS.get(result.verdict, "v-neutral"),
        raw=result.raw_score, mult=result.macro_multiplier, macro=result.macro_risk,
        component_rows=rows, is_hit=combo.hit_rate, oos_hit=wf.oos_hit_rate,
        degrad=degrad, n_bull=wf.n_bull_signals, persistence=persistence,
        warnings_html=warns,
    )
    path = Path(out_dir) / f"confluence_{ticker.replace('^','').replace('=','_')}.html"
    path.write_text(html, encoding="utf-8")
    return str(path)
