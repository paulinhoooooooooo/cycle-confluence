# Cycle Confluence Analyser

Surcouche de **fiabilisation et de confirmation multi-sources** du projet
[cycle_analyser](https://github.com/paulinhoooooooooo/cycle_analyser).
Le moteur original de détection de cycles (FFT + ajustement sinusoïdal) est
**réutilisé tel quel, sans modification** (dossier `core/`, vendored).

## Pourquoi ce projet

Le `cycle_analyser` original ajuste les sinusoïdes sur **tout** l'historique,
puis calcule les hit rates sur ce **même** historique. C'est un biais de
look-ahead : les performances affichées sont structurellement optimistes.
Test sur données synthétiques contenant de vrais cycles : hit rate in-sample
100 %, hit rate hors échantillon 67 % (−32 points).

Ce projet ajoute :

| Couche | Rôle | Source |
|---|---|---|
| **Walk-forward** | Hit rate *hors échantillon* : à chaque barre, les cycles sont ré-ajustés uniquement sur le passé, la prédiction est comparée au rendement réel du lendemain. Mesure la dégradation IS→OOS et pénalise le sur-ajustement. | calcul local |
| **Persistance** | Vérifie que le cycle était déjà détectable avec 70/80/90 % de l'historique (un cycle qui n'apparaît qu'avec la dernière barre est suspect). | calcul local |
| **TradingView** | Notation technique agrégée (~26 indicateurs) sur 1D/1W/1M, via l'endpoint public du scanner. | scanner.tradingview.com |
| **Yahoo analystes** | Consensus strongBuy→strongSell, upside vers l'objectif de cours moyen, short interest. | yfinance |
| **Smart money** | Transactions d'initiés (SEC Form 4, 6 mois), % institutionnel, et **COT CFTC** (positionnement net des hedge funds sur les futures, percentile 1 an). | yfinance + cftc.gov |
| **Polymarket** | Score de risque macro dérivé des marchés prédictifs actifs (récession, crash, Fed, guerre…). Agit en **modulateur global** ×0,70 – ×1,10. | gamma-api.polymarket.com |

Toutes les sources sont gratuites, sans clé API. Chaque source indisponible
(ex. pas d'analystes sur un indice) voit son poids **redistribué** plutôt que
compté comme neutre.

## Score de confluence

```
score = 35% cycle + 20% fiabilité WF + 15% TradingView + 15% Yahoo + 15% smart money
final = score × modulateur macro Polymarket
```

| Score | Verdict |
|---|---|
| ≥ 75 | ACHAT FORT |
| 60–74 | ACHAT |
| 45–59 | NEUTRE |
| 30–44 | ÉVITER |
| < 30 | VENTE / SHORT |

Des avertissements de divergence sont émis (ex. cycle haussier mais initiés
vendeurs, ou momentum TradingView baissier).

## Installation

```bash
git clone https://github.com/paulinhoooooooooo/cycle-confluence.git
cd cycle-confluence
pip install -r requirements.txt
```

## Usage

```bash
# Analyse complète (cycles auto + toutes les confirmations)
python confluence_analyser.py AAPL

# Périodes imposées (mêmes que la watchlist du projet original)
python confluence_analyser.py SOXQ --cycles 121,80 --period 5y

# Rapide, sans sources externes (cycles + walk-forward seulement)
python confluence_analyser.py BTC-USD --no-external --wf-step 4

# Rapport HTML
python confluence_analyser.py AAPL --html
```

Options : `--period` (5y), `--interval` (1d), `--wf-step` (2 ; 1 = plus précis
mais plus lent), `--no-cot` (saute le téléchargement CFTC, ~10 Mo).

## Structure

```
core/          moteur original (vendored, non modifié)
validation/    walk_forward.py — hit rate OOS, persistance
signals/       polymarket.py, tradingview.py, yahoo_analysts.py, smart_money.py
confluence/    scorer.py — agrégation pondérée + verdict
reporting/     html_report.py
confluence_analyser.py   CLI
```

## Limites connues

- Les endpoints TradingView et Polymarket sont publics mais non documentés
  officiellement : ils peuvent changer sans préavis (le code se replie alors
  en neutre, sans planter).
- Le COT ne couvre que les grands futures (S&P, Nasdaq, BTC, or, pétrole…).
- Le walk-forward valide le *timing* du signal barre par barre, pas une
  stratégie complète avec stop-loss (voir le projet original pour les SL).

**Outil d'aide à la décision — pas un conseil en investissement.**
