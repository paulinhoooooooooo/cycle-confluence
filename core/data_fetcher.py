import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

# Common aliases → canonical Yahoo Finance ticker
# Useful when the shell eats the '^' prefix (e.g. Windows CMD)
_TICKER_ALIASES: dict = {
    # S&P 500
    "SPX": "^GSPC", "GSPC": "^GSPC", "SP500": "^GSPC", "^SPX": "^GSPC",
    # Nasdaq 100
    "NDX": "^NDX", "NASDAQ": "^IXIC", "COMP": "^IXIC",
    # Dow Jones
    "DJI": "^DJI", "DOW": "^DJI", "DJIA": "^DJI",
    # US indices
    "DAX": "^GDAXI", "GDAXI": "^GDAXI",
    "CAC": "^FCHI", "CAC40": "^FCHI", "FCHI": "^FCHI",
    "FTSE": "^FTSE",
    "NIKKEI": "^N225", "N225": "^N225",
    "VIX": "^VIX",
    "RUT": "^RUT",
    "SOX": "^SOX",
    # European indices (^ eaten by Windows CMD)
    "STOXX50E": "^STOXX50E", "STOXX50": "^STOXX50E", "EUROSTOXX50": "^STOXX50E",
    "STOXX600": "^STOXX", "STOXX": "^STOXX",
    "AEX": "^AEX",
    "BEL20": "^BFX", "BFX": "^BFX",
    "IBEX": "^IBEX", "IBEX35": "^IBEX",
    "SMI": "^SSMI", "SSMI": "^SSMI",
    "MIB": "FTSEMIB.MI",
    "OMX": "^OMXS30",
    # Other
    "HANGSENG": "^HSI", "HSI": "^HSI",
    "ASX": "^AXJO", "ASX200": "^AXJO",
}


def _resolve_ticker(ticker: str) -> str:
    """Return the canonical Yahoo Finance symbol, resolving common aliases."""
    return _TICKER_ALIASES.get(ticker.upper(), ticker)


def fetch_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    ticker = _resolve_ticker(ticker)
    # auto_adjust=False keeps both Close (split-adjusted) and Adj Close (+ dividends)
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if data.empty:
        hint = ""
        if "^" in ticker:
            hint = ' Sur Windows CMD, entourez le ticker de guillemets : "^GSPC"'
        raise ValueError(
            f"Aucune donnée trouvée pour '{ticker}'. Vérifiez le symbole Yahoo Finance.{hint}"
        )
    data = data.dropna()
    if len(data) < 50:
        raise ValueError(f"Pas assez de données ({len(data)} barres). Essayez une période plus longue.")
    return data


def load_from_csv(file_path: str) -> pd.DataFrame:
    """Load OHLCV data from a local CSV file. The file must have a 'Close' column and a date index."""
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Fichier introuvable : {file_path}")

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Normalize column names
    df.columns = [c.strip().title() for c in df.columns]

    if "Close" not in df.columns:
        # Try to find a close-like column
        close_candidates = [c for c in df.columns if "close" in c.lower() or "fermeture" in c.lower() or "adj" in c.lower()]
        if close_candidates:
            df = df.rename(columns={close_candidates[0]: "Close"})
        else:
            raise ValueError(f"Colonne 'Close' introuvable. Colonnes disponibles : {list(df.columns)}")

    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]

    if len(df) < 50:
        raise ValueError(f"Pas assez de données ({len(df)} barres). Le fichier doit contenir au moins 50 lignes.")

    return df


def get_ticker_info(ticker: str) -> dict:
    ticker = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "name": info.get("longName", info.get("shortName", ticker)),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
            "type": info.get("quoteType", ""),
        }
    except Exception:
        return {"name": ticker, "currency": "", "exchange": "", "type": ""}


def _extract_col(data: pd.DataFrame, col: str) -> np.ndarray:
    """Extract a column from a potentially MultiIndex DataFrame."""
    if isinstance(data.columns, pd.MultiIndex):
        return data[col].iloc[:, 0].values.astype(float)
    return data[col].values.astype(float)


def get_close_prices(data: pd.DataFrame) -> np.ndarray:
    """Split-adjusted close prices (no dividend adjustment) — consistent with TradingView."""
    return _extract_col(data, "Close")


def get_dates(data: pd.DataFrame) -> pd.DatetimeIndex:
    return data.index
