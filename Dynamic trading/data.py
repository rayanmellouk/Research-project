from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_TICKERS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",
    "Natural Gas": "NG=F",
}
DEFAULT_MULTIPLIERS = {
    "GC=F": 100,
    "SI=F": 5000,
    "CL=F": 1000,
    "NG=F": 10000,
}
START_DATE = "2008-01-01"
END_DATE = "2019-01-01"


def download_market_data(
    tickers: Iterable[str], start: str, end: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    closes = data["Close"].sort_index()
    volumes = data["Volume"].sort_index()
    return closes, volumes


def standardized_rolling_mean(dP: pd.DataFrame, window: int) -> pd.DataFrame:
    rolling = dP.rolling(window=window)
    signal = (rolling.mean() / rolling.std(ddof=0)).shift(1)
    return signal.dropna(how="all")


def build_predictors(dP: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "f_5d": standardized_rolling_mean(dP, 5),
        "f_1y": standardized_rolling_mean(dP, 252),
        "f_5y": standardized_rolling_mean(dP, 1260),
    }


def fit_return_model(
    dP: pd.DataFrame, predictors: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    target = dP.shift(-1)
    stacked = {name: predictors[name].stack() for name in ("f_5d", "f_1y", "f_5y")}
    df = pd.DataFrame({"r": target.stack(), **stacked}).dropna()
    y = df.pop("r").values
    X = np.column_stack([np.ones(len(df)), df.values])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return {
        "const": float(beta[0]),
        "f_5d": float(beta[1]),
        "f_1y": float(beta[2]),
        "f_5y": float(beta[3]),
    }
