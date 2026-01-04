from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class SimulationResult:
    markowitz_positions: pd.DataFrame
    static_positions: pd.DataFrame
    dynamic_positions: pd.DataFrame
    brut_pnl_markowitz: pd.Series
    brut_pnl_static: pd.Series
    brut_pnl_dynamic: pd.Series
    net_pnl_markowitz: pd.Series
    net_pnl_static: pd.Series
    net_pnl_dynamic: pd.Series
    brut_sharpe_markowitz: float
    brut_sharpe_static: float
    brut_sharpe_dynamic: float
    net_sharpe_markowitz: float
    net_sharpe_static: float
    net_sharpe_dynamic: float


def _download_market_data(
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


def _standardized_rolling_mean(dP: pd.DataFrame, window: int) -> pd.DataFrame:
    rolling = dP.rolling(window=window)
    signal = (rolling.mean() / rolling.std(ddof=0)).shift(1)
    return signal.dropna(how="all")


def _build_predictors(dP: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "f_5d": _standardized_rolling_mean(dP, 5),
        "f_1y": _standardized_rolling_mean(dP, 252),
        "f_5y": _standardized_rolling_mean(dP, 1260),
    }


def _fit_return_model(
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


def _expected_returns(
    params: Dict[str, float], predictors: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    mu = pd.DataFrame(
        params["const"],
        index=predictors["f_5d"].index,
        columns=predictors["f_5d"].columns,
    )
    for key in ("f_5d", "f_1y", "f_5y"):
        mu += params[key] * predictors[key]
    return mu.dropna()


def _estimate_covariance(dP: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vol = dP.std(ddof=1)
    corr = dP.corr()
    corr_shrunk = 0.5 * corr + 0.5 * pd.DataFrame(
        np.eye(len(corr)), index=corr.index, columns=corr.columns
    )
    Sigma = pd.DataFrame(
        np.outer(vol, vol) * corr_shrunk.values, index=vol.index, columns=vol.index
    )
    Sigma_inv = pd.DataFrame(
        np.linalg.inv(Sigma.values), index=vol.index, columns=vol.index
    )
    return Sigma, Sigma_inv


def _position_pnl(position: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    common_idx = returns.index.intersection(position.index)
    pos = position.loc[common_idx]
    ret = returns.loc[common_idx]
    pnl = (pos.shift(1) * ret).sum(axis=1)
    return pnl.dropna()


def _calculate_net_pnl(
    positions: pd.DataFrame,
    price_changes: pd.DataFrame,
    lambda_val: float,
    sigma_matrix: pd.DataFrame,
) -> pd.Series:
    gross_pnl = (positions.shift(1) * price_changes).sum(axis=1)

    trades = positions.diff().fillna(0)

    quad_form = np.einsum(
        "ij, jk, ik -> i", trades.values, sigma_matrix.values, trades.values
    )

    costs = 0.5 * lambda_val * quad_form

    net_pnl = gross_pnl - pd.Series(costs, index=positions.index)

    return net_pnl.dropna()


def _estimate_phi(predictor: pd.Series) -> float:
    """Estimate AR(1) mean reversion speed: dX = -phi * X"""
    # The paper specifies: delta_f_{t+1} = -phi * f_t + error
    df = pd.DataFrame({"f": predictor, "df": predictor.diff().shift(-1)}).dropna()
    # Simple OLS without intercept (since signals are standardized to mean 0)
    phi = -np.linalg.lstsq(df[["f"]], df["df"], rcond=None)[0][0]
    return float(phi)


def _annualized_sharpe(pnl: pd.Series, periods: int = 260) -> float:
    daily_mean = pnl.mean()
    daily_std = pnl.std(ddof=1)
    if daily_std == 0:
        return float("nan")
    return float(daily_mean / daily_std * np.sqrt(periods))


def simulate_quadratic_trajectory(
    price_changes: pd.DataFrame = None,
    save_data: bool = False,
    tickers: Dict[str, str] = DEFAULT_TICKERS,
    multipliers: Dict[str, int] = DEFAULT_MULTIPLIERS,
    start: str = START_DATE,
    end: str = END_DATE,
    estimate_phi: bool = True,
) -> SimulationResult:

    if price_changes is None:
        closes, volumes = _download_market_data(tickers.values(), start, end)
        price_df = (
            closes.mul(pd.Series(multipliers), axis=1).dropna(how="all").dropna(axis=1)
        )
        dP = price_df.diff().dropna()
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if save_data:
            dP.to_csv(DATA_DIR / "price_changes_GC_SI_CL_NG.csv")
    else:
        raise ValueError(
            "Passing price_changes directly is not supported yet: "
            "we also need price levels and volumes for λ calibration."
        )

    predictors = _build_predictors(dP)
    common_idx = predictors["f_5d"].index
    for df in predictors.values():
        common_idx = common_idx.intersection(df.index)
    dP = dP.loc[common_idx]
    price_df = price_df.loc[common_idx]
    volumes = volumes.loc[common_idx]
    predictors = {k: v.loc[common_idx] for k, v in predictors.items()}

    params = _fit_return_model(dP, predictors)
    Bf = _expected_returns(params, predictors)
    idx = dP.index.intersection(Bf.index)
    dP = dP.loc[idx]
    price_df = price_df.loc[idx]
    volumes = volumes.loc[idx]
    predictors = {k: v.loc[idx] for k, v in predictors.items()}
    Bf = Bf.loc[idx]

    Sigma, Sigma_inv = _estimate_covariance(dP)
    gamma = 1e-7
    rho = 1 - np.exp(-0.02 / 260)

    sigma = dP.std(ddof=1)
    mean_price = price_df.mean()
    mean_volume = volumes.mean()
    f = 0.0159
    impact = 0.001
    lambda_asset = 2 * (impact * mean_price) / (f * mean_volume * (sigma**2))
    lambda_high = float(lambda_asset.max())

    dates = dP.index
    cols = dP.columns

    markowitz_pos = pd.DataFrame(
        (Bf.values @ Sigma_inv.values.T) / gamma, index=dates, columns=cols
    )

    alpha_static = gamma / (gamma + lambda_high)

    static_pos = pd.DataFrame(index=dates, columns=cols, dtype=float)
    prev_static = np.zeros(len(cols))
    for t in dates:
        target = markowitz_pos.loc[t].values
        prev_static = prev_static + alpha_static * (target - prev_static)
        static_pos.loc[t] = prev_static

    rho_bar = 1 - rho
    a_num = -(gamma * rho_bar + lambda_high * rho)
    disc = (gamma * rho_bar + lambda_high * rho) ** 2 + 4 * gamma * lambda_high * (
        rho_bar**2
    )
    a = (a_num + np.sqrt(disc)) / (2 * rho_bar)
    trade_speed = a / lambda_high

    phi = (
        {"f_5d": 0.2519, "f_1y": 0.0034, "f_5y": 0.0010}
        if not estimate_phi
        else {
            name: _estimate_phi(predictors[name].stack())
            for name in ("f_5d", "f_1y", "f_5y")
        }
    )
    r_dyn = pd.DataFrame(params["const"], index=dates, columns=cols)
    for name in ("f_5d", "f_1y", "f_5y"):
        beta_dyn = params[name] / (1 + phi[name] * a / gamma)
        r_dyn += beta_dyn * predictors[name]

    aim_dyn = pd.DataFrame(
        (r_dyn.values @ Sigma_inv.values.T) / gamma, index=dates, columns=cols
    )
    dynamic_pos = pd.DataFrame(index=dates, columns=cols, dtype=float)
    prev_dynamic = np.zeros(len(cols))
    for t in dates:
        target = aim_dyn.loc[t].values
        prev_dynamic = (1 - trade_speed) * prev_dynamic + trade_speed * target
        dynamic_pos.loc[t] = prev_dynamic

    brut_pnl_markowitz = _position_pnl(markowitz_pos, dP)
    brut_pnl_static = _position_pnl(static_pos, dP)
    brut_pnl_dynamic = _position_pnl(dynamic_pos, dP)

    brut_sharpe_m = _annualized_sharpe(brut_pnl_markowitz)
    brut_sharpe_s = _annualized_sharpe(brut_pnl_static)
    brut_sharpe_d = _annualized_sharpe(brut_pnl_dynamic)

    net_pnl_markowitz = _calculate_net_pnl(markowitz_pos, dP, lambda_high, Sigma)
    net_pnl_static = _calculate_net_pnl(static_pos, dP, lambda_high, Sigma)
    net_pnl_dynamic = _calculate_net_pnl(dynamic_pos, dP, lambda_high, Sigma)

    net_sharpe_m = _annualized_sharpe(net_pnl_markowitz)
    net_sharpe_s = _annualized_sharpe(net_pnl_static)
    net_sharpe_d = _annualized_sharpe(net_pnl_dynamic)

    return SimulationResult(
        markowitz_positions=markowitz_pos,
        static_positions=static_pos,
        dynamic_positions=dynamic_pos,
        brut_pnl_markowitz=brut_pnl_markowitz,
        brut_pnl_static=brut_pnl_static,
        brut_pnl_dynamic=brut_pnl_dynamic,
        net_pnl_markowitz=net_pnl_markowitz,
        net_pnl_static=net_pnl_static,
        net_pnl_dynamic=net_pnl_dynamic,
        brut_sharpe_dynamic=brut_sharpe_d,
        brut_sharpe_markowitz=brut_sharpe_m,
        brut_sharpe_static=brut_sharpe_s,
        net_sharpe_dynamic=net_sharpe_d,
        net_sharpe_markowitz=net_sharpe_m,
        net_sharpe_static=net_sharpe_s,
    )


if __name__ == "__main__":
    result = simulate_quadratic_trajectory(estimate_phi=True)
    print("Annualized Sharpe ratios:")
    print(f"Markowitz : {result.net_sharpe_markowitz:.4f}")
    print(f"Static    : {result.net_sharpe_static:.4f}")
    print(f"Dynamic   : {result.net_sharpe_dynamic:.4f}")
    print(f"Brut Markowitz : {result.brut_sharpe_markowitz:.4f}")
    print(f"Brut Static    : {result.brut_sharpe_static:.4f}")
    print(f"Brut Dynamic   : {result.brut_sharpe_dynamic:.4f}")
    # print(result.markowitz_positions.head())
    # print(result.static_positions.head())

    # print(result.dynamic_positions.head())
    sec_res = simulate_quadratic_trajectory(estimate_phi=False)
    print("\nWith fixed phi values:")
    print(f"Markowitz : {sec_res.net_sharpe_markowitz:.4f}")
    print(f"Static    : {sec_res.net_sharpe_static:.4f}")
    print(f"Dynamic   : {sec_res.net_sharpe_dynamic:.4f}")
    print(f"Brut Markowitz : {sec_res.brut_sharpe_markowitz:.4f}")
    print(f"Brut Static    : {sec_res.brut_sharpe_static:.4f}")
    print(f"Brut Dynamic   : {sec_res.brut_sharpe_dynamic:.4f}")
