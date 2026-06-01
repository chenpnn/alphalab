import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 250):
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    total_return = (1 + returns).prod()
    years = len(returns) / periods_per_year
    ann_ret = total_return ** (1 / years) - 1
    return ann_ret


def annualized_volatility(returns: pd.Series, periods_per_year: int = 250):
    returns = returns.dropna()
    annual_vol = np.nan if returns.empty else returns.std() * np.sqrt(periods_per_year)
    return annual_vol


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 250,
):
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    
    annual_ret = annualized_return(returns, periods_per_year)
    annual_vol = annualized_volatility(returns, periods_per_year)
    sharpe_ratio = (annual_ret - risk_free) / annual_vol if annual_vol != 0 else np.nan
    return sharpe_ratio


def max_drawdown(returns: pd.Series):
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    nav = (1 + returns).cumprod()
    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    return drawdown.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 250):
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)

    if pd.isna(mdd) or mdd == 0:
        return np.nan

    return ann_ret / abs(mdd)


def win_rate(returns: pd.Series):
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return (returns > 0).mean()


def profit_loss_ratio(returns: pd.Series):
    returns = returns.dropna()
    gain = returns[returns > 0].mean()
    loss = returns[returns < 0].mean()

    if pd.isna(loss) or loss == 0:
        return np.nan

    return gain / abs(loss)