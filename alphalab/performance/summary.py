import numpy as np
import pandas as pd

from .metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    max_drawdown,
    profit_loss_ratio,
    sharpe_ratio,
    win_rate,
)


def _single_performance_summary(
    returns: pd.Series,
    periods_per_year: int = 250,
    risk_free: float = 0.0,
) -> pd.Series:
    returns = returns.dropna()

    if returns.empty:
        return pd.Series(
            {
                "开始日期": pd.NaT,
                "结束日期": pd.NaT,
                "区间收益": np.nan,
                "年化收益率": np.nan,
                "年化波动率": np.nan,
                "夏普比率": np.nan,
                "最大回撤": np.nan,
                "卡玛比率": np.nan,
                "胜率": np.nan,
                "盈亏比": np.nan,
            }
        )

    return pd.Series(
        {
            "开始日期": returns.index.min().date(),
            "结束日期": returns.index.max().date(),
            "区间收益": (1 + returns).prod() - 1,
            "年化收益率": annualized_return(
                returns,
                periods_per_year=periods_per_year,
            ),
            "年化波动率": annualized_volatility(
                returns,
                periods_per_year=periods_per_year,
            ),
            "夏普比率": sharpe_ratio(
                returns,
                risk_free=risk_free,
                periods_per_year=periods_per_year,
            ),
            "最大回撤": max_drawdown(returns),
            "卡玛比率": calmar_ratio(
                returns,
                periods_per_year=periods_per_year,
            ),
            "胜率": win_rate(returns),
            "盈亏比": profit_loss_ratio(returns),
        }
    )


def performance_summary(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int = 250,
    risk_free: float = 0.0,
) -> pd.Series | pd.DataFrame:
    """

    """
    if isinstance(returns, pd.Series):
        return _single_performance_summary(
            returns,
            periods_per_year=periods_per_year,
            risk_free=risk_free,
        )

    if isinstance(returns, pd.DataFrame):
        summaries = {
            col: _single_performance_summary(
                returns[col],
                periods_per_year=periods_per_year,
                risk_free=risk_free,
            )
            for col in returns.columns
        }

        return pd.DataFrame(summaries).T

    raise TypeError("returns must be a pandas Series or DataFrame.")