from dataclasses import dataclass
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series
    nav: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame
    target_weights: pd.DataFrame
    benchmark_returns: pd.Series | None = None
    excess_returns: pd.Series | None = None
    excess_nav: pd.Series | None = None