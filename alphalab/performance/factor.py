import numpy as np
import pandas as pd


def calc_forward_returns(
    df_comp_price: pd.DataFrame,
    forward_days: int = 1,
) -> pd.DataFrame:
    """
    Calculate forward returns from wide price data.

    forward_return[t] = price[t + forward_days] / price[t] - 1
    """
    if forward_days <= 0:
        raise ValueError("forward_days must be positive.")

    price = df_comp_price.copy()
    price.index = pd.to_datetime(price.index)
    price = price.sort_index()
    price.columns = price.columns.astype(str)

    return price.shift(-forward_days) / price - 1


def calc_forward_returns_long(
    df_comp_price: pd.DataFrame,
    forward_days: int = 1,
    *,
    date_col: str = "date",
    code_col: str = "code",
    return_col: str = "forward_return",
) -> pd.Series:
    """
    Calculate forward returns and return a Series indexed by date and code.
    """
    forward_returns = calc_forward_returns(
        df_comp_price,
        forward_days=forward_days,
    )

    forward_returns = forward_returns.stack(dropna=True)
    forward_returns.index.names = [date_col, code_col]
    forward_returns.name = return_col

    return forward_returns


def prepare_factor_data(
    df_fct: pd.DataFrame,
    df_comp_price: pd.DataFrame,
    *,
    forward_days: int = 1,
    date_col: str = "date",
    code_col: str = "code",
    factor_col: str = "factor",
) -> pd.DataFrame:
    """
    Join long-form factor data with future returns.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date and code, with columns:
        - factor
        - forward_return
    """
    required_cols = {date_col, code_col, factor_col}
    missing_cols = required_cols - set(df_fct.columns)
    if missing_cols:
        raise ValueError(f"df_fct missing columns: {sorted(missing_cols)}")

    factor = df_fct[[date_col, code_col, factor_col]].copy()
    factor[date_col] = pd.to_datetime(factor[date_col])
    factor[code_col] = factor[code_col].astype(str)

    factor = (
        factor.groupby([date_col, code_col], as_index=False)[factor_col]
        .mean()
        .set_index([date_col, code_col])
        .sort_index()
        .rename(columns={factor_col: "factor"})
    )

    forward_returns = calc_forward_returns_long(
        df_comp_price,
        forward_days=forward_days,
        date_col=date_col,
        code_col=code_col,
        return_col="forward_return",
    )

    data = factor.join(forward_returns, how="inner")
    return data.dropna(subset=["factor", "forward_return"])


def calc_ic(
    df_fct: pd.DataFrame,
    df_comp_price: pd.DataFrame,
    *,
    forward_days: int = 1,
    method: str = "pearson",
    min_periods: int = 30,
    date_col: str = "date",
    code_col: str = "code",
    factor_col: str = "factor",
) -> pd.Series:
    """
    Calculate cross-sectional IC by date.
    """
    data = prepare_factor_data(
        df_fct,
        df_comp_price,
        forward_days=forward_days,
        date_col=date_col,
        code_col=code_col,
        factor_col=factor_col,
    )

    if data.empty:
        return pd.Series(dtype=float, name=f"ic_{forward_days}d")

    grouped = data.groupby(level=date_col)
    valid_count = grouped.size()
    ic = grouped.apply(
        lambda x: x["factor"].corr(x["forward_return"], method=method)
    )
    ic = ic.where(valid_count >= min_periods)

    return ic.rename(f"ic_{forward_days}d")


def calc_rank_ic(
    df_fct: pd.DataFrame,
    df_comp_price: pd.DataFrame,
    *,
    forward_days: int = 1,
    min_periods: int = 30,
    date_col: str = "date",
    code_col: str = "code",
    factor_col: str = "factor",
) -> pd.Series:
    """
    Calculate cross-sectional RankIC by date.
    """
    return calc_ic(
        df_fct,
        df_comp_price,
        forward_days=forward_days,
        method="spearman",
        min_periods=min_periods,
        date_col=date_col,
        code_col=code_col,
        factor_col=factor_col,
    )


def quantile_returns(
    df_fct: pd.DataFrame,
    df_comp_price: pd.DataFrame,
    *,
    forward_days: int = 1,
    n_quantiles: int = 5,
    min_periods: int = 30,
    date_col: str = "date",
    code_col: str = "code",
    factor_col: str = "factor",
) -> pd.DataFrame:
    """
    Calculate average forward return by factor quantile for each date.
    """
    data = prepare_factor_data(
        df_fct,
        df_comp_price,
        forward_days=forward_days,
        date_col=date_col,
        code_col=code_col,
        factor_col=factor_col,
    )

    if data.empty:
        return pd.DataFrame()

    min_count = max(min_periods, n_quantiles)
    valid_count = data.groupby(level=date_col)["factor"].transform("count")
    data = data.loc[valid_count >= min_count].copy()

    if data.empty:
        return pd.DataFrame()

    data["quantile"] = data.groupby(level=date_col)["factor"].transform(
        lambda x: pd.qcut(
            x,
            q=n_quantiles,
            labels=False,
            duplicates="drop",
        )
    )

    qret = (
        data.dropna(subset=["quantile"])
        .groupby([pd.Grouper(level=date_col), "quantile"])["forward_return"]
        .mean()
        .unstack("quantile")
    )

    qret.columns = [f"Q{int(col) + 1}" for col in qret.columns]
    qret.index.name = date_col

    return qret


def long_short_returns(
    df_fct: pd.DataFrame,
    df_comp_price: pd.DataFrame,
    *,
    forward_days: int = 1,
    n_quantiles: int = 5,
    min_periods: int = 30,
    date_col: str = "date",
    code_col: str = "code",
    factor_col: str = "factor",
) -> pd.Series:
    """
    Calculate top-minus-bottom quantile forward returns.
    """
    qret = quantile_returns(
        df_fct,
        df_comp_price,
        forward_days=forward_days,
        n_quantiles=n_quantiles,
        min_periods=min_periods,
        date_col=date_col,
        code_col=code_col,
        factor_col=factor_col,
    )

    top_col = f"Q{n_quantiles}"
    if qret.empty or "Q1" not in qret or top_col not in qret:
        return pd.Series(dtype=float, name=f"long_short_{forward_days}d")

    return (qret[top_col] - qret["Q1"]).rename(f"long_short_{forward_days}d")


def factor_summary(
    df_fct: pd.DataFrame,
    df_comp_price: pd.DataFrame,
    *,
    forward_days: int = 1,
    n_quantiles: int = 5,
    min_periods: int = 30,
    date_col: str = "date",
    code_col: str = "code",
    factor_col: str = "factor",
) -> pd.Series:
    """
    Summarize IC, RankIC, and long-short factor performance.
    """
    ic = calc_ic(
        df_fct,
        df_comp_price,
        forward_days=forward_days,
        min_periods=min_periods,
        date_col=date_col,
        code_col=code_col,
        factor_col=factor_col,
    )
    rank_ic = calc_rank_ic(
        df_fct,
        df_comp_price,
        forward_days=forward_days,
        min_periods=min_periods,
        date_col=date_col,
        code_col=code_col,
        factor_col=factor_col,
    )
    long_short = long_short_returns(
        df_fct,
        df_comp_price,
        forward_days=forward_days,
        n_quantiles=n_quantiles,
        min_periods=min_periods,
        date_col=date_col,
        code_col=code_col,
        factor_col=factor_col,
    )

    ic_std = ic.std()
    rank_ic_std = rank_ic.std()
    long_short_std = long_short.std()

    return pd.Series(
        {
            "ic_mean": ic.mean(),
            "ic_std": ic_std,
            "ic_ir": ic.mean() / ic_std if ic_std != 0 else np.nan,
            "rank_ic_mean": rank_ic.mean(),
            "rank_ic_std": rank_ic_std,
            "rank_ic_ir": (
                rank_ic.mean() / rank_ic_std if rank_ic_std != 0 else np.nan
            ),
            "long_short_mean": long_short.mean(),
            "long_short_std": long_short_std,
            "long_short_ir": (
                long_short.mean() / long_short_std
                if long_short_std != 0
                else np.nan
            ),
            "long_short_win_rate": (long_short > 0).mean(),
        }
    )
