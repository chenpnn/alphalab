import pandas as pd
from .result import BacktestResult
from tqdm import tqdm
import warnings

def run_backtest(
    df_wgt: pd.DataFrame,
    df_comp_price: pd.DataFrame,
    *,
    date_col: str = "date",
    code_col: str = "code",
    weight_col: str = "weight",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cost_rate: float = 0.0,
    normalize: bool = True,
):
    """
    运行回测并返回结果对象
    """
    if df_wgt.empty:
        raise ValueError("df_wgt is empty.")

    if df_comp_price.empty:
        raise ValueError("df_comp_price is empty.")
    
    required_cols = {date_col, code_col, weight_col}
    missing_cols = required_cols - set(df_wgt.columns)
    if missing_cols:
        raise ValueError(f"df_wgt missing columns: {sorted(missing_cols)}")

    weights_long = df_wgt[[date_col, code_col, weight_col]].copy()
    weights_long[date_col] = pd.to_datetime(weights_long[date_col])
    weights_long[code_col] = weights_long[code_col].astype(str)

    price = df_comp_price.copy()
    price.index = pd.to_datetime(price.index)
    price = price.sort_index()
    price.columns = price.columns.astype(str)

    start_date = weights_long[date_col].min() if start_date is None else pd.to_datetime(start_date)
    end_date = price.index.max() if end_date is None else pd.to_datetime(end_date)

    target_weights = (
        weights_long
        .pivot_table(
            index=date_col,
            columns=code_col,
            values=weight_col,
            aggfunc="sum",
        )
        .sort_index()
        .fillna(0.0)
    ).loc[start_date: end_date]

    common_codes = target_weights.columns.intersection(price.columns)
    missing_codes = target_weights.columns.difference(price.columns)
    if len(missing_codes) > 0:
        warnings.warn(f"Codes in target weights not found in price data: {sorted(missing_codes)}. These codes will be ignored.")

    target_weights = target_weights[common_codes]
    price = price[common_codes]

    if normalize:
        weight_sum = target_weights.abs().sum(axis=1)
        target_weights = target_weights.div(weight_sum.where(weight_sum != 0), axis=0)
        target_weights = target_weights.fillna(0.0)
    
    asset_returns = price.pct_change().loc[start_date: end_date].iloc[1:]

    rebal_dates = target_weights.index
    all_dates = asset_returns.index

    gross_returns = pd.Series(0.0, index=all_dates, name="gross_returns")
    actual_weights = pd.DataFrame(0.0, index=all_dates, columns=common_codes)

    for i, rebal_date in tqdm(enumerate(rebal_dates), total=len(rebal_dates), desc="回测"):
        next_rebal_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else end_date

        period_dates = all_dates[(all_dates > rebal_date) & (all_dates <= next_rebal_date)]

        if len(period_dates) == 0:
            continue

        weight = target_weights.loc[rebal_date]
        period_returns = asset_returns.loc[period_dates, common_codes].fillna(0.0)

        # Buy-and-hold portfolio return within the period.
        asset_nav = (1.0 + period_returns).cumprod()
        port_nav = asset_nav.mul(weight, axis=1).sum(axis=1)
        gross_returns.loc[period_dates] = pd.Series([1] + port_nav.tolist()).pct_change().dropna().values

        # 每日权重漂移
        drifted_value = asset_nav.mul(weight, axis=1)
        total_value = drifted_value.sum(axis=1)
        actual_weights.loc[period_dates] = drifted_value.div(
            total_value.where(total_value != 0),
            axis=0,
        ).fillna(0.0)
    
    # 换手率
    turnover = target_weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = target_weights.iloc[0].abs().sum()

    # 手续费
    costs_on_rebal_dates = turnover * cost_rate
    costs = pd.Series(0.0, index=all_dates, name="costs")

    for rebal_date, cost in costs_on_rebal_dates.items():
        eligible_dates = all_dates[all_dates > rebal_date]
        if len(eligible_dates) > 0:
            costs.loc[eligible_dates[0]] += cost
    
    net_returns = gross_returns - costs
    net_returns.name = "returns"

    nav = (1.0 + net_returns).cumprod()
    nav.name = "nav"

    return BacktestResult(
        returns=net_returns,
        nav=nav,
        weights=actual_weights,
        target_weights=target_weights,
        turnover=turnover,
        costs=costs,
        gross_returns=gross_returns,
    )

