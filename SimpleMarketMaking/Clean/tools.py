import pandas as pd
from pandas import DataFrame

import SimpleMarketMaking.Clean.config


def get_id_from_symbol(symbol: str) -> int:
    config: DataFrame = SimpleMarketMaking.Clean.config.config
    symbol_id = int(config[config['symbol'] == symbol]['id'])
    return symbol_id


def _get_sampled_indices(series: pd.Series, bar_size: int, start_on_round_multiple: bool) -> pd.DataFrame:
    t = series.iloc[0] + 1
    if start_on_round_multiple:
        t = bar_size * ((series.iloc[0] // bar_size) + 1)
    i = 0
    n = len(series)
    values = []
    indices = []
    while i < n:
        while i < n and series.iloc[i] < t:
            i = i + 1
        if i < n:
            indices.append(series.index[i - 1])
            values.append(t)
            t = t + bar_size
    data_frame = pd.DataFrame()
    data_frame['value'] = values
    data_frame['index'] = indices
    return data_frame


def get_time_sampled_indices(timestamp_millis: pd.Series, bar_size_in_millis: int,
                             start_on_the_dot: bool) -> pd.DataFrame:
    indices = _get_sampled_indices(timestamp_millis, bar_size_in_millis, start_on_the_dot)
    return indices


def get_volume_sampled_indices(sizes: pd.Series, volume: int) -> pd.DataFrame:
    cumulative_sizes = sizes.cumsum()
    indices = _get_sampled_indices(cumulative_sizes, volume, False)
    return indices


def get_dollar_sampled_indices(prices: pd.Series, sizes: pd.Series, dollar):
    cumulative_dollars = prices.multiply(sizes).cumsum()
    indices = _get_sampled_indices(cumulative_dollars, dollar, False)
    return indices
