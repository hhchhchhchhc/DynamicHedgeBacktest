import pandas as pd
from pandas import DataFrame

import SimpleMarketMaking.Clean.config


def get_id_from_symbol(symbol: str) -> int:
    config: DataFrame = SimpleMarketMaking.Clean.config.config
    symbol_id = int(config[config['symbol'] == symbol]['id'])
    return symbol_id


def get_sample_indices(timestamp_millis: pd.Series, bar_size_in_millis: int, start_on_the_dot: bool) -> [int]:
    t = bar_size_in_millis * ((timestamp_millis[0] // bar_size_in_millis) + 1) if start_on_the_dot else \
        timestamp_millis[0] + 1
    i = 0
    n = len(timestamp_millis)
    indices = []
    while i < n:
        while i < n and timestamp_millis[i] < t:
            i = i + 1
        if i < n:
            indices.append(i - 1)
            t = t + bar_size_in_millis
    return indices
