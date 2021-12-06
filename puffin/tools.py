import numpy as np
import pandas as pd
from pandas import DataFrame
import puffin.config as con
import boto3
import os


def get_id_from_symbol(symbol: str) -> int:
    config: DataFrame = con.config
    symbol_id = int(config[config['symbol'] == symbol]['id'])
    return symbol_id


def get_tick_size_from_symbol(symbol):
    config: DataFrame = con.config
    tick_size = float(config[config['symbol'] == symbol]['tick_size'])
    return tick_size


def get_step_size_from_symbol(symbol):
    config: DataFrame = con.config
    step_size = float(config[config['symbol'] == symbol]['step_size'])
    return step_size


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


def _get_imbalance_sampled_indices(b: pd.Series) -> pd.DataFrame:
    t = 10
    theta = 0
    ema_t = t
    ema_imbalance = np.abs(b.head(t).mean())
    threshold = ema_t * ema_imbalance
    ema_memory_weight: float = 0.5
    indices = []
    values = []
    bar_start_index = t
    i = t + 1
    n = len(b)
    while i < n:
        theta = theta + b.iloc[i]
        if np.abs(theta) >= threshold:
            t = i - bar_start_index
            average_imbalance = np.abs(b[bar_start_index + 1:i + 1].mean())
            ema_t = (ema_memory_weight * ema_t) + ((1 - ema_memory_weight) * t)
            ema_imbalance = (ema_memory_weight * ema_imbalance) + \
                            ((1 - ema_memory_weight) * average_imbalance)
            threshold = ema_t * ema_imbalance
            bar_start_index = i + 1
            indices.append(b.index[i])
            values.append(t)
            theta = 0
        i = i + 1
    data_frame = pd.DataFrame()
    data_frame['value'] = values
    data_frame['index'] = indices
    return data_frame


def get_trade_side_imbalance_sampled_indices(givens: pd.Series) -> pd.DataFrame:
    b = givens.apply(lambda g: -1 if g else 1)
    data_frame = _get_imbalance_sampled_indices(b)
    return data_frame


def get_tick_imbalance_sampled_indices(prices: pd.Series) -> pd.DataFrame:
    b = _get_ticks(prices)
    data_frame = _get_imbalance_sampled_indices(b)
    return data_frame


def _get_ticks(prices: pd.Series) -> pd.Series:
    b: list[float] = [np.nan]
    for i in range(1, len(prices)):
        delta_price = prices.iloc[i] - prices.iloc[i - 1]
        new_b = b[-1]
        if delta_price != 0:
            new_b = abs(delta_price) / delta_price
        b.append(new_b)
    return pd.Series(index=prices.index, data=b)


def get_volume_tick_imbalance_sampled_indices(prices: pd.Series, sizes: pd.Series) -> pd.DataFrame:
    b = _get_ticks(prices)
    b = b.multiply(sizes)
    data_frame = _get_imbalance_sampled_indices(b)
    return data_frame


def get_volume_trade_side_imbalance_sampled_indices(trades_formatted: pd.DataFrame) -> pd.DataFrame:
    b = trades_formatted['given'].apply(lambda g: -1 if g else 1)
    b = b.multiply(trades_formatted['size'])
    data_frame = _get_imbalance_sampled_indices(b)
    return data_frame


def get_dollar_tick_imbalance_sampled_indices(prices: pd.Series, sizes: pd.Series) -> pd.DataFrame:
    b = _get_ticks(prices)
    b = b.multiply(sizes)
    b = b.multiply(prices)
    data_frame = _get_imbalance_sampled_indices(b)
    return data_frame


def get_dollar_trade_side_imbalance_sampled_indices(trades_formatted: pd.DataFrame) -> pd.DataFrame:
    b = trades_formatted['given'].apply(lambda g: -1 if g else 1)
    b = b.multiply(trades_formatted['size'])
    b = b.multiply(trades_formatted['price'])
    data_frame = _get_imbalance_sampled_indices(b)
    return data_frame


def _get_runs_sampled_indices(b: pd.Series) -> pd.DataFrame:
    t = 10
    theta = 0
    ema_t = t
    first_bar = b.head(t).dropna()
    first_bar_positive = first_bar.where(first_bar > 0)
    first_bar_negative = first_bar.where(first_bar < 0)
    ema_run = max((first_bar_positive.count() / len(first_bar_positive)) * first_bar_positive.mean(),
                  (first_bar_negative.count() / len(first_bar_negative)) * -first_bar_negative.mean())
    threshold = ema_t * ema_run
    ema_memory_weight = 0.5
    indices = []
    values = []
    bar_start_index = t
    i = t + 1
    n = len(b)
    while i < n:
        theta = theta + b.iloc[i]
        if np.abs(theta) >= threshold:
            t = i - bar_start_index
            this_bar = b[bar_start_index + 1:i + 1]
            this_bar_positive = this_bar.where(this_bar > 0)
            this_bar_negative = this_bar.where(this_bar < 0)
            run = max((this_bar_positive.count() / len(this_bar_positive)) * this_bar_positive.mean(),
                      (this_bar_negative.count() / len(this_bar_negative)) * -this_bar_negative.mean())
            ema_t = (ema_memory_weight * ema_t) + ((1 - ema_memory_weight) * t)
            ema_run = (ema_memory_weight * ema_run) + ((1 - ema_memory_weight) * run)
            threshold = ema_t * ema_run
            bar_start_index = i + 1
            indices.append(b.index[i])
            values.append(t)
            theta = 0
        i = i + 1
    data_frame = pd.DataFrame()
    data_frame['value'] = values
    data_frame['index'] = indices
    return data_frame


def get_tick_runs_sampled_indices(prices: pd.Series) -> pd.DataFrame:
    b = _get_ticks(prices)
    data_frame = _get_runs_sampled_indices(b)
    return data_frame


def get_trade_side_runs_sampled_indices(givens: pd.Series) -> pd.DataFrame:
    b = givens.apply(lambda g: -1 if g else 1)
    data_frame = _get_runs_sampled_indices(b)
    return data_frame


def get_volume_tick_runs_sampled_indices(prices: pd.Series, sizes: pd.Series) -> pd.DataFrame:
    b = _get_ticks(prices)
    b = b.multiply(sizes)
    data_frame = _get_runs_sampled_indices(b)
    return data_frame


def get_volume_trade_side_runs_sampled_indices(trades_formatted: pd.DataFrame) -> pd.DataFrame:
    b = trades_formatted['given'].apply(lambda g: -1 if g else 1)
    b = b.multiply(trades_formatted['size'])
    data_frame = _get_runs_sampled_indices(b)
    return data_frame


def get_dollar_tick_runs_sampled_indices(prices: pd.Series, sizes: pd.Series) -> pd.DataFrame:
    b = _get_ticks(prices)
    b = b.multiply(sizes)
    b = b.multiply(prices)
    data_frame = _get_runs_sampled_indices(b)
    return data_frame


def get_dollar_trade_side_runs_sampled_indices(trades_formatted: pd.DataFrame) -> pd.DataFrame:
    b = trades_formatted['given'].apply(lambda g: -1 if g else 1)
    b = b.multiply(trades_formatted['size'])
    b = b.multiply(trades_formatted['price'])
    data_frame = _get_runs_sampled_indices(b)
    return data_frame


def s3_download_file(local_filename, bucket_name, remote_filename):
    print("DOWNLOAD FILE FROM S3: s3://" + bucket_name + '/' + remote_filename + ' -> ' + local_filename)
    s3_cl = boto3.client('s3', endpoint_url=os.getenv('BOTO3_S3_ENDPOINT_URL'))
    with open(local_filename, 'wb') as f:
        s3_cl.download_fileobj(bucket_name, remote_filename, f)


def s3_upload_file(local_filename, bucket_name, remote_filename):
    print("UPLOAD FILE TO S3: " + local_filename + ' -> ' + 's3://' + bucket_name + '/' + remote_filename)
    s3_cl = boto3.client('s3', endpoint_url=os.getenv('BOTO3_S3_ENDPOINT_URL'))
    with open(local_filename, 'rb') as f:
        s3_cl.upload_fileobj(f, bucket_name, remote_filename)


def s3_list(bucket_name, prefix_path):
    print("LIST FROM S3: s3://" + bucket_name + "/" + prefix_path)
    s3_cl = boto3.client('s3', endpoint_url=os.getenv('BOTO3_S3_ENDPOINT_URL'))
    keys = []
    # paginated shite
    paginator = s3_cl.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix_path)
    for page in pages:
        for obj in page['Contents']:
            keys.append({'name': obj['Key'], 'size': obj['Size']})
    return keys


def s3_download_directory(local_directory, bucket_name, remote_directory):
    print("DOWNLOAD DIRECTORY FROM S3: s3://" + bucket_name + '/' + remote_directory + ' -> ' + local_directory)
    remote_files = s3_list(bucket_name, remote_directory)
    prefix = remote_directory + '/'
    for rf in remote_files:
        remote_file = rf['name']
        size = rf['size']
        if size == 0:
            continue
        stripped_path = remote_file[len(prefix):]
        local_file = os.path.join(local_directory, stripped_path)
        local_file_dir = os.path.dirname(local_file)
        os.makedirs(local_file_dir, exist_ok=True)
        s3_download_file(local_file, bucket_name, remote_file)
