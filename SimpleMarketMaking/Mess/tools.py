import datetime
import math

import boto3 as boto3
import pyarrow.parquet as pq
import pandas as pd
import os

from botocore.exceptions import ClientError

pd.options.mode.chained_assignment = None
price_tick_size = 1e-5
minimum_size_increment = 1


def format_top_of_book_pandas_data_frame(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame['bid_price'] = data_frame['bid_price'] / price_tick_size
    data_frame['bid_price'] = data_frame['bid_price'].astype(int)
    data_frame['ask_price'] = data_frame['ask_price'] / price_tick_size
    data_frame['ask_price'] = data_frame['ask_price'].astype(int)
    data_frame['bid_qty'] = data_frame['bid_qty'] / minimum_size_increment
    data_frame['bid_qty'] = data_frame['bid_qty'].astype(int)
    data_frame['ask_qty'] = data_frame['ask_qty'] / minimum_size_increment
    data_frame['ask_qty'] = data_frame['ask_qty'].astype(int)
    data_frame = data_frame[['timestamp_ms', 'bid_price', 'bid_qty', 'ask_price', 'ask_qty']]
    data_frame['datetime'] = data_frame['timestamp_ms'].apply(
        lambda milliseconds_since_epoch: datetime.datetime.utcfromtimestamp(milliseconds_since_epoch / 1000))
    data_frame = data_frame.rename(
        columns={'timestamp_ms': 'milliseconds_since_epoch', 'bid_qty': 'bid_size', 'ask_qty': 'ask_size'})
    data_frame = data_frame[['milliseconds_since_epoch', 'datetime', 'bid_price', 'bid_size', 'ask_price', 'ask_size']]
    return data_frame


def write_data_to_file(my_date: datetime.date, directory: str) -> None:
    date_string = my_date.strftime('%Y%m%d')
    print('Writing DOGEUSDT date ' + date_string + '.')
    file = date_string + '_4656_tob.parquet'
    path = directory + 'parquet/' + file
    data_frame = create_pandas_data_frame_from_parquet_file(path)
    data_frame = format_top_of_book_pandas_data_frame(data_frame)
    data_frame.to_csv(directory + 'market_data_top_of_book_dogeusdt_' + date_string + '.csv', index=False)
    return None


def create_pandas_data_frame_from_parquet_file(path: str) -> pd.DataFrame:
    data_frame = pq.read_table(path).to_pandas()
    return data_frame


def drop_successive_duplicates(data_frame: pd.DataFrame) -> pd.DataFrame:
    short_data_frame = pd.DataFrame(columns=data_frame.columns)
    short_data_frame['timestamp_ms'] = short_data_frame['timestamp_ms'].astype(int)
    short_data_frame['bid_price'] = short_data_frame['bid_price'].astype(int)
    short_data_frame['ask_price'] = short_data_frame['ask_price'].astype(int)
    short_data_frame = short_data_frame.append(data_frame.iloc[0])
    progress_pct = 0
    number_of_rows = len(data_frame)
    n = len(data_frame)
    for i in range(1, number_of_rows):
        if math.floor(100 * i / n) > progress_pct:
            print(str('{:d}'.format(progress_pct)) + '% ', end='')
            progress_pct = progress_pct + 10
        if data_frame.iloc[i]['bid_price'] != short_data_frame['bid_price'].values[-1] or \
                data_frame.iloc[i]['ask_price'] != short_data_frame['ask_price'].values[-1]:
            short_data_frame = short_data_frame.append(data_frame.iloc[i])
    return short_data_frame


def format_top_of_book_price_change_only_pandas_data_frame(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame = data_frame[['timestamp_ms', 'bid_price', 'ask_price']]
    data_frame['bid_price'] = data_frame['bid_price'] / price_tick_size
    data_frame['bid_price'] = data_frame['bid_price'].astype(int)
    data_frame['ask_price'] = data_frame['ask_price'] / price_tick_size
    data_frame['ask_price'] = data_frame['ask_price'].astype(int)
    data_frame = drop_successive_duplicates(data_frame)
    data_frame = data_frame.rename(
        columns={'timestamp_ms': 'milliseconds_since_epoch', 'bid_qty': 'bid_size', 'ask_qty': 'ask_size'})
    data_frame['datetime'] = data_frame['milliseconds_since_epoch'].apply(
        lambda milliseconds_since_epoch: datetime.datetime.utcfromtimestamp(milliseconds_since_epoch / 1000))
    return data_frame


def extract_data_frame_interval(data_frame: pd.DataFrame, start: datetime.datetime, end: datetime.datetime):
    timestamp_ms_start = start.timestamp() * 1000
    timestamp_ms_end = end.timestamp() * 1000
    data_frame = data_frame[(data_frame['timestamp_ms'] >= timestamp_ms_start) and
                            (data_frame['timestamp_ms'] < timestamp_ms_end)]
    return data_frame


def s3_download_file(local_filename, bucket_name, remote_filename):
    print("DOWNLOAD FILE FROM S3: s3://" + bucket_name + '/' + remote_filename + ' -> ' + local_filename)
    s3_cl = boto3.client('s3', endpoint_url=os.getenv('BOTO3_S3_ENDPOINT_URL'))
    try:
        with open(local_filename, 'wb') as f:
            s3_cl.download_fileobj(bucket_name, remote_filename, f)
    except ClientError as e:
        raise ValueError("Error occurred in downloadFileFromS3 method: " + str(e))


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
