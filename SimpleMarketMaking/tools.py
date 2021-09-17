import datetime
import pyarrow.parquet as pq
import pandas as pd

pd.options.mode.chained_assignment = None
price_tick_size = 1e-4
minimum_size_increment = 0.1


def create_pandas_data_frame_from_parquet_file(path: str) -> pd.DataFrame:
    data_frame = pq.read_table(path).to_pandas()
    return data_frame


def drop_successive_duplicates(data_frame: pd.DataFrame) -> pd.DataFrame:
    short_data_frame = pd.DataFrame(columns=data_frame.columns)
    short_data_frame['timestamp_ms'] = short_data_frame['timestamp_ms'].astype(int)
    short_data_frame['bid_price'] = short_data_frame['bid_price'].astype(int)
    short_data_frame['ask_price'] = short_data_frame['ask_price'].astype(int)
    short_data_frame = short_data_frame.append(data_frame.iloc[0])
    progress = 10000
    for i in range(1, len(data_frame)):
        if i > progress:
            print(i / len(data_frame))
            progress = progress + 10000
        if data_frame.iloc[i]['bid_price'] != short_data_frame['bid_price'].values[-1] or \
                data_frame.iloc[i]['ask_price'] != short_data_frame['ask_price'].values[-1]:
            short_data_frame = short_data_frame.append(data_frame.iloc[i])
    return short_data_frame


def format_top_of_book_pandas_data_frame(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame = data_frame[['timestamp_ms', 'bid_price', 'ask_price']]
    data_frame['bid_price'] = data_frame['bid_price'] / price_tick_size
    data_frame['bid_price'] = data_frame['bid_price'].astype(int)
    data_frame['ask_price'] = data_frame['ask_price'] / price_tick_size
    data_frame['ask_price'] = data_frame['ask_price'].astype(int)
    data_frame = drop_successive_duplicates(data_frame)
    data_frame = data_frame.rename(
        columns={'timestamp_ms': 'millisecondsSinceEpoch', 'bid_price': 'bidPrice', 'bid_qty': 'bidSize',
                 'ask_price': 'askPrice', 'ask_qty': 'askSize'})
    data_frame['datetime'] = data_frame['millisecondsSinceEpoch'].apply(
        lambda milliseconds_since_epoch: datetime.datetime.utcfromtimestamp(milliseconds_since_epoch / 1000))
    return data_frame


def extract_data_frame_interval(data_frame: pd.DataFrame, start: datetime.datetime, end: datetime.datetime):
    timestamp_ms_start = start.timestamp() * 1000
    timestamp_ms_end = end.timestamp() * 1000
    data_frame = data_frame[(data_frame['timestamp_ms'] >= timestamp_ms_start) and
                            (data_frame['timestamp_ms'] < timestamp_ms_end)]
    return data_frame
