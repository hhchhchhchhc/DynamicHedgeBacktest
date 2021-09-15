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
    for i in range(1, len(data_frame)):
        if data_frame['bid_price'][i] != short_data_frame['bid_price'].values[-1] or data_frame['ask_price'][i] != \
                short_data_frame['ask_price'].values[-1]:
            short_data_frame = short_data_frame.append(data_frame.iloc[i])
    return short_data_frame


def format_top_of_book_pandas_data_frame(data_frame: pd.DataFrame) -> pd.DataFrame:
    # data_frame = data_frame[['timestamp_ms', 'bid_price', 'bid_qty', 'ask_price', 'ask_qty']]
    # data_frame['bid_qty'] = data_frame['bid_qty'] / minimum_size_increment
    # data_frame['bid_qty'] = data_frame['bid_qty'].astype(int)
    # data_frame['ask_qty'] = data_frame['ask_qty'] / minimum_size_increment
    # data_frame['ask_qty'] = data_frame['ask_qty'].astype(int)
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
    data_frame = data_frame[(data_frame['datetime'] >= start) & (data_frame['datetime'] < end)]
    return data_frame
