from tardis_client import TardisClient, Channel
from tardis_dev import datasets
import nest_asyncio
nest_asyncio.apply()
import pandas as pd
import puffin.config as con
import puffin.tools as tools


def batch_download(exchange: str, symbol_list: list, start_date:str, end_date:str, data_types: list) -> None:
    datasets.download(
    exchange=exchange,
        data_types=data_types,
        from_date=start_date,
        to_date=end_date,
        symbols=symbol_list,
        api_key="TD.vMgiu14USvCy4rls.HekfEfNiFbEcqFU.JorCa2us4AzKJtP.4EuMcxQVTvts8Ok.p3QzOkFjmrTe7iD.tnR1",
    )

def generate_tob_data_from_gzip(symbol: str, input_path:str, output_path:str = None) -> pd.DataFrame:
    df = pd.read_csv(input_path, compression='gzip')
    new_colnames = ["instrument_id", "receive_timestamp_nanos", "exchange_timestamp_nanos", "bid_price", "bid_qty", "ask_price", "ask_qty"]

    #convert timestamp to nanos from microseconds
    df["timestamp"] = df["timestamp"]*1e3
    df["timestamp"] = df["timestamp"].astype(int)
    df["local_timestamp"] = df["local_timestamp"]*1e3
    df["local_timestamp"] = df["local_timestamp"].astype(int)
    df['instrument_id'] = tools.get_id_from_symbol(symbol)
    df = df[["instrument_id", "local_timestamp", "timestamp", "bid_price", "bid_amount", "ask_price", "ask_amount"]]
    df.columns = new_colnames
    return df

def generate_trade_data_from_gzip(symbol: str, input_path:str, output_path:str = None) -> pd.DataFrame:
    df = pd.read_csv(input_path, compression='gzip')
    new_colnames = ["instrument_id", "receive_timestamp_nanos", "exchange_timestamp_nanos", "price", "qty", "buyer_is_market_maker", "trade_id"]

    #convert timestamp to nanos from microseconds
    df["timestamp"] = df["timestamp"]*1e3
    df["timestamp"] = df["timestamp"].astype(int)
    df["local_timestamp"] = df["local_timestamp"]*1e3
    df["local_timestamp"] = df["local_timestamp"].astype(int)
    df['instrument_id'] = tools.get_id_from_symbol(symbol)
    df['buyer_is_market_maker'] = df['side'] == 'sell'
    df = df[["instrument_id", "local_timestamp", "timestamp", "price", "amount", "buyer_is_market_maker", "id"]]
    df.columns = new_colnames
    return df