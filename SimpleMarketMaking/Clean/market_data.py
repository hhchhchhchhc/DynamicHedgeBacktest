import pandas as pd
import SimpleMarketMaking.Clean.tools
import SimpleMarketMaking.Clean.config
import pyarrow.parquet as pq


class MarketData:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.top_of_book_raw = self._get_top_of_book_sample()
        self.top_of_book_formatted = self._format_top_of_book()
        self.trades_raw = self._get_trades_sample()

    def _get_top_of_book_sample(self) -> pd.DataFrame:
        symbol_id: int = SimpleMarketMaking.Clean.tools.get_id_from_symbol(self.symbol)
        symbol_id_string = str(symbol_id)
        data = pq.read_table(SimpleMarketMaking.Clean.config.source_directory +
                             '20210901_' + symbol_id_string + '_tob.parquet').to_pandas()
        data = data.head(10000)
        return data

    def _format_top_of_book(self) -> pd.DataFrame:
        formatted_data = pd.DataFrame()
        formatted_data['timestamp_millis'] = self.top_of_book_raw['exchange_timestamp_nanos'].div(1000000)
        formatted_data['timestamp_millis'] = formatted_data['timestamp_millis'].astype('int64')
        config: pd.DataFrame = SimpleMarketMaking.Clean.config.config
        tick_size = float(config[config['symbol'] == self.symbol]['tick_size'])
        formatted_data['bid_price'] = self.top_of_book_raw['bid_price'].div(tick_size)
        formatted_data['bid_price'] = formatted_data['bid_price'].astype('int64')
        formatted_data['ask_price'] = self.top_of_book_raw['ask_price'].div(tick_size)
        formatted_data['ask_price'] = formatted_data['ask_price'].astype('int64')
        step_size = float(config[config['symbol'] == self.symbol]['step_size'])
        formatted_data['bid_size'] = self.top_of_book_raw['bid_qty'].div(step_size)
        formatted_data['bid_size'] = formatted_data['bid_size'].astype('int64')
        formatted_data['ask_size'] = self.top_of_book_raw['ask_qty'].div(step_size)
        formatted_data['ask_size'] = formatted_data['ask_size'].astype('int64')
        return formatted_data

    def time_sampled_top_of_book(self, millis: int):
        indices = SimpleMarketMaking.Clean.tools.get_sample_indices(self.top_of_book_formatted['timestamp_millis'],
                                                                    millis, False)
        return self.top_of_book_formatted.iloc[indices]

    def _get_trades_sample(self) -> pd.DataFrame:
        symbol_id: int = SimpleMarketMaking.Clean.tools.get_id_from_symbol(self.symbol)
        symbol_id_string = str(symbol_id)
        data = pq.read_table(SimpleMarketMaking.Clean.config.source_directory +
                             '20210901_' + symbol_id_string + '_trade.parquet').to_pandas()
        data = data[data['exchange_timestamp_nanos'].between(
            self.top_of_book_formatted['timestamp_millis'].iloc[0] * 1000000,
            self.top_of_book_formatted['timestamp_millis'].iloc[-1] * 1000000)]
        return data
