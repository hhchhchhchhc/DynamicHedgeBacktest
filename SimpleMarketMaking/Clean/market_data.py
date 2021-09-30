import numpy as np
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
        self.trades_formatted = self._format_trades()

    def _get_top_of_book_sample(self) -> pd.DataFrame:
        symbol_id: int = SimpleMarketMaking.Clean.tools.get_id_from_symbol(self.symbol)
        symbol_id_string = str(symbol_id)
        data = pq.read_table(SimpleMarketMaking.Clean.config.source_directory +
                             '20210901_' + symbol_id_string + '_tob.parquet').to_pandas()
        data = data.head(100000)
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
        indices = SimpleMarketMaking.Clean.tools.get_time_sampled_indices(
            self.top_of_book_formatted['timestamp_millis'],
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

    def _format_trades(self) -> pd.DataFrame:
        formatted_data = pd.DataFrame()
        formatted_data['timestamp_millis'] = self.trades_raw['exchange_timestamp_nanos'].div(1000000)
        formatted_data['timestamp_millis'] = formatted_data['timestamp_millis'].astype('int64')
        config: pd.DataFrame = SimpleMarketMaking.Clean.config.config
        tick_size = float(config[config['symbol'] == self.symbol]['tick_size'])
        formatted_data['price'] = self.trades_raw['price'].div(tick_size)
        formatted_data['price'] = formatted_data['price'].astype('int64')
        step_size = float(config[config['symbol'] == self.symbol]['step_size'])
        formatted_data['size'] = self.trades_raw['qty'].div(step_size)
        formatted_data['size'] = formatted_data['size'].astype('int64')
        formatted_data['given'] = self.trades_raw['buyer_is_market_maker']
        formatted_data['given'] = formatted_data['given'].astype('bool')
        return formatted_data

    def _get_bars(self, indices: pd.DataFrame):
        bars = pd.DataFrame()
        bars['value'] = indices['value'].iloc[:-1]
        n = len(bars.index)
        first_timestamp_millis = []
        last_timestamp_millis = []
        vwaps = []
        opens = []
        closes = []
        highs = []
        lows = []
        volumes = []
        for i in range(n):
            first_index = indices['index'].iloc[i] + 1
            last_index = indices['index'].iloc[i + 1]
            if first_index > last_index:
                first_timestamp_millis.append(np.nan)
                last_timestamp_millis.append(np.nan)
                vwaps.append(np.nan)
                opens.append(np.nan)
                closes.append(np.nan)
                highs.append(np.nan)
                lows.append(np.nan)
                volumes.append(0)
            else:
                prices = self.trades_formatted.loc[first_index:last_index, 'price']
                sizes = self.trades_formatted.loc[first_index:last_index, 'size']
                vwap = int(prices.multiply(sizes).sum() / sizes.sum())
                first_timestamp_millis.append(self.trades_formatted.loc[first_index, 'timestamp_millis'])
                last_timestamp_millis.append(self.trades_formatted.loc[last_index, 'timestamp_millis'])
                vwaps.append(vwap)
                opens.append(prices.iloc[0])
                closes.append(prices.iloc[-1])
                highs.append(prices.max())
                lows.append(prices.min())
                volumes.append(sizes.sum())
        bars['first_timestamp_millis'] = first_timestamp_millis
        bars['last_timestamp_millis'] = last_timestamp_millis
        bars['vwap'] = vwaps
        bars['open'] = opens
        bars['close'] = closes
        bars['high'] = highs
        bars['low'] = lows
        bars['volume'] = volumes
        bars = bars.dropna()
        bars['vwap'] = bars['vwap'].astype('int64')
        bars['open'] = bars['open'].astype('int64')
        bars['close'] = bars['close'].astype('int64')
        bars['high'] = bars['high'].astype('int64')
        bars['low'] = bars['low'].astype('int64')
        bars['volume'] = bars['volume'].astype('int64')
        return bars

    def get_time_bars(self, bar_size_in_millis: int) -> pd.DataFrame:
        indices = SimpleMarketMaking.Clean.tools.get_time_sampled_indices(self.trades_formatted['timestamp_millis'],
                                                                          bar_size_in_millis, True)
        time_bars = self._get_bars(indices)
        time_bars = time_bars.rename(columns={'value': 'timestamp_millis'})
        return time_bars

    def get_volume_bars(self, volume: int) -> pd.DataFrame:
        indices = SimpleMarketMaking.Clean.tools.get_volume_sampled_indices(self.trades_formatted['size'], volume)
        volume_bars = self._get_bars(indices)
        volume_bars = volume_bars.rename(columns={'value': 'cumulative_size'})
        return volume_bars

    def get_dollar_bars(self, dollar: int) -> pd.DataFrame:
        indices = SimpleMarketMaking.Clean.tools.get_dollar_sampled_indices(self.trades_formatted['price'],
                                                                            self.trades_formatted['size'], dollar)
        dollar_bars = self._get_bars(indices)
        dollar_bars = dollar_bars.rename(columns={'value': 'cumulative_dollar'})
        return dollar_bars

    def get_tick_imbalance_bars(self) -> pd.DataFrame:
        pass

    def get_trade_side_imbalance_bars(self) -> pd.DataFrame:
        b = []
        for g in self.trades_formatted['given']:
            if g:
                b.append(-1)
            else:
                b.append(1)

        return pd.DataFrame()
