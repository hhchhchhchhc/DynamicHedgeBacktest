import datetime
import numpy as np
import pandas as pd
import tools
import config as con


class MarketData:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.top_of_book_raw = pd.DataFrame()
        self.top_of_book_formatted = pd.DataFrame()
        self.trades_raw = pd.DataFrame()
        self.trades_formatted = pd.DataFrame()

    def load_top_of_book_data_from_parquet(self, date: datetime.date) -> None:
        symbol_id: int = tools.get_id_from_symbol(self.symbol)
        symbol_id_string = str(symbol_id)
        date_string = date.strftime('%Y%m%d')
        self.top_of_book_raw = pd.read_parquet(con.source_directory + 'raw/parquet/' +
                                               date_string + '_' + symbol_id_string + '_tob.parquet')

    def load_trade_data_from_parquet(self, date: datetime.date) -> None:
        symbol_id: int = tools.get_id_from_symbol(self.symbol)
        symbol_id_string = str(symbol_id)
        date_string = date.strftime('%Y%m%d')
        self.trades_raw = pd.read_parquet(
            con.source_directory + 'raw/parquet/' + date_string
            + '_' + symbol_id_string + '_trade.parquet')

    def load_formatted_trade_data_from_csv(self, date: datetime.date) -> None:
        date_string = date.strftime('%Y%m%d')
        self.trades_formatted = pd.read_csv(con.source_directory + 'formatted/trades/' +
                                            date_string + '_Binance_' + self.symbol + '_trades.csv')

    def generate_formatted_top_of_book_data(self) -> None:
        self.top_of_book_formatted = pd.DataFrame()
        self.top_of_book_formatted['timestamp_millis'] = self.top_of_book_raw['exchange_timestamp_nanos'].apply(
            lambda t: int(t / 1000000))
        self.top_of_book_formatted['timestamp_millis'] = self.top_of_book_formatted['timestamp_millis'].astype('int64')
        config: pd.DataFrame = con.config
        tick_size = float(config[config['symbol'] == self.symbol]['tick_size'])
        self.top_of_book_formatted['bid_price'] = self.top_of_book_raw['bid_price'].div(tick_size)
        self.top_of_book_formatted['bid_price'] = self.top_of_book_formatted['bid_price'].astype('int64')
        self.top_of_book_formatted['ask_price'] = self.top_of_book_raw['ask_price'].div(tick_size)
        self.top_of_book_formatted['ask_price'] = self.top_of_book_formatted['ask_price'].astype('int64')
        step_size = float(config[config['symbol'] == self.symbol]['step_size'])
        self.top_of_book_formatted['bid_size'] = self.top_of_book_raw['bid_qty'].div(step_size)
        self.top_of_book_formatted['bid_size'] = self.top_of_book_formatted['bid_size'].astype('int64')
        self.top_of_book_formatted['ask_size'] = self.top_of_book_raw['ask_qty'].div(step_size)
        self.top_of_book_formatted['ask_size'] = self.top_of_book_formatted['ask_size'].astype('int64')

    def generate_formatted_trades_data(self) -> None:
        formatted_data = pd.DataFrame()
        formatted_data['timestamp_millis'] = self.trades_raw['exchange_timestamp_nanos'].apply(
            lambda t: int(t / 1000000))
        config: pd.DataFrame = con.config
        tick_size = float(config[config['symbol'] == self.symbol]['tick_size'])
        formatted_data['price'] = self.trades_raw['price'].div(tick_size)
        formatted_data['price'] = formatted_data['price'].astype('int64')
        step_size = float(config[config['symbol'] == self.symbol]['step_size'])
        formatted_data['size'] = self.trades_raw['qty'].div(step_size)
        formatted_data['size'] = formatted_data['size'].astype('int64')
        formatted_data['given'] = self.trades_raw['buyer_is_market_maker']
        formatted_data['given'] = formatted_data['given'].astype('bool')
        self.trades_formatted = formatted_data

    def time_sampled_top_of_book(self, millis: int):
        indices = tools.get_time_sampled_indices(
            self.top_of_book_formatted['timestamp_millis'],
            millis, False)
        return self.top_of_book_formatted.iloc[indices]

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
        volume_givens = []
        for i in range(n):
            first_index = indices['index'].loc[i] + 1
            last_index = indices['index'].loc[i + 1]
            if first_index > last_index:
                first_timestamp_millis.append(np.nan)
                last_timestamp_millis.append(np.nan)
                vwaps.append(np.nan)
                opens.append(np.nan)
                closes.append(np.nan)
                highs.append(np.nan)
                lows.append(np.nan)
                volumes.append(0)
                volume_givens.append(0)
            else:
                prices = self.trades_formatted.loc[first_index: last_index, 'price']
                sizes = self.trades_formatted.loc[first_index: last_index, 'size']
                givens = self.trades_formatted.loc[first_index: last_index, 'given']
                vwap = int(prices.multiply(sizes).sum() / sizes.sum())
                first_timestamp_millis.append(self.trades_formatted.loc[first_index, 'timestamp_millis'])
                last_timestamp_millis.append(self.trades_formatted.loc[last_index, 'timestamp_millis'])
                vwaps.append(vwap)
                opens.append(prices.iloc[0])
                closes.append(prices.iloc[-1])
                highs.append(prices.max())
                lows.append(prices.min())
                volumes.append(sizes.sum())
                volume_givens.append(sizes.multiply(givens).sum())
        bars['first_timestamp_millis'] = first_timestamp_millis
        bars['last_timestamp_millis'] = last_timestamp_millis
        bars['vwap'] = vwaps
        bars['open'] = opens
        bars['close'] = closes
        bars['high'] = highs
        bars['low'] = lows
        bars['volume'] = volumes
        bars['volume_given'] = volume_givens
        bars = bars.dropna()
        bars['vwap'] = bars['vwap'].astype('int64')
        bars['open'] = bars['open'].astype('int64')
        bars['close'] = bars['close'].astype('int64')
        bars['high'] = bars['high'].astype('int64')
        bars['low'] = bars['low'].astype('int64')
        bars['volume'] = bars['volume'].astype('int64')
        return bars

    def get_time_bars(self, bar_size_in_millis: int) -> pd.DataFrame:
        indices = tools.get_time_sampled_indices(self.trades_formatted['timestamp_millis'], bar_size_in_millis, True)
        time_bars = self._get_bars(indices)
        time_bars = time_bars.rename(columns={'value': 'timestamp_millis'})
        return time_bars

    def get_tick_bars(self, number_of_ticks_per_bar: int):
        indices = pd.DataFrame()
        indices['value'] = self.trades_formatted[::number_of_ticks_per_bar].index
        indices['index'] = indices['value']
        tick_bars = self._get_bars(indices)
        tick_bars = tick_bars.rename(columns={'value': 'tick'})
        return tick_bars

    def get_volume_bars(self, volume: int) -> pd.DataFrame:
        indices = tools.get_volume_sampled_indices(self.trades_formatted['size'], volume)
        volume_bars = self._get_bars(indices)
        volume_bars = volume_bars.rename(columns={'value': 'cumulative_size'})
        return volume_bars

    def get_dollar_bars(self, dollar: int) -> pd.DataFrame:
        indices = tools.get_dollar_sampled_indices(
            self.trades_formatted['price'], self.trades_formatted['size'], dollar)
        dollar_bars = self._get_bars(indices)
        dollar_bars = dollar_bars.rename(columns={'value': 'cumulative_dollar'})
        return dollar_bars

    def get_tick_imbalance_bars(self) -> pd.DataFrame:
        indices = tools.get_tick_imbalance_sampled_indices(self.trades_formatted['price'])
        tick_imbalance_bars = self._get_bars(indices)
        tick_imbalance_bars = tick_imbalance_bars.rename(columns={'value': 'tick_imbalance'})
        return tick_imbalance_bars

    def get_trade_side_imbalance_bars(self) -> pd.DataFrame:
        indices = tools.get_trade_side_imbalance_sampled_indices(
            self.trades_formatted['given'])
        trade_side_imbalance_bars = self._get_bars(indices)
        trade_side_imbalance_bars = trade_side_imbalance_bars.rename(columns={'value': 'trade_side_imbalance'})
        return trade_side_imbalance_bars

    def get_volume_tick_imbalance_bars(self) -> pd.DataFrame:
        indices = tools.get_volume_tick_imbalance_sampled_indices(
            self.trades_formatted['price'],
            self.trades_formatted['size'])
        volume_tick_imbalance_bars = self._get_bars(indices)
        volume_tick_imbalance_bars = volume_tick_imbalance_bars.rename(columns={'value': 'volume_tick_imbalance'})
        return volume_tick_imbalance_bars

    def get_volume_trade_side_imbalance_bars(self) -> pd.DataFrame:
        indices = tools.get_volume_trade_side_imbalance_sampled_indices(
            self.trades_formatted)
        volume_tick_imbalance_bars = self._get_bars(indices)
        volume_tick_imbalance_bars = volume_tick_imbalance_bars.rename(columns={'value': 'volume_trade_side_imbalance'})
        return volume_tick_imbalance_bars

    def get_dollar_tick_imbalance_bars(self) -> pd.DataFrame:
        indices = tools.get_dollar_tick_imbalance_sampled_indices(
            self.trades_formatted['price'],
            self.trades_formatted['size'])
        dollar_tick_imbalance_bars = self._get_bars(indices)
        dollar_tick_imbalance_bars = dollar_tick_imbalance_bars.rename(columns={'value': 'dollar_tick_imbalance'})
        return dollar_tick_imbalance_bars

    def get_dollar_trade_side_imbalance_bars(self) -> pd.DataFrame:
        indices = tools.get_dollar_trade_side_imbalance_sampled_indices(
            self.trades_formatted)
        dollar_tick_imbalance_bars = self._get_bars(indices)
        dollar_tick_imbalance_bars = dollar_tick_imbalance_bars.rename(columns={'value': 'dollar_trade_side_imbalance'})
        return dollar_tick_imbalance_bars

    def get_tick_runs_bars(self) -> pd.DataFrame:
        indices = tools.get_tick_runs_sampled_indices(self.trades_formatted['price'])
        tick_run_bars = self._get_bars(indices)
        tick_run_bars = tick_run_bars.rename(columns={'value': 'tick_runs'})
        return tick_run_bars

    def get_trade_side_runs_bars(self) -> pd.DataFrame:
        indices = tools.get_trade_side_runs_sampled_indices(self.trades_formatted['given'])
        trade_side_runs_bars = self._get_bars(indices)
        trade_side_runs_bars = trade_side_runs_bars.rename(columns={'value': 'trade_side_runs'})
        return trade_side_runs_bars

    def get_volume_tick_runs_bars(self) -> pd.DataFrame:
        indices = tools.get_volume_tick_runs_sampled_indices(
            self.trades_formatted['price'],
            self.trades_formatted['size'])
        volume_tick_runs_bars = self._get_bars(indices)
        volume_tick_runs_bars = volume_tick_runs_bars.rename(columns={'value': 'volume_tick_runs'})
        return volume_tick_runs_bars

    def get_volume_trade_side_runs_bars(self) -> pd.DataFrame:
        indices = tools.get_volume_trade_side_runs_sampled_indices(
            self.trades_formatted)
        volume_trade_side_runs_bars = self._get_bars(indices)
        volume_trade_side_runs_bars = volume_trade_side_runs_bars.rename(columns={'value': 'volume_trade_side_runs'})
        return volume_trade_side_runs_bars

    def get_dollar_tick_runs_bars(self) -> pd.DataFrame:
        indices = tools.get_dollar_tick_runs_sampled_indices(
            self.trades_formatted['price'],
            self.trades_formatted['size'])
        dollar_tick_runs_bars = self._get_bars(indices)
        dollar_tick_runs_bars = dollar_tick_runs_bars.rename(columns={'value': 'dollar_tick_runs'})
        return dollar_tick_runs_bars

    def get_dollar_trade_side_runs_bars(self) -> pd.DataFrame:
        indices = tools.get_dollar_trade_side_runs_sampled_indices(
            self.trades_formatted)
        dollar_trade_side_imbalance_bars = self._get_bars(indices)
        dollar_trade_side_imbalance_bars = dollar_trade_side_imbalance_bars.rename(
            columns={'value': 'dollar_trade_side_imbalance'})
        return dollar_trade_side_imbalance_bars
