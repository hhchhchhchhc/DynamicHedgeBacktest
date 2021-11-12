import puffin.tardis_importer as ti
import puffin.config
import random
import pandas as pd
import numpy as np


class Backtest:
    def __init__(self):
        self.vwap_horizon = 60
        exchange_spot_trades_file = puffin.config.source_directory + 'data/inputs/ftx_trades_2021-10-14_BTC-USD.csv.gz'
        self.exchange_spot_trades = ti.generate_trade_data_from_gzip('BTC/USD', exchange_spot_trades_file)
        exchange_spot_tobs_file = puffin.config.source_directory + 'data/inputs/ftx_quotes_2021-10-14_BTC-USD.csv.gz'
        self.exchange_spot_tobs = ti.generate_tob_data_from_gzip('BTC/USD', exchange_spot_tobs_file)
        exchange_future_trades_file = puffin.config.source_directory + 'data/inputs/ftx_trades_2021-10-14_BTC-PERP' \
                                                                       '.csv.gz '
        self.exchange_future_trades = ti.generate_trade_data_from_gzip('BTC/USD', exchange_future_trades_file)
        exchange_future_tobs_file = puffin.config.source_directory + 'data/inputs/ftx_quotes_2021-10-14_BTC-PERP.csv.gz'
        self.exchange_future_tobs = ti.generate_tob_data_from_gzip('BTC/USD', exchange_future_tobs_file)
        self.first_time = max(self.exchange_spot_trades['exchange_timestamp_nanos'].iloc[0],
                              self.exchange_spot_tobs['exchange_timestamp_nanos'].iloc[0],
                              self.exchange_future_trades['exchange_timestamp_nanos'].iloc[0],
                              self.exchange_future_tobs['exchange_timestamp_nanos'].iloc[0])
        self.last_time = max(self.exchange_spot_trades['exchange_timestamp_nanos'].iloc[-1],
                             self.exchange_spot_tobs['exchange_timestamp_nanos'].iloc[-1],
                             self.exchange_future_trades['exchange_timestamp_nanos'].iloc[-1],
                             self.exchange_future_tobs['exchange_timestamp_nanos'].iloc[-1])
        self.current_time = self.first_time + max(60 * 60 * 1000000000, random.randrange(self.last_time -
                                                                                         self.first_time))
        self.current_time = ((self.current_time // 1000000000) + 2) * 1000000000
        self.minimum_basis_spread = self._compute_minimum_basis_spread_at_time(self.current_time)
        self.exchange_spot_trades = self.exchange_spot_trades[
            self.exchange_spot_trades.exchange_timestamp_nanos > self.current_time]
        self.exchange_spot_tobs = self.exchange_spot_tobs[
            self.exchange_spot_tobs.exchange_timestamp_nanos > self.current_time]
        self.exchange_future_trades = self.exchange_future_trades[
            self.exchange_future_trades.exchange_timestamp_nanos > self.current_time]
        self.exchange_future_tobs = self.exchange_future_tobs[
            self.exchange_future_tobs.exchange_timestamp_nanos > self.current_time]
        self.internal_processing_latency = 1000000
        self.strategy_to_exchange_latency = 10000000
        self.quote_size = 1
        self.target_size = 100
        self.backtest_results = pd.DataFrame(columns=['timestamp', 'spot_position', 'future_position'])
        self.backtest_spot_trades = pd.DataFrame(columns=['timestamp', 'spot_trade_price', 'spot_trade_size',
                                                          'spot_trade_side'])
        self.backtest_future_trades = pd.DataFrame(columns=['timestamp', 'future_trade_price', 'future_trade_size',
                                                            'future_trade_side'])
        self.vwap_buffer_size = 60
        self.spot_vwap_buffer = []
        self.future_vwap_buffer = []

    def run(self):
        while self.current_time < self.last_time:
            self._update_spot_vwap_buffer()
            self._update_future_vwap_buffer()
            if len(self.spot_vwap_buffer) >= self.vwap_buffer_size:
                print('bla')
            self.current_time += 1000000000

    def _compute_minimum_basis_spread_at_time(self, random_start_time: int) -> float:
        spot = self.exchange_spot_tobs[self.exchange_spot_tobs['exchange_timestamp_nanos'].between(
            random_start_time - 60 * 60 * 1000000000, random_start_time)]
        future = self.exchange_future_tobs[self.exchange_future_tobs.exchange_timestamp_nanos.between(
            random_start_time - 60 * 60 * 1000000000, random_start_time)]
        spread = pd.merge(spot[['exchange_timestamp_nanos', 'bid_price']],
                          future[['exchange_timestamp_nanos', 'ask_price']], how='outer',
                          left_on='exchange_timestamp_nanos', right_on='exchange_timestamp_nanos')
        spread = spread.sort_values(by=['exchange_timestamp_nanos'])
        spread = spread.fillna(method='ffill')
        spread = spread.dropna(how='any')
        spread['spread'] = 2 * (spread['ask_price'] - spread['bid_price']) / (
                spread['ask_price'] + spread['bid_price'])
        minimum_basis_spread = np.quantile(spread['spread'], .5)
        return minimum_basis_spread

    def _compute_spot_vwap(self) -> float:
        exchange_spot_trades_for_vwap = self.exchange_spot_trades[self.exchange_spot_trades[
            'exchange_timestamp_nanos'].between(
            self.current_time - 1000000000 - self.strategy_to_exchange_latency,
            self.current_time - self.strategy_to_exchange_latency)]
        spot_vwap = np.nan
        if len(exchange_spot_trades_for_vwap):
            spot_prices_for_vwap = np.array(exchange_spot_trades_for_vwap['price'])
            spot_sizes_for_vwap = np.array(exchange_spot_trades_for_vwap['size'])
            spot_vwap = np.average(spot_prices_for_vwap, weights=spot_sizes_for_vwap)
        return spot_vwap

    def _update_spot_vwap_buffer(self) -> None:
        spot_vwap = self._compute_spot_vwap()
        if np.isnan(spot_vwap):
            if len(self.spot_vwap_buffer):
                self.spot_vwap_buffer.append(self.spot_vwap_buffer[-1])
        else:
            self.spot_vwap_buffer.append(spot_vwap)
        if len(self.spot_vwap_buffer) > self.vwap_buffer_size:
            self.spot_vwap_buffer.pop(0)

    def _compute_future_vwap(self) -> float:
        exchange_future_trades_for_vwap = self.exchange_future_trades[self.exchange_future_trades[
            'exchange_timestamp_nanos'].between(
            self.current_time - 1000000000 - self.strategy_to_exchange_latency,
            self.current_time - self.strategy_to_exchange_latency)]
        future_vwap = np.nan
        if len(exchange_future_trades_for_vwap):
            future_prices_for_vwap = np.array(exchange_future_trades_for_vwap['price'])
            future_sizes_for_vwap = np.array(exchange_future_trades_for_vwap['size'])
            future_vwap = np.average(future_prices_for_vwap, weights=future_sizes_for_vwap)
        return future_vwap

    def _update_future_vwap_buffer(self) -> None:
        future_vwap = self._compute_future_vwap()
        if np.isnan(future_vwap):
            if len(self.future_vwap_buffer):
                self.future_vwap_buffer.append(self.future_vwap_buffer[-1])
        else:
            self.future_vwap_buffer.append(future_vwap)
        if len(self.future_vwap_buffer) > self.vwap_buffer_size:
            self.future_vwap_buffer.pop(0)
