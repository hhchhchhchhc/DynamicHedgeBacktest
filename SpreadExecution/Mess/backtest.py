import datetime

import puffin.tardis_importer as ti
import puffin.config
import random
import pandas as pd
import numpy as np
from enum import Enum
import os

np.seterr(all='raise')


class OrderState(Enum):
    PENDING_NEW = 1
    NEW = 2
    PENDING_FILL = 3
    FILLED = 4
    PENDING_REPLACE = 5
    PENDING_CANCEL = 6
    CANCELED = 7


def _compute_autocorrelation(delta_vwaps: np.array) -> float:
    n = len(delta_vwaps)
    x = delta_vwaps[:n - 1]
    y = delta_vwaps[1:]
    covariance_matrix = np.cov(x, y)
    correlation = covariance_matrix[0, 1]
    variances = covariance_matrix[0, 0] * covariance_matrix[1, 1]
    if variances > 0:
        correlation /= np.sqrt(variances)
    return correlation


def _compute_trend(prices: np.array):
    return prices[-1] - prices[0]


class Backtest:
    def __init__(self, data_dir: str, date: datetime.datetime, base_asset: str, quote_asset: str):
        self.latency = 0
        date_str = date.strftime('%Y-%m-%d')
        symbol = base_asset + '/' + quote_asset
        spot_filename_suffix = '_' + base_asset + '-' + quote_asset + '.csv.gz'
        perp_filename_suffix = '_' + base_asset + '-PERP.csv.gz'
        exchange_spot_trades_file = os.path.join(data_dir, 'ftx_trades_' + date_str + spot_filename_suffix)
        self.exchange_spot_trades = ti.generate_trade_data_from_gzip(symbol, exchange_spot_trades_file)
        exchange_spot_tobs_file = os.path.join(data_dir, 'ftx_quotes_' + date_str + spot_filename_suffix)
        self.exchange_spot_tobs = ti.generate_tob_data_from_gzip(symbol, exchange_spot_tobs_file)
        exchange_future_trades_file = os.path.join(data_dir, 'ftx_trades_' + date_str + perp_filename_suffix)
        self.exchange_future_trades = ti.generate_trade_data_from_gzip(symbol, exchange_future_trades_file)
        exchange_future_tobs_file = os.path.join(data_dir, 'ftx_quotes_' + date_str + perp_filename_suffix)
        self.exchange_future_tobs = ti.generate_tob_data_from_gzip(symbol, exchange_future_tobs_file)
        self.first_time = max(self.exchange_spot_trades['exchange_timestamp_nanos'].iloc[0],
                              self.exchange_spot_tobs['exchange_timestamp_nanos'].iloc[0],
                              self.exchange_future_trades['exchange_timestamp_nanos'].iloc[0],
                              self.exchange_future_tobs['exchange_timestamp_nanos'].iloc[0])
        self.last_time = max(self.exchange_spot_trades['exchange_timestamp_nanos'].iloc[-1],
                             self.exchange_spot_tobs['exchange_timestamp_nanos'].iloc[-1],
                             self.exchange_future_trades['exchange_timestamp_nanos'].iloc[-1],
                             self.exchange_future_tobs['exchange_timestamp_nanos'].iloc[-1])
        self.start_time = self.first_time + max(60 * 60 * 1000000000, random.randrange(self.last_time -
                                                                                       self.first_time))
        self.start_time = ((self.start_time // 1000000000) + 2) * 1000000000
        self.current_time = self.start_time
        self.spot_minimum_basis_spread_buffer = pd.DataFrame(columns=['timestamp', 'price'])
        self.future_minimum_basis_spread_buffer = pd.DataFrame(columns=['timestamp', 'price'])
        self._compute_minimum_basis_spread_at_time()
        self.exchange_spot_trades = self.exchange_spot_trades[
            self.exchange_spot_trades.exchange_timestamp_nanos > self.start_time]
        self.exchange_spot_tobs = self.exchange_spot_tobs[
            self.exchange_spot_tobs.exchange_timestamp_nanos > self.start_time]
        self.exchange_future_trades = self.exchange_future_trades[
            self.exchange_future_trades.exchange_timestamp_nanos > self.start_time]
        self.exchange_future_tobs = self.exchange_future_tobs[
            self.exchange_future_tobs.exchange_timestamp_nanos > self.start_time]
        self.quote_size = 1
        self.backtest_spot_trades = pd.DataFrame(columns=['timestamp', 'spot_trade_price', 'spot_trade_size',
                                                          'spot_trade_side', 'is_passive'])
        self.backtest_future_trades = pd.DataFrame(columns=['timestamp', 'future_trade_price', 'future_trade_size',
                                                            'future_trade_side', 'is_passive'])
        self.vwap_buffer_size = 60
        self.target_position = 10
        self.spot_vwap_buffer = []
        self.future_vwap_buffer = []
        self.spot_autocorrelation = None
        self.future_autocorrelation = None
        self.spot_trend = None
        self.future_trend = None
        self.delta = 0
        self.spot_position = 0
        self.future_position = 0
        self.spot_bid_order_state = None
        self.spot_bid_order_price = None
        self.spot_bid_order_state_timestamp = None
        self.future_ask_order_state = None
        self.future_ask_order_price = None
        self.future_ask_order_state_timestamp = None
        self.strategy_spot_best_bid_price = None
        self.strategy_future_best_ask_price = None
        self.run_start_time = self.current_time
        self.autocorrelation_threshold = -0.15
        self.use_autocorrelation = False

    def run(self, latency: int, use_autocorrelation: bool):
        self.use_autocorrelation = use_autocorrelation
        self.latency = latency
        self.backtest_spot_trades = pd.DataFrame(columns=['timestamp', 'spot_trade_price', 'spot_trade_size',
                                                          'spot_trade_side', 'is_passive'])
        self.backtest_future_trades = pd.DataFrame(columns=['timestamp', 'future_trade_price', 'future_trade_size',
                                                            'future_trade_side', 'is_passive'])
        self.spot_vwap_buffer = []
        self.future_vwap_buffer = []
        self.spot_autocorrelation = None
        self.future_autocorrelation = None
        self.spot_trend = None
        self.future_trend = None
        self.delta = 0
        self.spot_position = 0
        self.future_position = 0
        self.spot_bid_order_state = None
        self.spot_bid_order_price = None
        self.spot_bid_order_state_timestamp = None
        self.future_ask_order_state = None
        self.future_ask_order_price = None
        self.future_ask_order_state_timestamp = None
        self.strategy_spot_best_bid_price = None
        self.strategy_future_best_ask_price = None
        self.run_start_time = self.start_time
        self.current_time = self.start_time
        while self.current_time < self.last_time and \
                ((self.spot_position < self.target_position) or (self.future_position > - self.target_position)):
            self._update_world_before_strategy_decision()
            self._update_spot_vwap_buffer()
            self._update_future_vwap_buffer()
            if (len(self.spot_vwap_buffer) >= self.vwap_buffer_size) and (len(self.future_vwap_buffer) >=
                                                                          self.vwap_buffer_size):
                self._compute_spot_autocorrelation()
                self._compute_future_autocorrelation()
                self._compute_spot_trend()
                self._compute_future_trend()
                self._compute_delta()
                self._compute_minimum_basis_spread_at_time()
                self._update_strategy_market_data()
                self._update_strategy_basis_spread()
                strategy_buys_spot = False
                strategy_sells_future = False
                if (self.delta == 0) and (self.strategy_basis_spread > self.minimum_basis_spread) \
                        and ((self.spot_autocorrelation < self.autocorrelation_threshold)
                             or not self.use_autocorrelation):
                    strategy_buys_spot = True
                if (self.delta < 0) and ((self.spot_autocorrelation < self.autocorrelation_threshold)
                                           or (self.spot_trend > 0) or not self.use_autocorrelation):
                    strategy_buys_spot = True
                if (self.delta == 0) and (self.strategy_basis_spread > self.minimum_basis_spread) \
                        and ((self.future_autocorrelation < self.autocorrelation_threshold)
                             or not self.use_autocorrelation):
                    strategy_sells_future = True
                if (self.delta > 0) and ((self.future_autocorrelation < self.autocorrelation_threshold)
                                          or (self.future_trend < 0) or not self.use_autocorrelation):
                    strategy_sells_future = True
                self._execute_strategy_spot(strategy_buys_spot)
                self._execute_strategy_future(strategy_sells_future)
            self.current_time += 1000000000
        #self._write_results_to_files()
        self._print_some_result_to_console()

    def _update_world_before_strategy_decision(self) -> None:
        if self.spot_bid_order_state == OrderState.PENDING_NEW:
            self._check_for_aggressive_spot_fill_at_order_entry()
            if self.spot_bid_order_state_timestamp + (2 * self.latency) <= self.current_time:
                self.spot_bid_order_state = OrderState.NEW
        if self.spot_bid_order_state == OrderState.NEW:
            self._check_for_passive_spot_fill()
        if self.spot_bid_order_state == OrderState.PENDING_FILL:
            if self.spot_bid_order_state_timestamp + (2 * self.latency) <= self.current_time:
                self.spot_bid_order_state = OrderState.FILLED
        if self.spot_bid_order_state == OrderState.PENDING_REPLACE:
            if self.spot_bid_order_state_timestamp + (2 * self.latency) <= self.current_time:
                self.spot_bid_order_state = OrderState.NEW
        if self.spot_bid_order_state == OrderState.PENDING_CANCEL:
            if self.spot_bid_order_state_timestamp + (2 * self.latency) <= self.current_time:
                self.spot_bid_order_state = OrderState.CANCELED

        if self.future_ask_order_state == OrderState.PENDING_NEW:
            self._check_for_aggressive_future_fill_at_order_entry()
            if self.future_ask_order_state_timestamp + (2 * self.latency) <= self.current_time:
                self.future_ask_order_state = OrderState.NEW
        if self.future_ask_order_state == OrderState.NEW:
            self._check_for_passive_future_fill()
        if self.future_ask_order_state == OrderState.PENDING_FILL:
            if self.future_ask_order_state_timestamp + (2 * self.latency) <= self.current_time:
                self.future_ask_order_state = OrderState.FILLED
        if self.future_ask_order_state == OrderState.PENDING_REPLACE:
            if self.future_ask_order_state_timestamp + (2 * self.latency) <= self.current_time:
                self.future_ask_order_state = OrderState.NEW
        if self.future_ask_order_state == OrderState.PENDING_CANCEL:
            if self.future_ask_order_state_timestamp + (2 * self.latency) <= self.current_time:
                self.future_ask_order_state = OrderState.CANCELED

    def _check_for_aggressive_spot_fill_at_order_entry(self):
        timestamp = self.spot_bid_order_state_timestamp + self.latency
        spot_tob_at_order_entry = self.exchange_spot_tobs[self.exchange_spot_tobs.exchange_timestamp_nanos <=
                                                          timestamp].tail(1)
        if self.spot_bid_order_price >= spot_tob_at_order_entry['ask_price'].values[0]:
            fill = {'timestamp': timestamp,
                    'spot_trade_price': spot_tob_at_order_entry['ask_price'].values[0],
                    'spot_trade_size': self.quote_size,
                    'spot_trade_side': 1,
                    'is_passive': False}
            self.backtest_spot_trades = self.backtest_spot_trades.append(fill, ignore_index=True)
            self.spot_bid_order_state_timestamp = timestamp
            if timestamp + self.latency <= self.current_time:
                self.spot_bid_order_state = OrderState.FILLED
            else:
                self.spot_bid_order_state = OrderState.PENDING_FILL
            self.spot_position += self.quote_size

    def _check_for_aggressive_future_fill_at_order_entry(self):
        timestamp = self.future_ask_order_state_timestamp + self.latency
        future_tob_at_order_entry = self.exchange_future_tobs[self.exchange_future_tobs.exchange_timestamp_nanos <=
                                                              timestamp].tail(1)
        if self.future_ask_order_price <= future_tob_at_order_entry['bid_price'].values[0]:
            fill = {'timestamp': timestamp,
                    'future_trade_price': future_tob_at_order_entry['bid_price'].values[0],
                    'future_trade_size': self.quote_size,
                    'future_trade_side': -1,
                    'is_passive': False}
            self.backtest_future_trades = self.backtest_future_trades.append(fill, ignore_index=True)
            self.future_ask_order_state_timestamp = timestamp
            if timestamp + self.latency <= self.current_time:
                self.future_ask_order_state = OrderState.FILLED
            else:
                self.future_ask_order_state = OrderState.PENDING_FILL
            self.future_position -= self.quote_size

    def _check_for_passive_spot_fill(self):
        spot_trades = self.exchange_spot_trades[self.exchange_spot_trades.exchange_timestamp_nanos.between(
            self.spot_bid_order_state_timestamp + self.latency,
            self.current_time)]
        matches = spot_trades[spot_trades.price <= self.spot_bid_order_price]
        if len(matches) > 0:
            timestamp = matches['exchange_timestamp_nanos'].head(1).values[0]
            fill = {'timestamp': timestamp,
                    'spot_trade_price': self.spot_bid_order_price,
                    'spot_trade_size': self.quote_size,
                    'spot_trade_side': 1,
                    'is_passive': True}
            self.backtest_spot_trades = self.backtest_spot_trades.append(fill, ignore_index=True)
            self.spot_bid_order_state_timestamp = timestamp
            if timestamp + self.latency <= self.current_time:
                self.spot_bid_order_state = OrderState.FILLED
            else:
                self.spot_bid_order_state = OrderState.PENDING_FILL
            self.spot_position += self.quote_size

    def _check_for_passive_future_fill(self):
        future_trades = self.exchange_future_trades[self.exchange_future_trades.exchange_timestamp_nanos.between(
            self.future_ask_order_state_timestamp + self.latency,
            self.current_time)]
        matches = future_trades[future_trades.price >= self.future_ask_order_price]
        if len(matches) > 0:
            timestamp = matches['exchange_timestamp_nanos'].head(1).values[0]
            fill = {'timestamp': timestamp,
                    'future_trade_price': self.future_ask_order_price,
                    'future_trade_size': self.quote_size,
                    'future_trade_side': -1,
                    'is_passive': True}
            self.backtest_future_trades = self.backtest_future_trades.append(fill, ignore_index=True)
            self.future_ask_order_state_timestamp = timestamp
            if timestamp + self.latency <= self.current_time:
                self.future_ask_order_state = OrderState.FILLED
            else:
                self.future_ask_order_state = OrderState.PENDING_FILL
            self.future_position -= self.quote_size

    def _compute_minimum_basis_spread_at_time(self) -> None:
        self._update_minimum_basis_spread_buffer()
        spread = pd.merge(self.spot_minimum_basis_spread_buffer, self.future_minimum_basis_spread_buffer, how='outer',
                          left_on='exchange_timestamp_nanos', right_on='exchange_timestamp_nanos')
        spread = spread.sort_values(by=['exchange_timestamp_nanos'])
        spread = spread.fillna(method='ffill')
        spread = spread.dropna(how='any')
        spread['spread'] = 2 * (spread['ask_price'] - spread['bid_price']) / (
                spread['ask_price'] + spread['bid_price'])
        self.minimum_basis_spread = np.quantile(spread['spread'], .5)

    def _update_minimum_basis_spread_buffer(self) -> None:
        self.spot_minimum_basis_spread_buffer = self.exchange_spot_tobs[self.exchange_spot_tobs[
            'exchange_timestamp_nanos'].between(self.current_time - self.latency -
                                                60 * 60 * 1000000000,
                                                self.current_time - self.latency)]
        self.spot_minimum_basis_spread_buffer = self.spot_minimum_basis_spread_buffer[['exchange_timestamp_nanos',
                                                                                       'bid_price']]
        self.future_minimum_basis_spread_buffer = self.exchange_future_tobs[self.exchange_future_tobs[
            'exchange_timestamp_nanos'].between(self.current_time - self.latency -
                                                60 * 60 * 1000000000,
                                                self.current_time - self.latency)]
        self.future_minimum_basis_spread_buffer = self.future_minimum_basis_spread_buffer[
            ['exchange_timestamp_nanos', 'ask_price']]

    def _compute_spot_vwap(self) -> float:
        exchange_spot_trades_for_vwap = self.exchange_spot_trades[self.exchange_spot_trades[
            'exchange_timestamp_nanos'].between(
            self.current_time - 1000000000 - self.latency,
            self.current_time - self.latency)]
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
            self.current_time - 1000000000 - self.latency,
            self.current_time - self.latency)]
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

    def _compute_spot_autocorrelation(self) -> None:
        spot_delta_vwaps = np.array(np.diff(self.spot_vwap_buffer))
        self.spot_autocorrelation = _compute_autocorrelation(spot_delta_vwaps)

    def _compute_future_autocorrelation(self) -> None:
        future_delta_vwaps = np.array(np.diff(self.future_vwap_buffer))
        self.future_autocorrelation = _compute_autocorrelation(future_delta_vwaps)

    def _compute_spot_trend(self) -> None:
        spot_prices = self.spot_vwap_buffer
        self.spot_trend = _compute_trend(spot_prices)

    def _compute_future_trend(self) -> None:
        future_prices = self.future_vwap_buffer
        self.future_trend = _compute_trend(future_prices)

    def _compute_delta(self):
        self.delta = self.spot_position + self.future_position

    def _update_strategy_market_data(self) -> None:
        spot_tob = self.exchange_spot_tobs[self.exchange_spot_tobs.exchange_timestamp_nanos <=
                                           (self.current_time - self.latency)].tail(1)
        future_tob = self.exchange_future_tobs[self.exchange_future_tobs.exchange_timestamp_nanos <=
                                               (self.current_time - self.latency)].tail(1)
        self.strategy_spot_best_bid_price = spot_tob['bid_price'].values[0]
        self.strategy_future_best_ask_price = future_tob['ask_price'].values[0]

    def _update_strategy_basis_spread(self) -> None:
        self.strategy_basis_spread = 2 * (self.strategy_future_best_ask_price -
                                          self.strategy_spot_best_bid_price) / (self.strategy_future_best_ask_price +
                                                                                self.strategy_spot_best_bid_price)

    def _execute_strategy_spot(self, strategy_buys_spot: bool) -> None:
        if strategy_buys_spot:
            if self.spot_bid_order_state is None or \
                    self.spot_bid_order_state in [OrderState.FILLED, OrderState.CANCELED]:
                self.spot_bid_order_state = OrderState.PENDING_NEW
                self.spot_bid_order_price = self.strategy_spot_best_bid_price
                self.spot_bid_order_state_timestamp = self.current_time
            if self.spot_bid_order_state == OrderState.NEW and \
                    self.spot_bid_order_price != self.strategy_spot_best_bid_price:
                self.spot_bid_order_state = OrderState.PENDING_REPLACE
                self.spot_bid_order_price = self.strategy_spot_best_bid_price
                self.spot_bid_order_state_timestamp = self.current_time
        else:
            if self.spot_bid_order_state == OrderState.NEW:
                self.spot_bid_order_state = OrderState.PENDING_CANCEL
                self.spot_bid_order_price = None
                self.spot_bid_order_state_timestamp = self.current_time

    def _execute_strategy_future(self, strategy_sells_future: bool) -> None:
        if strategy_sells_future:
            if self.future_ask_order_state is None or \
                    self.future_ask_order_state in [OrderState.FILLED, OrderState.CANCELED]:
                self.future_ask_order_state = OrderState.PENDING_NEW
                self.future_ask_order_price = self.strategy_future_best_ask_price
                self.future_ask_order_state_timestamp = self.current_time
            if self.future_ask_order_state == OrderState.NEW and \
                    self.future_ask_order_price != self.strategy_future_best_ask_price:
                self.future_ask_order_state = OrderState.PENDING_REPLACE
                self.future_ask_order_price = self.strategy_future_best_ask_price
                self.future_ask_order_state_timestamp = self.current_time
        else:
            if self.future_ask_order_state == OrderState.NEW:
                self.future_ask_order_state = OrderState.PENDING_CANCEL
                self.future_ask_order_price = None
                self.future_ask_order_state_timestamp = self.current_time

    def _write_results_to_files(self) -> None:
        self.backtest_spot_trades.to_csv(puffin.config.source_directory +
                                         'data/outputs/spread_execution_spot_trades_latency_' + str(self.latency) +
                                         '.csv')
        self.backtest_future_trades.to_csv(puffin.config.source_directory +
                                           'data/outputs/spread_execution_future_trades_latency_' + str(self.latency) +
                                           '.csv')

    def _print_some_result_to_console(self) -> None:
        spot_vwap = np.average(self.backtest_spot_trades['spot_trade_price'].values,
                               weights=self.backtest_spot_trades['spot_trade_size'].values)
        future_vwap = np.average(self.backtest_future_trades['future_trade_price'].values,
                                 weights=self.backtest_future_trades['future_trade_size'].values)
        executed_premium = 2 * (future_vwap - spot_vwap) / (future_vwap + spot_vwap)
        passive = np.sum(self.backtest_spot_trades['is_passive']) + np.sum(self.backtest_future_trades['is_passive'])
        execution_duration = datetime.timedelta(microseconds=int((self.current_time - self.run_start_time) // 1000))
        # print('')
        # print('Start at ' + datetime.datetime.fromtimestamp(self.start_time // 1000000000).strftime(
        #     '%Y-%m-%d %H:%M:%S'))
        # print('Latency: ' + str(self.latency / 1000000) + ' millis.')
        # print('Use Autocorrelation: ' + str(self.use_autocorrelation) + '.')
        # print('Executed premium: ' + str(1e4 * executed_premium) + ' bp.')
        # print('number passive execution: ' + str(passive) + '.')
        # print('Execution duration: ' + str(execution_duration) + '.')
        # print('')
        print(str(self.use_autocorrelation) + ',' + str(1e4 * executed_premium) + ',' + str(self.latency / 1000000) + ',' + str(passive) + ',' + str(execution_duration) + ',' + str(self.target_position) + ',' + str(self.start_time))
