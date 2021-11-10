import datetime
import numpy as np
import pandas as pd
import puffin.tardis_importer as ti
import puffin.config
import random


def _compute_autocovariance(spot_delta_vwaps: np.array) -> float:
    autocovariance = 0
    mean = np.mean(spot_delta_vwaps)
    for i in range(len(spot_delta_vwaps) - 1):
        autocovariance += (spot_delta_vwaps[i] - mean) * (spot_delta_vwaps[i + 1] - mean)
    autocovariance /= len(spot_delta_vwaps) - 1
    return autocovariance


class Backtest:
    def __init__(self) -> None:
        self.minimum_basis_spread = 5e-4
        self.vwap_horizon = 60
        self.spot_tobs = ti.generate_trade_data_from_gzip('BTC/USD', puffin.config.source_directory +
                                                          'data/inputs/ftx_trades_2021-10-14_BTC-USD.csv.gz')
        self.spot_trades = ti.generate_tob_data_from_gzip('BTC/USD', puffin.config.source_directory +
                                                          'data/inputs/ftx_quotes_2021-10-14_BTC-USD.csv.gz')
        self.future_tobs = ti.generate_trade_data_from_gzip('BTC/USD', puffin.config.source_directory +
                                                            'data/inputs/ftx_trades_2021-10-14_BTC-PERP.csv.gz')
        self.future_trades = ti.generate_tob_data_from_gzip('BTC/USD', puffin.config.source_directory +
                                                            'data/inputs/ftx_quotes_2021-10-14_BTC-PERP.csv.gz')
        self.time_bar_start_timestamp = None
        self.spot_market_bid = None
        self.spot_market_ask = None
        self.future_market_bid = None
        self.future_market_ask = None
        self.spot_bid = None
        self.spot_ask = None
        self.future_bid = None
        self.future_ask = None
        self.spot_position = 0
        self.future_position = 0
        self.spot_price_buffer = []
        self.future_price_buffer = []
        self.spot_size_buffer = []
        self.future_size_buffer = []
        self.spot_vwap_buffer = []
        self.future_vwap_buffer = []
        self.cash = 0
        self.quote_size = 1

    def run(self):
        spot_tobs_index = 0
        spot_trades_index = 0
        spot_tobs_n = len(self.spot_tobs.index)
        spot_trades_n = len(self.spot_trades.index)
        future_tobs_index = 0
        future_trades_index = 0
        future_tobs_n = len(self.future_tobs.index)
        future_trades_n = len(self.future_trades.index)
        progress_pct = 0
        while all([spot_tobs_index < spot_tobs_n, spot_trades_index < spot_trades_n,
                   future_tobs_index < future_tobs_n, future_trades_index < future_trades_n]):
            if np.floor(100 * spot_trades_index / spot_tobs_n) > progress_pct:
                progress_pct = progress_pct + 5
                print(str(progress_pct) + '% ', end='')
            spot_trades_timestamp = self.spot_trades['exchange_timestamp_nanos'].iloc[spot_trades_index]
            spot_tobs_timestamp = self.spot_tobs['exchange_timestamp_nanos'].iloc[spot_tobs_index]
            future_trades_timestamp = self.future_trades['exchange_timestamp_nanos'].iloc[future_trades_index]
            future_tobs_timestamp = self.future_tobs['exchange_timestamp_nanos'].iloc[future_tobs_index]
            events_list = [{'timestamp': spot_trades_timestamp, 'function': self._spot_trade_event},
                           {'timestamp': spot_tobs_timestamp, 'function': self._spot_tob_event},
                           {'timestamp': future_trades_timestamp, 'function': self._future_trade_event},
                           {'timestamp': future_tobs_timestamp, 'function': self._future_tob_event}]
            events_dict = {}
            for event in events_list:
                if event['timestamp'] in events_dict.keys():
                    events_dict[event['timestamp']].append(event['function'])
                else:
                    events_dict[event['timestamp']] = [event['function']]
            first_timestamp = min(events_dict)
            candidate_event_functions = events_dict[first_timestamp]
            chosen_event_function = random.choice(candidate_event_functions)
            chosen_event_function()

    def _spot_trade_event(self, spot_trade: pd.Series) -> None:
        self._check_end_of_bar(spot_trade.timestamp_millis)
        self.spot_price_buffer.append(spot_trade.price)
        self.spot_size_buffer.append(spot_trade.size)

    def _future_trade_event(self, future_trade: pd.Series) -> None:
        self._check_end_of_bar(future_trade.timestamp_millis)
        self.future_price_buffer.append(future_trade.price)
        self.future_size_buffer.append(future_trade.size)

    def _spot_tob_event(self, spot_tob: pd.Series) -> None:
        self._check_end_of_bar(spot_tob.timestamp_millis)
        self.spot_market_bid = spot_tob.bid_price
        self.spot_market_ask = spot_tob.ask_price
        if all([self.spot_bid is not None, self.spot_market_ask is not None, self.spot_bid >= self.spot_market_ask]):
            self._passive_spot_buy()
        if all([self.spot_ask is not None, self.spot_market_bid is not None, self.spot_ask <= self.spot_market_bid]):
            self._passive_spot_sell()

    def _future_tob_event(self, future_tob: pd.Series) -> None:
        self._check_end_of_bar(future_tob.timestamp_millis)
        self.future_market_bid = future_tob.bid_price
        self.future_market_ask = future_tob.ask_price
        if all([self.future_bid is not None, self.future_market_ask is not None,
                self.future_bid >= self.future_market_ask]):
            self._passive_future_buy()
        if all([self.future_ask is not None, self.future_market_bid is not None,
                self.future_ask <= self.future_market_bid]):
            self._passive_future_sell()

    def _check_end_of_bar(self, timestamp: datetime.datetime) -> None:
        if self.time_bar_start_timestamp is None:
            self.time_bar_start_timestamp = timestamp
        if timestamp - self.time_bar_start_timestamp >= 1e9:
            self._end_of_bar()
            self.time_bar_start_timestamp = timestamp

    def _end_of_bar(self) -> None:
        if self.spot_price_buffer:
            self.spot_vwap_buffer.append(np.average(self.spot_price_buffer, weights=self.spot_size_buffer))
        elif self.spot_vwap_buffer:
            self.spot_vwap_buffer.append(self.spot_vwap_buffer[-1])
        if self.future_price_buffer:
            self.future_vwap_buffer.append(np.average(self.future_price_buffer, weights=self.future_size_buffer))
        elif self.future_vwap_buffer:
            self.future_vwap_buffer.append(self.future_vwap_buffer[-1])
        if len(self.spot_vwap_buffer) > self.vwap_horizon:
            self.spot_vwap_buffer.pop(0)
        if len(self.future_vwap_buffer) > self.vwap_horizon:
            self.future_vwap_buffer.pop(0)
        if len(self.spot_vwap_buffer) > self.vwap_horizon and len(self.future_vwap_buffer) > self.vwap_horizon:
            spot_delta_vwaps = np.array(np.diff(self.spot_vwap_buffer))
            future_delta_vwaps = np.array(np.diff(self.future_vwap_buffer))
            spot_delta_vwaps = spot_delta_vwaps[~np.isnan(spot_delta_vwaps)]
            future_delta_vwaps = future_delta_vwaps[~np.isnan(future_delta_vwaps)]
            spot_autocovariance = _compute_autocovariance(spot_delta_vwaps)
            future_autocovariance = _compute_autocovariance(future_delta_vwaps)
            spot_trend = self.spot_vwap_buffer[-1] - self.spot_vwap_buffer[0]
            future_trend = self.future_vwap_buffer[-1] - self.future_vwap_buffer[0]
            delta = self.spot_position + self.future_position
            basis_spread = 2 * (self.future_market_ask - self.spot_market_bid) / (self.future_market_ask +
                                                                                  self.spot_market_bid)
            enough_basis_spread = basis_spread > self.minimum_basis_spread
            if (delta == 1 or (delta == 0 and enough_basis_spread)) and (future_autocovariance < 0 or future_trend > 0):
                self.future_ask = self.future_market_ask
            if (delta == -1 or (delta == 0 and enough_basis_spread)) and (spot_autocovariance < 0 or spot_trend < 0):
                self.spot_bid = self.spot_market_bid

    def _passive_spot_buy(self) -> None:
        self._spot_buy(self.spot_bid)

    def _passive_future_buy(self) -> None:
        self._future_buy(self.future_bid)

    def _passive_spot_sell(self) -> None:
        self._spot_sell(self.spot_ask)

    def _passive_future_sell(self) -> None:
        self._future_sell(self.future_ask)

    def _aggressive_spot_buy(self) -> None:
        self._spot_buy(self.spot_market_ask)

    def _aggressive_future_buy(self) -> None:
        self._future_buy(self.future_market_ask)

    def _aggressive_spot_sell(self) -> None:
        self._spot_sell(self.spot_market_bid)

    def _aggressive_future_sell(self) -> None:
        self._future_sell(self.future_market_bid)

    def _spot_buy(self, price: int) -> None:
        self.cash = self.cash - (self.quote_size * price)
        self.spot_position += self.quote_size
        self.spot_bid = None

    def _future_buy(self, price: int) -> None:
        self.cash = self.cash - (self.quote_size * price)
        self.future_position += self.quote_size
        self.future_bid = None

    def _spot_sell(self, price: int) -> None:
        self.cash = self.cash + (self.quote_size * price)
        self.spot_position -= self.quote_size
        self.spot_ask = None

    def _future_sell(self, price: int) -> None:
        self.cash = self.cash + (self.quote_size * price)
        self.future_position -= self.quote_size
        self.future_ask = None
