import datetime
import math
from typing import Optional

import pandas as pd
import numpy as np


class Backtest:
    def __init__(self, data_frame: Optional[pd.DataFrame]) -> None:
        self.data_frame = data_frame

        self.target_spread = 30
        self.max_position = 10
        self.target_position = 1
        self.price_change_threshold = 15
        self.skew_multiplier = 1
        self.aggressive_brokerage = 0.0004
        self.passive_brokerage = 0.0002

        self.position = 0
        self.cash = 0
        self.last_quoted_bid = 0
        self.last_quoted_ask = 0
        self.last_market_bid = 0
        self.last_market_ask = 0
        self.market_bid_at_last_quote = 0
        self.market_ask_at_last_quote = 0
        self.last_market_mid = 0
        self.bid_skew = 0
        self.ask_skew = 0
        self.trade_side = 0
        self.trade_size = 0
        self.trade_price = 0

        self.cumulative_gross_pnl = 0
        self.cumulative_fee = 0
        self.cumulative_net_pnl = 0
        self.cumulative_usd_volume = 0
        self.cumulative_number_of_quote = 0
        self.cumulative_number_of_trade = 0

        self.market_bids = []
        self.market_asks = []
        self.market_mids = []
        self.bid_skews = []
        self.ask_skews = []
        self.quoted_bids = []
        self.quoted_asks = []
        self.positions = []
        self.cashs = []

        self.gross_pnls = []
        self.net_pnls = []
        self.usd_volumes = []
        self.fees = []
        self.number_of_quotes = []
        self.number_of_trades = []

        self.trade_sides = []
        self.trade_sizes = []
        self.trade_prices = []

        self.hourly_indices = []
        self.hourly_timestamps = []
        self.hourly_pnls = []
        self.hourly_volumes = []
        self.hourly_returns = []

    def run(self) -> None:
        n = len(self.data_frame)
        progress_pct = 0
        for i in range(n):
            if math.floor(100 * i / n) > progress_pct:
                progress_pct = progress_pct + 5
                print(str(progress_pct) + '% ', end='')
            self.trade_side = 0
            self.trade_size = 0
            self.trade_price = 0
            self.last_market_bid = self.data_frame['bid_price'].iloc[i]
            self.last_market_ask = self.data_frame['ask_price'].iloc[i]
            self.last_market_mid = 0.5 * (self.last_market_bid + self.last_market_ask)
            if self.last_quoted_bid > 0 and self.last_quoted_bid >= self.last_market_ask:
                self._execute_passive_buy(self.target_position)
            if 0 < self.last_quoted_ask <= self.last_market_bid:
                self._execute_passive_sell(self.target_position)
            if i < n - 1:
                self.compute_skew()
                if self.last_quoted_bid == 0 or \
                        np.abs(self.last_market_bid - self.market_bid_at_last_quote) >= self.price_change_threshold:
                    self.market_bid_at_last_quote = self.last_market_bid
                    self.last_quoted_bid = np.max(np.ceil(self.last_market_mid - 0.5 * self.target_spread -
                                                          self.bid_skew), 0)
                    if self.last_quoted_bid > 0:
                        self.cumulative_number_of_quote = self.cumulative_number_of_quote + 1
                if self.last_quoted_ask == 0 or \
                        np.abs(self.last_market_ask - self.market_ask_at_last_quote) >= self.price_change_threshold:
                    self.market_ask_at_last_quote = self.last_market_ask
                    self.last_quoted_ask = np.ceil(self.last_market_mid + 0.5 * self.target_spread - self.ask_skew)
                    if np.isinf(self.last_quoted_ask):
                        self.last_quoted_ask = 0
                    if self.last_quoted_ask > 0:
                        self.cumulative_number_of_quote = self.cumulative_number_of_quote + 1
                if self.last_quoted_bid > 0 and self.last_quoted_bid >= self.last_market_ask:
                    self._execute_aggressive_buy(self.target_position)
                if 0 < self.last_quoted_ask <= self.last_market_bid:
                    self._execute_aggressive_sell(self.target_position)
            else:
                if self.position > 0:
                    self._execute_aggressive_sell(self.position)
                if self.position < 0:
                    self._execute_aggressive_buy(-self.position)
            self.cumulative_gross_pnl = self.cash + self.position * self.last_market_mid
            self.cumulative_net_pnl = self.cumulative_gross_pnl - self.cumulative_fee
            self.market_bids.append(self.last_market_bid)
            self.market_asks.append(self.last_market_ask)
            self.market_mids.append(self.last_market_mid)
            self.quoted_bids.append(self.last_quoted_bid)
            self.quoted_asks.append(self.last_quoted_ask)
            self.bid_skews.append(self.bid_skew)
            self.ask_skews.append(self.ask_skew)
            self.positions.append(self.position)
            self.gross_pnls.append(self.cumulative_gross_pnl)
            self.fees.append(self.cumulative_fee)
            self.net_pnls.append(self.cumulative_net_pnl)
            self.trade_sides.append(self.trade_side)
            self.trade_sizes.append(self.trade_size)
            self.trade_prices.append(self.trade_price)
            self.usd_volumes.append(self.cumulative_usd_volume)
            self.number_of_quotes.append(self.cumulative_number_of_quote)
            self.number_of_trades.append(self.cumulative_number_of_trade)
        self.data_frame['market_bid'] = self.market_bids
        self.data_frame['market_ask'] = self.market_asks
        self.data_frame['market_mid'] = self.market_mids
        self.data_frame['quoted_bid'] = self.quoted_bids
        self.data_frame['quoted_ask'] = self.quoted_asks
        self.data_frame['bid_skew'] = self.bid_skews
        self.data_frame['ask_skew'] = self.ask_skews
        self.data_frame['position'] = self.positions
        self.data_frame['trade_side'] = self.trade_sides
        self.data_frame['trade_size'] = self.trade_sizes
        self.data_frame['trade_price'] = self.trade_prices
        self.data_frame['usd_volume'] = self.usd_volumes
        self.data_frame['gross_pnl'] = self.gross_pnls
        self.data_frame['fee'] = self.fees
        self.data_frame['net_pnl'] = self.net_pnls
        self.data_frame['number_of_quotes'] = self.cumulative_number_of_quote
        self.data_frame['number_of_trades'] = self.cumulative_number_of_trade

    def get_total_gross_yield(self) -> float:
        total_gross_yield = 1e6 * self.cumulative_gross_pnl / self.cumulative_usd_volume
        return total_gross_yield

    def get_total_net_yield(self) -> float:
        total_net_yield = 1e6 * self.cumulative_net_pnl / self.cumulative_usd_volume
        return total_net_yield

    def load_market_data(self, directory: str, symbol: str, start_date: datetime.date, number_of_days: int) -> None:
        market_data = pd.DataFrame()
        for d in range(number_of_days):
            date = start_date
            date = date + datetime.timedelta(days=d)
            date_string = date.strftime('%Y%m%d')
            data_frame = pd.read_csv(directory + 'market_data_top_of_book_' + symbol + '_' +
                                     date_string + '.csv')
            market_data = market_data.append(data_frame)
        self.data_frame = market_data

    def load_market_data_from_path(self, path: str) -> None:
        market_data = pd.read_csv(path)
        self.data_frame = market_data

    def get_maximum_drawdown(self) -> float:
        maximum_drawdown = 0
        maximum_pnl = 0
        minimum_pnl = 0
        for pnl in self.gross_pnls:
            if pnl > maximum_pnl:
                maximum_pnl = pnl
                minimum_pnl = pnl
            if pnl < minimum_pnl:
                minimum_pnl = pnl
                if (maximum_pnl - minimum_pnl) > maximum_drawdown:
                    maximum_drawdown = maximum_pnl - minimum_pnl
        return maximum_drawdown

    def generate_hourly_indices_and_timestamps(self) -> None:
        i = 0
        timestamp = self.data_frame['milliseconds_since_epoch'].iloc[0]
        timestamp = timestamp - (timestamp % 3600000) + 3600000
        n = len(self.data_frame.index)
        self.hourly_indices = []
        self.hourly_timestamps = []
        while i < n:
            while i < n and self.data_frame['milliseconds_since_epoch'].iloc[i] < timestamp:
                i = i + 1
            self.hourly_indices.append(i - 1)
            self.hourly_timestamps.append(timestamp)
            timestamp = timestamp + 3600000

    def generate_hourly_pnls(self) -> None:
        if not self.hourly_indices:
            self.generate_hourly_indices_and_timestamps()
        self.hourly_pnls = []
        for i in range(len(self.hourly_indices) - 1):
            self.hourly_pnls.append(self.gross_pnls[self.hourly_indices[i + 1]] -
                                    self.gross_pnls[self.hourly_indices[i]])

    def generate_hourly_volumes(self) -> None:
        if not self.hourly_indices:
            self.generate_hourly_indices_and_timestamps()
        self.hourly_volumes = []
        for i in range(len(self.hourly_indices) - 1):
            self.hourly_volumes.append(self.usd_volumes[self.hourly_indices[i + 1]] -
                                       self.usd_volumes[self.hourly_indices[i]])

    def generate_hourly_returns(self) -> None:
        if not self.hourly_pnls:
            self.generate_hourly_pnls()
        if not self.hourly_volumes:
            self.generate_hourly_volumes()
        self.hourly_returns = []
        for i in range(len(self.hourly_pnls)):
            if self.hourly_volumes[i] > 0:
                self.hourly_returns.append(self.hourly_pnls[i] / self.hourly_volumes[i])
            else:
                self.hourly_returns.append(np.nan)

    def get_number_of_trades(self) -> int:
        number_of_trades = 0
        for trade in self.trade_sides:
            number_of_trades = number_of_trades + np.abs(trade)
        return number_of_trades

    def generate_trades(self) -> pd.DataFrame:
        trades = self.data_frame[self.data_frame['trade_side'] != 0]
        return trades

    def compute_skew(self) -> None:
        self.bid_skew = 0
        self.ask_skew = 0
        if self.position >= self.max_position:
            self.bid_skew = np.inf
        if self.position <= -self.max_position:
            self.ask_skew = np.inf
        if 0 < self.position < self.max_position:
            self.bid_skew = int(self.skew_multiplier * self.target_spread * self.position / self.max_position)
        if -self.max_position < self.position < 0:
            self.ask_skew = int(self.skew_multiplier * self.target_spread * self.position / self.max_position)

    def _execute_aggressive_buy(self, size: int) -> None:
        self.cumulative_fee = self.cumulative_fee + size * self.last_market_ask * self.aggressive_brokerage
        self._execute_buy(size)

    def _execute_aggressive_sell(self, size: int) -> None:
        self.cumulative_fee = self.cumulative_fee + size * self.last_market_bid * self.aggressive_brokerage
        self._execute_sell(size)

    def _execute_passive_buy(self, size: int) -> None:
        self.cumulative_fee = self.cumulative_fee + size * self.last_quoted_bid * self.passive_brokerage
        self._execute_buy(size)

    def _execute_passive_sell(self, size: int) -> None:
        self.cumulative_fee = self.cumulative_fee + size * self.last_quoted_ask * self.passive_brokerage
        self._execute_sell(size)

    def _execute_buy(self, size: int) -> None:
        self.last_quoted_bid = 0
        self.position = self.position + size
        self.cash = self.cash - size * self.last_market_ask
        self.trade_side = 1
        self.trade_size = size
        self.trade_price = self.last_market_ask
        self.cumulative_usd_volume = self.cumulative_usd_volume + self.trade_size * self.trade_price
        self.cumulative_number_of_trade = self.cumulative_number_of_trade + 1

    def _execute_sell(self, size: int) -> None:
        self.last_quoted_ask = 0
        self.position = self.position - size
        self.cash = self.cash + size * self.last_market_bid
        self.trade_side = -1
        self.trade_size = size
        self.trade_price = self.last_market_bid
        self.cumulative_usd_volume = self.cumulative_usd_volume + self.trade_size * self.trade_price
        self.cumulative_number_of_trade = self.cumulative_number_of_trade + 1

    def generate_hourly_data(self) -> pd.DataFrame:
        hourly_data: pd.DataFrame = pd.DataFrame()
        if not self.hourly_indices:
            self.generate_hourly_indices_and_timestamps()
        hourly_data['index'] = self.hourly_indices[1:]
        hourly_data['timestamp'] = self.hourly_timestamps[1:]
        if not self.hourly_volumes:
            self.generate_hourly_volumes()
        hourly_data['volume'] = self.hourly_volumes
        if not self.hourly_pnls:
            self.generate_hourly_pnls()
        hourly_data['pnl'] = self.hourly_pnls
        if not self.hourly_returns:
            self.generate_hourly_returns()
        hourly_data['return'] = self.hourly_returns
        return hourly_data

    def annualise(self, x):
        return 1000 * 60 * 60 * 24 * 365.24 * x / \
               (self.data_frame['milliseconds_since_epoch'].iloc[-1] -
                self.data_frame['milliseconds_since_epoch'].iloc[0])

    def generate_stats(self) -> pd.DataFrame:
        stats: pd.DataFrame = pd.DataFrame()
        stats['target_spread'] = [self.target_spread]
        stats['price_change_threshold'] = [self.price_change_threshold]
        stats['target_position'] = [self.target_position]
        stats['skew_multiplier'] = [self.skew_multiplier]
        stats['gross_pnl'] = [self.cumulative_gross_pnl]
        stats['fees'] = [self.cumulative_fee]
        stats['net_pnl'] = [self.cumulative_net_pnl]
        stats['usd_volume'] = [self.cumulative_usd_volume]
        stats['number_of_quotes'] = [self.cumulative_number_of_quote]
        stats['number_of_trades'] = [self.cumulative_number_of_trade]
        stats['MDD'] = [self.get_maximum_drawdown()]
        stats['gross_yield'] = [self.get_total_gross_yield()]
        stats['net_yield'] = [self.get_total_net_yield()]
        return stats
