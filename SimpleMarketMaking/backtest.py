import datetime
import math
from typing import Optional

import pandas as pd
import numpy as np


class Backtest:
    def __init__(self, data_frame: Optional[pd.DataFrame]) -> None:
        self.data_frame = data_frame

        self.target_spread = 30
        self.max_pos = 200
        self.target_position = 20
        self.mid_change_threshold = 15
        self.aggressive_brokerage = 0.0004
        self.passive_brokerage = 0.0002

        self.position = 0
        self.cash = 0
        self.last_quoted_bid = 0
        self.last_quoted_ask = 0
        self.last_quoted_mid = 0
        self.last_market_bid = 0
        self.last_market_ask = 0
        self.last_market_mid = 0
        self.skew = 0
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
        self.skews = []
        self.quoted_bids = []
        self.quoted_asks = []
        self.quoted_mids = []
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
                progress_pct = progress_pct + 10
                print(str(progress_pct) + '% ', end='')
            self.trade_side = 0
            self.trade_size = 0
            self.trade_price = 0
            self.last_market_bid = self.data_frame['bidPrice'].iloc[i]
            self.last_market_ask = self.data_frame['askPrice'].iloc[i]
            self.last_market_mid = 0.5 * (self.last_market_bid + self.last_market_ask)
            if self.last_quoted_bid > 0 and self.last_quoted_bid >= self.last_market_ask:
                self._execute_passive_buy(self.target_position)
            if 0 < self.last_quoted_ask <= self.last_market_bid:
                self._execute_passive_sell(self.target_position)
            self.cumulative_number_of_quote = 0
            if i < n - 1:
                self.compute_skew()
                if self.last_quoted_bid == 0 or self.last_quoted_ask == 0 or \
                        np.abs(self.last_market_mid - self.last_quoted_mid) >= self.mid_change_threshold:
                    self.last_quoted_bid = math.ceil(self.last_market_mid + self.skew - 0.5 * self.target_spread)
                    self.last_quoted_ask = math.floor(self.last_market_mid + self.skew + 0.5 * self.target_spread)
                    self.last_quoted_mid = 0.5 * (self.last_quoted_bid + self.last_quoted_ask)
                    self.cumulative_number_of_quote = 2
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
            self.quoted_mids.append(self.last_quoted_mid)
            self.skews.append(self.skew)
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
        self.data_frame['market_bid'] = list(map(lambda x: x / 10000, self.market_bids))
        self.data_frame['market_ask'] = list(map(lambda x: x / 10000, self.market_asks))
        self.data_frame['market_mid'] = list(map(lambda x: x / 10000, self.market_mids))
        self.data_frame['quoted_bid'] = list(map(lambda x: x / 10000, self.quoted_bids))
        self.data_frame['quoted_ask'] = list(map(lambda x: x / 10000, self.quoted_asks))
        self.data_frame['quoted_mid'] = list(map(lambda x: x / 10000, self.quoted_mids))
        self.data_frame['skew'] = self.skews
        self.data_frame['position'] = self.positions
        self.data_frame['trade_side'] = self.trade_sides
        self.data_frame['trade_size'] = self.trade_sizes
        self.data_frame['trade_price'] = list(map(lambda x: x / 10000, self.trade_prices))
        self.data_frame['usd_volume'] = list(map(lambda x: x / 10000, self.usd_volumes))
        self.data_frame['gross_pnl'] = list(map(lambda x: x / 10000, self.gross_pnls))
        self.data_frame['fee'] = list(map(lambda x: x / 10000, self.fees))
        self.data_frame['net_pnl'] = list(map(lambda x: x / 10000, self.net_pnls))

    def get_total_gross_yield(self) -> float:
        total_gross_yield = 1e6 * self.cumulative_gross_pnl / self.cumulative_usd_volume
        return total_gross_yield

    def get_total_net_yield(self) -> float:
        total_net_yield = 1e6 * self.cumulative_net_pnl / self.cumulative_usd_volume
        return total_net_yield

    def load_market_data(self, directory: str, start_date: datetime.date, number_of_days: int) -> None:
        market_data = pd.DataFrame()
        for d in range(number_of_days):
            date = start_date
            date = date + datetime.timedelta(days=d)
            date_string = date.strftime('%Y%m%d')
            data_frame = pd.read_csv(directory + 'market_data_xrpusdt_' + date_string + '.csv')
            market_data = market_data.append(data_frame)
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
        timestamp = self.data_frame['millisecondsSinceEpoch'].iloc[0]
        timestamp = timestamp - (timestamp % 3600000) + 3600000
        n = len(self.data_frame.index)
        self.hourly_indices = []
        self.hourly_timestamps = []
        while i < n:
            while i < n and self.data_frame['millisecondsSinceEpoch'].iloc[i] < timestamp:
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
        self.skew = 0
        if np.abs(self.position) >= self.max_pos:
            self.skew = int(-2 * np.sign(self.position) * self.target_spread)
        else:
            self.skew = int(-1 * np.sign(self.position) * np.floor(self.target_spread * np.abs(self.position) /
                                                                   self.max_pos))

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
        hourly_data['index'] = self.hourly_indices[1:]
        hourly_data['timestamp'] = self.hourly_timestamps[1:]
        hourly_data['volume'] = self.hourly_volumes
        hourly_data['pnl'] = self.hourly_pnls
        hourly_data['return'] = self.hourly_returns
        return hourly_data

    def annualise(self, x):
        return 1000 * 60 * 60 * 24 * 365.24 * x / \
               (self.data_frame['millisecondsSinceEpoch'].iloc[-1] -
                self.data_frame['millisecondsSinceEpoch'].iloc[0])
