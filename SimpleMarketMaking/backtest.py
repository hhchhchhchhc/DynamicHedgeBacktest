import datetime
import math
from typing import Optional

import pandas as pd
import numpy as np


class Backtest:
    def __init__(self, data_frame: Optional[pd.DataFrame]) -> None:
        self.data_frame = data_frame

        self.symbol = 'self.data_frame'
        self.target_spread = 20
        self.skew_at_20_pct = 0
        self.skew_at_40_pct = 0
        self.skew_at_max_pos = 0
        self.max_pos = 100
        self.target_pos = 20
        self.mid_change_threshold = 10

        self.position = 0
        self.cash = 0
        self.bid = 0
        self.ask = 0
        self.mid = 0
        self.last_mid = 0
        self.pnl = 0
        self.mid_changed = False
        self.skew = 0

        self.market_mids = []
        self.skews = []
        self.bids = []
        self.asks = []
        self.positions = []
        self.cashs = []
        self.pnls = []

        self.trade_sides = []
        self.trade_sizes = []
        self.trade_prices = []

        self.hourly_indices = []
        self.hourly_timestamps = []
        self.hourly_pnls = []
        self.hourly_volumes = []

    def run(self) -> None:
        n = len(self.data_frame)
        progress_pct = 0
        for i in range(n):
            if math.floor(100 * i / n) > progress_pct:
                progress_pct = progress_pct + 10
                print(str(progress_pct) + '% ', end='')
            bid_price = self.data_frame['bidPrice'].iloc[i]
            ask_price = self.data_frame['askPrice'].iloc[i]
            if i < n - 1:
                self.mid = 0.5 * (bid_price + ask_price)
                self.market_mids.append(self.mid)
                if self.bid > 0 and ask_price <= self.bid:
                    self.position = self.position + self.target_pos
                    self.cash = self.cash - self.target_pos * self.bid
                    self.trade_sides.append(1)
                    self.trade_sizes.append(self.target_pos)
                    self.trade_prices.append(self.bid)
                    self.bid = 0
                elif 0 < self.ask <= bid_price:
                    self.position = self.position - self.target_pos
                    self.cash = self.cash + self.target_pos * self.ask
                    self.trade_sides.append(-1)
                    self.trade_sizes.append(self.target_pos)
                    self.trade_prices.append(self.ask)
                    self.ask = 0
                else:
                    self.trade_sides.append(0)
                    self.trade_sizes.append(0)
                    self.trade_prices.append(0)
                self.positions.append(self.position)
                self.mid_changed = abs(self.mid - self.last_mid) >= self.mid_change_threshold
                self.last_mid = self.mid
                self.skew = 0
                if 0.2 * self.max_pos <= self.position < 0.4 * self.max_pos:
                    self.skew = -self.skew_at_20_pct
                elif 0.4 * self.max_pos <= self.position < self.max_pos:
                    self.skew = -self.skew_at_40_pct
                elif self.position >= self.max_pos:
                    self.skew = -self.skew_at_max_pos
                if -0.4 * self.max_pos < self.position <= -0.2 * self.max_pos:
                    self.skew = self.skew_at_20_pct
                elif -self.max_pos < self.position <= -0.4 * self.max_pos:
                    self.skew = self.skew_at_40_pct
                elif self.position <= -self.max_pos:
                    self.skew = self.skew_at_max_pos
                self.skews.append(self.skew)
                if self.mid_changed or self.bid == 0 or self.ask == 0:
                    self.bid = math.ceil(self.mid + self.skew - 0.5 * self.target_spread)
                    self.ask = math.floor(self.mid + self.skew + 0.5 * self.target_spread)
            else:
                self.market_mids.append(self.last_mid)
                self.skews.append(0)
                if self.position > 0:
                    self.cash = self.cash + self.position * bid_price
                    self.trade_sides.append(-1)
                    self.trade_sizes.append(self.position)
                    self.trade_prices.append(bid_price)
                elif self.position < 0:
                    self.cash = self.cash - self.position * ask_price
                    self.trade_sides.append(1)
                    self.trade_sizes.append(-self.position)
                    self.trade_prices.append(ask_price)
                else:
                    self.trade_sides.append(0)
                    self.trade_sizes.append(0)
                    self.trade_prices.append(0)
                self.position = 0
                self.positions.append(self.position)
            self.bids.append(self.bid)
            self.asks.append(self.ask)
            self.pnl = self.cash + self.position * self.mid
            self.pnls.append(self.pnl)
            self.cashs.append(self.cash)
        self.data_frame['market_mid'] = self.market_mids
        self.data_frame['bid'] = self.bids
        self.data_frame['ask'] = self.asks
        self.data_frame['position'] = self.positions
        self.data_frame['cash'] = self.cashs
        self.data_frame['skew'] = self.skews
        self.data_frame['pnl'] = self.pnls
        self.data_frame['trade_side'] = self.trade_sides
        self.data_frame['trade_size'] = self.trade_sizes
        self.data_frame['trade_price'] = self.trade_prices

    def get_total_volume_traded(self) -> int:
        volume = sum(self.trade_sizes)
        return volume

    def get_yield(self) -> float:
        y = 1e2 * self.pnl / self.get_total_volume_traded()
        return y

    def load_market_data(self, directory: str, start_date: datetime.date, number_of_days: int) -> None:
        market_data = pd.DataFrame()
        for d in range(number_of_days):
            date = start_date
            date = date + datetime.timedelta(days=d)
            date_string = date.strftime('%Y%m%d')
            data_frame = pd.read_csv(directory + 'market_data_xrpusdt_' + date_string + '.csv')
            market_data = market_data.append(data_frame)
        self.data_frame = market_data

    def get_average_hourly_return(self) -> float:
        if len(self.hourly_returns) == 0:
            self.generate_hourly_returns()
        average_hourly_returns = float(np.average(self.hourly_returns))
        return average_hourly_returns

    def generate_hourly_indices(self) -> None:
        i = 0
        timestamp = self.data_frame['millisecondsSinceEpoch'].iloc[0]
        timestamp = timestamp - (timestamp % 3600000) + 3600000
        n = len(self.data_frame.index)
        while i < n:
            while i < n and self.data_frame['millisecondsSinceEpoch'].iloc[i] < timestamp:
                i = i + 1
            self.hourly_indices.append(i - 1)
            self.hourly_timestamps.append(timestamp)
            timestamp = timestamp + 3600000

    def get_standard_deviation_of_hourly_returns(self) -> float:
        if len(self.hourly_returns) == 0:
            self.generate_hourly_returns()
        standard_deviation_of_hourly_returns = float(np.std(self.hourly_returns))
        return standard_deviation_of_hourly_returns

    def get_annualised_average_hourly_return(self) -> float:
        annualised_average_hourly_return = 24 * 365.24 * self.get_average_hourly_return()
        return annualised_average_hourly_return

    def get_annualised_standard_deviation_of_hourly_returns(self) -> float:
        annualised_standard_deviation_of_hourly_returns = np.sqrt(
            24 * 365.24) * self.get_standard_deviation_of_hourly_returns()
        return annualised_standard_deviation_of_hourly_returns

    def get_annualised_sharp_ratio(self) -> float:
        sharp_ratio = self.get_annualised_average_hourly_return() / \
                      self.get_annualised_standard_deviation_of_hourly_returns()
        return sharp_ratio

    def get_maximum_drawdown(self) -> float:
        maximum_drawdown = 0
        maximum_pnl = 0
        minimum_pnl = 0
        for pnl in self.pnls:
            if pnl > maximum_pnl:
                maximum_pnl = pnl
                if maximum_pnl - minimum_pnl > maximum_drawdown:
                    maximum_drawdown = (maximum_pnl - minimum_pnl)/maximum_pnl
            if pnl < minimum_pnl:
                minimum_pnl = pnl
        return maximum_drawdown

    def generate_hourly_returns(self) -> None:
        if len(self.hourly_indices) == 0:
            self.generate_hourly_indices()
        n = len(self.hourly_indices)
        for i in range(n - 1):
            self.hourly_returns.append((self.pnls[self.hourly_indices[i + 1]] -
                                        self.pnls[self.hourly_indices[i]]) /
                                       float(np.abs(self.pnls[self.hourly_indices[i]])))

    def get_number_of_trades(self) -> int:
        number_of_trades = 0
        for trade in self.trade_sides:
            number_of_trades = number_of_trades + np.abs(trade)
        return number_of_trades

    def generate_trades(self) -> pd.DataFrame:
        trades = self.data_frame[self.data_frame['trade_side'] != 0]
        return trades
