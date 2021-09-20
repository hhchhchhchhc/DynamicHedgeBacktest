import datetime
import math
from typing import Optional

import pandas as pd


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

        self.trade_datetimes = []
        self.trade_sides = []
        self.trade_sizes = []
        self.trade_prices = []

    def run(self) -> None:
        n = len(self.data_frame)
        progress_pct = 0
        for i in range(n):
            if math.floor(100 * i / n) > progress_pct:
                progress_pct = progress_pct + 10
                print(str(progress_pct) + '% ', end='')
            date_time = self.data_frame['datetime'].iloc[i]
            bid_price = self.data_frame['bidPrice'].iloc[i]
            ask_price = self.data_frame['askPrice'].iloc[i]
            if i < n - 1:
                self.mid = 0.5 * (bid_price + ask_price)
                self.market_mids.append(self.mid)
                if self.bid > 0 and ask_price <= self.bid:
                    self.position = self.position + self.target_pos
                    self.cash = self.cash - self.target_pos * self.bid
                    self.trade_datetimes.append(date_time)
                    self.trade_sides.append(1)
                    self.trade_sizes.append(self.target_pos)
                    self.trade_prices.append(self.bid)
                    self.bid = 0
                if 0 < self.ask <= bid_price:
                    self.position = self.position - self.target_pos
                    self.cash = self.cash + self.target_pos * self.ask
                    self.trade_datetimes.append(date_time)
                    self.trade_sides.append(-1)
                    self.trade_sizes.append(self.target_pos)
                    self.trade_prices.append(self.ask)
                    self.ask = 0
                self.positions.append(self.position)
                self.cashs.append(self.cash)
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
                    self.cashs.append(self.cash)
                    self.trade_datetimes.append(date_time)
                    self.trade_sides.append(-1)
                    self.trade_sizes.append(self.position)
                    self.trade_prices.append(bid_price)
                if self.position < 0:
                    self.pnl = self.pnl - self.position * ask_price
                    self.cashs.append(self.cash)
                    self.trade_datetimes.append(date_time)
                    self.trade_sides.append(1)
                    self.trade_sizes.append(-self.position)
                    self.trade_prices.append(ask_price)
                self.position = 0
                self.positions.append(self.position)
            self.bids.append(self.bid)
            self.asks.append(self.ask)
            self.pnl = self.cash + self.position * self.mid
            self.pnls.append(self.pnl)

    def get_trades(self):
        data_frame = pd.DataFrame({'datetime': self.trade_datetimes,
                                   'side': self.trade_sides,
                                   'size': self.trade_sizes,
                                   'price': self.trade_prices})
        return data_frame

    def get_total_volume_traded(self) -> int:
        volume = sum(self.trade_sizes)
        return volume

    def get_yield(self) -> float:
        y = 1e2 * self.pnl / self.get_total_volume_traded()
        return y

    def get_number_of_trades(self) -> int:
        number_of_trades = len(self.trade_datetimes)
        return number_of_trades

    def load_market_data(self, directory: str, start_date: datetime.date, number_of_days: int) -> None:
        market_data = pd.DataFrame()
        for d in range(number_of_days):
            date = start_date
            date = date + datetime.timedelta(days=d)
            date_string = date.strftime('%Y%m%d')
            data_frame = pd.read_csv(directory + 'market_data_xrpusdt_' + date_string + '.csv')
            market_data = market_data.append(data_frame)
        self.data_frame = market_data

    def add_results_to_data_frame(self):
        self.data_frame['market_mids'] = self.market_mids
        self.data_frame['bids'] = self.bids
        self.data_frame['asks'] = self.asks
        self.data_frame['positions'] = self.positions
        self.data_frame['cashs'] = self.cashs
        self.data_frame['skews'] = self.skews
        self.data_frame['pnls'] = self.pnls

    def generate_trades(self) -> pd.DataFrame:
        trades = pd.DataFrame()
        trades['trade_datetimes'] = self.trade_datetimes
        trades['trade_sides'] = self.trade_sides
        trades['trade_sizes'] = self.trade_sizes
        return trades
