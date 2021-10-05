import datetime
import random
import numpy as np
import pandas as pd
import SimpleMarketMaking.Clean.tools


class Backtest:
    def __init__(self, symbol: str, date: datetime.date, k: float, gamma: float, horizon: int) -> None:
        self.symbol = symbol
        date_string = date.strftime('%Y%m%d')
        self.tobs = pd.read_csv('C:/Users/Tibor/Data/formatted/tob/' + date_string +
                                '_Binance_BTCUSDT_tob.csv')
        self.trades = pd.read_csv('C:/Users/Tibor/Data/formatted/trades/' + date_string +
                                  '_Binance_BTCUSDT_trades.csv')
        self.k = k
        self.gamma = gamma
        self.horizon = horizon
        self.tick_size = SimpleMarketMaking.Clean.tools.get_tick_size_from_symbol(self.symbol)
        self.step_size = SimpleMarketMaking.Clean.tools.get_step_size_from_symbol(self.symbol)
        self.time_bar_start_timestamp = None
        self.price_buffer = []
        self.size_buffer = []
        self.bid = None
        self.ask = None
        self.position = 0
        self.sigma_squared = None
        self.market_bid = None
        self.market_ask = None
        self.cash = 0
        self.backtest_timestamps = []
        self.backtest_positions = []
        self.backtest_pnls = []
        self.backtest_bids = []
        self.backtest_asks = []
        self.spread = None
        self.skew = None
        self.backtest_spreads = []
        self.backtest_skews = []
        self.backtest_market_bids = []
        self.backtest_market_asks = []
        self.backtest_sigma_squareds = []
        self.vwap_buffer = []

    def run(self) -> pd.DataFrame:
        trades_index = 0
        tobs_index = 0
        trades_n = len(self.trades.index)
        tobs_n = len(self.tobs.index)
        progress_pct = 0
        while trades_index < trades_n and tobs_index < tobs_n:
            if np.floor(100 * trades_index / trades_n) > progress_pct:
                progress_pct = progress_pct + 5
                print(str(progress_pct) + '% ', end='')
            trades_timestamp = self.trades['timestamp_millis'].iloc[trades_index]
            tobs_timestamp = self.tobs['timestamp_millis'].iloc[tobs_index]
            is_trade_event = False
            if trades_timestamp < tobs_timestamp:
                is_trade_event = True
            elif tobs_timestamp < trades_timestamp:
                is_trade_event = False
            elif trades_timestamp == tobs_timestamp:
                is_trade_event = random.choice([False, True])
            if is_trade_event:
                self._trade_event(self.trades.iloc[trades_index])
                trades_index = trades_index + 1
            else:
                self._tob_event(self.tobs.iloc[tobs_index])
                tobs_index = tobs_index + 1
        print()
        backtest_results = pd.DataFrame()
        backtest_results['timestamp_millis'] = self.backtest_timestamps
        backtest_results['position'] = np.multiply(self.backtest_positions, self.step_size)
        backtest_results['pnl'] = np.multiply(self.backtest_pnls, self.tick_size)
        backtest_results['bid'] = list(map(lambda x: None if x is None else x * self.tick_size, self.backtest_bids))
        backtest_results['ask'] = list(map(lambda x: None if x is None else x * self.tick_size, self.backtest_asks))
        backtest_results['market_bid'] = np.multiply(self.backtest_market_bids, self.tick_size)
        backtest_results['market_ask'] = np.multiply(self.backtest_market_asks, self.tick_size)
        backtest_results['spread'] = list(map(
            lambda x: None if x is None else x * self.tick_size, self.backtest_spreads))
        backtest_results['skew'] = list(map(
            lambda x: None if x is None else x * self.tick_size, self.backtest_skews))
        backtest_results['sigma_squared'] = list(map(
            lambda x: None if x is None else x * self.tick_size * self.tick_size, self.backtest_sigma_squareds))
        return backtest_results

    def _trade_event(self, trade: pd.Series) -> None:
        if self.time_bar_start_timestamp is None:
            self.time_bar_start_timestamp = trade.timestamp_millis
        if trade.timestamp_millis - self.time_bar_start_timestamp >= 1000:
            self._end_of_bar()
            self.time_bar_start_timestamp = trade.timestamp_millis
        self.price_buffer.append(trade.price)
        self.size_buffer.append(trade.size)

    def _tob_event(self, tob: pd.Series) -> None:
        if self.time_bar_start_timestamp is None:
            self.time_bar_start_timestamp = tob.timestamp_millis
        if tob.timestamp_millis - self.time_bar_start_timestamp >= 1000:
            self._end_of_bar()
            self.time_bar_start_timestamp = tob.timestamp_millis
        self.market_bid = tob.bid_price
        self.market_ask = tob.ask_price
        if self.bid is not None and self.market_ask is not None and self.bid >= self.market_ask:
            self._passive_buy()
        if self.ask is not None and self.market_bid is not None and self.ask <= self.market_bid:
            self._passive_sell()

    def _end_of_bar(self) -> None:
        if len(self.price_buffer) > 0:
            vwap = np.average(self.price_buffer, weights=self.size_buffer)
            self.vwap_buffer.append(vwap)
            if len(self.vwap_buffer) > self.horizon:
                self.vwap_buffer.pop()
            if len(self.vwap_buffer) >= self.horizon:
                self.sigma_squared = np.diff(self.vwap_buffer).var()
            if self.sigma_squared is not None:
                self.skew = (self.step_size * self.position) * self.gamma * (
                        self.tick_size * self.tick_size * self.sigma_squared) * self.horizon / self.tick_size
                self.spread = (2 / self.gamma) * np.log(1 + (self.gamma / self.k)) / self.tick_size
                self.bid = int(vwap - (self.spread / 2) - self.skew)
                self.ask = int(vwap + (self.spread / 2) - self.skew)
                if self.bid is not None and self.market_ask is not None and self.bid >= self.market_ask:
                    self._aggressive_buy()
                if self.ask is not None and self.market_bid is not None and self.ask <= self.market_bid:
                    self._aggressive_sell()
        self.price_buffer = []
        self.size_buffer = []
        self._record_backtest_results()

    def _passive_buy(self) -> None:
        quote_size = int((np.log(1 + (self.gamma / self.k)) / (10 * self.gamma * self.gamma * (
                self.tick_size * self.tick_size * self.sigma_squared) * self.horizon)) / self.step_size)
        self.cash = self.cash - (quote_size * self.bid)
        self.position = self.position + quote_size
        self.bid = None

    def _passive_sell(self) -> None:
        quote_size = int((np.log(1 + (self.gamma / self.k)) / (10 * self.gamma * self.gamma * (
                self.tick_size * self.tick_size * self.sigma_squared) * self.horizon)) / self.step_size)
        self.cash = self.cash + (quote_size * self.ask)
        self.position = self.position - quote_size
        self.ask = None

    def _aggressive_buy(self) -> None:
        quote_size = int((np.log(1 + (self.gamma / self.k)) / (10 * self.gamma * self.gamma * (
                self.tick_size * self.tick_size * self.sigma_squared) * self.horizon)) / self.step_size)
        self.cash = self.cash - (quote_size * self.market_ask)
        self.position = self.position + quote_size
        self.bid = None

    def _aggressive_sell(self) -> None:
        quote_size = int((np.log(1 + (self.gamma / self.k)) / (10 * self.gamma * self.gamma * (
                self.tick_size * self.tick_size * self.sigma_squared) * self.horizon)) / self.step_size)
        self.cash = self.cash + (quote_size * self.market_bid)
        self.position = self.position - quote_size
        self.ask = None

    def _record_backtest_results(self) -> None:
        self.backtest_timestamps.append(self.time_bar_start_timestamp)
        self.backtest_positions.append(self.position)
        pnl = self.cash + (self.position * ((self.market_bid + self.market_ask) / 2))
        self.backtest_pnls.append(pnl)
        self.backtest_bids.append(self.bid)
        self.backtest_asks.append(self.ask)
        self.backtest_market_bids.append(self.market_bid)
        self.backtest_market_asks.append(self.market_ask)
        self.backtest_spreads.append(self.spread)
        self.backtest_skews.append(self.skew)
        self.backtest_sigma_squareds.append(self.sigma_squared)
