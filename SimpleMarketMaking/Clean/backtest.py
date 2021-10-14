import datetime
import random
import numpy as np
import pandas as pd
import tools


class Backtest:
    def _get_tobs_from_csvs(self, start_date: datetime.date, number_of_days: int) -> pd.DataFrame:
        tobs = pd.DataFrame()
        for days in range(number_of_days):
            date = start_date + datetime.timedelta(days=days)
            date_string = date.strftime('%Y%m%d')
            tob = pd.read_csv(
                'C:/Users/Tibor/Data/formatted/tobs/' + date_string + '_Binance_' + self.symbol + '_tobs.csv')
            tobs = tobs.append(tob)
        return tobs

    def _get_trades_from_csvs(self, start_date: datetime.date, number_of_days: int) -> pd.DataFrame:
        trades = pd.DataFrame()
        for days in range(number_of_days):
            date = start_date + datetime.timedelta(days=days)
            date_string = date.strftime('%Y%m%d')
            trade = pd.read_csv(
                'C:/Users/Tibor/Data/formatted/trades/' + date_string + '_Binance_' + self.symbol + '_trades.csv')
            trades = trades.append(trade)
        return trades

    def __init__(self, symbol: str, start_date: datetime.date, number_of_days: int, c: int) -> None:
        self.symbol = symbol
        self.c = c
        self.horizon = 60
        self.tobs = self._get_tobs_from_csvs(start_date, number_of_days)
        self.trades = self._get_trades_from_csvs(start_date, number_of_days)
        self.tick_size = tools.get_tick_size_from_symbol(self.symbol)
        self.step_size = tools.get_step_size_from_symbol(self.symbol)
        self.quote_size = 1 / self.step_size
        self.max_position = 10 / self.step_size
        self.time_bar_start_timestamp = None
        self.price_buffer = []
        self.size_buffer = []
        self.bid = None
        self.ask = None
        self.position = 0
        self.sigma = None
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
        self.backtest_sigmas = []
        self.vwap_buffer = []
        self.backtest_cumulative_volumes = []
        self.backtest_cumulative_volume_dollars = []
        self.volume_traded_buffer = []
        self.volume_traded_dollar_buffer = []
        self.cumulative_volume = 0
        self.cumulative_volume_dollar = 0

    def run(self) -> pd.DataFrame:
        trades_index = 0
        tobs_index = 0
        trades_n = len(self.trades.index)
        tobs_n = len(self.tobs.index)
        progress_pct = 0
        while trades_index < trades_n and tobs_index < tobs_n:
            if np.floor(100 * trades_index / trades_n) > progress_pct:
                progress_pct = progress_pct + 1
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
        backtest_results['cumulative_volume'] = np.multiply(self.backtest_cumulative_volumes, self.step_size)
        backtest_results['cumulative_volume_dollar'] = np.multiply(
            self.backtest_cumulative_volume_dollars, self.tick_size * self.step_size)
        backtest_results['position'] = list(map(lambda x: None if x is None else x * self.step_size,
                                                self.backtest_positions))
        backtest_results['pnl'] = np.multiply(self.backtest_pnls, self.tick_size * self.step_size)
        backtest_results['bid'] = list(map(lambda x: None if x is None else x * self.tick_size, self.backtest_bids))
        backtest_results['ask'] = list(map(lambda x: None if x is None else x * self.tick_size, self.backtest_asks))
        backtest_results['market_bid'] = np.multiply(self.backtest_market_bids, self.tick_size)
        backtest_results['market_ask'] = np.multiply(self.backtest_market_asks, self.tick_size)
        backtest_results['spread'] = list(
            map(lambda x: None if x is None else x * self.tick_size, self.backtest_spreads))
        backtest_results['skew'] = list(
            map(lambda x: None if x is None else x * self.tick_size, self.backtest_skews))
        backtest_results['sigma'] = list(
            map(lambda x: None if x is None else x * self.tick_size, self.backtest_sigmas))
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
                self.vwap_buffer.pop(0)
            if len(self.vwap_buffer) >= self.horizon:
                self.sigma = np.diff(self.vwap_buffer).std()
            if self.sigma is not None:
                self.skew = (self.c * (self.sigma * self.tick_size) * self.position / (
                    self.max_position)) / self.tick_size
                self.spread = self.c * self.sigma
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
        quote_size = self.quote_size
        self.cash = self.cash - (quote_size * self.bid)
        self.position = self.position + quote_size
        self.volume_traded_buffer.append(quote_size)
        self.volume_traded_dollar_buffer.append(quote_size * self.bid)
        self.bid = None

    def _passive_sell(self) -> None:
        quote_size = self.quote_size
        self.cash = self.cash + (quote_size * self.ask)
        self.position = self.position - quote_size
        self.volume_traded_buffer.append(quote_size)
        self.volume_traded_dollar_buffer.append(quote_size * self.ask)
        self.ask = None

    def _aggressive_buy(self) -> None:
        quote_size = self.quote_size
        self.cash = self.cash - (quote_size * self.market_ask)
        self.position = self.position + quote_size
        self.volume_traded_buffer.append(quote_size)
        self.volume_traded_dollar_buffer.append(quote_size * self.market_ask)
        self.bid = None

    def _aggressive_sell(self) -> None:
        quote_size = self.quote_size
        self.cash = self.cash + (quote_size * self.market_bid)
        self.position = self.position - quote_size
        self.volume_traded_buffer.append(quote_size)
        self.volume_traded_dollar_buffer.append(quote_size * self.market_bid)
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
        self.backtest_sigmas.append(self.sigma)
        self.cumulative_volume = self.cumulative_volume + sum(self.volume_traded_buffer)
        self.backtest_cumulative_volumes.append(self.cumulative_volume)
        self.volume_traded_buffer = []
        self.cumulative_volume_dollar = self.cumulative_volume_dollar + sum(self.volume_traded_dollar_buffer)
        self.backtest_cumulative_volume_dollars.append(self.cumulative_volume_dollar)
        self.volume_traded_dollar_buffer = []
