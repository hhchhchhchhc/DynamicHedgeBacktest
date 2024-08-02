from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import numpy
import pandas

from deribit_marketdata import Currency, Market
from utils.blackscholes import black_scholes
import numpy as np
import pandas as pd


class CashFlow:
    def __init__(self, settlement_time: datetime, currency: Currency, notional: float, drop_from_pv: bool=True):
        self.settlement_time: datetime = settlement_time
        self.currency: Currency = currency
        self.drop_from_pv: bool = drop_from_pv
        self.notional: float = notional

    def value(self, market: Market):
        return self.notional * (market.spot if self.currency == market.currency else 1)


class Instrument(ABC):
    """everything is coin margined so beware confusions:
    self.notional is in USD, except cash.
    all greeks are in coin and have signature: (self,market,optionnal **kwargs)
    +ve Delta still means long coin, though, and is in USD"""

    def __init__(self):
        self.symbol = None

    @abstractmethod
    def pv(self, market):
        raise NotImplementedError


class Cash(Instrument):
    greeks_available = ['delta']
    def __init__(self, currency: Currency):
        super().__init__()
        self.symbol = f'{currency}-CASH:'
        self.currency: Currency = currency

    def pv(self, market):
        return market.spot

    def delta(self, market):
        return 1


class InverseFuture(Instrument):
    """a long perpetual (size in USD, strike in coinUSD) is a short USDcoin(notional in USD, 1/strike in USDcoin
    what we implement is more like a fwd..
    """
    greeks_available = ['delta', 'theta']  # ,'IR01'

    def __init__(self, underlying: str, maturity: datetime, settlement_currency: Currency):
        super().__init__()
        self.maturity: datetime = maturity  # timestamp. TODO: in fact only fri 8utc for 1w,2w,1m,and 4 IMM
        self.settlement_currency: Currency = settlement_currency
        self.symbol: str = f'{underlying}-FUTURE:' + str(pd.to_datetime(maturity, unit='s'))

    def pv(self, market):
        return 0

    def delta(self, market, maturity_timestamp=None):
        """for a unit move"""
        if self.maturity <= market.t:
            return 0
        if maturity_timestamp and abs(maturity_timestamp - self.maturity) > timedelta(minutes=1):
            return 0.0

        fwd = market.fwdcurve.interpolate(self.maturity)
        return fwd / market.spot

    def theta(self, market):
        """for 1d"""
        if self.maturity <= market.t:
            return 0
        T = (self.maturity - market.t).total_seconds() / 3600 / 24 / 365.25
        fwd = market.fwdcurve.interpolate(self.maturity)
        return np.log(market.spot / fwd) / T / 24 / 365.25

    def margin_value(self, market):
        '''
        https://www.deribit.com/kb/futures
        Initial margin = 4% + (Position Size in BTC) * 0.005%
        Maintenance Margin = 2% + (Position Size in BTC) * 0.005%
        '''
        return -4e-2

    def cash_flow(self, market: Market, prev_market: Market) -> list[CashFlow]:
        margin_payment = market.fwdcurve.interpolate(self.maturity) - prev_market.fwdcurve.interpolate(self.maturity)
        return [CashFlow(settlement_time=market.t,
                         currency=self.settlement_currency,
                         notional=margin_payment / market.spot,
                         drop_from_pv=True)]


class InversePerpetual(Instrument):
    """a long perpetual (size in USD, strike in coinUSD) is a short USDcoin(notional in USD, 1/strike in USDcoin
    what we implement is more like a fwd.."""
    greeks_available = ['delta']

    def __init__(self, underlying: str, settlement_currency: Currency):
        super().__init__()
        self.symbol: str = f'{underlying}-PERPETUAL:'
        self.settlement_currency: Currency = settlement_currency

    def pv(self, market):
        return 0

    def delta(self, market):
        """for a unit move"""
        return 1

    def cash_flow(self, market: Market, prev_market: Market) -> list[CashFlow]:
        funding_payment = - market.spot * market.funding_rate * (market.t - prev_market.t).total_seconds() / 3600 / 8 / 365.25
        margin_payment = market.spot - prev_market.spot
        return [CashFlow(settlement_time=market.t,
                         currency=self.settlement_currency,
                         notional=(funding_payment + margin_payment) / market.spot,
                         drop_from_pv=False)]

    def margin_value(self, market):
        '''
        https://www.deribit.com/kb/futures
        Initial margin = 4% + (Position Size in BTC) * 0.005%
        Maintenance Margin = 2% + (Position Size in BTC) * 0.005%
        '''
        return -4e-2


class Option(Instrument):
    """a coinUSD call(size in coin, strike in coinUSD) is a USDcoin put(strike*size in USD,1/strike in USDTBC)
    notional in USD, strike in coin, maturity in sec, callput = C or P """
    greeks_available = ['delta', 'gamma', 'vega', 'theta']  # , 'IR01']

    def __init__(self, underlying: str, strike_coinUSD: float, maturity: datetime, call_put: str, settlement_currency: Currency):
        super().__init__()
        self.strike = strike_coinUSD
        self.maturity = maturity  # TODO: in fact only 8utc for 1d,2d, fri 1w,2w,3w,1m,2m,3m and 4 IMM
        if call_put == 'C':
            self.call_put = 'P'
        elif call_put == 'P':
            self.call_put = 'C'
        elif call_put == 'S':
            self.call_put = 'S'
        else:
            raise ValueError
        self.symbol = (
                f'{underlying}-{call_put}:'
                + str(pd.to_datetime(maturity, unit='s'))
                + '+'
                + str(int(strike_coinUSD))
        )
        self.settlement_currency: Currency = settlement_currency

    def pv(self, market):
        '''pv in usd for 1 coin notional'''
        if self.maturity <= market.t:
            return 0
        T = (self.maturity - market.t).total_seconds() / 3600 / 24 / 365.25
        fwd = market.fwdcurve.interpolate(self.maturity)
        vol = market.vol.interpolate(self.strike, self.maturity)
        return black_scholes.pv(fwd, self.strike, vol, T, self.call_put)

    def delta(self, market, maturity_timestamp=None):
        '''for unit move of spot
        parallel delta by default.
        delta by tenor is maturity_timestamp (within 1min)'''
        if self.maturity <= market.t:
            return 0
        if maturity_timestamp and abs(maturity_timestamp - self.maturity) > timedelta(minutes=1):
            return 0.0
        T = (self.maturity - market.t).total_seconds() / 3600 / 24 / 365.25
        fwd = market.fwdcurve.interpolate(self.maturity)
        vol = market.vol.interpolate(self.strike, self.maturity)
        return fwd / market.spot * black_scholes.delta(fwd, self.strike, vol, T, self.call_put)

    def gamma(self, market, maturity_timestamp=None):
        '''delta move for unit move of spot'''
        if self.maturity <= market.t:
            return 0
        if maturity_timestamp and abs(maturity_timestamp - self.maturity) > timedelta(minutes=1):
            return 0.0
        T = (self.maturity - market.t).total_seconds() / 3600 / 24 / 365.25
        fwd = market.fwdcurve.interpolate(self.maturity)
        vol = market.vol.interpolate(self.strike, self.maturity)
        return (fwd / market.spot)**2 * black_scholes.gamma(fwd, self.strike, vol, T, self.call_put)

    def vega(self, market, maturity_timestamp=None):
        '''vega by tenor maturity_timestamp(within 1s)'''
        if self.maturity <= market.t:
            return 0
        if maturity_timestamp and abs(maturity_timestamp - self.maturity) > timedelta(minutes=1):
            return 0.0
        T = (self.maturity - market.t).total_seconds() / 3600 / 24 / 365.25
        fwd = market.fwdcurve.interpolate(self.maturity)
        vol = market.vol.interpolate(self.strike, self.maturity)
        return black_scholes.vega(fwd, self.strike, vol, T, self.call_put)

    def theta(self, market) -> float:
        '''for 1d'''
        if self.maturity <= market.t:
            return 0
        T = (self.maturity - market.t).total_seconds() / 3600 / 24 / 365.25
        fwd = market.fwdcurve.interpolate(self.maturity)
        vol = market.vol.interpolate(self.strike, self.maturity)
        return black_scholes.theta(fwd, self.strike, vol, T, self.call_put)

    def cash_flow(self, market: Market, prev_market: Market) -> list[CashFlow]:
        if prev_market.t < self.maturity <= market.t:
            cash_flow = 0
            if self.call_put == 'C':
                cash_flow = max(0, (market.spot - self.strike))
            elif self.call_put == 'P':
                cash_flow = max(0, (self.strike - market.spot))
            elif self.call_put == 'S':
                cash_flow = abs(self.strike - market.spot)
            return [CashFlow(settlement_time=market.t,
                             currency=self.settlement_currency,
                             notional=cash_flow / market.spot,
                             drop_from_pv=True)]
        else:
            return []

    def margin_value(self, market, notional):
        if self.maturity <= market.t:
            return 0
        fwd = market.spot
        intrisinc = max(0.15 + (fwd - self.strike) * (1 if self.call_put == 'C' else -1) * np.sign(notional),
                        0.1) / market.spot
        mark = self.pv(market)
        vega = self.vega(market) * 0.45 * np.power(30 * 24 * 3600 / (self.maturity - market.t).total_seconds(), 0.3)
        pin = 0.01 / market.spot if notional < 0 else 0
        return - (intrisinc + mark + vega + pin)