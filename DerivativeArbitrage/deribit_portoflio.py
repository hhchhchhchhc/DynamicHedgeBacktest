import pandas as pd

from deribit_history import *
import numpy as np
import scipy
import math

class black_scholes:
    @staticmethod
    def d1(S, K, V, T):
        return (math.log(S / float(K)) + (V ** 2 / 2) * T) / (V * math.sqrt(T))

    @staticmethod
    def d2(S, K, V, T):
        return black_scholes.d1(S, K, V, T) - (V * math.sqrt(T))

    @staticmethod
    def pv(S, K, V, T, cp):
        if cp == 'C':
            return S * scipy.stats.norm.cdf(black_scholes.d1(S, K, V, T)) - K * scipy.stats.norm.cdf(
                black_scholes.d2(S, K, V, T))
        else:
            return K * scipy.stats.norm.cdf(-black_scholes.d2(S, K, V, T)) - S * scipy.stats.norm.cdf(
                -black_scholes.d1(S, K, V, T))

    @staticmethod
    def delta(S, K, V, T, cp):
        if cp == 'C':
            delta = scipy.stats.norm.cdf(black_scholes.d1(S, K, V, T))
        elif cp == 'P':
            delta = scipy.stats.norm.cdf(black_scholes.d1(S, K, V, T)) - 1
        else:
            delta = 1
        return delta

    @staticmethod
    def vega(S, K, V, T):
        vega = (S * math.sqrt(T) * scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T))) / 100
        return vega

    @staticmethod
    def theta(S, K, V, T):
        theta = -((S * V * scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T))) / (2 * math.sqrt(T))) / god
        return theta

    @staticmethod
    def gamma(S, K, V, T):
        gamma = scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T)) / (S * V * math.sqrt(T))
        return gamma

delta_hedge_slippage = 0.0
vega_slippage = 0.0

class Instrument:
    '''everything is BTC margined so beware confusions:
    self.notional is in USD (except for cash), self.mtm always BTC.
    +ve Delta still means long BTC, though.
    we don't support USDC perp'''

    def __init__(self, notional):
        self.notional = notional
        self.mtm = 0

    def process_cash_flows(self, market):
        previous_mtm = self.mtm
        self.mtm = self.pv(market)
        return self.mtm - previous_mtm
    def pv(self, market):
        raise NotImplementedError
    def delta(self, market):
        '''for 1% up in BTCUSD'''
        raise NotImplementedError
    def cash_flow(self, market):
        raise NotImplementedError
    def margin_value(self, market):
        raise NotImplementedError


class Cash(Instrument):
    '''BTC cash'''

    def __init__(self, notional_BTC):
        self.notional = notional_BTC
        self.mtm = notional_BTC

    def pv(self,market):
        return self.notional
    def delta(self,market):
        return 0
    def cash_flow(self,market):
        return self.notional * market['borrow'] * (market['t'] - market['prev_t'])/3600/8
    def margin_value(self,market):
        return self.notional


class InversePerpetual(Instrument):
    '''a long perpetual (size in USD, strike in BTCUSD) is a short USDBTC(notional in USD, 1/strike in USDBTC
    what we implement is more like a fwd..'''

    def __init__(self, notional_USD, strike_BTCUSD):
        self.notional = notional_USD
        self.strike = 1 / strike_BTCUSD
        self.mtm = 0

    def pv(self, market):
        return self.notional * (self.strike - 1 / market['fwd'])
    def delta(self, market):
        return self.notional
    def cash_flow(self, market):
        '''accrues every millisecond.
        8h Funding Rate = Maximum (0.05%, Premium Rate) + Minimum (-0.05%, Premium Rate)
        TODO: doesn't handle intra period changes'''
        return - self.notional * market['fundingRate'] * (market['t'] - market['prev_t'])/3600/8
    def margin_value(self, market):
        '''TODO: approx, in fact it's converted at spot not fwd'''
        return -6e-3 * market['fwd'] * self.notional

class Option(Instrument):
    '''a BTCUSD call(size in btc, strike in BTCUSD) is a USDBTC put(strike*size in USD,1/strike in USDTBC)
    notional in USD, strike in BTC, maturity datetime - > sec, callput = C or P '''

    def __init__(self, notional_USD, strike_BTCUSD, maturity, call_put):
        self.notional = notional_USD
        self.strike = 1 / strike_BTCUSD
        self.maturity = maturity
        self.call_put = 'P' if call_put == 'C' else 'C'
        self.mtm = 0 # temporary

    def pv(self, market):
        return self.notional * black_scholes.pv(1 / market['fwd'], self.strike, market['vol'],
                                                (self.maturity - market['t']) / 3600 / 24 / 365.25, self.call_put) if self.maturity > market['t'] else 0
    def delta(self, market):
        return self.notional * black_scholes.delta(1 / market['fwd'], self.strike, market['vol'],
                                                   (self.maturity - market['t']) / 3600 / 24 / 365.25, self.call_put) if self.maturity > market['t'] else 0
    def gamma(self, market):
        return self.notional * black_scholes.gamma(1 / market['fwd'], self.strike, market['vol'],
                                                   (self.maturity - market['t']) / 3600 / 24 / 365.25) if self.maturity > market['t'] else 0
    def vega(self, market):
        return self.notional * black_scholes.vega(1 / market['fwd'], self.strike, market['vol'],
                                                  (self.maturity - market['t']) / 3600 / 24 / 365.25) if self.maturity > market['t'] else 0
    def cash_flow(self, market):
        '''cash settled in BTC'''
        if market['prev_t'] < self.maturity and market['t'] >= self.maturity:
            return max(0,(1 if self.call_put == 'C' else -1)*(1 / market['fwd'] - self.strike))
        else:
            return 0
    def margin_value(self, market):
        return - (self.notional * max(0.15 - (1 / market['fwd'] - self.strike) * (1 if self.call_put == 'C' else -1),
                                      0.1) + self.mtm) if self.maturity > market['t'] else 0

class Portfolio(Instrument):
    def __init__(self,notional_BTC):
        self.instruments = [Cash(notional_BTC)]
        self.notional = 1  # scaler, usually stays 1
        self.mtm = 1

    def greek(self, greek, market):
        return self.notional * sum([getattr(instrument, greek)(market) for instrument in self.instruments if hasattr(instrument,greek)])

    def add(self, instrument, price_BTC, mtm):
        assert not isinstance(instrument, Cash)
        instrument.notional /= self.notional
        instrument.mtm = mtm
        self.instruments += [instrument]
        self.instruments[0].notional -= price_BTC / self.notional

    def delta_hedge(self, hedge_instrument, market):
        portfolio_delta = self.greek('delta',market)
        hedge_instrument_delta = hedge_instrument.delta(market)
        if hedge_instrument in self.instruments:
            hedge_instrument.notional -= portfolio_delta / hedge_instrument_delta / self.notional
            self.instruments[0].notional -= portfolio_delta / hedge_instrument_delta * delta_hedge_slippage
        else:
            hedge_instrument.notional = -portfolio_delta / hedge_instrument_delta / self.notional
            self.add(hedge_instrument,0,-delta_hedge_slippage)

    def process_cash_flows(self, market):
        '''uses instruments.mtm and not self.mtm. Still maintains it though'''
        margin_call = self.notional * sum(
            [getattr(instrument, 'process_cash_flows')(market) for instrument in self.instruments])
        self.mtm = self.notional * sum(instrument.mtm for instrument in self.instruments)

        cash_instrument = self.instruments[0]
        cash_instrument.notional += margin_call / self.notional
        cash_instrument.notional += self.greek('cash_flow',market)

        return 0

if __name__ == "__main__":
    history = deribit_history_main('just use',['BTC'],'deribit','cache')[0]
    tenor_columns = history.filter(like='rate/T').columns
    mark_columns = history.filter(like='mark/c').columns
    mark_columns = [r for r in mark_columns if 'PERPETUAL' not in r]

    history.reset_index(inplace=True)
    history['1mFwd'] = history.apply(lambda t: scipy.interpolate.interp1d(
        x=np.array([t[tenor] for tenor in tenor_columns if not pd.isna(t[tenor])]),
        y=np.array([t[rate] for rate in mark_columns if not pd.isna(t[rate])]),
        kind='linear', fill_value='extrapolate')(1/12), axis=1)
    history.rename(columns={'index':'t',
                            '1mFwd':'fwd',
                            'BTC/volindex/c':'vol',
                            'BTC-PERPETUAL/indexes/c':'spot',
                            'BTC-PERPETUAL/rate/funding':'fundingRate'},inplace=True)
    history['t'] = history['t'].apply(lambda t:t.timestamp())
    history['prev_t'] = history['t'].shift(1)
    history['vol'] = history['vol']/100
    history['borrow'] = 0
    history.ffill(limit=2,inplace=True)

    portoflio = Portfolio(notional_BTC=1)

    mkt_0 = history.iloc[0]
    genesis_call = Option(notional_USD=-mkt_0['fwd'],
                         strike_BTCUSD=mkt_0['fwd'],
                         maturity=mkt_0['t'] + 3600 * 24 * 30,
                         call_put='C')
    portoflio.add(genesis_call,
                  price_BTC= genesis_call.pv(mkt_0),
                  mtm= genesis_call.pv(mkt_0) - vega_slippage * genesis_call.vega(mkt_0))
    genesis_put = Option(notional_USD=-mkt_0['fwd'],
                         strike_BTCUSD=mkt_0['fwd'],
                         maturity=mkt_0['t'] + 3600 * 24 * 30,
                         call_put='P')
    portoflio.add(genesis_put,
                  price_BTC= genesis_put.pv(mkt_0),
                  mtm= genesis_put.pv(mkt_0) - vega_slippage * genesis_call.vega(mkt_0))
    future_hedge = InversePerpetual(notional_USD=1,strike_BTCUSD=mkt_0['fwd'])

    display = pd.DataFrame()
    for _,mkt in history.tail(-1).iterrows():
        new_data = pd.concat([mkt[['prev_t','t','spot','fwd','vol','fundingRate','borrow']],
                             pd.Series({greek: portoflio.greek(greek=greek,market=mkt) for greek in ['pv','delta']}),
                             pd.Series({instrument.__class__: instrument.notional for instrument in portoflio.instruments})])
        new_data.name = pd.to_datetime(mkt['t'], unit='s')
        display = pd.concat([display,new_data],axis=1)
        portoflio.process_cash_flows(mkt)
        portoflio.delta_hedge(future_hedge,mkt)

    display.T.to_excel('Runtime/runs/deribit.xlsx')

if False:
    btc = pd.read_csv("C:/Users/david/pyCharmProjects/SystematicCeFi/DerivativeArbitrage/Runtime/Deribit_Mktdata_database/deribit_options_chain_2019-07-01_BTC.csv")
    btc['expiry'] = btc['expiration'].apply(lambda t: pd.to_datetime(int(t), unit='us'))
    btc['timestamp'] = btc['timestamp'].apply(lambda t: pd.to_datetime(int(t), unit='us'))
    btc['dist_2_fwd'] = btc.apply(lambda f: np.abs(np.log(f['underlying_price']/f['strike_price'])),axis=1)
    smiles = btc.groupby(by=['expiry', 'timestamp'])[['timestamp','delta','mark_iv']]
    #.sort_values(by='strike_price',
                                                             #key=lambda f: np.abs(f['underlying_price'] / f['strike']))[-1]
