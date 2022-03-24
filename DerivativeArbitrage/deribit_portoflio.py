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
    self.notional is in USD (except for cash), self.mark always BTC.
    +ve Delta still means long BTC, though.
    we don't support USDC perp'''
    def __init__(self,mark):
        self.mark = mark
    def process_margin_call(self, market):
        previous_mark = self.mark
        self.mark = self.pv(market)
        return self.mark - previous_mark

    def pv(self, market):
        raise NotImplementedError
    def delta(self, market):
        '''for 1% up in BTCUSD'''
        raise NotImplementedError
    def cash_flow(self, market):
        raise NotImplementedError
    def margin_value(self, market):
        raise NotImplementedError

class InversePerpetual(Instrument):
    '''a long perpetual (size in USD, strike in BTCUSD) is a short USDBTC(notional in USD, 1/strike in USDBTC
    what we implement is more like a fwd..'''
    def __init__(self, strike_BTCUSD, market):
        self.strike = 1 / strike_BTCUSD
        self.mark = self.pv(market)

    def pv(self, market):
        return market['df'] * (self.strike - 1 / market['fwd'])
    def delta(self, market):
        return market['df']
    def process_cash_flow(self, market):
        '''accrues every millisecond.
        8h Funding Rate = Maximum (0.05%, Premium Rate) + Minimum (-0.05%, Premium Rate)
        TODO: doesn't handle intra period changes'''
        return - market['fundingRate'] * (market['t'] - market['prev_t'])/3600/8
    def margin_value(self, market):
        '''TODO: approx, in fact it's converted at spot not fwd'''
        return -6e-3 * market['spot']

class Option(Instrument):
    '''a BTCUSD call(size in btc, strike in BTCUSD) is a USDBTC put(strike*size in USD,1/strike in USDTBC)
    notional in USD, strike in BTC, maturity datetime - > sec, callput = C or P '''

    def __init__(self, strike_BTCUSD, maturity, call_put,market):
        self.strike = 1 / strike_BTCUSD
        self.maturity = maturity
        self.call_put = 'P' if call_put == 'C' else 'C'
        self.mark = self.pv(market)

    def pv(self, market):
        return market['df'] * black_scholes.pv(1 / market['fwd'], self.strike, market['vol'],
                                                (self.maturity - market['t']) / 3600 / 24 / 365.25, self.call_put) if self.maturity > market['t'] else 0
    def delta(self, market):
        return market['df'] * black_scholes.delta(1 / market['fwd'], self.strike, market['vol'],
                                                   (self.maturity - market['t']) / 3600 / 24 / 365.25, self.call_put) if self.maturity > market['t'] else 0
    def gamma(self, market):
        return market['df'] * black_scholes.gamma(1 / market['fwd'], self.strike, market['vol'],
                                                   (self.maturity - market['t']) / 3600 / 24 / 365.25) if self.maturity > market['t'] else 0
    def vega(self, market):
        return market['df'] * black_scholes.vega(1 / market['fwd'], self.strike, market['vol'],
                                                  (self.maturity - market['t']) / 3600 / 24 / 365.25) if self.maturity > market['t'] else 0
    def theta(self, market):
        return market['df'] * black_scholes.theta(1 / market['fwd'], self.strike, market['vol'],
                                                  (self.maturity - market['t']) / 3600 / 24 / 365.25) if self.maturity > market['t'] else 0

    def process_cash_flow(self, market):
        '''cash settled in BTC'''
        if market['prev_t'] < self.maturity and market['t'] >= self.maturity:
            cash_flow = max(0,(1 if self.call_put == 'C' else -1)*(1 / market['fwd'] - self.strike))
            self.mark = 0
            return cash_flow
        else:
            return 0
    def margin_value(self, market):
        return - (max(0.15 - (1 / market['fwd'] - self.strike) * (1 if self.call_put == 'C' else -1),
                                      0.1) + self.mark) if self.maturity > market['t'] else 0

class Position:
    def __init__(self,instrument,notional_USD):
        self.instrument = instrument
        self.notional = notional_USD
    def apply(self, greek, market):
        return self.notional * getattr(self.instrument, greek)(market) if hasattr(self.instrument,greek) else 0

class Portfolio():
    def __init__(self,notional_BTC):
        self.cash = notional_BTC
        self.positions = []

    def apply(self, greek, market):
        result = sum([position.apply(greek,market) for position in self.positions])
        if greek == 'pv':
            result += self.cash
        return result

    def add(self, instrument, notional_USD, price_BTC, market=None):
        position = next(iter(x for x in self.positions if x.instrument == instrument), 'new_position')
        if position == 'new_position':
            # mark it
            if instrument.mark is None:
                if market:
                    instrument.mark = instrument.pv(market)
                else:
                    raise Exception("need a market to open a new position without preexisting mark")
            # add it
            position = Position(instrument,notional_USD)
            self.positions += [position]
        else:
            position.notional += notional_USD
        # pay slippage
        self.cash -= notional_USD * (price_BTC - instrument.mark)

    def delta_hedge(self, hedge_instrument, market):
        portfolio_delta = self.apply('delta',market)
        hedge_instrument_delta = hedge_instrument.delta(market)
        hedge_notional = -portfolio_delta / hedge_instrument_delta

        self.add(hedge_instrument,
                 hedge_notional,
                 hedge_instrument.mark + delta_hedge_slippage/market['spot'])

    def process_cash_flows(self, market):
        '''cash_flows before margin calls !!'''
        cash_flows = sum([position.notional * getattr(position.instrument, 'process_cash_flow')(market) 
                          for position in self.positions])
        margin_call = sum([position.notional * getattr(position.instrument, 'process_margin_call')(market) 
                           for position in self.positions])
        self.cash += cash_flows + margin_call

if __name__ == "__main__":

    ## get history
    history = deribit_history_main('just use',['BTC'],'deribit','cache')[0]

    # 1mFwd
    tenor_columns = history.filter(like='rate/T').columns
    mark_columns = history.filter(like='mark/c').columns
    mark_columns = [r for r in mark_columns if 'PERPETUAL' not in r]
    history['1mFwd'] = history.apply(lambda t: scipy.interpolate.interp1d(
        x=np.array([t[tenor] for tenor in tenor_columns if not pd.isna(t[tenor])]),
        y=np.array([t[rate] for rate in mark_columns if not pd.isna(t[rate])]),
        kind='linear', fill_value='extrapolate')(1/12), axis=1)

    ## format history
    history.reset_index(inplace=True)
    history.rename(columns={'index':'t',#TODO: more consistent with /o
                            '1mFwd':'fwd',
                            'BTC/volindex/c':'vol',
                            'BTC-PERPETUAL/indexes/c':'spot',
                            'BTC-PERPETUAL/rate/funding':'fundingRate'},inplace=True)
    history['t'] = history['t'].apply(lambda t:t.timestamp())
    history['prev_t'] = history['t'].shift(1)
    history['vol'] = history['vol']/100
    history['borrow'] = 0
    history['df'] = 1
    history.ffill(limit=2,inplace=True)

    ## new portfolio
    portoflio = Portfolio(notional_BTC=1)

    # at time 0 sell a straddle
    mkt_0 = history.iloc[0]
    hedge_instrument = InversePerpetual(strike_BTCUSD=mkt_0['fwd'],market=mkt_0)
    genesis_call = Option(strike_BTCUSD=mkt_0['fwd'],
                          maturity=mkt_0['t'] + 3600 * 24 * 30,
                          call_put='C',
                          market=mkt_0)
    portoflio.add(instrument= genesis_call,
                  notional_USD= -mkt_0['fwd'],
                  price_BTC= genesis_call.mark - vega_slippage * genesis_call.vega(mkt_0))

    genesis_put = Option(strike_BTCUSD=mkt_0['fwd'],
                         maturity=mkt_0['t'] + 3600 * 24 * 30,
                         call_put='P',
                         market=mkt_0)
    portoflio.add(instrument= genesis_put,
                  notional_USD= -mkt_0['fwd'],
                  price_BTC= genesis_put.mark - vega_slippage * genesis_put.vega(mkt_0))

    # run backtest
    greek_list = ['pv','delta','gamma','vega']
    display = pd.DataFrame()
    for _,mkt in history.tail(-1).iterrows():
        portoflio.process_cash_flows(mkt)

        new_data = pd.concat([mkt[['prev_t','t','spot','fwd','vol','fundingRate','borrow','df']],
                             pd.Series({greek: portoflio.apply(greek=greek,market=mkt)
                                        for greek in greek_list}),
                             pd.Series({greek+'_'+str(position.instrument.__class__)+'_'+str(i): position.apply(greek,mkt)
                                        for greek in greek_list for (i,position) in enumerate(portoflio.positions)})])
        new_data.name = pd.to_datetime(mkt['t'], unit='s')
        display = pd.concat([display,new_data],axis=1)

        portoflio.delta_hedge(hedge_instrument,mkt)

    display.T.to_excel('Runtime/runs/deribit.xlsx')

if False:
    btc = pd.read_csv("C:/Users/david/pyCharmProjects/SystematicCeFi/DerivativeArbitrage/Runtime/Deribit_Mktdata_database/deribit_options_chain_2019-07-01_BTC.csv")
    btc['expiry'] = btc['expiration'].apply(lambda t: pd.to_datetime(int(t), unit='us'))
    btc['timestamp'] = btc['timestamp'].apply(lambda t: pd.to_datetime(int(t), unit='us'))
    btc['dist_2_fwd'] = btc.apply(lambda f: np.abs(np.log(f['underlying_price']/f['strike_price'])),axis=1)
    smiles = btc.groupby(by=['expiry', 'timestamp'])[['timestamp','delta','mark_iv']]
    #.sort_values(by='strike_price',
                                                             #key=lambda f: np.abs(f['underlying_price'] / f['strike']))[-1]
