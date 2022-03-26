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
        elif cp == 'P':
            return K * scipy.stats.norm.cdf(-black_scholes.d2(S, K, V, T)) - S * scipy.stats.norm.cdf(
                -black_scholes.d1(S, K, V, T))
        else:
            return black_scholes.pv(S, K, V, T, 'P') + black_scholes.pv(S, K, V, T, 'C')

    @staticmethod
    def delta(S, K, V, T, cp):
        '''for a 1% move'''
        delta = scipy.stats.norm.cdf(black_scholes.d1(S, K, V, T))
        if cp == 'C':
            delta = delta
        elif cp == 'P':
            delta = (delta - 1)
        elif cp =='S':
            delta = (2 * delta - 1)

        return delta * S * 0.01

    @staticmethod
    def gamma(S, K, V, T, cp):
        '''for a 1% move'''
        gamma = scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T)) / (S * V * math.sqrt(T))
        return gamma * S * 0.01 * S * 0.01 * (1 if cp != 'S' else 2)

    @staticmethod
    def vega(S, K, V, T, cp):
        '''for a 10% move'''
        vega = (S * math.sqrt(T) * scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T))) / 100
        return vega * V * 0.1 * (1 if cp != 'S' else 2)

    @staticmethod
    def theta(S, K, V, T, cp):
        '''for 1h'''
        theta = -((S * V * scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T))) / (2 * math.sqrt(T))) / 24/365.25
        return theta * (1 if cp != 'S' else 2)

slippage = {'delta':0, # 1 means 1%
            'gamma':0, # 1 means 1%
            'vega':0,# 1 means 10% relative
            'theta':0, # 1 means 1d
            'rho':0}

class Instrument:
    '''everything is BTC margined so beware confusions:
    self.notional is in USD
    self.mark and all greeks are in BTC
    +ve Delta still means long BTC, though, and is in USD'''
    def process_margin_call(self, market):
        previous_mark = self.mark
        self.mark = self.pv(market)
        return self.mark - previous_mark

class InverseFuture(Instrument):
    '''a long perpetual (size in USD, strike in BTCUSD) is a short USDBTC(notional in USD, 1/strike in USDBTC
    what we implement is more like a fwd..'''
    greeks_available = ['delta','theta']
    def __init__(self, strike_BTCUSD, maturity, market):
        self.strike = 1 / strike_BTCUSD
        self.maturity = maturity
        self.mark = self.pv(market)
        self.symbol = 'BTC-1m'# + str(pd.to_datetime(maturity, unit='s'))

    def pv(self, market):
        '''for a 1% move'''
        T = (self.maturity - market['t'])
        df = np.exp(-market['r']*T)
        return df * (self.strike - 1 / market['fwd']) / market['fwd']
    def delta(self, market):
        T = (self.maturity - market['t'])
        return np.exp(-market['r']*T) * 0.01
    def theta(self, market):
        T = (self.maturity - market['t'])
        df = np.exp(-market['r'] * T)
        return - market['r'] * df / 24/365.25
    def process_cash_flow(self, market):
        '''cash settled in BTC'''
        if market['prev_t'] < self.maturity and market['t'] >= self.maturity:
            cash_flow = (1 / market['fwd'] - self.strike)
            self.mark = 0
            return cash_flow
        else:
            return 0
    def margin_value(self, market):
        '''TODO: approx, in fact it's converted at spot not fwd'''
        return -6e-3

class InversePerpetual(Instrument):
    '''a long perpetual (size in USD, strike in BTCUSD) is a short USDBTC(notional in USD, 1/strike in USDBTC
    what we implement is more like a fwd..'''
    greeks_available = ['delta']
    def __init__(self, strike_BTCUSD, market):
        self.strike = 1 / strike_BTCUSD
        self.mark = self.pv(market)
        self.symbol = 'BTC-PERPETUAL'

    def pv(self, market):
        '''for a 1% move'''
        return (self.strike - 1 / market['fwd']) / market['fwd']
    def delta(self, market):
        return 0.01
    def process_cash_flow(self, market):
        '''accrues every millisecond.
        8h Funding Rate = Maximum (0.05%, Premium Rate) + Minimum (-0.05%, Premium Rate)
        TODO: doesn't handle intra period changes'''
        return - market['fundingRate'] * (market['t'] - market['prev_t'])/3600/8 / market['spot']
    def margin_value(self, market):
        '''TODO: approx, in fact it's converted at spot not fwd'''
        return -6e-3

class Option(Instrument):
    '''a BTCUSD call(size in btc, strike in BTCUSD) is a USDBTC put(strike*size in USD,1/strike in USDTBC)
    notional in USD, strike in BTC, maturity in sec, callput = C or P '''
    greeks_available = ['delta', 'gamma', 'vega', 'theta']
    def __init__(self, strike_BTCUSD, maturity, call_put,market):
        self.strike = 1 / strike_BTCUSD
        self.maturity = maturity
        if call_put == 'C':
            self.call_put = 'P'
        elif call_put == 'P':
            self.call_put = 'C'
        elif call_put == 'S':
            self.call_put = 'S'
        else: raise Exception("unknown option type")
        self.mark = self.pv(market)
        self.symbol = 'BTC-' + call_put + '-1m'# + str(pd.to_datetime(maturity, unit='s')) + '+' + str(int(strike_BTCUSD))

    def pv(self, market):
        T = (self.maturity - market['t'])
        df = np.exp(-market['r']*T)
        return df * black_scholes.pv(1 / market['fwd'], self.strike, market['vol'],
                                                T / 3600 / 24 / 365.25, self.call_put) if self.maturity > market['t'] else 0
    def delta(self, market):
        T = (self.maturity - market['t'])
        df = np.exp(-market['r']*T)
        return df * black_scholes.delta(1 / market['fwd'], self.strike, market['vol'],
                                                   T / 3600 / 24 / 365.25, self.call_put) if self.maturity > market['t'] else 0
    def gamma(self, market):
        T = (self.maturity - market['t'])
        df = np.exp(-market['r']*T)
        return df * black_scholes.gamma(1 / market['fwd'], self.strike, market['vol'],
                                                   T / 3600 / 24 / 365.25, self.call_put) if self.maturity > market['t'] else 0
    def vega(self, market):
        T = (self.maturity - market['t'])
        df = np.exp(-market['r'] * T)
        return df * black_scholes.vega(1 / market['fwd'], self.strike, market['vol'],
                                                  T / 3600 / 24 / 365.25, self.call_put) if self.maturity > market['t'] else 0
    def theta(self, market):
        T = (self.maturity - market['t'])
        df = np.exp(-market['r']*T)
        theta_fwd = df * black_scholes.theta(1 / market['fwd'], self.strike, market['vol'],
                                                  T / 3600 / 24 / 365.25, self.call_put) if self.maturity > market['t'] else 0
        return theta_fwd - market['r'] * df / 24/365.25

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

    def new_trade(self, instrument, notional_USD, price_BTC = 'slippage', market = None):
        position = next(iter(x for x in self.positions if x.instrument == instrument), 'new_position')
        if position == 'new_position':
            position = Position(instrument,notional_USD)
            self.positions += [position]
        elif np.abs(position.notional + notional_USD)<1e-18:
            self.positions.remove(position)
        else:
            position.notional += notional_USD

        # slippage
        if price_BTC == 'slippage':
            assert market is not None
            mid = instrument.pv(market)
            slippage_cost = sum(
                slippage[greek] * np.abs(self.apply(greek,market)) for greek in instrument.__class__.greeks_available)
            price_BTC = mid + slippage_cost * np.sign(notional_USD)
            
        # pay premium. will be mopped up by next margin call
        self.cash -= notional_USD * price_BTC

    def delta_hedge(self, hedge_instrument, market):
        portfolio_delta = self.apply('delta',market)
        hedge_instrument_delta = hedge_instrument.delta(market)
        hedge_notional = -portfolio_delta / hedge_instrument_delta

        self.new_trade(hedge_instrument,
                       hedge_notional,
                       price_BTC= 'slippage',
                       market = market)

    def gamma_target(self, target, market, hedge_instrument = 'atm', hedge_mode = 'add'):
        '''hedge_instrument in ['atm,an Option]'''
        if hedge_instrument == 'atm':
            hedge_instrument = Option(strike_BTCUSD=market['fwd'],
                      maturity=market['t'] + 3600 * 24 * 30,
                      call_put='S',
                      market=market)
        hedge_instrument_gamma = hedge_instrument.gamma(market)
        assert hedge_instrument_gamma

        if hedge_mode == 'replace':
            [self.new_trade(position.instrument,
                            -position.notional,
                            price_BTC= 'slippage',
                            market = market
                            ) for position in self.positions if isinstance(position.instrument,Option)]
        portfolio_gamma = self.apply('gamma', market)

        hedge_notional = (target-portfolio_gamma) / hedge_instrument_gamma

        self.new_trade(hedge_instrument,
                       hedge_notional,
                       price_BTC= 'slippage',
                       market = market)

    def process_cash_flows(self, market):
        '''cash_flows before margin calls !!'''
        cash_flows = sum([position.notional * getattr(position.instrument, 'process_cash_flow')(market) 
                          for position in self.positions])
        margin_call = sum([position.notional * getattr(position.instrument, 'process_margin_call')(market) 
                           for position in self.positions])
        self.cash += cash_flows + margin_call

def display_current(portfolio,market,prev_market):
    predictor = {'pv':lambda greek: greek - portfolio.apply(greek='pv', market=prev_market),# this is actual pnl in fact
                 'delta':lambda greek: greek*(market['fwd']/prev_market['fwd']-1)*100,
                 'gamma':lambda greek: greek*(market['fwd']/prev_market['fwd']-1)**2 *100*100,
                 'vega':lambda greek: greek*(market['vol']/prev_market['vol']-1)*10,
                 'theta':lambda greek: greek*(market['t']-prev_market['t'])/3600}

    display_market = pd.Series({('mkt',data,'total'):market[data] for data in ['prev_t', 't', 'spot', 'fwd', 'dF/F', 'vol', 'fundingRate', 'borrow', 'r']})
    total_greeks = pd.Series(
        {('risk',greek,'total'): portfolio.apply(greek=greek, market=market)
                   for greek in predictor.keys()})
    greeks_per_instrument = pd.Series(
        {('risk',greek,position.instrument.symbol): position.apply(greek, market)
                   for greek in predictor.keys() for (i, position) in enumerate(portoflio.positions)}
                  | {('risk','pv','cash'): portoflio.cash})
    total_predict = pd.Series(
        {('predict',greek,'total'): predictor(portfolio.apply(greek=greek, market=market))
                   for greek,predictor in predictor.items()})
    predict_per_instrument = pd.Series(
        {('predict',greek,position.instrument.symbol): predictor(position.apply(greek, market))
                   for greek,predictor in predictor.items() for (i, position) in enumerate(portoflio.positions)})

    new_data = pd.concat([display_market,total_greeks,total_predict,greeks_per_instrument,predict_per_instrument],axis=0)
    new_data.name = pd.to_datetime(market['t'], unit='s')
    return new_data

if __name__ == "__main__":

    ## get history
    history = deribit_history_main('just use',['BTC'],'deribit','cache')[0]

    # 1mFwd
    if False: # not enough history unless we fetch all expired futures
        tenor_columns = history.filter(like='rate/T').columns
        mark_columns = history.filter(like='mark/c').columns
        mark_columns = [r for r in mark_columns if 'PERPETUAL' not in r]
        history['1mFwd'] = history.apply(lambda t: scipy.interpolate.interp1d(
            x=np.array([t[tenor] for tenor in tenor_columns if not pd.isna(t[tenor])]),
            y=np.array([t[rate] for rate in mark_columns if not pd.isna(t[rate])]),
            kind='linear', fill_value='extrapolate')(1/12), axis=1)
    history['1mFwd'] = history['BTC-PERPETUAL/mark/c']

    ## format history
    history.reset_index(inplace=True)
    history.rename(columns={'index':'t',#TODO: more consistent with /o
                            '1mFwd':'fwd',
                            'BTC/volindex/c':'vol',
                            'BTC-PERPETUAL/indexes/c':'spot',
                            'BTC-PERPETUAL/rate/funding':'fundingRate'},inplace=True)
    history['dF/F'] = history['fwd'].apply(lambda f:np.log(f)).diff()
    history['t'] = history['t'].apply(lambda t:t.timestamp())
    history['prev_t'] = history['t'].shift(1)
    history['vol'] = history['vol']/100
    history['borrow'] = 0
    history['r'] = history['borrow']
    history.ffill(limit=2,inplace=True)

    ## new portfolio
    portoflio = Portfolio(notional_BTC=1)

    # at time 0 sell a straddle
    prev_mkt = history.iloc[0]
    hedge_instrument = InversePerpetual(strike_BTCUSD=prev_mkt['fwd'],market=prev_mkt)
    straddle = Option(strike_BTCUSD=prev_mkt['fwd'],
                      maturity=prev_mkt['t'] + 3600 * 24 * 30,
                      call_put='S',
                      market=prev_mkt)
    portoflio.new_trade(instrument= straddle,
                        notional_USD= -prev_mkt['fwd'],
                        price_BTC= 'slippage',
                        market= prev_mkt)

    ## run backtest
    series = []
    for _,market in history.iterrows():
        portoflio.process_cash_flows(market)
        portoflio.gamma_target(portoflio.apply('gamma',market),market,hedge_instrument='atm',hedge_mode='replace')
        portoflio.delta_hedge(hedge_instrument,market)
        
        series += [display_current(portoflio,market,prev_mkt)]
        prev_mkt = market

    display = pd.concat(series,axis=1)
    display.sort_index(axis=0,level=0).T.to_excel('Runtime/runs/deribit.xlsx')

if False:
    btc = pd.read_csv("C:/Users/david/pyCharmProjects/SystematicCeFi/DerivativeArbitrage/Runtime/Deribit_Mktdata_database/deribit_options_chain_2019-07-01_BTC.csv")
    btc['expiry'] = btc['expiration'].apply(lambda t: pd.to_datetime(int(t), unit='us'))
    btc['timestamp'] = btc['timestamp'].apply(lambda t: pd.to_datetime(int(t), unit='us'))
    btc['dist_2_fwd'] = btc.apply(lambda f: np.abs(np.log(f['underlying_price']/f['strike_price'])),axis=1)
    smiles = btc.groupby(by=['expiry', 'timestamp'])[['timestamp','delta','mark_iv']]
    #.sort_values(by='strike_price',
                                                             #key=lambda f: np.abs(f['underlying_price'] / f['strike']))[-1]
