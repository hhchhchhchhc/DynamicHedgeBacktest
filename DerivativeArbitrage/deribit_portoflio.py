import pandas as pd

from deribit_history import *
import numpy as np
import scipy
import math
import copy

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
        vega = (S * math.sqrt(T) * scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T)))
        return vega * V * 0.1 * (1 if cp != 'S' else 2)

    @staticmethod
    def theta(S, K, V, T, cp):
        '''for 1h'''
        theta = -((S * V * scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T))) / (2 * math.sqrt(T)))
        return theta / 24/365.25 * (1 if cp != 'S' else 2)

slippage = {'delta':0, # 1 means 1%
            'gamma':0, # 1 means 1%
            'vega':0,# 1 means 10% relative
            'theta':0, # 1 means 1d
            'rho':0}

class Instrument:
    '''everything is coin margined so beware confusions:
    self.notional is in USD
    all greeks are in coin
    +ve Delta still means long coin, though, and is in USD'''
    def __init__(self,market):
        self.symbol = None

class Cash(Instrument):
    def __init__(self,currency,market):
        self.symbol = currency+'-CASH:'
        self.currency = currency
    def pv(self, market):
        return 1
    def cash_flow(self, market,prev_market):
        return {'drop_from_pv':0,'accrues':0}

class InverseFuture(Instrument):
    '''a long perpetual (size in USD, strike in coinUSD) is a short USDcoin(notional in USD, 1/strike in USDcoin
    what we implement is more like a fwd..'''
    greeks_available = ['delta','theta']
    def __init__(self, underlying, strike_coinUSD, maturity, market):
        self.strike = 1 / strike_coinUSD
        self.maturity = maturity
        self.symbol = underlying+'-FUTURE:' + str(pd.to_datetime(maturity, unit='s'))

    def pv(self, market):
        T = (self.maturity - market['t'])
        df = np.exp(-market['r']*T)
        return df * (self.strike - 1 / market['fwd'])
    def delta(self, market):
        '''for a 1% move'''
        T = (self.maturity - market['t'])
        return np.exp(-market['r']*T) * 0.01 / market['fwd']
    def theta(self, market):
        '''for 1 day'''
        T = (self.maturity - market['t'])
        df = np.exp(-market['r'] * T)
        return - market['r'] * df / 24/365.25
    def cash_flow(self, market, prev_market):
        '''cash settled in coin'''
        if prev_market['t'] < self.maturity and market['t'] >= self.maturity:
            cash_flow = (1 / market['fwd'] - self.strike)
            return {'drop_from_pv':cash_flow,'accrues':0}
        else:
            return 0
    def margin_value(self, market):
        '''TODO: approx, in fact it's converted at spot not fwd'''
        return -6e-3

class InversePerpetual(Instrument):
    '''a long perpetual (size in USD, strike in coinUSD) is a short USDcoin(notional in USD, 1/strike in USDcoin
    what we implement is more like a fwd..'''
    greeks_available = ['delta']
    def __init__(self, underlying, strike_coinUSD, market):
        self.strike = 1 / strike_coinUSD
        self.symbol = underlying+'-PERPETUAL:'

    def pv(self, market):
        return (self.strike - 1 / market['fwd'])
    def delta(self, market):
        '''for a -1% move of 1/f'''
        return 0.01 / market['fwd']
    def cash_flow(self, market, prev_market):
        '''accrues every millisecond.
        8h Funding Rate = Maximum (0.05%, Premium Rate) + Minimum (-0.05%, Premium Rate)
        TODO: doesn't handle intra period changes'''
        cash_flow = market['fundingRate'] * (market['t'] - prev_market['t'])/3600/24/365.25 / market['spot']
        return {'drop_from_pv': 0, 'accrues': cash_flow}
    def margin_value(self, market):
        '''TODO: approx, in fact it's converted at spot not fwd'''
        return -6e-3 / market['spot']

class Option(Instrument):
    '''a coinUSD call(size in coin, strike in coinUSD) is a USDcoin put(strike*size in USD,1/strike in USDTBC)
    notional in USD, strike in coin, maturity in sec, callput = C or P '''
    greeks_available = ['delta', 'gamma', 'vega', 'theta']
    def __init__(self, underlying,strike_coinUSD, maturity, call_put,market):
        self.strike = 1 / strike_coinUSD
        self.maturity = maturity
        if call_put == 'C':
            self.call_put = 'P'
        elif call_put == 'P':
            self.call_put = 'C'
        elif call_put == 'S':
            self.call_put = 'S'
        else:
            raise Exception("unknown option type")
        self.symbol = underlying+'-' + call_put + ':' + str(pd.to_datetime(maturity, unit='s')) + '+' + str(int(strike_coinUSD))

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

    def cash_flow(self, market, prev_market):
        '''cash settled in coin'''
        if prev_market['t'] < self.maturity and market['t'] >= self.maturity:
            cash_flow = max(0,(1 if self.call_put == 'C' else -1)*(1 / market['fwd'] - self.strike))
        else:
            cash_flow = 0
        return {'drop_from_pv': cash_flow, 'accrues': 0}

    def margin_value(self, market):
        return - (max(0.15 - (1 / market['fwd'] - self.strike) * (1 if self.call_put == 'C' else -1),
                                      0.1) + self.pv(market)) if self.maturity > market['t'] else 0

class Position:
    def __init__(self,instrument,notional_USD):
        self.instrument = instrument
        self.notional = notional_USD
        self.new_deal_pnl = 0
    def apply(self, greek, market):
        return self.notional * getattr(self.instrument, greek)(market) if hasattr(self.instrument,greek) else 0
    def process_margin_call(self, market, prev_market):
        '''apply to prev_portfolio !! '''
        self.new_deal_pnl = 0
        return self.notional * (self.instrument.pv(market) - self.instrument.pv(prev_market))
    def cash_flow(self, market, prev_market):
        return {mode: self.notional * self.instrument.cash_flow(market,prev_market)[mode] for mode in ['drop_from_pv','accrues']}
        
class Portfolio():
    def __init__(self,currency, notional_coin,market):
        self.positions = [Position(Cash(currency,market),notional_coin)]
        self.greeks_cache = dict()

    def apply(self, greek, market):
        result = sum([position.apply(greek,market) for position in self.positions])
        return result

    def new_trade(self, instrument, notional_USD, price_coin = 'slippage', market = None):
        position = next(iter(x for x in self.positions if x.instrument.symbol == instrument.symbol), 'new_position')
        if position == 'new_position':
            position = Position(instrument,notional_USD)
            self.positions += [position]
        else:
            position.notional += notional_USD

        # slippage
        if price_coin == 'slippage':
            assert market is not None
            slippage_cost = sum(
                slippage[greek] * np.abs(self.apply(greek,market)) for greek in instrument.__class__.greeks_available)
            new_deal_pnl = slippage_cost * np.sign(notional_USD)
        else:
            new_deal_pnl = price_coin - instrument.pv(market)

        # pay cost.
        position.new_deal_pnl = new_deal_pnl
        self.positions[0].notional -= notional_USD * new_deal_pnl

    def delta_hedge(self, hedge_instrument, market):
        portfolio_delta = self.apply('delta',market)
        hedge_instrument_delta = hedge_instrument.delta(market)
        hedge_notional = -portfolio_delta / hedge_instrument_delta

        self.new_trade(hedge_instrument,
                       hedge_notional,
                       price_coin= 'slippage',
                       market = market)

    def gamma_target(self, target, market, hedge_instrument = 'atm', hedge_mode = 'add'):
        '''hedge_instrument in ['atm,an Option]'''
        if hedge_instrument == 'atm':
            hedge_instrument = Option(
                underlying=self.positions[0].instrument.currency,
                strike_coinUSD=market['fwd'],
                maturity=market['t'] + 3600 * 24 * 30,
                call_put='S',
                market=market)
        hedge_instrument_gamma = hedge_instrument.gamma(market)
        assert hedge_instrument_gamma

        if hedge_mode == 'replace':
            [self.new_trade(position.instrument,
                            -position.notional,
                            price_coin= 'slippage',
                            market = market
                            ) for position in self.positions if isinstance(position.instrument,Option)]
        portfolio_gamma = self.apply('gamma', market)

        hedge_notional = (target-portfolio_gamma) / hedge_instrument_gamma

        self.new_trade(hedge_instrument,
                       hedge_notional,
                       price_coin= 'slippage',
                       market = market)

    def process_settlements(self, market, prev_market):
        '''apply to prev_portfolio !'''
        new_positions = copy.deepcopy(self.positions)

        # settle cash_flows before margin calls
        cash_flows = sum([getattr(position, 'cash_flow')(market, prev_market)[mode]
                           for position in new_positions for mode in ['drop_from_pv','accrues']])
        margin_call = sum([getattr(position, 'process_margin_call')(market, prev_market)
                           for position in new_positions])
        new_positions[0].notional += cash_flows + margin_call

        # clean-up unwound positions
        for position in new_positions[1:]:
            if (hasattr(position.instrument, 'maturity') and market['t'] > position.instrument.maturity) \
                    or np.abs(position.notional) < 1e-18:
                new_positions.remove(position)

        return new_positions

def display_current(portfolio,prev_portfolio, market,prev_market):
    predictors = {'delta':lambda x: x.apply(greek='delta', market=prev_market)*(1-prev_market['fwd']/market['fwd'])*100,
                  'gamma':lambda x: 0.5 * x.apply(greek='gamma', market=prev_market)*(1-prev_market['fwd']/market['fwd'])**2 *100*100,
                  'vega':lambda x: x.apply(greek='vega', market=prev_market)*(market['vol']/prev_market['vol']-1)*10,
                  'theta':lambda x: x.apply(greek='theta', market=prev_market)*(market['t']-prev_market['t'])/3600}

    # 1: mkt
    display_market = pd.Series({('mkt',data,'total'):market[data] for data in ['t', 'spot', 'fwd', 'vol', 'fundingRate', 'borrow', 'r']})
    
    # 2 : greeks
    greeks = pd.Series(
        {('risk',greek,position.instrument.symbol.split(':')[0]):
             position.apply(greek, market)
         for greek in ['pv'] + list(predictors.keys()) for position in portfolio.positions})

    # 3: predict = mkt move for prev_portfolio
    predict = pd.Series(
        {('predict',greek,position.instrument.symbol.split(':')[0]):
             predictor(position)
         for greek,predictor in predictors.items() for position in prev_portfolio.positions[1:]})

    # 4: actual = prev_portfolio cash_flows + new deal + unexplained
    actual = pd.Series(
        {('actual', 'unexplained', p0.instrument.symbol.split(':')[0]):
             p0.apply('pv',market) - p0.apply('pv',prev_market)
             - p0.cash_flow(market,prev_market)['drop_from_pv']
             - predict[('predict',slice(None),p0.instrument.symbol.split(':')[0])].sum()
         for p0 in prev_portfolio.positions[1:]}
        |{('actual', 'cash_flow', p0.instrument.symbol.split(':')[0]):
              p0.cash_flow(market,prev_market)['drop_from_pv']+p0.cash_flow(market,prev_market)['accrues']
          for p0 in prev_portfolio.positions[1:]}
        |{('actual', 'new_deal', p1.instrument.symbol.split(':')[0]):
              p1.new_deal_pnl
          for p1 in portfolio.positions[1:]})

    # add totals and mkt
    new_data = pd.concat([greeks,predict,actual],axis=0).unstack(level=2)
    new_data['total'] = new_data.sum(axis=1)
    new_data = new_data.stack()
    new_data = pd.concat([display_market,new_data],axis=0)

    new_data.name = pd.to_datetime(market['t'], unit='s')
    return new_data.sort_index(axis=0,level=[0,1,2])

if __name__ == "__main__":
    if len(sys.argv)<2:
        currency = 'BTC'
    else:
        currency = sys.argv[1]

    ## get history
    history = deribit_history_main('just use',[currency],'deribit','cache')[0]

    # 1mFwd
    if False: # not enough history unless we fetch all expired futures
        tenor_columns = history.filter(like='rate/T').columns
        mark_columns = history.filter(like='mark/c').columns
        mark_columns = [r for r in mark_columns if 'PERPETUAL' not in r]
        history['1mFwd'] = history.apply(lambda t: scipy.interpolate.interp1d(
            x=np.array([t[tenor] for tenor in tenor_columns if not pd.isna(t[tenor])]),
            y=np.array([t[rate] for rate in mark_columns if not pd.isna(t[rate])]),
            kind='linear', fill_value='extrapolate')(1/12), axis=1)
    history['1mFwd'] = history[currency+'-PERPETUAL/mark/c']

    ## format history
    history.reset_index(inplace=True)
    history.rename(columns={'index':'t',#TODO: more consistent with /o
                            '1mFwd':'fwd',
                            currency+'/volindex/c':'vol',
                            currency+'-PERPETUAL/indexes/c':'spot',
                            currency+'-PERPETUAL/rate/funding':'fundingRate'},inplace=True)
    history['t'] = history['t'].apply(lambda t:t.timestamp())
    history['vol'] = history['vol']/100
    history['borrow'] = 0
    history['r'] = history['borrow']
    history.ffill(limit=2,inplace=True)
    history.bfill(inplace=True)

    prev_mkt = history.iloc[0]

    ## new portfolio
    portfolio = Portfolio(currency,notional_coin=1,market=prev_mkt)
    hedge_instrument = InversePerpetual(underlying=currency,
                                        strike_coinUSD=prev_mkt['fwd'],
                                        market=prev_mkt)
    straddle = Option(underlying=currency,
                      strike_coinUSD=prev_mkt['fwd'],
                      maturity=prev_mkt['t'] + 3600 * 24 * 30,
                      call_put='C',
                      market=prev_mkt)
    portfolio.new_trade(instrument= straddle,
                        notional_USD= -prev_mkt['fwd'],
                        price_coin= 'slippage',
                        market= prev_mkt)
    target_gamma = portfolio.apply('gamma',prev_mkt)
    prev_portfolio = copy.deepcopy(portfolio)

    ## run backtest
    series = []
    for i,market in history.iterrows():
        portfolio.positions = prev_portfolio.process_settlements(market, prev_mkt)
        portfolio.gamma_target(target_gamma,market,hedge_instrument='atm',hedge_mode='replace')
        portfolio.delta_hedge(hedge_instrument,market)

        series += [display_current(portfolio,prev_portfolio,market,prev_mkt)]
        prev_mkt = market
        prev_portfolio = copy.deepcopy(portfolio)

        if i%1000==0:
            print(i)

    display = pd.concat(series,axis=1)
    #display.loc[(['predict','actual'],slice(None),slice(None))] = display.loc[(['predict','actual'],slice(None),slice(None))].cumsum(axis=1)
    display.T.to_excel('Runtime/runs/deribit.xlsx')
    display.to_pickle('Runtime/runs/deribit.pickle')

if False:
    coin = pd.read_csv("C:/Users/david/pyCharmProjects/SystematicCeFi/DerivativeArbitrage/Runtime/Deribit_Mktdata_database/deribit_options_chain_2019-07-01_coin.csv")
    coin['expiry'] = coin['expiration'].apply(lambda t: pd.to_datetime(int(t), unit='us'))
    coin['timestamp'] = coin['timestamp'].apply(lambda t: pd.to_datetime(int(t), unit='us'))
    coin['dist_2_fwd'] = coin.apply(lambda f: np.abs(np.log(f['underlying_price']/f['strike_price'])),axis=1)
    smiles = coin.groupby(by=['expiry', 'timestamp'])[['timestamp','delta','mark_iv']]
    #.sort_values(by='strike_price',
                                                             #key=lambda f: np.abs(f['underlying_price'] / f['strike']))[-1]
