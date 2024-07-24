import copy
import os
import typing
from abc import abstractmethod, ABC
import argparse
import yaml

from deribit_history import *
from deribit_marketdata import kaiko_history, Market
from utils.blackscholes import black_scholes


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

    def cash_flow(self, market, prev_market):
        return {'drop_from_pv': 0, 'accrues': 0}


class Cash(Instrument):
    greeks_available = ['delta']
    def __init__(self, currency):
        super().__init__()
        self.symbol = f'{currency}-CASH:'
        self.currency = currency

    def pv(self, market):
        return market.spot

    def delta(self, market):
        return 1


class InverseFuture(Instrument):
    """a long perpetual (size in USD, strike in coinUSD) is a short USDcoin(notional in USD, 1/strike in USDcoin
    what we implement is more like a fwd..
    """
    greeks_available = ['delta', 'theta']  # ,'IR01'

    def __init__(self, underlying: str, maturity: datetime):
        super().__init__()
        self.maturity: datetime = maturity  # timestamp. TODO: in fact only fri 8utc for 1w,2w,1m,and 4 IMM
        self.symbol: str = f'{underlying}-FUTURE:' + str(pd.to_datetime(maturity, unit='s'))

    def pv(self, market):
        return 0

    def delta(self, market, maturity_timestamp=None):
        """for a unit move"""
        if self.maturity < market.t:
            return 0
        if maturity_timestamp and abs(maturity_timestamp - self.maturity) > timedelta(minutes=1):
            return 0.0

        fwd = market.fwdcurve.interpolate(self.maturity)
        return fwd / market.spot

    def theta(self, market) -> np.float64:
        """for 1d"""
        if self.maturity < market.t:
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

    def cash_flow(self, market, prev_market):
        margin_payment = market.fwdcurve.interpolate(self.maturity) - prev_market.fwdcurve.interpolate(self.maturity)
        return {'drop_from_pv': 0, 'accrues': margin_payment}

class InversePerpetual(Instrument):
    """a long perpetual (size in USD, strike in coinUSD) is a short USDcoin(notional in USD, 1/strike in USDcoin
    what we implement is more like a fwd.."""
    greeks_available = ['delta']

    def __init__(self, underlying):
        super().__init__()
        self.symbol = f'{underlying}-PERPETUAL:'

    def pv(self, market):
        return 0

    def delta(self, market):
        """for a unit move"""
        return 1

    def cash_flow(self, market, prev_market):
        """accrues every millisecond.
        8h Funding Rate = Maximum (0.05%, Premium Rate) + Minimum (-0.05%, Premium Rate)
        TODO: doesn't handle intra period changes"""
        funding_payment = - market.spot * market.funding_rate * (market.t - prev_market.t).total_seconds() / 3600 / 8 / 365.25
        margin_payment = market.spot - prev_market.spot
        return {'drop_from_pv': margin_payment, 'accrues': funding_payment}

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

    def __init__(self, currency: str, strike_coinUSD: float, maturity: datetime, call_put: str):
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
                f'{currency}-{call_put}:'
                + str(pd.to_datetime(maturity, unit='s'))
                + '+'
                + str(int(strike_coinUSD))
        )

    def pv(self, market):
        '''pv in usd for 1 coin notional'''
        if self.maturity < market.t:
            return 0
        T = (self.maturity - market.t).total_seconds() / 3600 / 24 / 365.25
        fwd = market.fwdcurve.interpolate(self.maturity)
        vol = market.vol.interpolate(self.strike, self.maturity)
        return black_scholes.pv(fwd, self.strike, vol, T, self.call_put)

    def delta(self, market, maturity_timestamp=None):
        '''for unit move of spot
        parallel delta by default.
        delta by tenor is maturity_timestamp (within 1min)'''
        if self.maturity < market.t:
            return 0
        if maturity_timestamp and abs(maturity_timestamp - self.maturity) > timedelta(minutes=1):
            return 0.0
        T = (self.maturity - market.t).total_seconds() / 3600 / 24 / 365.25
        fwd = market.fwdcurve.interpolate(self.maturity)
        vol = market.vol.interpolate(self.strike, self.maturity)
        return black_scholes.delta(fwd, self.strike, vol, T, self.call_put) + self.pv(market) / market.spot

    def gamma(self, market, maturity_timestamp=None):
        '''delta move for unit move of spot'''
        if self.maturity < market.t:
            return 0
        if maturity_timestamp and abs(maturity_timestamp - self.maturity) > timedelta(minutes=1):
            return 0.0
        T = (self.maturity - market.t).total_seconds() / 3600 / 24 / 365.25
        fwd = market.fwdcurve.interpolate(self.maturity)
        vol = market.vol.interpolate(self.strike, self.maturity)
        return black_scholes.gamma(fwd, self.strike, vol, T, self.call_put)

    def vega(self, market, maturity_timestamp=None):
        '''vega by tenor maturity_timestamp(within 1s)'''
        if self.maturity < market.t:
            return 0
        if maturity_timestamp and abs(maturity_timestamp - self.maturity) > timedelta(minutes=1):
            return 0.0
        T = (self.maturity - market.t).total_seconds() / 3600 / 24 / 365.25
        fwd = market.fwdcurve.interpolate(self.maturity)
        vol = market.vol.interpolate(self.strike, self.maturity)
        return black_scholes.vega(fwd, self.strike, vol, T, self.call_put)

    def theta(self, market) -> float:
        '''for 1d'''
        if self.maturity < market.t:
            return 0
        T = (self.maturity - market.t).total_seconds() / 3600 / 24 / 365.25
        fwd = market.fwdcurve.interpolate(self.maturity)
        vol = market.vol.interpolate(self.strike, self.maturity)
        return black_scholes.theta(fwd, self.strike, vol, T, self.call_put)

    def cash_flow(self, market, prev_market):
        '''value in usd, note cash is settled in coin'''
        cash_flow = 0
        if prev_market.t < self.maturity <= market.t:
            if self.call_put == 'C':
                cash_flow = max(0, (market.spot - self.strike))
            elif self.call_put == 'P':
                cash_flow = max(0, (self.strike - market.spot))
            elif self.call_put == 'S':
                cash_flow = abs(self.strike - market.spot)

        return {'drop_from_pv': cash_flow, 'accrues': 0}

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


class Position:
    def __init__(self, instrument, notional, label=None):
        self.instrument: Instrument = instrument
        self.notional: float = notional
        self.new_deal_pnl: float = 0
        self.label: str = label  # this str is handy to describe why position is here, eg gamma_hedge, delta_hedge...

    def apply(self, greek, market, **kwargs):
        return self.notional * getattr(self.instrument, greek)(market, **kwargs) if hasattr(self.instrument,
                                                                                            greek) else 0

    def margin_value(self, market):
        kwargs = {'notional': self.notional} if type(self.instrument) == Option else {}
        return self.notional * getattr(self.instrument, 'margin_value')(market, **kwargs)

    def cash_flow(self, market, prev_market):
        return {mode: self.notional * self.instrument.cash_flow(market, prev_market)[mode] for mode in
                ['drop_from_pv', 'accrues']}


class DeltaStrategy(ABC):
    @abstractmethod
    def delta_hedge(self, delta: float, context) -> float:
        raise NotImplementedError


class DeltaThresholdStrategy(DeltaStrategy):
    def __init__(self, target: float, threshold: float):
        self.target = target
        self.threshold = threshold

    def delta_hedge(self, delta: float, context) -> float:
        return self.target - delta if abs(delta - self.target) > self.threshold else 0


class Strategy(ABC):
    """
    a list of position mixin
    hedging buisness logic is here
    """

    def __init__(self, currency, equity):
        # TODO: book object when we implement portfolio margining
        self.positions = [Position(Cash(currency), equity, label='initial_cash')]
        self.greeks_cache = {}

    @staticmethod
    def build_strategy(config: dict, market: Market):
        strategy_type = config.pop('type')
        return globals()[strategy_type](market=market, **config)

    def target_greek(self, market,
                     greek,
                     greek_params,
                     hedge_instrument,
                     target,
                     hedge_params=None) -> None:
        #TODO: make it multidimentional
        """
        targets a value of a greek
        first unwinds all instruments of type hedge_mode['replace'], or just adds if hedge_mode['replace']={}
        """
        hedge_instrument_greek = getattr(hedge_instrument, greek)(market, **greek_params)
        assert hedge_instrument_greek, f'zero {greek} {hedge_instrument.symbol}'

        if hedge_params and ('replace_label' in hedge_params):
            positions_to_replace = [position for position in self.positions if
                                    position.label == hedge_params['replace_label']]

            [self.new_trade(position.instrument,
                            -position.notional,
                            price_coin='slippage',
                            market=market,
                            label=hedge_params['replace_label']
                            ) for position in positions_to_replace]
        portfolio_greek = self.apply(greek, market, **greek_params)

        hedge_notional = (target - portfolio_greek) / hedge_instrument_greek
        self.new_trade(hedge_instrument,
                       hedge_notional,
                       price_coin='slippage',
                       market=market,
                       label=f'{greek}_hedge')

    @abstractmethod
    def set_rebalancing_rules(self, market, underlying, **kwargs):
        """
        This is where the hedging steps are detailed
        """
        raise NotImplementedError

    def process_settlements(self, market, prev_market):
        cash_flows = sum(
            getattr(position, 'cash_flow')(market, prev_market)[mode]
            for position in self.positions
            for mode in ['drop_from_pv', 'accrues']
        )
        if 'USD' not in self.positions[0].instrument.currency:
            conversion_from_USD = market.spot
        else:
            conversion_from_USD = 1
        self.positions[0].notional += cash_flows / conversion_from_USD

    def cleanup_positions(self, market):
        # clean-up unwound positions
        for position in self.positions[1:]:
            if (hasattr(position.instrument, 'maturity') and market.t > position.instrument.maturity) \
                    or np.abs(position.notional) < 1e-18:
                self.positions.remove(position)

    def apply(self, greek, market, **kwargs):
        return sum(
            position.apply(greek, market, **kwargs) for position in self.positions
        )

    def new_trade(self, instrument, notional, price_coin: typing.Union[str, float] = 'slippage', market=None,
                  label=None):
        '''this assumes '''
        position = next(iter(x for x in self.positions if x.instrument.symbol == instrument.symbol), 'new_position')
        if position == 'new_position':
            position = Position(instrument, notional, label)
            self.positions += [position]
        else:
            position.notional += notional

        # slippage
        if price_coin == 'slippage':
            assert market is not None, 'market is None'
            # empty: slippage using parallel greeks
            kwargs = {}
            slippage_cost = sum(
                market.slippage[greek] * np.abs(position.apply(greek, market, **kwargs)) for greek in
                instrument.__class__.greeks_available)
            price_coin = instrument.pv(market) + slippage_cost * np.sign(notional)

        new_deal_pnl = notional * (instrument.pv(market) - price_coin)
        # pay cost.
        position.new_deal_pnl = new_deal_pnl
        if 'USD' not in self.positions[0].instrument.currency:
            conversion_from_USD = market.spot
        else:
            conversion_from_USD = 1
        self.positions[0].notional -= notional * price_coin / conversion_from_USD


class ShortGamma(Strategy):
    def __init__(self, **kwargs):
        '''equity need to be converted to coin'''
        super().__init__(kwargs['currency'], kwargs['equity'] / kwargs['market'].spot)
        self.gamma_tenor = kwargs['gamma_tenor']*24*3600
        self.theta_target = kwargs['yield_target'] * kwargs['equity'] / 365.25
        self.delta_strategy: DeltaStrategy = DeltaThresholdStrategy(**kwargs['delta_hedging'])

    def set_rebalancing_rules(self, market, currency, **kwargs):
        """
        This is where the hedging steps are detailed
        """

        def select_hedging_instrument(threshold=0.5):
            # benchmark is the theta of the atm at gamma_tenor
            benchmark_expiry = market.t + timedelta(seconds=self.gamma_tenor)
            benchmark_strike = market.fwdcurve.interpolate(benchmark_expiry)
            benchmark_theta = Option(currency, benchmark_strike, benchmark_expiry, 'S').theta(market)

            # find the most theta in current book
            options = [option
                       for option in self.positions[1:]
                       if type(option.instrument) == Option]
            closest_option = min(options, key=lambda p: p.instrument.theta(market)) if options else None

            # if better than half the benchmark, keep it, other use closest.
            if closest_option and closest_option.instrument.theta(market) / benchmark_theta > threshold:
                hedge_instrument = closest_option.instrument
                hedge_params = None
            else:
                strike = market.nearest_strike(option_maturity)
                hedge_instrument = Option(currency, strike, option_maturity, 'S')
                hedge_params = {'replace_label': 'theta_hedge'}
            return hedge_instrument, hedge_params

        # 1: target theta
        # find shortest options that is longer than gamma_tenor
        # TODO: need 2 expiries + 2 strikes
        option_maturity = market.t + timedelta(seconds=next((T for T in market.vol.dataframe.columns if T > self.gamma_tenor),
                        max(market.vol.dataframe.columns)))

        hedge_instrument, hedge_params = select_hedging_instrument()
        # TODO: need to invert a matrix of hedge_instruments
        self.target_greek(market,
                          greek='theta', greek_params={},
                          hedge_instrument=hedge_instrument, target=self.theta_target, hedge_params=hedge_params)

        # 3: hedge delta, replacing InversePerpetual
        # should rather find closest but usually the same
        # future_maturity_sec = next((T for T in market.fwdcurve.series.index if market.t + timedelta(seconds=T) >= option_maturity),
        #                 max(market.fwdcurve.series.index))
        # future_maturity = market.t + timedelta(seconds=future_maturity_sec)
        hedge_instrument = InversePerpetual(currency)
        self.target_greek(market,
                          greek='delta', greek_params={},
                          hedge_instrument=hedge_instrument, target=0)


class VolSteepener(Strategy):
    def set_rebalancing_rules(self, market, underlying, vega_target, vol_maturity_timestamp, gamma_maturity_timestamp):
        """
        This is where the hedging steps are detailed
        """

        # 1: target vega, replacing back legs
        hedge_instrument = Option(market, underlying, market.fwdcurve.interpolate(vol_maturity_timestamp),
                                  vol_maturity_timestamp, 'S')
        self.target_greek(market,
                          greek='vega', greek_params={'maturity_timestamp': vol_maturity_timestamp},
                          hedge_instrument=hedge_instrument,
                          hedge_params={'replace_label': 'vega_leg'}, target=vega_target)

        # 2: hedge gamma, replacing front legs
        hedge_instrument = Option(market, underlying, market.fwdcurve.interpolate(gamma_maturity_timestamp),
                                  gamma_maturity_timestamp, 'S')
        self.target_greek(market,
                          greek='gamma', greek_params={},
                          hedge_instrument=hedge_instrument,
                          hedge_params={'replace_label': 'gamma_hedge'}, target=0)

        # 3: hedge delta, replacing InverseForward
        hedge_instrument = InverseFuture(market, underlying, market.fwdcurve.interpolate(vol_maturity_timestamp),
                                         vol_maturity_timestamp)
        self.target_greek(market,
                          greek='delta', greek_params={},
                          hedge_instrument=hedge_instrument,
                          hedge_params={'replace_label': 'delta_hedge'}, target=0)

def display_current(portfolio, prev_portfolio, market, prev_market):
    predictors = {'delta': lambda x: x.apply(greek='delta', market=prev_market) * (
            market.spot - prev_market.spot),
                  'gamma': lambda x: 0.5 * x.apply(greek='gamma', market=prev_market) * (
                          market.spot - prev_market.spot) ** 2,
                  'vega': lambda x: x.apply(greek='vega', market=prev_market) * (
                          market.vol.interpolate(x.instrument.strike, x.instrument.maturity) -
                          prev_market.vol.interpolate(x.instrument.strike, x.instrument.maturity)) if type(
                      x.instrument) == Option else 0,
                  'theta': lambda x: x.apply(greek='theta', market=prev_market) * (
                          market.t - prev_market.t).total_seconds() / 3600 / 24,
                  'IR01': lambda x: x.apply(greek='delta', market=prev_market) * (
                          1 - market.spot / prev_market.spot * prev_market.fwdcurve.interpolate(
                      x.instrument.maturity) / market.fwdcurve.interpolate(x.instrument.maturity)) * 100 if type(
                      x.instrument) in [InverseFuture, Option] else 0}

    # 1: mkt
    option_mat = next((
        position.instrument.maturity for position in portfolio.positions if type(position.instrument) == Option), None)
    option_strike = next((
        position.instrument.strike for position in portfolio.positions if type(position.instrument) == Option), None)
    display_market = pd.Series(
        {('mkt', data, 'total'): getattr(market, data) for data in ['t', 'spot', 'funding_rate']}
        | {('mkt', 'vol', 'total'): market.vol.interpolate(option_strike, option_mat) if option_mat else None})

    # 2 : inventory
    inventory = pd.Series(
        {('portfolio', position.label, 'symbol'): position.instrument.symbol
         for position in portfolio.positions}
        | {('portfolio', position.label, 'notional'): position.notional
           for position in portfolio.positions})

    # 3 : greeks
    greeks = pd.Series(
        {('risk', greek, position.label):
             position.apply(greek, market)
         for greek in ['pv'] + list(predictors.keys())
         for position in portfolio.positions})

    # 4: predict = mkt move for prev_portfolio
    predict = pd.Series(
        {('predict', greek, position.label): predictor(position)
         for greek, predictor in predictors.items()
         for position in prev_portfolio.positions})

    # 5: actual = prev_portfolio cash_flows + new deal + unexplained
    actual = pd.Series(
        {('actual', 'unexplained',
          p0.label):  # the unexplained = dPV-predict, for old portoflio only
             p0.apply('pv', market) - p0.apply('pv', prev_market)  # dPV
             + p0.cash_flow(market, prev_market)['drop_from_pv']
             - predict[('predict', slice(None), p0.label)].sum()  # - the predict
         for p0 in prev_portfolio.positions}  # ....only for old positions. New positions contribute to new_deal only.
        | {('actual', 'cash_flow', p0.label):
               p0.cash_flow(market, prev_market)['drop_from_pv'] + p0.cash_flow(market, prev_market)['accrues']
           for p0 in prev_portfolio.positions}
        | {('actual', 'new_deal', p1.label):
               p1.new_deal_pnl  # new deals only contribute here.
           for p1 in portfolio.positions})

    # tx by symbols
    # tx = pd.Series(
    #     {('tx', 'common', p.symbol):
    #          p.notional for p in set(prev_portfolio.positions).intersection(portfolio.positions)})
    # if issues := [
    #     p0.label
    #     for p0 in prev_portfolio.positions[1:]
    #     if p0.new_deal_pnl
    # ]:
    #     print('new_deal_pnl double count' + ''.join(
    #         [p0.label for p0 in prev_portfolio.positions[1:]
    #          if p0.new_deal_pnl]))

    # add totals and mkt
    new_data = pd.concat([greeks, predict, actual], axis=0).unstack(level=2)
    new_data['total'] = new_data.sum(axis=1)
    new_data = new_data.stack()
    new_data = pd.concat([display_market, new_data, inventory], axis=0)

    new_data.name = pd.to_datetime(market.t, unit='s')
    return new_data.sort_index(axis=0, level=[0, 1, 2])


def strategies_main(*argv):
    parser = argparse.ArgumentParser(description='Read a configuration file.')
    parser.add_argument('config_file', help='Path to the configuration file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)


    ## get history
    if config["mktdata"]["source"] == "deribit":
        history = deribit_history_main('get', config["strategy"]["currency"], 'deribit', config["backtest"]["backtest_window"],
                                       **config)
    elif config["mktdata"]["source"] == "kaiko":
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=config["backtest"]["backtest_window"])
        history = kaiko_history(config["strategy"]["currency"], start, end, config["mktdata"])

    ## new portfolio
    prev_mkt = history[0]
    # need to hack that to have equity in usd...
    portfolio = Strategy.build_strategy(market=prev_mkt, config=config['strategy'])
    prev_portfolio = copy.deepcopy(portfolio)

    ## run backtest
    series = []
    for run_i, market in enumerate(history[1:]):
        # all book-keeping
        prev_portfolio.process_settlements(market, prev_mkt)

        # all the hedging steps
        portfolio.set_rebalancing_rules(market, **config['strategy'])
        portfolio.cleanup_positions(market)

        # all the info
        series += [display_current(portfolio, prev_portfolio, market, prev_mkt)]

        # ...and move on
        prev_mkt = market
        prev_portfolio = copy.deepcopy(portfolio)

        if run_i % 100 == 0:
            print(f'{run_i} at {datetime.now(timezone.utc)}')

    display = pd.concat(series, axis=1)
    display.loc[(['predict', 'actual'], slice(None), slice(None))] = display.loc[
        (['predict', 'actual'], slice(None), slice(None))].cumsum(axis=1)
    display.sort_index(axis=0, inplace=True)

    filename = os.path.join(os.sep, os.getcwd(),'logs', f'run_{datetime.now(timezone.utc).strftime("%Y-%m-%d-%Hh")}')  # should be linear 1 to 1
    with pd.ExcelWriter(f'{filename}.xlsx', engine='xlsxwriter', mode='w') as writer:
        display.columns = [t.replace(tzinfo=None) for t in display.columns]
        display = display.map(lambda x: x.replace(tzinfo=None) if isinstance(x, datetime) else x)
        display.T.to_excel(writer, sheet_name=f"{config['strategy']['currency']}_{config['strategy']['gamma_tenor']}")
        pd.DataFrame(
            index=['params'],
            data={
                'currency': [config['strategy']['currency']],
                'gamma_tenor': [config['strategy']['gamma_tenor']],
            },
        ).to_excel(writer, sheet_name='params')
    display.to_pickle(f'{filename}.pickle')


if __name__ == "__main__":
    strategies_main(*sys.argv[1:])
