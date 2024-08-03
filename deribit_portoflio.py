import copy
from abc import abstractmethod, ABC
import argparse
import yaml

from deribit_history import *
from deribit_marketdata import kaiko_history, Market, Currency
from instrument import Instrument, Cash, InverseFuture, InversePerpetual, Option, CashFlow


class Position:
    def __init__(self, instrument, notional, label, cash_flow_position):
        self.instrument: Instrument = instrument
        self.notional: float = notional
        self.new_deal_pnl: float = 0
        self.label: str = label  # this str is handy to describe why position is here, eg gamma_hedge, delta_hedge...
        self.cash_flows: list[CashFlow] = []
        self.cash_flow_position: Position = cash_flow_position  # can be None

    def apply(self, greek, market, **kwargs):
        return self.notional * getattr(self.instrument, greek)(market, **kwargs) if hasattr(self.instrument,
                                                                                            greek) else 0

    def margin_value(self, market):
        kwargs = {'notional': self.notional} if type(self.instrument) == Option else {}
        return self.notional * getattr(self.instrument, 'margin_value')(market, **kwargs)

    def get_cash_flow(self, market, prev_market) -> list[CashFlow]:
        return [cash_flow for cash_flow in self.cash_flows if prev_market.t < cash_flow.settlement_time <= market.t]


class DeltaStrategy(ABC):
    @abstractmethod
    def delta_hedge(self, delta: float) -> float:
        raise NotImplementedError


class DeltaThresholdStrategy(DeltaStrategy):
    def __init__(self, target_inf: float, target_sup: float):
        assert target_inf < target_sup, 'target[0] should be < target[1]'
        self.target_inf: float = target_inf
        self.target_sup: float = target_sup
        self.target: float = 0.5*(target_inf + target_sup)

    def delta_hedge(self, delta: float) -> float:
        if self.target_inf <= delta <= self.target_sup:
            return 0
        else:
            return self.target - delta


class Strategy(ABC):
    """
    a list of position mixin
    hedging buisness logic is here
    """

    def __init__(self, currency, equity):
        # TODO: book object when we implement portfolio margining
        self.positions = [Position(Cash(currency), equity, label='initial_cash', cash_flow_position=None)]
        self.greeks_cache = {}

    @staticmethod
    def build_strategy(config: dict, market: Market):
        strategy_type = config.pop('type')
        return globals()[strategy_type](market=market, **config)

    def cash_flows(self, settlement_time) -> list[CashFlow]:
        return [cash_flow
                for position in self.positions
                for cash_flow in position.cash_flows
                if cash_flow.settlement_time == settlement_time]

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
            positions_to_closeout = [position for position in self.positions if
                                    position.label == hedge_params['replace_label']]

            [self.new_trade(position.instrument,
                            -position.notional,
                            price_coin='slippage',
                            market=market,
                            label=hedge_params['replace_label']
                            ) for position in positions_to_closeout]
        portfolio_greek = self.apply(greek, market, **greek_params)

        hedge_notional = (target - portfolio_greek) / hedge_instrument_greek
        self.new_trade(hedge_instrument,
                       hedge_notional,
                       price_coin='slippage',
                       market=market,
                       label=f'{greek}_hedge')

    @abstractmethod
    def rebalance(self, market, underlying, **kwargs):
        """
        This is where the hedging steps are detailed
        """
        raise NotImplementedError

    def process_settlements(self, market, prev_market):
        for position in self.positions:
            if hasattr(position.instrument, 'cash_flow'):
                # generate cashflows
                for cash_flow in getattr(position.instrument, 'cash_flow')(market, prev_market):
                    # append to list of cash flows
                    cash_flow.notional *= position.notional
                    position.cash_flows.append(cash_flow)

                    # increment cash flow position notional
                    position.cash_flow_position.notional += cash_flow.notional


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

    def new_trade(self, instrument, notional, price_coin: typing.Union[str, float], market,
                  label):
        '''this assumes '''
        # adjust price for slippage
        if price_coin == 'slippage':
            assert market is not None, 'market is None'
            # empty: slippage using parallel greeks
            kwargs = {}
            slippage_cost = sum(
                market.slippage[greek] * np.abs(getattr(instrument, greek)(market, **kwargs)) for greek in
                instrument.__class__.greeks_available)
            price_coin = instrument.pv(market) + slippage_cost * np.sign(notional)

        # create position and its cash flow if not exist
        position = next(iter(x for x in self.positions
                             if x.instrument.symbol == instrument.symbol
                             and x.label == label), None)
        if not position:
            cash_flow_position = Position(instrument=Cash(instrument.settlement_currency),
                                          notional=0,
                                          label=f'cash_flow_{label}',
                                          cash_flow_position=None)
            position = Position(instrument, 0, label, cash_flow_position)
            self.positions += [position, cash_flow_position]

        # assign new deal, increment cash flow for price, and notional
        position.new_deal_pnl = notional * (instrument.pv(market) - price_coin)
        position.cash_flow_position.notional -= notional * price_coin / (market.spot if instrument.settlement_currency == market.currency else 1)
        position.notional += notional


class ShortGamma(Strategy):
    def __init__(self, **kwargs):
        # equity need to be converted to coin
        super().__init__(kwargs['currency'], kwargs['equity'] / kwargs['market'].spot)

        # delta limits need to be converted to delta per unit spot move
        delta_inf = kwargs['delta_hedging']['target'][0] * kwargs['equity'] / kwargs['market'].spot
        delta_sup = kwargs['delta_hedging']['target'][1] * kwargs['equity'] / kwargs['market'].spot
        self.delta_strategy: DeltaStrategy = DeltaThresholdStrategy(delta_inf, delta_sup)

        # theta target need to be converted to theta per day
        self.theta_target_inf = kwargs['theta_hedge']['target'][0] * kwargs['equity'] / 365.25
        self.theta_target_sup = kwargs['theta_hedge']['target'][1] * kwargs['equity'] / 365.25
        self.theta_target = (self.theta_target_inf + self.theta_target_sup) / 2

    def rebalance(self, market, currency: Currency, **kwargs):
        """
        This is where the hedging steps are detailed
        """
        # 1: target theta
        self.theta_hedge(market, currency, **kwargs)

        # 3: hedge delta, replacing InversePerpetual
        # should rather find closest but usually the same
        # future_maturity_sec = next((T for T in market.fwdcurve.series.index if market.t + timedelta(seconds=T) >= option_maturity),
        #                 max(market.fwdcurve.series.index))
        # future_maturity = market.t + timedelta(seconds=future_maturity_sec)
        self.delta_hedge(market, currency)

    def theta_hedge(self, market: Market, currency: Currency, **kwargs):
        '''
        to buy theta: hedge using either the benchmark, or an existing instrument if one has enough theta.
        to sell theta: reduce size by order of position theta
        :param market:
        :param currency:
        :param kwargs:
        :return:
        '''
        # find the biggest theta instrument in the current book
        biggest_instrument_theta = 0
        total_theta = 0
        closest_option: Option = None
        for option in self.positions[1:]:
            if type(option.instrument) == Option:
                theta = option.instrument.theta(market)
                total_theta += theta * option.notional
                if -theta > biggest_instrument_theta:
                    biggest_instrument_theta = -theta
                    closest_option = option.instrument

        # to buy theta: hedge using either the benchmark, or the biggest theta instrument if it has enough theta.
        if total_theta < self.theta_target_inf:
            # benchmark_theta is the theta of the atm at gamma_tenor
            benchmark_expiry = market.t + timedelta(seconds=kwargs['theta_hedge']['gamma_tenor'] * 24 * 3600)
            benchmark_strike = market.nearest_strike(benchmark_expiry)
            benchmark_option = Option(currency, benchmark_strike, benchmark_expiry, 'S', settlement_currency=currency)
            benchmark_theta = benchmark_option.theta(market)

            if biggest_instrument_theta > kwargs['theta_hedge']['theta_hurdle'] * -benchmark_theta:
                hedge_instrument = closest_option
            else:
                hedge_instrument = benchmark_option
            notional = (self.theta_target - total_theta) / hedge_instrument.theta(market)
            self.new_trade(hedge_instrument, notional=notional, price_coin='slippage', market=market,
                           label='theta_hedge')
        elif total_theta > self.theta_target_sup:
            cur_theta = total_theta
            for option in reversed(sorted([option for option in self.positions[1:]
                                  if type(option.instrument) == Option],
                                 key=lambda o: o.apply('theta', market))):
                theta = option.apply('theta', market)
                # if unwinding the whole position is not enough
                if cur_theta - self.theta_target > theta:
                    notional = -option.notional
                    self.new_trade(instrument=option.instrument,
                                   notional=notional,
                                   price_coin='slippage',
                                   market=market,
                                   label='theta_hedge')
                    cur_theta -= theta
                # if unwinding the whole position is too much, unwind partially and stop
                else:
                    notional = - option.notional * (cur_theta - self.theta_target) / theta
                    self.new_trade(instrument=option.instrument,
                                   notional=notional,
                                   price_coin='slippage',
                                   market=market,
                                   label='theta_hedge')
                    break

    def delta_hedge(self, market: Market, currency: Currency, **kwargs):
        hedge_instrument = InversePerpetual(currency, settlement_currency=market.currency)
        delta = self.apply('delta', market)
        notional = self.delta_strategy.delta_hedge(delta)
        self.new_trade(instrument=hedge_instrument,
                       notional=notional,
                       price_coin='slippage',
                       market=market,
                       label='delta_hedge')

def display_current(portfolio, prev_portfolio, market, prev_market):
    predictors = {'delta': lambda x: x.apply(greek='delta', market=prev_market) * (
            market.spot - prev_market.spot),
                  'gamma': lambda x: 0.5 * x.apply(greek='gamma', market=prev_market) * (
                          market.spot - prev_market.spot) ** 2,
                  'vega': lambda x: x.apply(greek='vega', market=prev_market) * (
                          market.vol.interpolate(x.instrument.strike, x.instrument.maturity) -
                          prev_market.vol.interpolate(x.instrument.strike, x.instrument.maturity))
                  if hasattr(x.instrument, 'vega') else 0,
                  'theta': lambda x: x.apply(greek='theta', market=prev_market) * (
                          market.t - prev_market.t).total_seconds() / 3600 / 24,
                  'IR01': lambda x: x.apply(greek='delta', market=prev_market) * (
                          1 - market.spot / prev_market.spot * prev_market.fwdcurve.interpolate(
                      x.instrument.maturity) / market.fwdcurve.interpolate(x.instrument.maturity)) * 100
                  if hasattr(x.instrument, 'IR01') else 0}
    labels = {position.label for position in portfolio.positions}

    # 1: mkt
    display_market = pd.Series(
        {('mkt', data, 'total'): getattr(market, data) for data in ['t', 'spot', 'funding_rate']}
        | {('mkt', 'vol', 'total'): sum(market.vol.interpolate(position.instrument.strike, position.instrument.maturity)
                                        * position.apply('vega', market)
                                        for position in portfolio.positions if type(position.instrument) == Option) /
                                    (1e-18 + sum(position.apply('vega', market) for position in portfolio.positions if
                                                 type(position.instrument) == Option))})

    # 2 : inventory
    inventory = pd.Series(
        {('portfolio', position.label, position.instrument.symbol):
             position.notional
         for position in portfolio.positions})

    # 3 : greeks
    greeks = pd.Series(
        {('risk', greek, label):
             sum(position.apply(greek, market) for position in portfolio.positions if position.label == label)
         for greek in ['pv'] + list(predictors.keys())
         for label in labels})

    # 4: predict = mkt move for prev_portfolio, and cash flows
    predict = pd.Series(
        {('predict', greek, label):
             sum(predictor(position) for position in prev_portfolio.positions if position.label == label)
         for greek, predictor in predictors.items()
         for label in labels}
        | {('predict', 'cash_flow', label): # should be on p0 but doesn't have cash flows yet ...
               sum(cf.value(market)
                   for p1 in portfolio.positions if p1.label == label
                   for cf in p1.get_cash_flow(market, prev_market))
           for label in labels})

    # 5: actual = prev_portfolio cash_flows + new deal + unexplained
    actual = pd.Series(
        {('actual', 'unexplained', label):  # the unexplained = dPV-predict, for old portoflio only
             sum(p0.apply('pv', market) - p0.apply('pv', prev_market)  # dPV ...
                 for p0 in prev_portfolio.positions
                 if p0.label == label)
             - predict[('predict', slice(None), label)].sum()  # ... - the predict
         for label in labels}  # ....only for old positions. New positions contribute to new_deal only.
        | {('actual', 'new_deal', label):
               sum(p1.new_deal_pnl
                   for p1 in portfolio.positions if p1.label == label)  # new deals only contribute here.
           for label in labels}
        | {('actual', 'portfolio_chg', label):
               sum(p1.apply('pv', market)
                   for p1 in portfolio.positions
                   if p1.label == label)
               - sum(p0.apply('pv', market)
                     for p0 in prev_portfolio.positions
                     if p0.label == label)
           for label in labels})

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
    series = [display_current(portfolio, prev_portfolio, prev_mkt, prev_mkt)]

    ## run backtest
    for run_i, market in enumerate(history[1:]):
        # all the hedging steps
        if run_i == 0 or not config['backtest']['test']:
            portfolio.rebalance(market, **config['strategy'])
        # all book-keeping
        portfolio.process_settlements(market, prev_mkt)
        portfolio.cleanup_positions(market)

        # all the info
        series += [display_current(portfolio, prev_portfolio, market, prev_mkt)]
        for position in portfolio.positions:
            position.new_deal_pnl = 0

        # ...and move on
        prev_mkt = market
        prev_portfolio = copy.deepcopy(portfolio)

        if run_i % 100 == 0:
            print(f'{run_i} at {datetime.now(timezone.utc)}')

    display = pd.concat(series, axis=1)
    if not config['backtest']['test']:
        display.loc[(['predict', 'actual'], slice(None), slice(None))] = display.loc[
            (['predict', 'actual'], slice(None), slice(None))].cumsum(axis=1)
    display.sort_index(axis=0, inplace=True)

    date_suffix = datetime.now(timezone.utc).strftime("%Y-%m-%d-%Hh")
    filename = os.path.join(os.sep, os.getcwd(),'logs', f'run_')  # should be linear 1 to 1
    with pd.ExcelWriter(f'{filename}.xlsx', engine='xlsxwriter', mode='w') as writer:
        display.columns = [t.replace(tzinfo=None) for t in display.columns]
        display = display.map(lambda x: x.replace(tzinfo=None) if isinstance(x, datetime) else x)
        display.T.to_excel(writer, header=[0,1,2], sheet_name=f"{config['strategy']['currency']}_{config['strategy']['theta_hedge']['gamma_tenor']}")
        pd.DataFrame(
            index=['params'],
            data={
                'currency': [config['strategy']['currency']],
                'gamma_tenor': [config['strategy']['theta_hedge']['gamma_tenor']],
            },
        ).to_excel(writer, sheet_name='params')
    filename = os.path.join(os.sep, os.getcwd(), 'logs', 'run.csv')
    display.stack().reset_index().rename(columns={'level_3': 'datetime', 0: 'value'}).to_csv(filename)


if __name__ == "__main__":
    strategies_main(*sys.argv[1:])
