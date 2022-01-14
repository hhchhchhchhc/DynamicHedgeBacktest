import numpy as np
import pandas as pd

from ftx_snap_basis import *
from ftx_utilities import *
from ftx_ftx import *

### History holds times series (fetched or read).
class RawHistory:
    def __init__(self, futures: pd.DataFrame) -> None:
        self._futures=futures ## need name, symbol, underlying, type, expiry
        self._timeframe='1h'
        self._start_date = datetime.now()
        self._end_date = datetime.now()
        self._history = pd.DataFrame()
    def build(self,exchange: Exchange,timeframe: str,
                 start_date: datetime.date,
                 end_date: datetime.date,
                filename: str='') -> None:
        self._timeframe = timeframe
        self._start_date = start_date
        self._end_date = end_date
        self._history = RawHistory.read_or_fetch_history(self._futures, exchange,
                                                            self._timeframe, self._start_date, self._end_date,
                                                            filename)
    @staticmethod
    def read_or_fetch_history(futures: pd.DataFrame, exchange: Exchange,
                 timeframe: str,
                 start_date: datetime.date,
                 end_date: datetime.date,
                 filename: str) -> pd.DataFrame:
        #### read history, fetch if absent
        try:
            hy_history = from_parquet(filename)
        except:
            hy_history = RawHistory.fetch_history(futures, exchange,
                                                            timeframe, start_date, end_date)
            to_parquet(hy_history, filename)
            return hy_history

        ### if more dates, fetch history
        if not ((start_date in hy_history.index)&(end_date in hy_history.index)): ## ignore timeframe...
            hy_history = RawHistory.fetch_history(futures, exchange,
                                                            timeframe, start_date, end_date)
            to_parquet(hy_history, filename)
        else:
            existing_futures = [name.split('/')[0] for name in hy_history.columns]
            new_futures = futures[futures['symbol'].isin(existing_futures) == False]
            ### if more futures, only fetch those
            if new_futures.empty == False:
                hy_history = pd.concat([hy_history,
                                    RawHistory.fetch_history(new_futures, exchange,
                                                            timeframe, start_date, end_date)],
                                       join='outer', axis=1)
                to_parquet(hy_history, filename)
        return hy_history

    @staticmethod
    def fetch_history(futures: pd.DataFrame, exchange: Exchange,
                 timeframe: str,
                 start: datetime.date,
                 end: datetime.date) -> pd.DataFrame:

        if futures[futures['type']=='perpetual'].empty: perp_funding_data=[]
        else:
            perp_funding_data=futures[futures['type']=='perpetual'].apply(lambda f:
                                RawHistory.fetch_funding_history(f,exchange,start,end),axis=1).to_list()

        future_rate_data = futures.apply(lambda f: RawHistory.fetch_rate_history(f, exchange,timeframe,start,end),axis=1).to_list()
        spot_data=futures.apply(lambda f: RawHistory.fetch_price_history(f, exchange,timeframe,start,end),axis=1).to_list()
        borrow_data=[RawHistory.fetch_borrow_history(f, exchange,start,end) for f in futures['underlying'].unique()]\
                    +[RawHistory.fetch_borrow_history('USD',exchange,start,end)]

        data= pd.concat(perp_funding_data
                        +future_rate_data
                        +spot_data
                        +borrow_data,join='outer',axis=1)
        return data

    ### only perps, only borrow and funding, only hourly
    @staticmethod
    def fetch_borrow_history(spot: str, exchange: Exchange,
                             start: datetime.date,
                             end: datetime.date) -> pd.DataFrame:
        max_funding_data = int(500)  # in hour. limit is 500 :(
        resolution = exchange.describe()['timeframes']['1h']
        print('borrow_history: '+spot)

        ### grab data per batch of 5000
        borrow_data=pd.DataFrame()
        end_time = end.timestamp()
        start_time = (datetime.fromtimestamp(end_time) - timedelta(hours=max_funding_data)).timestamp()

        while end_time > start.timestamp():
            if start_time<start.timestamp(): start_time=start.timestamp()

            datas=fetch_borrow_rate_history(exchange,spot,start_time,end_time)
            if len(datas)==0: break
            borrow_data= pd.concat([borrow_data,datas], join='outer', axis=0)

            end_time = (datetime.fromtimestamp(start_time) - timedelta(hours=1)).timestamp()
            start_time = (datetime.fromtimestamp(end_time) - timedelta(hours=max_funding_data)).timestamp()

        if len(borrow_data)>0:
            borrow_data = borrow_data.astype(dtype={'time': 'int64'}).set_index(
                'time')[['coin','rate','size']]
            borrow_data = borrow_data[(borrow_data.index.duplicated() == False)&(borrow_data['coin']==spot)]
            data = pd.DataFrame()
            data[spot+'/rate/borrow'] = borrow_data['rate']
            data[spot+'/rate/size'] = borrow_data['size']
            data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
            data = data[~data.index.duplicated()].sort_index()
        else: data=pd.DataFrame()

        return data

    ######### annualized funding for perps
    @staticmethod
    def fetch_funding_history(future: pd.Series, exchange: Exchange,
                 start: datetime.date,
                 end: datetime.date) -> pd.DataFrame:
        max_funding_data = int(500)  # in hour. limit is 500 :(
        resolution = exchange.describe()['timeframes']['1h']
        print('funding_history: ' + future['name'])

        ### grab data per batch of 5000
        funding_data=pd.DataFrame()
        end_time = end.timestamp()
        start_time = (datetime.fromtimestamp(end_time) - timedelta(hours=max_funding_data)).timestamp()

        while end_time > start.timestamp():
            if start_time<start.timestamp(): start_time=start.timestamp()

            data = fetch_funding_rate_history(exchange, future, start_time, end_time)
            if len(data) == 0: break
            funding_data = pd.concat([funding_data, data], join='outer', axis=0)

            end_time = (datetime.fromtimestamp(start_time) - timedelta(hours=1)).timestamp()
            start_time = (datetime.fromtimestamp(end_time) - timedelta(hours=max_funding_data)).timestamp()

        if len(funding_data) > 0:
            funding_data = funding_data.astype(dtype={'time': 'int64'}).set_index(
                'time')[['rate']]
            funding_data = funding_data[(funding_data.index.duplicated() == False)]
            data = pd.DataFrame()
            data[future['name'] + '/rate/funding'] = funding_data['rate']
            data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
            data = data[~data.index.duplicated()].sort_index()
        else:
            data = pd.DataFrame()

        return data

    #### annualized rates for futures and perp
    @staticmethod
    def fetch_rate_history(future: pd.Series, exchange: Exchange,
                 timeframe: str,
                 start: datetime.date,
                 end: datetime.date) -> pd.DataFrame:
        max_mark_data = int(1500)
        resolution = exchange.describe()['timeframes'][timeframe]
        print('rate_history: ' + future['name'])

        indexes = []
        mark = []
        data = []
        end_time = end.timestamp()
        start_time = (datetime.fromtimestamp(end_time) - timedelta(seconds=max_mark_data * int(resolution))).timestamp()

        while end_time >= start.timestamp():
            if start_time < start.timestamp(): start_time = start.timestamp()
            new_mark = fetch_ohlcv(exchange, future['symbol'], timeframe=timeframe, start=start_time, end=end_time)
            new_indexes = exchange.publicGetIndexesMarketNameCandles(
                params={'start_time': start_time, 'end_time': end_time, 'market_name': future['underlying'],
                        'resolution': resolution})['result']

            if (len(new_mark) == 0): break
            mark.extend(new_mark)
            indexes.extend(new_indexes)
            end_time = (datetime.fromtimestamp(start_time) - timedelta(seconds=int(resolution))).timestamp()
            start_time = (datetime.fromtimestamp(end_time) - timedelta(
                seconds=max_mark_data * int(resolution))).timestamp()

        column_names = ['t', 'o', 'h', 'l', 'c', 'volume']

        ###### indexes
        indexes = pd.DataFrame(indexes, dtype=float).astype(dtype={'time': 'int64'}).set_index('time')
        indexes = indexes.drop(columns=['startTime', 'volume'])

        ###### marks
        mark = pd.DataFrame(columns=column_names, data=mark).astype(dtype={'t': 'int64'}).set_index('t')

        mark.columns = ['mark/' + column for column in mark.columns]
        indexes.columns = ['indexes/' + column for column in indexes.columns]
        data = mark.join(indexes, how='inner')

        ########## rates from index to mark
        if future['type'] == 'future':
            expiry_time = dateutil.parser.isoparse(future['expiry']).timestamp()
            data['rate/T'] = data.apply(lambda t: (expiry_time - int(t.name) / 1000) / 3600 / 24 / 365, axis=1)

            data['rate/c'] = data.apply(
                lambda y: calc_basis(y['mark/c'],
                                     indexes.loc[y.name, 'indexes/close'], future['expiryTime'],
                                     datetime.fromtimestamp(int(y.name / 1000), tz=None)), axis=1)
            data['rate/h'] = data.apply(
                lambda y: calc_basis(y['mark/h'], indexes.loc[y.name, 'indexes/high'], future['expiryTime'],
                                     datetime.fromtimestamp(int(y.name / 1000), tz=None)), axis=1)
            data['rate/l'] = data.apply(
                lambda y: calc_basis(y['mark/l'], indexes.loc[y.name, 'indexes/low'], future['expiryTime'],
                                     datetime.fromtimestamp(int(y.name / 1000), tz=None)), axis=1)
        elif future['type'] == 'perpetual': ### 1h funding = (mark/spot-1)/24
            data['rate/c'] = (mark['mark/c'] / indexes['indexes/close'] - 1)*365.25
            data['rate/h'] = (mark['mark/h'] / indexes['indexes/high'] - 1)*365.25
            data['rate/l'] = (mark['mark/l'] / indexes['indexes/low'] - 1)*365.25
        else:
            print('what is ' + future['symbol'] + ' ?')
            return
        data.columns = [future['symbol'] + '/' + c for c in data.columns]
        data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
        data = data[~data.index.duplicated()].sort_index()

        return data

    @staticmethod
    def fetch_price_history(symbol: str, exchange: Exchange,
                 timeframe: str,
                 start: datetime.date,
                 end: datetime.date) -> pd.DataFrame:
        max_mark_data = int(5000)
        resolution = exchange.describe()['timeframes'][timeframe]
        print('price_history: ' + symbol)

        spot =[]
        end_time = end.timestamp()
        start_time = (datetime.fromtimestamp(end_time) - timedelta(seconds=max_mark_data * int(resolution))).timestamp()

        while end_time >= start.timestamp():
            if start_time < start.timestamp(): start_time = start.timestamp()
            new_spot = fetch_ohlcv(exchange, symbol, timeframe=timeframe, start=start_time, end=end_time)

            if (len(new_spot) == 0): break
            spot.extend(new_spot)
            end_time = (datetime.fromtimestamp(start_time) - timedelta(seconds=int(resolution))).timestamp()
            start_time = (datetime.fromtimestamp(end_time) - timedelta(
                seconds=max_mark_data * int(resolution))).timestamp()

        column_names = ['t', 'o', 'h', 'l', 'c', 'volume']

        ###### spot
        data = pd.DataFrame(columns=column_names, data=spot).astype(dtype={'t': 'int64', 'volume': 'float'}).set_index('t')
        data.columns = [symbol.replace('USD/','') + '/price/' + column for column in data.columns]
        data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
        data = data[~data.index.duplicated()].sort_index()

        return data

### EstimationModel describes how to model distribution of assets. Currently historical ewm but could add hw2f.
class EstimationModel:
    def __init__(self,signal_horizon: timedelta) ->None:
        self._signal_horizon=signal_horizon
### AssetDefinition describes atomic positions. Currently only \int{perp-cash} per (name,direction)
class Asset:
    def __init__(self,holding_period: timedelta,direction: int,future: pd.Series) ->None:
        self._holding_period=holding_period ## TODO this rather belongs to Strategy
        self._direction=direction
        self._future=future
    def carry(self,start,end) ->float:
        return 0 # TODO carry
    def transactionCost(self) ->float:
        return 0  # TODO transactionCost
### ModelledAssets holds risk/reward of atomic positions, whose weights will then be optimized by PortfolioBuilder.
### Needs RawHistory, EstimationParameters (eg: ewm window), Asset (eg: integral over holding period)
class ModelledAssets:
    def __init__(self, signal_horizon: timedelta, holding_period: timedelta) -> None:
        self._futures = raw_history._futures
        self._assets = [Asset(holding_period, direction, future)
                        for future in raw_history._futures for direction in [-1,1]]
        self._estimator = EstimationModel(signal_horizon)

    ### Constant rate slippage calculation. Reads order book, or applies override.
    def fetch_rate_slippage(self,exchange: Exchange,
                        slippage_override: int=-999, slippage_orderbook_depth: float=0, slippage_scaler: float=1.0,
                        params: dict={'override_slippage':True}) -> None:
        # -------------------- transaction costs---------------
        # add slippage, fees and speed. Pls note we will trade both legs, at entry and exit.
        # Unless slippage override, calculate from orderbook (override Only live is supported).
        holding_period=self._assets._holding_period
        point_in_time=datetime.now()
        markets=exchange.fetch_markets()
        futures = self._futures
        if slippage_override != -999:
            futures['bid_rate_slippage'] = slippage_override
            futures['ask_rate_slippage'] = slippage_override
        else:
            fees=(exchange.fetch_trading_fees()['taker']+exchange.fetch_trading_fees()['maker']*0)#maker fees 0 with 26 FTT staked
            ### relative semi-spreads incl fees, and speed
            if slippage_orderbook_depth == 0.0:
                futures['spot_ask'] = fees + futures.apply(lambda f: 0.5 * (
                            float(find_spot_ticker(markets, f, 'ask')) / float(
                        find_spot_ticker(markets, f, 'bid')) - 1), axis=1) * slippage_scaler
                futures['spot_bid'] = -futures['spot_ask']
                futures['future_ask'] = fees + 0.5 * (
                            futures['ask'].astype(float) / futures['bid'].astype(float) - 1) * slippage_scaler
                futures['future_bid'] = -futures['future_ask']
                ### forget speed for now..
                # futures['speed'] = 0  ##*futures['future_ask'] ### just 0
            else:
                futures['spot_ask'] = futures['spot_ticker'].apply(
                    lambda x: mkt_at_size(exchange, x, 'asks', slippage_orderbook_depth)['slippage']) * slippage_scaler + fees
                futures['spot_bid'] = futures['spot_ticker'].apply(
                    lambda x: mkt_at_size(exchange, x, 'bids', slippage_orderbook_depth)['slippage']) * slippage_scaler - fees
                futures['future_ask'] = futures['name'].apply(
                    lambda x: mkt_at_size(exchange, x, 'asks', slippage_orderbook_depth)['slippage']) * slippage_scaler + fees
                futures['future_bid'] = futures['name'].apply(
                    lambda x: mkt_at_size(exchange, x, 'bids', slippage_orderbook_depth)['slippage']) * slippage_scaler - fees
                ### forget speed for now..
                # futures['speed'] = futures['name'].apply(
                #    lambda x: mkt_speed(exchange, x, depths).seconds)

            #### rate slippage assuming perps are rolled every perp_holding_period
            #### use centred bid ask for robustness
            futures['expiryTime'] = futures.apply(lambda x:
                                                  x['expiryTime'] if x['type'] == 'future'
                                                  else point_in_time + holding_period,
                                                  axis=1)  # .replace(tzinfo=timezone.utc)

            futures['bid_rate_slippage_in_' + str(slippage_orderbook_depth)] = futures.apply(lambda f: \
                         (f['future_bid_in_' + str(slippage_orderbook_depth)] - f['spot_ask_in_' + str(slippage_orderbook_depth)]) \
                         / np.max([1, (f['expiryTime'] - point_in_time).seconds/3600])
                                  *365.25*24,axis=1) # no less than 1h
            futures['ask_rate_slippage_in_' + str(slippage_orderbook_depth)] = futures.apply(lambda f: \
                         (f['future_ask_in_' + str(slippage_orderbook_depth)] - f['spot_bid_in_' + str(slippage_orderbook_depth)]) \
                         / np.max([1, (f['expiryTime'] - point_in_time).seconds/3600])
                                  *365.25*24,axis=1) # no less than 1h
        ### not very parcimonious...
            self._futures['bid_rate_slippage']=futures['bid_rate_slippage_in_' + str(slippage_orderbook_depth)]
            self._futures['ask_rate_slippage']=futures['ask_rate_slippage_in_' + str(slippage_orderbook_depth)]

    def build_assets_history(self,raw_history: RawHistory) -> None:
        self.fetch_rate_slippage(futures, exchange, holding_period,
                                             slippage_override,slippage_orderbook_depth,slippage_scaler,
                                             params)
        ### remove blanks for this
        history = raw_history._history.fillna(method='ffill', limit=2, inplace=False)

        # ---------- compute max leveraged \int{carry moments}, long and short
        # for perps, compute carry history to estimate moments.
        # for future, funding is deterministic because rate change is compensated by carry change (well, modulo funding...)
        LongCarry = self._futures.apply(lambda f:
                                      f['bid_rate_slippage'] - history['USD/rate/borrow'] +
                                      history[f['name'] + '/rate/funding'] if f['type'] == 'perpetual'
                                      else np.NaN,
                                  axis=1).T
        LongCarry.columns = ('long/'+self._futures['name']).tolist()

        ShortCarry = self._futures.apply(lambda f:
                                       f['ask_rate_slippage'] - history['USD/rate/borrow'] +
                                       history[f['name'] + '/rate/funding'] if f['type'] == 'perpetual'
                                       else np.NaN
                                        + history[f['underlying'] + '/rate/borrow'],
                                   axis=1).T
        ShortCarry.columns = ('short/'+self._futures['name']).tolist()

        ### not sure how to define window as a timedelta :(
        hours=int(self._assets._holding_period.total_seconds() / 3600)
        intLongCarry = LongCarry.rolling(hours).mean().shift(periods=-hours)
        intShortCarry = ShortCarry.rolling(hours).mean().shift(periods=-hours)
        intUSDborrow = history['USD/rate/borrow'].rolling(hours).mean().shift(periods=-hours)
        self._history=pd.concat([intLongCarry,intShortCarry,intUSDborrow],axis=1)

    def build_assets_estimates(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        hours=int(self._assets._holding_period.total_seconds() / 3600)
        ### E_int and C_int are moments of the integral of annualized carry.
        Expectation = self._history.rolling(hours).median()
        Covariance = self._history.rolling(hours).cov()
        return Expectation,Covariance

class Strategy:
    def __init__(self,holding_period: timedelta) ->None:
        _holding_period=holding_period

def strategyOO():
    exchange = open_exchange('ftx')
    futures = pd.DataFrame(fetch_futures(exchange, includeExpired=False))

    funding_threshold = 1e4
    volume_threshold = 1e6
    type_allowed = 'perpetual'
    max_nb_coins = 10
    carry_floor = 0.4
    slippage_override = 2e-4  #### this is given by mktmaker
    #    slippage_scaler=0.5
    #    slippage_orderbook_depth=0
    signal_horizon = timedelta(days=3)
    backtest_window = timedelta(days=200)
    holding_period = timedelta(days=2)
    concentration_limit = 0.25
    loss_tolerance = 0.01

    enriched = enricher(exchange, futures)
    pre_filtered = enriched[
        (enriched['expired'] == False)
        & (enriched['funding_volume'] * enriched['mark'] > funding_threshold)
        & (enriched['volumeUsd24h'] > volume_threshold)
        & (enriched['tokenizedEquity'] != True)
        & (enriched['type'].isin(type_allowed))]

    today=datetime.today().replace(hour=0,minute=0,second=0,microsecond=0)
    history=RawHistory(pre_filtered)
    history.build(exchange, '1h',today-timedelta(days=30),today,"test.parquet")
    carries=ModelledAssets(history,signal_horizon,holding_period)
    carries.build_assets_history(history,exchange,today)
    carries.build_assets_processes()[0].to_excel("testE.xlsx")
    carries.build_assets_processes()[1].to_excel("testC.xlsx")

strategyOO()

#### naive plex of a perp,assuming cst size.
def perp_carry_backtest(future,rates_history,
                   end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                   start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30)):
    prekey=future['name']+'/PnL/'
    rates_history=rates_history[start:end]
    USDborrow = rates_history['USD/rate/borrow']
    pnlHistory=pd.DataFrame()
    pnlHistory[prekey+'funding'] = (rates_history[future['name']+'/rate/funding']*future['maxPos'])/365.25/24
    pnlHistory[prekey+'borrow'] = (-USDborrow*future['maxPos']+
                rates_history[future['underlying']+'/rate/borrow']*np.min(future['maxPos'],0))/365.25/24
    pnlHistory[prekey+'maxCarry'] = pnlHistory[prekey+'funding']+pnlHistory[prekey+'borrow']
    return pnlHistory

#### naive plex of a perp,assuming cst size and no rate move.
def future_carry_backtest(future,rates_history,
                   end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                   start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30)):
    prekey = future['name'] + '/PnL/'
    rates_history=rates_history[start:end]
    USDborrow = rates_history['USD/rate/borrow']
    pnlHistory = pd.DataFrame(index=rates_history.index)
    #### ignores future curves since future no cst maturity
    pnlHistory[prekey +'borrow'] = (-USDborrow * future['maxPos'] -
                rates_history[future['underlying'] + '/rate/borrow']*np.min(future['maxPos'],0))/365.25/24.0
    pnlHistory[prekey+'funding'] = (rates_history.loc[start+timedelta(hours=1),future['name']+'/rate/c']
                *future['maxPos'])/365.25/24.0
    pnlHistory[prekey+'maxCarry'] = pnlHistory[prekey+'funding']+pnlHistory[prekey+'borrow']
    return pnlHistory

def carry_backtest(future,rates_history,
                   end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                   start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30)):
    if future['type'] == 'perpetual':
        return perp_carry_backtest(future, rates_history, start=start, end=end)
    elif future['type'] == 'future':
        return future_carry_backtest(future, rates_history, start=start, end=end)

def max_leverage_carry(futures,rates_history,
                       end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                       start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30)):
    data = pd.concat(futures.apply(lambda f:carry_backtest(f,rates_history,start=start,end=end), axis = 1).to_list(),join='inner', axis=1)

    return data