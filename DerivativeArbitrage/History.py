import pandas as pd

from ftx_utilities import *
from ftx_ftx import *

### History holds times series (fetched or read).
class RawHistory:
    def __init__(self, futures: pd.DataFrame, exchange: Exchange) -> None:
        self.futures=futures ## need name, symbol, underlying, type, expiry
        self.exchange=exchange
        self.timeframe=str()
        self.start_date = datetime()
        self.end_date = datetime()
        self.hy_history = pd.DataFrame()
    def build(self,timeframe: str,
                 start_date: datetime.date,
                 end_date: datetime.date) -> None:
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.hy_history = RawHistory.read_or_fetch_history(futures, exchange, timeframe, start_date, end_date)
    @staticmethod
    def read_or_fetch_history(futures: pd.DataFrame, exchange: Exchange,
                 timeframe: str,
                 start_date: datetime.date,
                 end_date: datetime.date) -> pd.DataFrame:
        #### read history, fetch if absent
        try:
            hy_history = from_parquet("history.parquet")
        except:
            hy_history = fetch_history(futures, exchange, timeframe, end_date, start_date)
            to_parquet(hy_history, "history.parquet")
            return hy_history

        ### if more dates, fetch history
        if not ((start_date in hy_history.index)&(end_date in hy_history.index)): ## ignore timeframe...
            hy_history = fetch_history(futures, exchange, timeframe, end_date, start_date)
            to_parquet(hy_history, "history.parquet")
        else:
            existing_futures = [name.split('/')[0] for name in hy_history.columns]
            new_futures = futures[futures['symbol'].isin(existing_futures) == False]
            ### if more futures, only fetch those
            if new_futures.empty == False:
                hy_history = pd.concat([hy_history,
                                        fetch_history(new_futures, exchange, timeframe, end_date, start_date)],
                                       join='outer', axis=1)
                to_parquet(hy_history, "history.parquet")
        return hy_history

    @staticmethod
    def fetch_history(self, futures: pd.DataFrame, exchange: Exchange,
                 timeframe: str,
                 start: datetime.date,
                 end: datetime.date) -> pd.DataFrame:

        if futures[futures['type']=='perpetual'].empty: perp_funding_data=[]
        else:
            perp_funding_data=futures[futures['type']=='perpetual'].apply(lambda f:fetch_funding_history(f,exchange),axis=1).to_list()

        future_rate_data = futures.apply(lambda f: fetch_rate_history(f, exchange),axis=1).to_list()
        spot_data=futures.apply(lambda f: fetch_spot_history(f, exchange,timeframe=timeframe),axis=1).to_list()
        borrow_data=[fetch_borrow_history(f, exchange) for f in futures['underlying'].unique()]\
                    +[fetch_borrow_history('USD',exchange)]

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

        return data

    @staticmethod
    def fetch_spot_history(future: pd.Series, exchange: Exchange,
                 timeframe: str,
                 start: datetime.date,
                 end: datetime.date) -> pd.DataFrame:
        max_mark_data = int(5000)
        resolution = exchange.describe()['timeframes'][timeframe]
        print('spot_history: ' + future['name'])

        spot =[]
        end_time = end.timestamp()
        start_time = (datetime.fromtimestamp(end_time) - timedelta(seconds=max_mark_data * int(resolution))).timestamp()

        while end_time >= start.timestamp():
            if start_time < start.timestamp(): start_time = start.timestamp()
            new_spot = fetch_ohlcv(exchange, future['symbol'], timeframe=timeframe, start=start_time, end=end_time)

            if (len(new_spot) == 0): break
            spot.extend(new_spot)
            end_time = (datetime.fromtimestamp(start_time) - timedelta(seconds=int(resolution))).timestamp()
            start_time = (datetime.fromtimestamp(end_time) - timedelta(
                seconds=max_mark_data * int(resolution))).timestamp()

        column_names = ['t', 'o', 'h', 'l', 'c', 'volume']

        ###### spot
        data = pd.DataFrame(columns=column_names, data=spot).astype(dtype={'t': 'int64', 'volume': 'float'}).set_index('t')
        data.columns = [future['symbol'] + '/spot/' + column for column in data.columns]
        data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]

        return data

### EstimationModel describes how to model distribution of assets. Currently historical ewm but could add hw2f.
class EstimationModel:
    def __init__(self,signal_horizon: datetime.timedelta) -->None:
        self.signal_horizon=signal_horizon
### AssetDefinition describes atomic positions. Currently only \int{perp-cash} per (name,direction)
class AssetDefinition:
    def __init__(self,holding_period: datetime.timedelta) -->None:
        self.holding_period=holding_period

### ModelledAssets holds risk/reward of atomic positions, whose weights will then be optimized by PortfolioBuilder.
### Needs RawHistory, EstimationParameters (eg: ewm window), AssetDefinition (eg: integral over holding period)
class ModelledAssets:
    def __init__(self, futures: pd.DataFrame, signal_horizon: datetime.timedelta, holding_period: datetime.timedelta) -> None:
        self.futures=futures
        self.asset_definition = AssetDefinition(holding_period)
        self.history=pd.DataFrame()
        self.estimation_parameter = EstimationModel(signal_horizon)

    ### Constant rate slippage calculation. Reads order book, or applies override.
    def fetch_rate_slippage(self,exchange: Exchange,
                      slippage_override: int=-999, depths: float=0, slippage_scaler: float=1.0) -> pd.DataFrame():
        # -------------------- transaction costs---------------
        # add slippage, fees and speed. Pls note we will trade both legs, at entry and exit.
        # Unless slippage override, calculate from orderbook (override Only live is supported).
        point_in_time=datetime.now()
        markets=exchange.fetch_markets()
        futures = self.futures
        if slippage_override != -999:
            futures['bid_rate_slippage'] = slippage_override
            futures['ask_rate_slippage'] = slippage_override
        else:
            fees = 2 * (exchange.fetch_trading_fees()['taker'] + exchange.fetch_trading_fees()['maker'])
            ### relative semi-spreads incl fees, and speed
            if depths == 0.0:
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
                    lambda x: mkt_depth(exchange, x, 'asks', depths)) * slippage_scaler + fees
                futures['spot_bid'] = futures['spot_ticker'].apply(
                    lambda x: mkt_depth(exchange, x, 'bids', depths)) * slippage_scaler - fees
                futures['future_ask'] = futures['name'].apply(
                    lambda x: mkt_depth(exchange, x, 'asks', depths)) * slippage_scaler + fees
                futures['future_bid'] = futures['name'].apply(
                    lambda x: mkt_depth(exchange, x, 'bids', depths)) * slippage_scaler - fees
                ### forget speed for now..
                # futures['speed'] = futures['name'].apply(
                #    lambda x: mkt_speed(exchange, x, depths).seconds)

            #### rate slippage assuming perps are rolled every perp_holding_period
            #### use centred bid ask for robustness
            futures['expiryTime'] = futures.apply(lambda x:
                                                  dateutil.parser.isoparse(x['expiry']).replace(tzinfo=None) if x[
                                                                                                                    'type'] == 'future'
                                                  else point_in_time + holding_period,
                                                  axis=1)  # .replace(tzinfo=timezone.utc)

            futures['bid_rate_slippage_in_' + str(size)] = futures.apply(lambda f: \
                                                                             (f['future_bid_in_' + str(size)] - f[
                                                                                 'spot_ask_in_' + str(size)]) \
                                                                             / np.max([1, (f[
                                                                                               'expiryTime'] - point_in_time).seconds * 365.25 * 24 * 3600]),
                                                                         axis=1)
            futures['ask_rate_slippage_in_' + str(size)] = futures.apply(lambda f: \
                                                                             (f['future_ask_in_' + str(size)] - f[
                                                                                 'spot_bid_in_' + str(size)]) \
                                                                             / np.max([1, (f[
                                                                                               'expiryTime'] - point_in_time).seconds * 365.25 * 24 * 3600]),
                                                                         axis=1)
            ### not very parcimonious...
            self.futures['expiryTime']=futures['expiryTime']
            self.futures['bid_rate_slippage']=futures['bid_rate_slippage']
            self.futures['ask_rate_slippage'] = futures['ask_rate_slippage']

    def build_assets_history(self,history: RawHistory, exchange: Exchange, point_in_time: datetime=datetime.now()) -> None:
        ModelledAssets.fetch_rate_slippage(exchange,slippage_override= 0.0002)
        ### remove blanks for this
        history = history.fillna(method='ffill', limit=2, inplace=False)

        # ---------- compute max leveraged \int{carry moments}, long and short
        # for perps, compute carry history to estimate moments.
        # for future, funding is deterministic because rate change is compensated by carry change (well, modulo funding...)
        LongCarry = self.futures.apply(lambda f:
                                      f['bid_rate_slippage'] - history['USD/rate/borrow'] +
                                      history[f['name'] + '/rate/funding'] if f['type'] == 'perpetual'
                                      else history.loc[point_in_time, f['name'] + '/rate/funding'],
                                  axis=1).T
        LongCarry.columns = ('long/'+self.futures['name']).tolist()

        ShortCarry = self.futures.apply(lambda f:
                                       f['ask_rate_slippage'] - history['USD/rate/borrow'] +
                                       history[f['name'] + '/rate/funding'] if f['type'] == 'perpetual'
                                       else history.loc[point_in_time, f['name'] + '/rate/funding']
                                            + history[f['underlying'] + '/rate/borrow'],
                                   axis=1).T
        ShortCarry.columns = ('short/'+self.futures['name']).tolist()

        ### not sure how to define window as a timedelta :(
        intLongCarry = LongCarry.rolling(int(self.asset_definition.holding_period.total_seconds() / 3600)).mean()
        intShortCarry = ShortCarry.rolling(int(self.asset_definition.holding_period.total_seconds() / 3600)).mean()
        intUSDborrow = history['USD/rate/borrow'].rolling(int(self.asset_definition.holding_period.total_seconds() / 3600)).mean()
        self.history=pd.concat([intLongCarry,intShortCarry,intUSDborrow])

    def build_assets_processes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ### E_int and C_int are moments of the integral of annualized carry.
        Expectation = self.history.ewm(times=self.history.index, halflife=self.estimation_parameter.signal_horizon, axis=0).mean()
        Covariance = self.history.ewm(times=self.history.index, halflife=self.estimation_parameter.signal_horizon, axis=0).cov()
        return Expectation,Covariance

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