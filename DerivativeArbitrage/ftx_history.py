from ftx_utilities import *
from ftx_ftx import *
from dateutil import rrule

def build_history(futures,exchange,
        timeframe='1h',
        end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
        start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30)):

    if futures[futures['type']=='perpetual'].empty: perp_funding_data=[]
    else:
        perp_funding_data=pd.concat([funding_history(f,exchange,start=start,end=end)
                                     for (i,f) in futures[futures['type']=='perpetual'].iterrows()])
        pd.DataFrame(perp_funding_data).to_pickle("allfundings.pickle")

    future_rate_data=pd.concat([rate_history(f, exchange, end, start, timeframe)
               for (i, f) in futures[futures['type'] == 'perpetual'].iterrows()],
              axis=1)
    spot_data=pd.concat([spot_history(f, exchange, end, start, timeframe)
               for (i, f) in futures[futures['type'] == 'perpetual'].iterrows()],
              axis=1)
    borrow_data=pd.concat([borrow_history(f, exchange, end, start)
               for f in futures['underlying'].unique()]
                          +[borrow_history('USD',exchange,end,start)],
              axis=1)

    borrow_data.to_pickle("allborrows.pickle")

    data= pd.concat([perp_funding_data,
                    future_rate_data,
                    spot_data,
                    borrow_data],join='outer',axis=1)
    return data

### only perps, only borrow and funding, only hourly
def borrow_history(spot,exchange,
                 end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30)):
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
def funding_history(future,exchange,
                 end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30)):
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
def rate_history(future,exchange,
                 end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                 timeframe='1h'):
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

    data.to_pickle(future['symbol']+"_futures.pickle")

    return data

def spot_history(future,exchange,
                 end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                 timeframe='1h'):
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

    data.to_pickle(future['symbol'] + "_spot.pickle")

    return data

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

