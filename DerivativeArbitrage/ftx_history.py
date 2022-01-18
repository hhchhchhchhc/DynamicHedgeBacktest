import dateutil.parser
import pandas as pd

from ftx_utilities import *
from ftx_ftx import *

# all rates annualized, all volumes daily
async def build_history(futures,exchange,
        timeframe='1h',
        end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
        start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                  dirname='Runtime/temporary_parquets'):

    perp_funding_data=pd.DataFrame()
    if not futures[futures['type']=='perpetual'].empty:
        parquet_filename = dirname+'/allfundings.parquet'
        if os.path.isfile(parquet_filename):
            perp_funding_data=from_parquet(parquet_filename)
            perp_funding_data = perp_funding_data[[c for c in perp_funding_data.columns if any( \
                                        coin == c.split('/')[0] for coin in list(futures['symbol'])
                                    )]]
        else:
            perp_funding_data=pd.concat(await asyncio.gather(*[funding_history(f,exchange,start,end,dirname)
                                     for (i,f) in futures[futures['type']=='perpetual'].iterrows()]),join='outer',axis=1)
            if dirname!='': perp_funding_data.to_parquet(parquet_filename)

    future_rate_data=pd.concat([rate_history(f, exchange, end, start, timeframe,dirname)
               for (i, f) in futures.iterrows()],
              join='outer',axis=1)
    spot_price_data = pd.concat([price_history(f+'/USD', exchange, end, start, timeframe,dirname)
                                   for f in futures['underlying'].unique()],
                                  join='outer', axis=1)

    parquet_filename = dirname+'/allborrows.parquet'
    borrow_data=pd.DataFrame()
    if os.path.isfile(parquet_filename):
        borrow_data = from_parquet(parquet_filename)
        borrow_data = borrow_data[[c for c in borrow_data.columns if any( \
                                       coin==c.split('/')[0] for coin in (['USD']+ list(futures['underlying']))
                                   )]]
    else:
        borrow_data1= [borrow_history(f, exchange, end, start,dirname)
               for f in futures.loc[futures['spotMargin'],'underlying'].unique()]
        borrow_data2= [pd.DataFrame(index=spot_price_data.index,columns=[f + '/rate/size', f + '/rate/borrow'],data=999)
               for f in futures.loc[~futures['spotMargin'],'underlying'].unique()]
        borrow_data3 = [borrow_history('USD',exchange,end,start,dirname)]

        borrow_data=pd.concat(borrow_data1+borrow_data2+borrow_data3,join='outer',axis=1)

        if dirname!='': borrow_data.to_parquet(parquet_filename)

    ## just couldn't figure out pd.concat...
    data = perp_funding_data.join(
                future_rate_data.join(
                    spot_price_data.join(borrow_data, how='outer'),
                how='outer'),
            how='outer')

    return data[~data.index.duplicated()].sort_index()

### only perps, only borrow and funding, only hourly
def borrow_history(spot,exchange,
                 end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                   dirname='Runtime/temporary_parquets'):
    parquet_filename = dirname+'/allborrows.parquet'
    if os.path.isfile(parquet_filename):
         return from_parquet(parquet_filename)[[spot+'/rate/borrow',spot+'/rate/size']]
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
        data = pd.DataFrame()
        data[spot+'/rate/borrow'] = borrow_data['rate']
        data[spot+'/rate/size'] = borrow_data['size']
        data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
        data=data[~data.index.duplicated()].sort_index()
    else: data=pd.DataFrame()

    return data

######### annualized funding for perps
async def funding_history(future,exchange,
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                    end=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)),
                    dirname='Runtime/temporary_parquets'):
    parquet_filename=dirname+'/allfundings.parquet'
    if os.path.isfile(parquet_filename): return from_parquet(parquet_filename)[[future+'/rate/funding']]
    max_funding_data = int(500)  # in hour. limit is 500 :(
    resolution = exchange.describe()['timeframes']['1h']
    print('funding_history: ' + future['symbol'])

    ### grab data per batch of 5000
    funding_data=pd.DataFrame()
    end_time = end.timestamp()
    start_time = (datetime.fromtimestamp(end_time) - timedelta(hours=max_funding_data)).timestamp()

    while end_time > start.timestamp():
        if start_time<start.timestamp(): start_time=start.timestamp()

        data = await fetch_funding_rate_history(exchange, future, start_time, end_time)
        if len(data) == 0: break
        funding_data = pd.concat([funding_data, data], join='outer', axis=0)

        end_time = (datetime.fromtimestamp(start_time) - timedelta(hours=1)).timestamp()
        start_time = (datetime.fromtimestamp(end_time) - timedelta(hours=max_funding_data)).timestamp()

    if len(funding_data) > 0:
        funding_data = funding_data.astype(dtype={'time': 'int64'}).set_index(
            'time')[['rate']]
        data = pd.DataFrame()
        data[future['symbol'] + '/rate/funding'] = funding_data['rate']
        data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
        data = data[~data.index.duplicated()].sort_index()
    else:
        data = pd.DataFrame()

    return data

def fetch_trades_history(symbol,exchange,
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                    end=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)),
                         frequency='1s',
                    dirname='Runtime/temporary_parquets'):
    parquet_filename = dirname + '/' + symbol.split('/USD')[0] + "_trades.parquet"
    if dirname!='':
        if os.path.isfile(parquet_filename): return from_parquet(parquet_filename)

    max_trades_data = int(5000)  # in trades. limit is 5000 :(
    print('trades_history: ' + symbol)

    ### grab data per batch of 5000, try weekly
    trades=[]
    end_time = end.timestamp()
    start_time = (datetime.fromtimestamp(end_time) - timedelta(weeks=1)).timestamp()

    while end_time > start.timestamp():
        if start_time<start.timestamp(): start_time=start.timestamp()

        new_trades =  exchange.publicGetMarketsMarketNameTrades(
            {'market_name': symbol, 'start_time': start_time, 'end_time': end_time}
                                                        )['result']
        if (len(new_trades) == 0): break
        trades.extend(new_trades)
        end_time = min(start_time,dateutil.parser.isoparse(new_trades[-1]['time']).timestamp())
        start_time = (datetime.fromtimestamp(end_time) - timedelta(
            weeks=1)).timestamp()

    data = pd.DataFrame(data=trades)
    data['size'] = data['size'].astype(float)
    data['volume'] = data['size'] * data['price'].astype(float)
    data['count'] = 1

    data['time']=data['time'].apply(dateutil.parser.isoparse)
    data.set_index('time',inplace=True)

    vwap=data[['size','volume','count']].resample(frequency).sum().ffill()
    vwap['vwap']=vwap['volume']/vwap['size']

    vwap.columns = [symbol.split('/USD')[0] + '/trades/' + column for column in vwap.columns]
    #data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
    vwap = vwap[~vwap.index.duplicated()].sort_index()


    if dirname != '': vwap.to_parquet(parquet_filename)

    return vwap


#### annualized rates for futures and perp, volumes are daily
def rate_history(future,exchange,
                 end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                 timeframe='1h',
                 dirname='Runtime/temporary_parquets'):
    if dirname!='':
        parquet_filename=dirname+'/'+future['symbol']+'_futures.parquet'
        if os.path.isfile(parquet_filename): return from_parquet(parquet_filename)

    max_mark_data = int(1500)
    resolution = exchange.describe()['timeframes'][timeframe]
    print('rate_history: ' + future['symbol'])

    indexes = []
    mark = []
    data = []
    end_time = end.timestamp()
    start_time = (datetime.fromtimestamp(end_time) - timedelta(seconds=max_mark_data * int(resolution))).timestamp()

    while end_time >= start.timestamp():
        if start_time < start.timestamp(): start_time = start.timestamp()
        new_mark = fetch_ohlcv(exchange, future['symbol'], timeframe=timeframe, start=start_time, end=end_time) # volume is for max_mark_data*resolution
        new_indexes = exchange.publicGetIndexesMarketNameCandles(
            params={'start_time': start_time, 'end_time': end_time, 'market_name': future['underlying'],
                    'resolution': resolution})['result']

        if ((len(new_indexes) == 0)|(len(new_mark) == 0)): break

        mark.extend(new_mark)
        indexes.extend(new_indexes)
        end_time = (datetime.fromtimestamp(start_time) - timedelta(seconds=int(resolution))).timestamp()
        start_time = (datetime.fromtimestamp(end_time) - timedelta(
            seconds=max_mark_data * int(resolution))).timestamp()
    if ((len(indexes) == 0) | (len(mark) == 0)):
        return pd.DataFrame(columns=
                         [future['symbol'] + '/mark/' + c for c in ['t', 'o', 'h', 'l', 'c', 'volume']]
                        +[future['symbol'] + '/indexes/'  + c for c in ['t', 'open', 'high', 'low', 'close', 'volume']]
                        +[future['symbol'] + '/rate/' + c for c in ['T','c','h','l']])
    column_names = ['t', 'o', 'h', 'l', 'c', 'volume']

    ###### indexes
    indexes = pd.DataFrame(indexes, dtype=float).astype(dtype={'time': 'int64'}).set_index('time')
    indexes['volume'] = indexes['volume']* 24 * 3600 / int(resolution)
    indexes = indexes.drop(columns=['startTime', 'volume']) # volume is often none

    ###### marks
    mark = pd.DataFrame(columns=column_names, data=mark).astype(dtype={'t': 'int64'}).set_index('t')
    mark['volume']=mark['volume']*24*3600/int(resolution)

    mark.columns = ['mark/' + column for column in mark.columns]
    indexes.columns = ['indexes/' + column for column in indexes.columns]
    data = mark.join(indexes, how='inner')

    ########## rates from index to mark
    if future['type'] == 'future':
        expiry_time = dateutil.parser.isoparse(future['expiry']).timestamp()
        data['rate/T'] = data.apply(lambda t: (expiry_time - int(t.name) / 1000) / 3600 / 24 / 365.25, axis=1)

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

    if dirname!='':data.to_parquet(dirname+'/'+future['symbol']+"_futures.parquet")

    return data

## populates future_price or spot_price depending on type
def price_history(symbol,exchange,
                 end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                 timeframe='1h',
                  dirname='Runtime/temporary_parquets'):
    if dirname!='':
        parquet_filename = dirname +'/' + symbol.replace('/USD','') + '_price.parquet'
        if os.path.isfile(parquet_filename): return from_parquet(parquet_filename)

    max_mark_data = int(1500)
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
    data['volume'] = data['volume'] * 24 * 3600 / int(resolution)
    data.columns = [symbol.replace('/USD','') + '/price/' + column for column in data.columns]
    data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
    data = data[~data.index.duplicated()].sort_index()

    if dirname!='': data.to_parquet(dirname+'/'+ symbol.replace('/USD','') + "_price.parquet")

    return data

