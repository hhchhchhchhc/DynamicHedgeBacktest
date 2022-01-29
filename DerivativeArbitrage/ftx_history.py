import asyncio

import dateutil.parser
import pandas as pd

from ftx_utilities import *
from ftx_ftx import *

# all rates annualized, all volumes daily
async def build_history(futures,exchange,
        timeframe='1h',
        end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
        start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                  dirname=''):
    print('building history for :')
    print(futures['symbol'].values)
    data=pd.concat(await asyncio.gather(*(
            [funding_history(f,exchange,start,end,dirname)
                for (i,f) in futures[futures['type']=='perpetual'].iterrows()]+
            [rate_history(f, exchange, end, start, timeframe,dirname)
                for (i, f) in futures.iterrows()]+
            [spot_history(f + '/USD', exchange, end, start, timeframe, dirname)
             for f in futures['underlying'].unique()]+
            [borrow_history(f, exchange, end, start, dirname)
                for f in (list(futures.loc[futures['spotMargin'], 'underlying'].unique())+['USD'])]
                                                      )),join='outer',axis=1)
    data=pd.concat([data]+[pd.DataFrame(index=data.index, columns=[f + '/rate/size', f + '/rate/borrow'], data=999)
                    for f in futures.loc[~futures['spotMargin'], 'underlying'].unique()],join='outer',axis=1)

    return data[~data.index.duplicated()].sort_index()

### only perps, only borrow and funding, only hourly
async def borrow_history(spot,exchange,
                 end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                   dirname=''):
    parquet_filename = dirname+'/allborrows.parquet'
    if os.path.isfile(parquet_filename):
         return from_parquet(parquet_filename)[[spot+'/rate/borrow',spot+'/rate/size']]
    max_funding_data = int(500)  # in hour. limit is 500 :(
    resolution = exchange.describe()['timeframes']['1h']


    e = end.timestamp()
    s = start.timestamp()
    f = max_funding_data * int(resolution)
    start_times = [e - k * f for k in range(1 + int((e - s) / f)) if e - k * f > s] + [s]

    data = pd.concat(await asyncio.gather(*[
        fetch_borrow_rate_history(exchange,spot,start_time,start_time+f-int(resolution))
            for start_time in start_times]),axis=0,join='outer')
    if data.empty:
        return data
    data = data.astype(dtype={'time': 'int64'}).set_index(
        'time')[['rate','size']]
    data.rename(columns={'rate':spot+'/rate/borrow','size':spot+'/rate/size'},inplace=True)
    data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
    data=data[~data.index.duplicated()].sort_index()

    if dirname != '': data.to_parquet(parquet_filename)

    return data

######### annualized funding for perps
async def funding_history(future,exchange,
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                    end=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)),
                    dirname=''):
    parquet_filename=dirname+'/allfundings.parquet'
    if os.path.isfile(parquet_filename): return from_parquet(parquet_filename)[[exchange.market(future['symbol'])['id']+'/rate/funding']]
    max_funding_data = int(500)  # in hour. limit is 500 :(
    resolution = exchange.describe()['timeframes']['1h']


    e = end.timestamp()
    s = start.timestamp()
    f = max_funding_data * int(resolution)
    start_times=[s+k*f for k in range(1+int((e-s)/f)) if s+k*f<e]

    lists = await asyncio.gather(*[
        exchange.fetch_funding_rate_history(exchange.market(future['symbol'])['symbol'], params={'start_time':start_time, 'end_time':start_time+f})
                                for start_time in start_times])
    funding = [y for x in lists for y in x]
    data = pd.DataFrame(funding)
    data['time']=data['timestamp'].astype(dtype='int64')
    data[exchange.market(future['symbol'])['id'] + '/rate/funding']=data['fundingRate']*365.25*24
    data=data[['time',exchange.market(future['symbol'])['id'] + '/rate/funding']].set_index('time')
    data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
    data = data[~data.index.duplicated()].sort_index()

    if dirname != '': data.to_parquet(parquet_filename)

    return data

async def fetch_trades_history(symbol,exchange,
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                    end=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)),
                         frequency=timedelta(minutes=1),
                    dirname=''):
    parquet_filename = dirname + '/' + symbol.split('/USD')[0] + "_trades.parquet"
    if dirname!='':
        if os.path.isfile(parquet_filename): return from_parquet(parquet_filename)

    max_trades_data = int(5000)  # in trades. limit is 5000 :(
    print('trades_history: ' + symbol)

    ### grab data per batch of 5000, try weekly
    trades=[]
    end_time = (start + timedelta(hours=1)).timestamp()
    start_time = start.timestamp()

    while start_time < end.timestamp():
        new_trades =  (await exchange.publicGetMarketsMarketNameTrades(
            {'market_name': symbol, 'start_time': start_time, 'end_time': end_time}
                                                        ))['result']
        trades.extend(new_trades)

        if len(new_trades) > 0:
            last_trade_time = dateutil.parser.isoparse(new_trades[0]['time']).timestamp()
            if last_trade_time > end.timestamp(): break
            if (len(new_trades)<max_trades_data)&(end_time>end.timestamp()): break
            start_time = last_trade_time if len(new_trades)==max_trades_data else end_time
        else:
            start_time=end_time
        end_time = (datetime.fromtimestamp(start_time) + timedelta(
            hours=1)).timestamp()

    if len(trades)==0: return pd.DataFrame()
    data = pd.DataFrame(data=trades)
    data['size'] = data['size'].astype(float)
    data['volume'] = data['size'] * data['price'].astype(float)
    data['count'] = 1

    data['time']=data['time'].apply(dateutil.parser.isoparse)
    data.set_index('time',inplace=True)

    vwap=data[['size','volume','count']].resample(frequency).sum()
    vwap['vwap']=vwap['volume']/vwap['size']

    vwap.columns = [symbol.split('/USD')[0] + '/trades/' + column for column in vwap.columns]
    #data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
    vwap = vwap[~vwap.index.duplicated()].sort_index().ffill()


    if dirname != '': vwap.to_parquet(parquet_filename)

    return {'symbol':exchange.market(symbol)['symbol'],'coin':exchange.market(symbol)['base'],'vwap':vwap}


#### annualized rates for futures and perp, volumes are daily
async def rate_history(future,exchange,
                 end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                 timeframe='1h',
                 dirname=''):
    if dirname!='':
        parquet_filename=dirname+'/'+exchange.market(future['symbol'])['id']+'_futures.parquet'
        if os.path.isfile(parquet_filename): return from_parquet(parquet_filename)

    max_mark_data = int(1500)
    resolution = exchange.describe()['timeframes'][timeframe]


    e = end.timestamp()
    s = start.timestamp()
    f = max_mark_data * int(resolution)
    start_times=[s+k*f for k in range(1+int((e-s)/f)) if s+k*f<e]

    mark_indexes = await asyncio.gather(*[
        exchange.fetch_ohlcv(exchange.market(future['symbol'])['symbol'], timeframe=timeframe, params=params) # volume is for max_mark_data*resolution
            for start_time in start_times
                for params in [{'start_time':start_time, 'end_time':start_time+f-int(resolution)},
                               {'start_time':start_time, 'end_time':start_time+f-int(resolution),'price':'index'}]])
    mark = [y for x in mark_indexes[::2] for y in x]
    indexes = [y for x in mark_indexes[1::2] for y in x]

    if ((len(indexes) == 0) | (len(mark) == 0)):
        return pd.DataFrame(columns=
                         [exchange.market(future['symbol'])['id'] + '/mark/' + c for c in ['t', 'o', 'h', 'l', 'c', 'volume']]
                        +[exchange.market(future['symbol'])['id'] + '/indexes/'  + c for c in ['t', 'o', 'h', 'l', 'c', 'volume']]
                        +[exchange.market(future['symbol'])['id'] + '/rate/' + c for c in ['T','c','h','l']])
    column_names = ['t', 'o', 'h', 'l', 'c', 'volume']

    ###### indexes
    indexes = pd.DataFrame([dict(zip(column_names,row)) for row in indexes], dtype=float).astype(dtype={'t': 'int64'}).set_index('t')
    indexes['volume'] = indexes['volume']* 24 * 3600 / int(resolution)

    ###### marks
    mark = pd.DataFrame([dict(zip(column_names,row)) for row in mark]).astype(dtype={'t': 'int64'}).set_index('t')
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
                                 indexes.loc[y.name, 'indexes/c'], future['expiryTime'],
                                 datetime.fromtimestamp(int(y.name / 1000), tz=None)), axis=1)
        data['rate/h'] = data.apply(
            lambda y: calc_basis(y['mark/h'], indexes.loc[y.name, 'indexes/h'], future['expiryTime'],
                                 datetime.fromtimestamp(int(y.name / 1000), tz=None)), axis=1)
        data['rate/l'] = data.apply(
            lambda y: calc_basis(y['mark/l'], indexes.loc[y.name, 'indexes/l'], future['expiryTime'],
                                 datetime.fromtimestamp(int(y.name / 1000), tz=None)), axis=1)
    elif future['type'] == 'perpetual': ### 1h funding = (mark/spot-1)/24
        data['rate/c'] = (mark['mark/c'] / indexes['indexes/c'] - 1)*365.25
        data['rate/h'] = (mark['mark/h'] / indexes['indexes/h'] - 1)*365.25
        data['rate/l'] = (mark['mark/l'] / indexes['indexes/l'] - 1)*365.25
    else:
        print('what is ' + future['symbol'] + ' ?')
        return
    data.columns = [exchange.market(future['symbol'])['id'] + '/' + c for c in data.columns]
    data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
    data = data[~data.index.duplicated()].sort_index()

    if dirname != '': data.to_parquet(parquet_filename)

    return data

## populates future_price or spot_price depending on type
async def spot_history(symbol, exchange,
                       end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                       start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                       timeframe='1h',
                       dirname=''):
    if dirname!='':
        parquet_filename = dirname +'/' + symbol.replace('/USD','') + '_price.parquet'
        if os.path.isfile(parquet_filename): return from_parquet(parquet_filename)

    max_mark_data = int(1500)
    resolution = exchange.describe()['timeframes'][timeframe]


    e = end.timestamp()
    s = start.timestamp()
    f = max_mark_data * int(resolution)
    start_times=[s+k*f for k in range(1+int((e-s)/f)) if s+k*f<e]

    spot_lists = await asyncio.gather(*[
        exchange.fetch_ohlcv(symbol, timeframe=timeframe, params={'start_time':start_time, 'end_time':start_time+f-int(resolution)})
                                for start_time in start_times])
    spot = [y for x in spot_lists for y in x]

    column_names = ['t', 'o', 'h', 'l', 'c', 'volume']

    ###### spot
    data = pd.DataFrame(columns=column_names, data=spot).astype(dtype={'t': 'int64', 'volume': 'float'}).set_index('t')
    data['volume'] = data['volume'] * 24 * 3600 / int(resolution)
    data.columns = [symbol.replace('/USD','') + '/price/' + column for column in data.columns]
    data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
    data = data[~data.index.duplicated()].sort_index()

    if dirname!='': data.to_parquet(parquet_filename)

    return data

async def ftx_history_main_wrapper(*argv):
    exchange=open_exchange(argv[2],'')
    futures = pd.DataFrame(await fetch_futures(exchange, includeExpired=True)).set_index('name')
    markets= await exchange.fetch_markets()
    await exchange.load_markets()

    #argv[1] is either 'all', either a universe name, or a list of currencies
    filename = 'Runtime/configs/universe.xlsx'
    try:
        universe_list=pd.read_excel(filename,sheet_name='screening_params',index_col=0).columns
    except:
        universe_list=pd.DataFrame()

    if isinstance(argv[1],str) and argv[1] in universe_list:
        universe = pd.read_excel(filename,sheet_name=argv[1],index_col=0).index
    elif argv[1]=='all': universe=exchange.markets_by_id
    else: universe = [id for id,data in exchange.markets_by_id.items() if data['base']==argv[1].upper() and data['derivative']]
    futures = futures[futures.index.isin(universe)]

    # volume screening
    hy_history = await build_history(futures, exchange,
                               timeframe='1h', end=datetime.now(), start=datetime.now()-timedelta(days=int(argv[3])),
                               dirname=argv[4])
    await exchange.close()
    return hy_history

def ftx_history_main(*argv):
    argv=list(argv)
    if len(argv) < 1:
        argv.extend(['build'])
    if len(argv) < 2:
        argv.extend(['wide']) # universe name, or list of currencies, or 'all'
    if len(argv) < 3:
        argv.extend(['ftx']) # exchange_name
    if len(argv) < 4:
        argv.extend([150])# nb days
    if len(argv) < 5:
        argv.extend([''])# cache directory

    return asyncio.run(ftx_history_main_wrapper(*argv))

if __name__ == "__main__":
    ftx_history_main(*sys.argv[1:])
