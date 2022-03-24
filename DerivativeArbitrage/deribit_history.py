#!/usr/bin/env python3
import logging

import pandas as pd
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import ccxt
from datetime import datetime,timezone,timedelta,date
import dateutil
import dateutil.parser

from ftx_utilities import from_parquet, to_parquet, calc_basis
history_start = datetime(2021, 4, 1)

###### this file is syncronous ###############
###### this file is syncronous ###############
###### this file is syncronous ###############
###### this file is syncronous ###############

def open_exchange(exchange_name,subaccount,config={}):
    if exchange_name == 'deribit':
        exchange = ccxt.deribit(config={
            'enableRateLimit': True,
            'apiKey': '4vc_41O4',
            'secret': 'viEFbpRQpQLgUAujPrwWleL6Xutq9I8YVUVMkEfQG1E',
        }|config)
    else: print('what exchange?')
    exchange.checkRequiredCredentials()  # raises AuthenticationError
    exchange.load_markets()
    exchange.load_fees()
    return exchange

def get_history(derivative, start = 'cache', end = datetime.now(tz=None).replace(minute=0,second=0,microsecond=0),
        dirname = 'Runtime/Deribit_Mktdata_database'):
    ''' all rates annualized, all volumes daily in usd'''
    data = pd.concat(
            [from_parquet(dirname+'/'+f+'_funding.parquet')
             for f in derivative.loc[derivative['type'] == 'swap','instrument_name']] +
            [from_parquet(dirname+'/'+f+'_derivative.parquet')
             for f in derivative['instrument_name']] +
            [from_parquet(dirname+'/'+f+'_volIndex.parquet')
             for f in derivative['base_currency'].unique()]
        , join='outer', axis=1)

    if start == 'cache':
        return data[~data.index.duplicated()].sort_index()
    elif isinstance(start,int):
        start = end - timedelta(hours=start)
        return data[~data.index.duplicated()].sort_index()[start:end]
    elif isinstance(start,datetime):
        return data[~data.index.duplicated()].sort_index()[start:end]
    else:
        raise Exception('invalid start mode')

def build_history(derivative,exchange,
        end = (datetime.now(tz=None).replace(minute=0,second=0,microsecond=0)),
        dirname = 'Runtime/Deribit_Mktdata_database'):
    '''for now, increments local files and then uploads to s3'''

    coroutines = []
    for _, f in derivative[derivative['type'] == 'swap'].iterrows():
        parquet_name = dirname + '/' + f['instrument_name'] + '_funding.parquet'
        parquet = from_parquet(parquet_name) if os.path.isfile(parquet_name) else None
        start = max(parquet.index)+timedelta(hours=1) if parquet is not None else history_start
        if start < end:
            coroutines.append(funding_history(f, exchange, start, end, dirname))

    for _, f in derivative.iterrows():
        parquet_name = dirname + '/' + f['instrument_name'] + '_derivative.parquet'
        parquet = from_parquet(parquet_name) if os.path.isfile(parquet_name) else None
        start = max(parquet.index) + timedelta(hours=1) if parquet is not None else history_start
        if start < end:
            coroutines.append(rate_history(f, exchange, end, start, '1h', dirname))

    for f in derivative['base_currency'].unique():
        parquet_name = dirname + '/' + f + '_volIndex.parquet'
        parquet = from_parquet(parquet_name) if os.path.isfile(parquet_name) else None
        start = max(parquet.index) + timedelta(hours=1) if parquet is not None else history_start
        if start < end:
            coroutines.append(vol_index_history(f, exchange, end, start , '1h', dirname))

    #os.system("aws s3 sync Runtime/Deribit_Mktdata_database/ s3://hourlyftx/Deribit_Mktdata_database")

def funding_history(future,exchange,
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                    end=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)),
                    dirname=''):
    '''annualized funding for perps'''
    max_funding_data = 500  # in hour. limit is 500 :(
    resolution = int(exchange.describe()['timeframes']['1h'])*60

    e = int(end.timestamp())
    s = int(start.timestamp())
    f = int(max_funding_data * resolution)
    start_times = [s+k*f for k in range(1+int((e-s)/f)) if s+k*f<e]

    logging.info(f'calling {sys._getframe(1).f_code.co_name} {len(start_times)} times')
    funding = [exchange.publicGetGetFundingRateHistory(params={'start_timestamp':start_time*1000, 'end_timestamp':(start_time+f)*1000,'instrument_name':future['instrument_name']})['result']
             for start_time in start_times]
    funding = [y for x in funding for y in x]

    if len(funding)==0:
        return pd.DataFrame(columns=[exchange.market(future['symbol'])['id'] + '/rate/funding'])

    data = pd.DataFrame(funding)
    data['time']=data['timestamp'].astype(dtype='int64')
    data[exchange.market(future['symbol'])['id'] + '/rate/funding'] = data['interest_1h'].astype(float) *365.25*24
    data=data[['time',exchange.market(future['symbol'])['id'] + '/rate/funding']].set_index('time')
    data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
    data = data[~data.index.duplicated()].sort_index()

    if dirname != '': to_parquet(data,dirname + '/' + exchange.market(future['symbol'])['id'] + '_funding.parquet',mode='a')

    return data

#### annualized rates for derivative and perp, volumes are daily
def rate_history(future,exchange,
                 end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                 timeframe='1h',
                 dirname=''):
    max_mark_data = 700
    resolution = int(exchange.describe()['timeframes'][timeframe])*60

    e = int(end.timestamp())
    s = int(start.timestamp())
    f = int(max_mark_data * resolution)
    start_times=[s+k*f for k in range(1+int((e-s)/f)) if s+k*f < e]

    logging.info(f'calling {sys._getframe(1).f_code.co_name} {len(start_times)} times')
    ## TODO: index does NOT work. No obvious endpoint
    mark_indexes = [
        exchange.fetch_ohlcv(future['symbol'], timeframe=timeframe, limit=999999999999999999, params=params) # volume is for max_mark_data*resolution
            for start_time in start_times
                for params in [{'start_timestamp':start_time*1000, 'end_timestamp':(start_time+f-int(resolution))*1000},
                               {'start_timestamp':start_time*1000, 'end_timestamp':(start_time+f-int(resolution))*1000,'price':'index'}]]
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
        expiry_time = future['expiryTime'].timestamp()
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
    elif future['type'] == 'swap': ### 1h funding = (mark/spot-1)/24
        data['rate/c'] = (mark['mark/c'] / indexes['indexes/c'] - 1)*365.25
        data['rate/h'] = (mark['mark/h'] / indexes['indexes/h'] - 1)*365.25
        data['rate/l'] = (mark['mark/l'] / indexes['indexes/l'] - 1)*365.25
    else:
        print('what is ' + future['symbol'] + ' ?')
        return
    data.columns = [exchange.market(future['symbol'])['id'] + '/' + c for c in data.columns]
    data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
    data = data[~data.index.duplicated()].sort_index()

    if dirname != '': to_parquet(data,dirname + '/' + exchange.market(future['symbol'])['id'] + '_derivative.parquet',mode='a')

    return data

## populates future_price or spot_price depending on type
def spot_history(symbol, exchange,
                       end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                       start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                       timeframe='1h',
                       dirname=''):
    max_mark_data = int(1500)
    resolution = int(exchange.describe()['timeframes'][timeframe])*60

    e = int(end.timestamp())
    s = int(start.timestamp())
    f = int(max_mark_data * resolution)
    start_times=[s+k*f for k in range(1+int((e-s)/f)) if s+k*f<e]

    logging.info(f'calling {sys._getframe(1).f_code.co_name} {len(start_times)} times')
    spot_lists = [
        exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit = 99999999, params={'start_timestamp':start_time*1000, 'end_timestamp':(start_time+f-int(resolution))*1000})
                                for start_time in start_times]
    spot = [y for x in spot_lists for y in x]

    column_names = ['t', 'o', 'h', 'l', 'c', 'volume']
    if len(spot)==0:
        return pd.DataFrame(columns=[symbol.replace('/USD','') + '/price/' + c for c in column_names])

    ###### spot
    data = pd.DataFrame(columns=column_names, data=spot).astype(dtype={'t': 'int64', 'volume': 'float'}).set_index('t')
    data['volume'] = data['volume'] * 24 * 3600 / int(resolution)
    data.columns = [symbol.replace('/USD','') + '/price/' + column for column in data.columns]
    data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
    data = data[~data.index.duplicated()].sort_index()
    if dirname!='': to_parquet(data,dirname + '/' + symbol.replace('/USD', '') + '_price.parquet',mode='a')

    return data

## populates future_price or spot_price depending on type
def vol_index_history(currency, exchange,
                       end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                       start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                       timeframe='1h',
                       dirname=''):
    max_mark_data = int(1000)
    resolution = int(exchange.describe()['timeframes'][timeframe])*60

    e = int(end.timestamp())
    s = int(start.timestamp())
    f = int(max_mark_data * resolution)
    start_times=[s+k*f for k in range(1+int((e-s)/f)) if s+k*f<e]

    logging.info(f'calling {sys._getframe(1).f_code.co_name} {len(start_times)} times')
    spot_lists = [
        exchange.publicGetGetVolatilityIndexData(params={'currency':currency,'resolution':resolution,
                                                      'start_timestamp':start_time*1000, 'end_timestamp':(start_time+f-int(resolution))*1000})['result']['data']
                                for start_time in start_times]
    spot = [y for x in spot_lists for y in x]

    column_names = ['t', 'o', 'h', 'l', 'c']
    if len(spot)==0:
        return pd.DataFrame(columns=[currency + '/volindex/' + c for c in column_names])

    ###### spot
    data = pd.DataFrame(columns=column_names, data=spot,dtype=float).astype(dtype={'t': 'int64'}).set_index('t')
    data.columns = [currency + '/volindex/' + column for column in data.columns]
    data.index = [datetime.fromtimestamp(x / 1000) for x in data.index]
    data = data[~data.index.duplicated()].sort_index()
    if dirname!='': to_parquet(data,dirname + '/' + currency + '_volindex.parquet',mode='a')

    return data

def deribit_history_main_wrapper(*argv):
    exchange = open_exchange('deribit','')
    currencies = argv[1]
    markets = {currency:exchange.fetch_tickers(params={'currency':currency,'kind':'future','expired':True})
               for currency in currencies}
    markets = {currency:
                   {symbol:data|data['info']
                    for symbol,data in currency_data.items()}
               for currency, currency_data in markets.items()}
    markets = {currency:pd.DataFrame(markets[currency]).T
               for currency in currencies}
    for market in markets.values():
        market['type'] = market['symbol'].apply(lambda f: exchange.market(f)['type'])
        market['expiryTime'] = market['symbol'].apply(lambda f: dateutil.parser.isoparse(exchange.market(f)['expiryDatetime']).replace(tzinfo=None))

    # volume screening
    if argv[0] == 'build':
        [build_history(markets[currency], exchange) for currency in argv[1]]

    if argv[3] == 'cache':
        hy_history = [get_history(markets[currency],'cache') for currency in argv[1]]
    else:
        hy_history = [get_history(markets[currency], 24 * int(argv[3])) for currency in argv[1]]

    return hy_history

def deribit_history_main(*argv):
    argv=list(argv)
    if len(argv) < 1:
        argv.extend(['build'])
    if len(argv) < 2:
        argv.extend([['BTC']]) # universe name, or list of currencies, or 'all'
    if len(argv) < 3:
        argv.extend(['deribit']) # exchange_name
    if len(argv) < 4:
        argv.extend([5000])# nb days

    return deribit_history_main_wrapper(*argv)

if __name__ == "__main__":
    history = deribit_history_main(*sys.argv[1:])
