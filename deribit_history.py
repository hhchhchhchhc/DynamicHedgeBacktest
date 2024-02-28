#!/usr/bin/env python3
import os
import typing

import numpy as np
import pandas as pd
from pandas import DatetimeIndex

from utils.io_utils import *
from utils.ccxt_utilities import *
from deribit_smile import MktCurve, VolSurface
from utils.blackscholes import black_scholes
import ccxt.async_support as ccxt
from datetime import datetime, timezone, timedelta
import sys

funding_start = datetime(2021, 3, 24, tzinfo=timezone.utc)
perp_start = datetime(2021, 3, 24, tzinfo=timezone.utc)
volIndex_start = datetime(2021, 3, 24, tzinfo=timezone.utc)
history_start = datetime(2021, 3, 24, tzinfo=timezone.utc)


###### this file is syncronous ###############
###### this file is syncronous ###############
###### this file is syncronous ###############
###### this file is syncronous ###############

async def open_exchange(exchange_name, subaccount, config={}):
    if exchange_name == 'deribit':
        exchange = ccxt.deribit(config={
                                           'enableRateLimit': True,
                                           'apiKey': '4vc_41O4',
                                           'secret': 'viEFbpRQpQLgUAujPrwWleL6Xutq9I8YVUVMkEfQG1E',
                                           "enableRateLimit": True,
                                           "rateLimit": 100,
                                           "timeout": 10000
                                       } | config)
    else:
        print('what exchange?')
    exchange.checkRequiredCredentials()  # raises AuthenticationError
    await exchange.load_markets()
    await exchange.load_fees()
    return exchange


async def get_history(derivative, start='cache', end=datetime.now(tz=timezone.utc),
                      dirname='Runtime/Deribit_Mktdata_database'):
    ''' all rates annualized, all volumes daily in usd'''

    def time_interval(data, start: typing.Union[str, int], end):
        if start == 'cache':
            return data[~data.index.duplicated()].sort_index()
        elif isinstance(start, int):
            start = end - timedelta(hours=start)
            return data[~data.index.duplicated()].sort_index()[start:end]
        elif isinstance(start, datetime):
            return data[~data.index.duplicated()].sort_index()[start:end]
        else:
            raise Exception('invalid start mode')

    parquet_coros = [async_wrap(from_parquet)(os.path.join(os.sep, os.getcwd(), dirname, f))
                     for f in os.listdir(dirname)
                     if f.endswith('.parquet')]
    data = pd.concat(await safe_gather(parquet_coros), join='outer', axis=1) if len(
        parquet_coros) > 0 else pd.DataFrame()

    data = time_interval(data, start, end)
    # data.index = DatetimeIndex(data.index)
    # data = data.resample('H').mean().interpolate('linear')  # forwsight bias, but not often...

    return data


async def build_history(derivative, exchange,
                        end=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)),
                        dirname='Runtime/Deribit_Mktdata_database',
                        max_moneyness: float = 2):
    ccy = derivative['base_currency'].unique()[0]

    funding_df, index_df = await build_perps(derivative, dirname, end, exchange)
    spot = index_df.tail(1).filter(like='indexes/c').squeeze()

    volIndex_df = await build_volIndex(ccy, dirname, end, exchange)
    vol = volIndex_df.tail(1).filter(like='volIndex/c').squeeze()

    futures_df = await build_futures(derivative, dirname, end, exchange)

    options_df = await build_options(derivative, dirname, end, exchange, max_moneyness, spot, vol)

    return options_df


async def build_options(derivative, dirname, end, exchange, max_moneyness: float, spot, vol) -> pd.DataFrame:
    coroutines3 = dict()
    for _, f in derivative[derivative['option']].iterrows():
        parquet_name = dirname + '/' + f['instrument_name'] + '_derivative.parquet'
        parquet = from_parquet(parquet_name) if os.path.isfile(parquet_name) else None
        start = max(parquet.index) + timedelta(hours=1) if parquet is not None else history_start
        # restrict to reasonable (current!) moneyness
        moneyness = np.log(spot / float(f['strike'])) / vol / np.sqrt(
            (f['expiryTime'] - end).total_seconds() / 3600 / 24 / 365.25)
        # exclude if not future (bit heavyhanded...)
        future_parquet_name = dirname + '/' + '-'.join(f['instrument_name'].split('-')[:2]) + '_derivative.parquet'
        if abs(moneyness) < max_moneyness and os.path.isfile(future_parquet_name):
            if start < end:
                coroutines3[(f['strike'], f['expiryTime'])] = rate_history(f, exchange, end, start, '1h', dirname)
            else:
                coroutines3[(f['strike'], f['expiryTime'])] = async_wrap(lambda x: x)(parquet)
    options_list = await safe_gather(coroutines3.values())
    options_df = pd.concat({key: data.filter(like='vol/c') for key, data in zip(coroutines3.keys(), options_list)},
                           join='outer', axis="columns")

    return options_df


async def build_futures(derivative, dirname, end, exchange) -> pd.DataFrame:
    coroutines2 = dict()
    for _, f in derivative[derivative['future']].iterrows():
        parquet_name = dirname + '/' + f['instrument_name'] + '_derivative.parquet'
        parquet = from_parquet(parquet_name) if os.path.isfile(parquet_name) else None
        start = max(parquet.index) + timedelta(hours=1) if parquet is not None else history_start
        if start < end:
            coroutines2[f['expiryTime']] = rate_history(f, exchange, end, start, '1h', dirname)
        else:
            coroutines2[f['expiryTime']] = async_wrap(lambda x: x)(parquet)
    futures_list = await safe_gather(coroutines2.values())
    futures_df = pd.concat(
        {expiration: data.filter(like='mark/c') for expiration, data in zip(coroutines2.keys(), futures_list)},
        join='outer', axis="columns")

    return futures_df


async def build_volIndex(ccy, dirname, end, exchange) -> pd.DataFrame:
    spot_vol = dict()
    # TODO: only on currency at a time
    parquet_name = dirname + '/' + ccy + '_volIndex.parquet'
    parquet = from_parquet(parquet_name) if os.path.isfile(parquet_name) else None
    start = max(parquet.index) + timedelta(hours=1) if parquet is not None else history_start
    if start < end:
        volIndex_df = await vol_index_history(ccy, exchange, end, start, '1h', dirname)
    else:
        volIndex_df = parquet
    return volIndex_df


async def build_perps(derivative, dirname, end, exchange) -> tuple[pd.DataFrame, pd.DataFrame]:
    coroutines = dict()
    for _, f in derivative[derivative['swap']].iterrows():
        parquet_name = dirname + '/' + f['instrument_name'] + '_funding.parquet'
        parquet = from_parquet(parquet_name) if os.path.isfile(parquet_name) else None
        start = max(parquet.index) + timedelta(hours=1) if parquet is not None else history_start
        if start < end:
            coroutines[f['instrument_name']] = funding_history(f, exchange, start, end, dirname)
        else:
            coroutines[f['instrument_name']] = async_wrap(lambda x: x)(parquet)
    perp_list = await safe_gather(coroutines.values())
    funding_df = pd.concat(
        {perp: data.filter(like='/rate/funding') for perp, data in zip(coroutines.keys(), perp_list)}, join='outer',
        axis="columns")
    index_df = pd.concat(
        {perp: data.filter(like='/indexes/c') for perp, data in zip(coroutines.keys(), perp_list)}, join='outer',
        axis="columns")

    return funding_df, index_df


async def build_FwdCurve():
    pass


@ignore_error
async def funding_history(future, exchange,
                          start=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)) - timedelta(
                              days=30),
                          end=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)),
                          dirname='') -> pd.DataFrame:
    '''annualized funding for perps'''
    future_id = future['instrument_name']
    max_funding_data = 500  # in hour. limit is 500 :(
    resolution = int(exchange.describe()['timeframes']['1h']) * 60

    e = int(end.timestamp())
    s = int(start.timestamp())
    f = int(max_funding_data * resolution)
    start_times = [s + k * f for k in range(1 + int((e - s) / f)) if s + k * f < e]

    logging.info(f'calling {sys._getframe(1).f_code.co_name} {len(start_times)} times')
    funding_coros = [exchange.publicGetGetFundingRateHistory(params={
        'start_timestamp': start_time * 1000,
        'end_timestamp': (start_time + f) * 1000,
        'instrument_name': future['instrument_name']})
        for start_time in start_times]
    funding = await safe_gather(funding_coros, n=1)
    funding = [y for x in funding for y in x['result']]

    if len(funding) == 0:
        return pd.DataFrame(columns=[future_id + '/rate/funding'])

    data = pd.DataFrame(funding)
    data['time'] = data['timestamp'].astype(dtype='int64')
    data[future_id + '/rate/funding'] = data['interest_1h'].astype(float) * 365.25 * 24
    data[future_id + '/indexes/o'] = data['prev_index_price'].astype(float)
    data[future_id + '/indexes/c'] = data['index_price'].astype(float)
    data = data[['time', future_id + '/rate/funding', future_id + '/indexes/o', future_id + '/indexes/c']].set_index(
        'time')
    data.index = [datetime.utcfromtimestamp(x / 1000).replace(tzinfo=timezone.utc) for x in data.index]
    data = data[~data.index.duplicated()].sort_index()

    if dirname != '':
        await async_wrap(to_parquet)(data, dirname + '/' + future['instrument_name'] + '_funding.parquet', mode='a')

    return data


#### annualized rates for derivative and perp, volumes are daily
@ignore_error
async def rate_history(future, exchange,
                       end=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)),
                       start=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)) - timedelta(
                           days=30),
                       timeframe='1h',
                       dirname=''  #
                       ) -> pd.DataFrame:
    max_mark_data = 700
    resolution = int(exchange.describe()['timeframes'][timeframe]) * 60

    e = int(end.timestamp())
    s = int(start.timestamp())
    f = int(max_mark_data * resolution)
    start_times = [s + k * f for k in range(1 + int((e - s) / f)) if s + k * f < e]

    logging.info(f'calling {sys._getframe(1).f_code.co_name} {len(start_times)} times')
    ## TODO: index does NOT work. No obvious endpoint
    mark_coros = [
        exchange.fetch_ohlcv(future['symbol'], timeframe=timeframe, limit=999999999999999999,
                             params={'start_timestamp': start_time * 1000,
                                     'end_timestamp': (start_time + f - int(resolution)) * 1000})
        # volume is for max_mark_data*resolution
        for start_time in start_times]
    mark = await safe_gather(mark_coros, n=1)
    mark = [y for x in mark for y in x]

    if (len(mark) == 0):
        return pd.DataFrame(columns=
                            [exchange.market(future['symbol'])['id'] + '/mark/' + c for c in
                             ['t', 'o', 'h', 'l', 'c', 'volume']]
                            + [exchange.market(future['symbol'])['id'] + '/rate/' + c for c in ['T', 'c', 'h', 'l']])
    column_names = ['t', 'o', 'h', 'l', 'c', 'volume']

    ###### marks
    data = pd.DataFrame([dict(zip(column_names, row)) for row in mark]).astype(dtype={'t': 'int64'}).set_index('t')
    data['volume'] = data['volume'] * 24 * 3600 / int(resolution)

    data.columns = ['mark/' + column for column in data.columns]

    ########## rates from index to mark
    if future['future']:
        expiry_time = future['expiryTime'].timestamp()
        data['rate/T'] = data.apply(lambda t: (expiry_time - int(t.name) / 1000) / 3600 / 24 / 365.25, axis=1)

        ## index is in funding......
        parquet_name = dirname + '/' + '{}{}-PERPETUAL'.format(future['base'],
                                                               '' if future['quote'] == 'USD' else '_' + future[
                                                                   'quote']) + '_funding.parquet'
        indexes = from_parquet(parquet_name)
        indexes.index = [datetime.timestamp(x) * 1000 for x in indexes.index]
        data['spot/o'] = pd.concat([indexes[future['base'] + '-PERPETUAL/indexes/o'], data],
                                   axis=1).sort_index().interpolate().ffill().loc[
            data.index, future['base'] + '-PERPETUAL/indexes/o'].dropna()

        data['rate/c'] = data.apply(
            lambda y: calc_basis(y['mark/c'], y['spot/o'], future['expiryTime'],
                                 datetime.utcfromtimestamp(int(y.name / 1000)).replace(tzinfo=timezone.utc)), axis=1)
    elif future['swap']:  ### 1h funding = (mark/spot-1)/24
        ## index is in funding......
        parquet_name = dirname + '/' + '{}{}-PERPETUAL'.format(future['base'],
                                                               '' if future['quote'] == 'USD' else '_' + future[
                                                                   'quote']) + '_funding.parquet'
        indexes = from_parquet(parquet_name)
        indexes.index = [datetime.timestamp(x) * 1000 for x in indexes.index]
        data['spot/o'] = pd.concat([indexes[future['base'] + '-PERPETUAL/indexes/o'], data],
                                   axis=1).sort_index().interpolate().ffill().loc[
            data.index, future['base'] + '-PERPETUAL/indexes/o'].dropna()

        data['rate/c'] = (data['mark/c'] / data['spot/o'] - 1) * 365.25
    elif future['option']:
        ## fwd is in future......
        future_name = '-'.join(future['instrument_name'].split('-')[:2])
        parquet_name = dirname + '/' + f'{future_name}_derivative.parquet'
        fwds = from_parquet(parquet_name)
        fwds.index = [int(datetime.timestamp(x) * 1000) for x in fwds.index]
        data['fwd/c'] = pd.concat([fwds, data], axis=1).sort_index().interpolate().ffill().loc[
            data.index, f'{future_name}/mark/c'].dropna()

        # ## volIndex for delta calc......
        # parquet_name = dirname + '/' + f"{future['base']}_volIndex.parquet"
        # volIndex = from_parquet(parquet_name)
        # volIndex.index = [int(datetime.timestamp(x)*1000) for x in volIndex.index]
        # data['volIndex/c'] = pd.concat([volIndex, data], axis=1).sort_index().interpolate().ffill().loc[
        #     data.index, f"{future['base']}/volIndex/c"].dropna()

        data['vol/c'] = data.apply(lambda y: black_scholes.bs_iv(y['mark/c'] / float(future['strike']),
                                                                 1 / y['fwd/c'],
                                                                 1 / float(future['strike']),
                                                                 (future[
                                                                      'expiry'] - y.name) / 1000 / 3600 / 24 / 365.25,
                                                                 'P' if future['optionType'] == 'call' else 'C') if y[
                                                                                                                        'fwd/c'] > 0 else np.NaN
                                   , axis=1)
    else:
        return pd.DataFrame()
    data.columns = [exchange.market(future['symbol'])['id'] + '/' + c for c in data.columns]
    data.index = [datetime.utcfromtimestamp(x / 1000).replace(tzinfo=timezone.utc) for x in data.index]
    data = data[~data.index.duplicated()].sort_index().dropna()

    if dirname != '':
        await async_wrap(to_parquet)(data, dirname + '/' + future['instrument_name'] + '_derivative.parquet', mode='a')

    return data


## populates future_price or spot_price depending on type
@ignore_error
async def spot_history(symbol, exchange,
                       end=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)),
                       start=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)) - timedelta(
                           days=30),
                       timeframe='1h',
                       dirname='') -> pd.DataFrame:
    max_mark_data = int(1500)
    resolution = int(exchange.describe()['timeframes'][timeframe]) * 60

    e = int(end.timestamp())
    s = int(start.timestamp())
    f = int(max_mark_data * resolution)
    start_times = [s + k * f for k in range(1 + int((e - s) / f)) if s + k * f < e]

    logging.info(f'calling {sys._getframe(1).f_code.co_name} {len(start_times)} times')
    spot_lists_coros = [
        exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=99999999,
                             params={'start_timestamp': start_time * 1000,
                                     'end_timestamp': (start_time + f - int(resolution)) * 1000})
        for start_time in start_times]
    spot_lists = await safe_gather(spot_lists_coros, n=1)
    spot = [y for x in spot_lists for y in x]

    column_names = ['t', 'o', 'h', 'l', 'c', 'volume']
    if len(spot) == 0:
        return pd.DataFrame(columns=[symbol.replace('/USD', '') + '/price/' + c for c in column_names])

    ###### spot
    data = pd.DataFrame(columns=column_names, data=spot).astype(dtype={'t': 'int64', 'volume': 'float'}).set_index('t')
    data['volume'] = data['volume'] * 24 * 3600 / int(resolution)
    data.columns = [symbol.replace('/USD', '') + '/price/' + column for column in data.columns]
    data.index = [datetime.utcfromtimestamp(x / 1000).replace(tzinfo=timezone.utc) for x in data.index]
    data = data[~data.index.duplicated()].sort_index()
    if dirname != '':
        await async_wrap(to_parquet)(data, dirname + '/' + symbol.replace('/USD', '') + '_price.parquet', mode='a')

    return data


## populates future_price or spot_price depending on type
@ignore_error
async def vol_index_history(currency, exchange,
                            end=(datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)),
                            start=(
                                  datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)) - timedelta(
                                days=30),
                            timeframe='1h',
                            dirname='') -> pd.DataFrame:
    max_mark_data = int(1000)
    resolution = int(exchange.describe()['timeframes'][timeframe]) * 60

    e = int(end.timestamp())
    s = int(start.timestamp())
    f = int(max_mark_data * resolution)
    start_times = [s + k * f for k in range(1 + int((e - s) / f)) if s + k * f < e]

    logging.info(f'calling {sys._getframe(1).f_code.co_name} {len(start_times)} times')
    volIndex_coros = [
        exchange.publicGetGetVolatilityIndexData(params={'currency': currency, 'resolution': resolution,
                                                         'start_timestamp': start_time * 1000,
                                                         'end_timestamp': (start_time + f - int(resolution)) * 1000})
        for start_time in start_times]
    volIndex = await safe_gather(volIndex_coros, n=1)
    volIndex = [y for x in volIndex for y in x['result']['data']]

    column_names = ['t', 'o', 'h', 'l', 'c']
    if not volIndex:
        return pd.DataFrame(columns=[currency + '/volIndex/' + c for c in column_names])

    ###### spot
    data = pd.DataFrame(columns=column_names, data=volIndex, dtype=float).astype(dtype={'t': 'int64'}).set_index(
        't') / 100
    data.columns = [currency + '/volIndex/' + column for column in data.columns]
    data.index = [datetime.utcfromtimestamp(x / 1000).replace(tzinfo=timezone.utc) for x in data.index]
    data = data[~data.index.duplicated()].sort_index()
    if dirname != '':
        await async_wrap(to_parquet)(data, dirname + '/' + currency + '_volIndex.parquet', mode='a')

    return data


async def deribit_history_main_wrapper(*argv):
    currency = argv[1]
    exchange = await open_exchange(argv[2], '')

    markets = {symbol: data | data['info']
               for symbol, data in exchange.markets.items()
               if data['base'] == currency
               and (data['swap'] or data['future'] or data['option'])}
    markets = pd.DataFrame(markets).T
    markets['expiryTime'] = markets['symbol'].apply(
        lambda f: dateutil.parser.isoparse(exchange.market(f)['expiryDatetime']).replace(tzinfo=timezone.utc))
    markets = markets[(markets['quote'] == 'USD') &
                      (markets['settlement_period'].isin(['month', 'perpetual'])) &
                      (~markets['kind'].isin(['future_combo', 'option_combo']))]

    if argv[0] == 'build':
        try:
            return await build_history(markets, exchange)
        finally:
            await exchange.close()
    elif argv[0] == 'get':
        pass
    else:
        raise Exception(f'unknown command{argv[0]}: use build,get')

    if argv[3] == 'cache':
        hy_history = await get_history(markets, 'cache')
    else:
        hy_history = await get_history(markets, 24 * int(argv[3]))

    return hy_history


def deribit_history_main(*argv):
    argv = list(argv)
    if len(argv) < 1:
        argv.extend(['build'])
    if len(argv) < 2:
        argv.extend(['ETH'])  # universe name, or list of currencies, or 'all'
    if len(argv) < 3:
        argv.extend(['deribit'])  # exchange_name
    if len(argv) < 4:
        argv.extend([5000])  # nb days

    return asyncio.run(deribit_history_main_wrapper(*argv))


if __name__ == "__main__":
    history = deribit_history_main(*sys.argv[1:])
