#!/usr/bin/env python3
import logging
import os
import sys
import asyncio
import functools
import aiofiles
import platform
if platform.system()=='Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import warnings
import shutil
import json
from typing import Tuple
import ccxtpro as ccxt
#import ccxt
import numpy as np
import pandas as pd
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
import xlsxwriter
import boto3
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import plotly.express as px

from datetime import datetime,timezone,timedelta,date
import dateutil
import itertools

safe_gather_limit = 50

def async_wrap(f):
    @functools.wraps(f)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        p = functools.partial(f, *args, **kwargs)
        return await loop.run_in_executor(executor, p)
    return run

async def safe_gather(tasks,n=safe_gather_limit,semaphore=None):
    semaphore = semaphore if semaphore else asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            return await task
    return await asyncio.gather(*(sem_task(task) for task in tasks))

def timedeltatostring(dt):
    return str(dt.days)+'d'+str(int(dt.seconds/3600))+'h'

if not 'Runtime' in os.listdir('.'):
    # notebooks run one level down...
    os.chdir('../')
    if not 'Runtime' in os.listdir('.'):
        raise Exception("This needs to run in DerivativesArbitrage, where Runtime/ is located")
static_params=pd.read_excel('Runtime/configs/static_params.xlsx',sheet_name='params',index_col='key')
NB_BLOWUPS = int(static_params.loc['NB_BLOWUPS','value'])#3)
SHORT_BLOWUP = float(static_params.loc['SHORT_BLOWUP','value'])# = 0.3
LONG_BLOWUP = float(static_params.loc['LONG_BLOWUP','value'])# = 0.15
EQUITY = str(static_params.loc['EQUITY','value']) #can be subaccount name, filename or number
OPEN_ORDERS_HEADROOM = float(static_params.loc['OPEN_ORDERS_HEADROOM','value'])#=1e5
SIGNAL_HORIZON = pd.Timedelta(static_params.loc['SIGNAL_HORIZON','value'])
HOLDING_PERIOD = pd.Timedelta(static_params.loc['HOLDING_PERIOD','value'])
SLIPPAGE_OVERRIDE = float(static_params.loc['SLIPPAGE_OVERRIDE','value'])
CONCENTRATION_LIMIT = float(static_params.loc['CONCENTRATION_LIMIT','value'])
MKTSHARE_LIMIT = float(static_params.loc['MKTSHARE_LIMIT','value'])
MINIMUM_CARRY = float(static_params.loc['MINIMUM_CARRY','value'])
EXCLUSION_LIST = [c for c in static_params.loc['REBASE_TOKENS','value'].split('+')]+[c for c in static_params.loc['EXCLUSION_LIST','value'].split('+')]
DELTA_BLOWUP_ALERT = float(static_params.loc['DELTA_BLOWUP_ALERT','value'])
UNIVERSE = str(static_params.loc['UNIVERSE','value'])
TYPE_ALLOWED = [c for c in static_params.loc['TYPE_ALLOWED','value'].split('+')]
print('read static_params')

########## only for dated futures
def calc_basis(f,s,T,t): # T is tring, t is date
    basis = np.log(float(f)/float(s))
    res= (T-t)
    return basis/np.max([1, res.days])*365.25

#index_list=['DEFI_PERP','SHIT_PERP','ALT_PERP','MID_PERP','DRGN_PERP','PRIV_PERP']
#publicGetIndexesIndexNameWeights()GET /indexes/{index_name}/weights
#index_table=pd.DataFrame()

def getUnderlyingType(coin_detail_item):
    if coin_detail_item['usdFungible'] == True:
        return 'usdFungible'
    if ('tokenizedEquity' in coin_detail_item.keys()):
        if (coin_detail_item['tokenizedEquity'] == True):
            return 'tokenizedEquity'
    if coin_detail_item['fiat'] == True:
        return 'fiat'

    return 'crypto'

def pickleit(object,filename,mode="ab+"):############ timestamp and append to pickle
    with open(filename,mode) as file:
        pickle.dump(object,file)
        file.close()
    return

def to_parquet(df,filename,mode="w"):
    if mode == 'a' and os.path.isfile(filename):
        previous = from_parquet(filename)
        df = pd.concat([previous,df],axis=0)
        df = df[~df.index.duplicated()].sort_index()
    pq_df = pa.Table.from_pandas(df)
    pq.write_table(pq_df, filename)
    return None
async def async_to_parquet(df,filename,mode="w"):
    coro = async_wrap(to_parquet)
    return await coro(df,filename,mode)

def from_parquet(filename):
    return pq.read_table(filename).to_pandas()
async def async_from_parquet(filename):
    coro = async_wrap(from_parquet)
    return await coro(filename)

def openit(filename,mode="rb"):#### to open pickle
    data=pd.DataFrame()
    with open(filename, mode) as file:
        try:
            while True:
                df=pickle.load(file)
                data=data.append(df)
        except EOFError:
            return data
    return data

def excelit(pickle_filename,excel_filename):
    data = pd.DataFrame(openit(pickle_filename))
    data.to_excel(excel_filename)
    return

def find_spot_ticker(markets,future,query):
    try:
        spot_found=next(item for item in markets if
                        (item['base'] == future['underlying'])
                        &(item['quote'] == 'USD')
                        &(item['type'] == 'spot'))
        return spot_found['info'][query]
    except:
        return np.NaN

def find_borrow(markets,future,query):
    try:
        spot_found=next(item for item in markets if
                        (item['base'] == future['underlying'])
                        &(item['quote'] == 'USD')
                        &(item['type'] == 'spot'))
        return spot_found['info'][query]
    except:
        return np.NaN

def outputit(data,datatype,exchange_name,params={'excelit':False,'pickleit':False}):
    if params['pickleit']:
        pickleit(data, "Runtime/temporary_parquets/" + exchange_name + datatype + ".pickle", "ab")
    if params['excelit']:
        try:
            data.to_excel("Runtime/temporary_parquets/" + exchange_name + datatype + ".xlsx")
        except:  ###usually because file is open
            data.to_excel("Runtime/temporary_parquets/" + exchange_name + datatype + "copy.xlsx")

apiKey='ZUWyqADqpXYFBjzzCQeUTSsxBZaMHeufPFgWYgQU'
secret='RC3lziT6QVS4jSTx2VrnT2NvKQB6E9WKVmnOcBCm'

async def open_exchange(exchange_name,subaccount,config={}):
    if exchange_name=='ftx':
        exchange = ccxt.ftx(config={ ## David personnal
            'enableRateLimit': True,
            'apiKey': 'ZUWyqADqpXYFBjzzCQeUTSsxBZaMHeufPFgWYgQU',
            'secret': 'RC3lziT6QVS4jSTx2VrnT2NvKQB6E9WKVmnOcBCm',
            'asyncio_loop': config['asyncio_loop'] if 'asyncio_loop' in config else asyncio.get_running_loop()
        }|config)
        if subaccount!='': exchange.headers= {'FTX-SUBACCOUNT': subaccount}

    elif exchange_name == 'binance':
        exchange = ccxt.binance(config={# subaccount convexity
        'enableRateLimit': True,
        'apiKey': 'V2KfGbMd9Zd9fATONTESrbtUtkEHFcVDr6xAI4KyGBjKs7z08pQspTaPhqITwh1M',
        'secret': '1rUn4jITdmck2vSSsnTDtR9vBS3oqLkNzx70SV1mijmr3BcVR2lwOkzWHpbULWc7',
    }|config)
    elif exchange_name == 'okex5':
        exchange = ccxt.okex5(config={
            'enableRateLimit': True,
            'apiKey': '6a72779d-0a4a-4554-a283-f28a17612747',
            'secret': '1F0CA69C0766727542A11EE602163883',
            'password': 'etvoilacestencoreokex'
        }|config)
        if subaccount != 'convexity':
            warnings.warn('subaccount override: convexity')
            exchange.headers = {'FTX-SUBACCOUNT': 'convexity'}
    elif exchange_name == 'huobi':
        exchange = ccxt.huobi(config={
            'enableRateLimit': True,
            'apiKey': 'b7d9d6f8-ce6a01b8-8b6ab42f-mn8ikls4qg',
            'secret': '40aefa97-cf88ddd6-dbb4171d-44360',
        }|config)
    elif exchange_name == 'deribit':
        exchange = ccxt.deribit(config={
            'enableRateLimit': True,
            'apiKey': '4vc_41O4',
            'secret': 'viEFbpRQpQLgUAujPrwWleL6Xutq9I8YVUVMkEfQG1E',
        }|config)
    elif exchange_name == 'kucoin':
        exchange = ccxt.kucoin(config={
                                           'enableRateLimit': True,
                                           'apiKey': '62091838bff2a30001b0d3f6',
                                           'secret': 'c789d699-a298-4b64-a5ff-b7be011c4233',
                                       } | config)
    #subaccount_list = pd.DataFrame((exchange.privateGetSubaccounts())['result'])
    else: print('what exchange?')
    exchange.checkRequiredCredentials()  # raises AuthenticationError
    await exchange.load_markets()
    await exchange.load_fees()
    return exchange

import collections
def flatten(dictionary, parent_key=False, separator='.'):
    """
    All credits to https://github.com/ScriptSmith
    Turn a nested dictionary into a flattened dictionary
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, collections.MutableMapping):
            items.extend(flatten(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)

def deepen(dictionary, parent_key=False, separator='.'):
    """
    All credits to https://github.com/ScriptSmith
    Turn a flattened dictionary into a nested dictionary
    :return: A flattened dictionary
    """
    top_keys = set(key.split(separator)[0] for key in dictionary.keys())
    result = {}
    for top_key in top_keys:
        sub_dict={}
        sub_result={}
        for key, value in dictionary.items():
            if key.split(separator)[0]==top_key:
                if separator in key:
                        sub_dict|={key.split(separator, 1)[1]:value}
                else:
                    sub_result |={key:value}
        if sub_dict != {}:
            result |= {top_key:deepen(sub_dict,parent_key=parent_key,separator=separator)}
        else:
            result |= sub_result

    return result

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.core.generic.NDFrame):
            return obj.to_json()
        return super(NpEncoder, self).default(obj)

def log_reader(prefix='latest',dirname='Runtime/logs'):
    path = f'{dirname}/{prefix}'
    with open(f'{path}_events.json', 'r') as file:
        d = json.load(file)
        events=pd.DataFrame(d)
    with open(f'{path}_risk_reconciliations.json', 'r') as file:
        d = json.load(file)
        risk = pd.DataFrame(d)
    with open(f'{path}_request.json', 'r') as file:
        d = json.load(file)
        request=pd.DataFrame(d)
    history = from_parquet(f'{path}_minutely.parquet')

    with pd.ExcelWriter(f'{dirname}/latest_exec.xlsx', engine='xlsxwriter', mode="w") as writer:
        if not events.empty:
            events.sort_values(by='in_flight_timestamp',ascending=True).to_excel(writer, sheet_name='order_lifecycle')
        if not risk.empty:
            risk.sort_values(by='timestamp', ascending=True).to_excel(writer, sheet_name='risk_recon')
        request.to_excel(writer, sheet_name='request')
        history.to_excel(writer, sheet_name='history')

if __name__ == "__main__":
    log_reader(*sys.argv[1:])