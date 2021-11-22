import os
from typing import Tuple
import ccxt
import numpy as np
import pandas as pd
import pickle
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import xlsxwriter
from s3 import *
import matplotlib
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import plotly.express as px

from datetime import datetime,timezone,timedelta,date
import dateutil
import itertools

def timedeltatostring(dt):
    return str(dt.days)+'d'+str(int(dt.seconds/3600))+'h'


if not 'Runtime' in os.listdir('.'): raise Exception("This needs to run in DerivativesArbitrage, where Runtime/ is located")
static_params=pd.read_excel('Runtime/configs/static_params.xlsx',index_col='key')
NB_BLOWUPS = int(static_params.loc['NB_BLOWUPS','value'])#3)
SHORT_BLOWUP = float(static_params.loc['SHORT_BLOWUP','value'])# = 0.3
LONG_BLOWUP = float(static_params.loc['LONG_BLOWUP','value'])# = 0.15
EQUITY = float(static_params.loc['EQUITY','value'])#=1e5
OPEN_ORDERS_HEADROOM = float(static_params.loc['OPEN_ORDERS_HEADROOM','value'])#=1e5
SIGNAL_HORIZON = pd.Timedelta(static_params.loc['SIGNAL_HORIZON','value'])
HOLDING_PERIOD = pd.Timedelta(static_params.loc['HOLDING_PERIOD','value'])
SLIPPAGE_OVERRIDE = float(static_params.loc['SLIPPAGE_OVERRIDE','value'])
CONCENTRATION_LIMIT = float(static_params.loc['CONCENTRATION_LIMIT','value'])
EXCLUSION_LIST = [c for c in static_params.loc['EXCLUSION_LIST','value'].split('+')]
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
    pq_df = pa.Table.from_pandas(df)
    pq.write_table(pq_df, filename)
    return None
def from_parquet(filename):
    return pq.read_table(filename).to_pandas()

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
        pickleit(data, "C:/Users/david/Dropbox/mobilier/crypto/" + exchange_name + datatype + ".pickle", "ab")
    if params['excelit']:
        try:
            data.to_excel("C:/Users/david/Dropbox/mobilier/crypto/" + exchange_name + datatype + ".xlsx")
        except:  ###usually because file is open
            data.to_excel("C:/Users/david/Dropbox/mobilier/crypto/" + exchange_name + datatype + "copy.xlsx")

def open_exchange(exchange_name):
    if exchange_name=='ftx':
        exchange = ccxt.ftx({ ## David personnal
            'enableRateLimit': True,
            'apiKey': 'SRHF4xLeygyOyi4Z_P_qB9FRHH9y73Y9jUk4iWvI',
            'secret': 'NHrASsA9azwQkvu_wOgsDrBFZOExb1E43ECXrZgV',
        })
    elif exchange_name == 'ftx_benoit':
        exchange = ccxt.ftx({  ## Benoit personnal
            'enableRateLimit': True,
            'apiKey': 'yJp-MCMT5wJW65CbD8myjkAZsAbUqlnXF3EeeZsZ',
            'secret': '6s2vWNcZrwoMc8otJN4h4semrdHyKBLohqaq2H3w',
            #'FTX-SUBACCOUNT': 'CashAndCarry'
        })
    elif exchange_name == 'binance':
        exchange = ccxt.binance({
        'enableRateLimit': True,
        'apiKey': 'pMaBWUoEVqsRJXZJoQ31JkA13QJHNRZyb6N0uZSAlwJscBMXprjgDQqKAfOLdGPK',
        'secret': 'neVVDD4oOyXbti1Xi5gI3nckEsIWz8BJ7CNd4UsRtK34GsWTMqS2D3xc0wY8mtxY',
    })
    else: print('what exchange?')
    print(exchange.requiredCredentials)  # prints required credentials
    exchange.checkRequiredCredentials()  # raises AuthenticationError
    #exchange['secret']='none of your buisness'
    return  exchange
