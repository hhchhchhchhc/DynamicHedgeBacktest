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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime,timezone,timedelta,date
import dateutil
import itertools

os.chdir('C:\\Users\\david\\Dropbox\\mobilier\\crypto')
sceening_mode = 'wide'
NB_BLOWUPS = 3
SHORT_BLOWUP = 0.3
LONG_BLOWUP = 0.15
BORROW_DECILE = 0.1
if sceening_mode=='wide':
    FUTURE_VOLUME_THRESHOLD = 2e5
    SPOT_VOLUME_THRESHOLD = 2e5
    BORROW_VOLUME_THRESHOLD = 2e5
elif sceening_mode=='tight':
    FUTURE_VOLUME_THRESHOLD = 5e6
    SPOT_VOLUME_THRESHOLD = 5e6
    BORROW_VOLUME_THRESHOLD = 5e6

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

def diagnosis_checkpoint(accruer,new_data,level_name,new_label):
    if not level_name in new_data.columns.names:
        new_accruer = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples([tuple([new_label]+[c]) for c in new_data.columns],
                                               names=[level_name]+new_data.columns.names))
        new_accruer[(new_label,)] = new_data
        accruer=accruer.join(new_accruer, how='outer')
    else:
        accruer[(new_label,)] = new_data
    return accruer

#a=openit('ftxStopout.pickle')
#outputit(a,'ftx','ftxstopout',params={'pickleit':False,'excelit':True})
#a.columns

#sd=from_parquet("history.parquet")
def compile_runs(dirname='runs/'):
    result=pd.DataFrame()
    for filename in [filename for filename in os.listdir(dirname) if "runs_" in filename]:
        df=pd.read_pickle(dirname+filename).T
        df['concentration_limit']=float(filename.split('_')[3])
        df['holding_period'] = pd.Timedelta(filename.split('_')[6])
        df['signal_horizon'] = pd.Timedelta(filename.split('_')[9].split('.')[0])
        df['time']=df.apply(lambda f: f.name[0],axis=1)
        df['field'] = df.apply(lambda f: f.name[1], axis=1)
        result=result.append(df.set_index(['concentration_limit',
                                             'holding_period',
                                             'signal_horizon',
                                             'time','field']))

    return result