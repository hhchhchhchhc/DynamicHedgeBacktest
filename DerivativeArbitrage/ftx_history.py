import numpy as np
import pandas as pd
import os
import sys
import pickle
from ftx_utilities import *
from ftx_ftx import *
import dateutil
from datetime import datetime,timezone,timedelta,date

volumeList = ['BTC','ETH','BNB','SOL','LTC','XRP','LINK','OMG','TRX','DOGE','SUSHI']

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')

def fine_history(futures,exchange,
        timeframe='1h',
        end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
        start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
        params={'excelit':False,'pickleit':False}):
    max_mark_data = int(1500)
    resolution = exchange.describe()['timeframes'][timeframe]
    #### reset pickle
    if params['pickleit']:
        pickleit(pd.DataFrame(),"C:/Users/david/Dropbox/mobilier/crypto/"+exchange.name + "partial.pickle","wb")
    futures.loc[futures['type'] == 'future','expiryTime'] = futures[futures['type'] == 'future'].apply(lambda x:
                        dateutil.parser.isoparse(x['expiry']).replace(tzinfo=None), axis=1)  # .replace(tzinfo=timezone.utc)

    ###### this is only needed to have getUnderlyingType, which is cosmetic.
    data=pd.DataFrame()
    for (useless,future) in futures.iterrows():
        indexes=[]
        mark=[]
        spot=[]
        new_data=[]
        end_time = end.timestamp()
        start_time = (datetime.fromtimestamp(end_time) - timedelta(seconds=max_mark_data*int(resolution))).timestamp()

        while end_time > start.timestamp():
            if datetime.fromtimestamp(end_time).month>datetime.fromtimestamp(start_time).month:
                print(future['symbol'] + ' on ' + str(datetime.fromtimestamp(start_time)))
            if start_time<start.timestamp(): start_time=start.timestamp()
            new_mark=fetch_ohlcv(exchange,future['symbol'], timeframe=timeframe,start=start_time,end=end_time)
            new_indexes=exchange.publicGetIndexesMarketNameCandles(
                params={'start_time': start_time, 'end_time': end_time, 'market_name': future['underlying'],'resolution':resolution})['result']
            new_spot=fetch_ohlcv(exchange,future['symbol'], timeframe=timeframe,start=start_time,end=end_time)

            if (len(new_mark)==0): break
            mark.extend(new_mark)
            indexes.extend(new_indexes)
            spot.extend(new_spot)
            end_time = (datetime.fromtimestamp(start_time) - timedelta(seconds=int(resolution))).timestamp()
            start_time = (datetime.fromtimestamp(end_time) - timedelta(seconds=max_mark_data*int(resolution))).timestamp()

        column_names=['t', 'o', 'h', 'l', 'c', 'volume']

        ###### spot
        spot=pd.DataFrame(columns=column_names, data=spot).astype(dtype={'t': 'int64','volume': 'float'}).set_index('t')
        ###### indexes
        indexes = pd.DataFrame(indexes,dtype=float).astype(dtype={'time':'int64'}).set_index('time')
        indexes = indexes.drop(columns=['startTime','volume'])

        ###### marks
        mark=pd.DataFrame(columns=column_names, data=mark).astype(dtype={'t': 'int64'}).set_index('t')

        spot.columns=['spot/'+column for column in spot.columns]
        mark.columns=['mark/'+column for column in mark.columns]
        indexes.columns=['indexes/'+column for column in indexes.columns]
        new_data=spot.join(mark,how='inner').join(indexes,how='inner')
        ########## rates from index to mark --> for futures stopout
        if future['type']=='future':
            expiry_time = dateutil.parser.isoparse(future['expiry']).timestamp()
            new_data['rate/T']=new_data.apply(lambda t: (expiry_time-int(t.name)/1000)/3600/24/365,axis=1)

            new_data['rate/c'] = new_data.apply(
                lambda y: calc_basis(y['mark/c'],
                                     indexes.loc[y.name,'indexes/close'], future['expiryTime'],
                                     datetime.fromtimestamp(int(y.name / 1000),tz=None)), axis=1)
            new_data['rate/h'] = new_data.apply(
                lambda y: calc_basis(y['mark/h'], indexes.loc[y.name,'indexes/high'], future['expiryTime'],
                                     datetime.fromtimestamp(int(y.name / 1000),tz=None)), axis=1)
            new_data['rate/l'] = new_data.apply(
                lambda y: calc_basis(y['mark/l'], indexes.loc[y.name,'indexes/low'], future['expiryTime'],
                                     datetime.fromtimestamp(int(y.name / 1000),tz=None)), axis=1)
        elif future['type']=='perpetual':
            new_data['rate/c']=mark['mark/c']/indexes['indexes/close']-1
            new_data['rate/h']=mark['mark/h']/indexes['indexes/high']-1
            new_data['rate/l']=mark['mark/l']/indexes['indexes/low']-1
        else:
            print('what is '+future['symbol'] + ' ?')
            return

        print(future['symbol'] + ' ' + str(len(new_data.index)))
        if params['pickleit']:
            pickleit(data,"C:/Users/david/Dropbox/mobilier/crypto/"+exchange.name + "partial.pickle","ab+")
        new_data.columns=[future['symbol']+'/'+c for c in new_data.columns]
        data = data.join(new_data, how='outer')

    data.index = [datetime.fromtimestamp(x/ 1000) for x in data.index]

    ### optional debug info
    datatype = "finehistory"
    exchange = "ftx"
    if  params['pickleit']:
        pickleit(data,"C:/Users/david/Dropbox/mobilier/crypto/"+exchange + datatype +".pickle","wb")
    if params['excelit']:
        try:
            data.to_excel("C:/Users/david/Dropbox/mobilier/crypto/" + exchange + datatype + ".xlsx")
        except:###usually because file is open
            data.to_excel("C:/Users/david/Dropbox/mobilier/crypto/" + exchange + datatype + "copy.xlsx")

    return data

### only perps, only borrow and funding, only hourly
def perp_rate_history(perps,exchange,
                 end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                 start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30),
                 params={'excelit':False,'pickleit':False}):
    perps=perps[perps['type']=='perpetual'] ####  only perps here
    max_funding_data = int(500)  # in hour. limit is 500 :(
    resolution = exchange.describe()['timeframes']['1h']

    ### USD only has borrow data
    spot='USD'
    borrow_data = pd.DataFrame()
    end_time = end.timestamp()
    start_time = (datetime.fromtimestamp(end_time) - timedelta(seconds=max_funding_data*int(resolution))).timestamp()
    while end_time > start.timestamp():
        if start_time < start.timestamp(): start_time = start.timestamp()
        new_borrows = fetch_borrow_rate_history(exchange,spot, start_time,end_time)#,resolution)  ####very wasteful as retrieves all coins

        if (len(new_borrows) == 0): break
        borrow_data= pd.concat([borrow_data,new_borrows], join='outer', axis=0)
        end_time = (datetime.fromtimestamp(start_time) - timedelta(hours=1)).timestamp()
        start_time = (datetime.fromtimestamp(end_time) - timedelta(hours=max_funding_data)).timestamp()

    borrow_data = borrow_data.astype(dtype={'time': 'int64'}).set_index('time')[['coin', 'rate','size']]
    borrow_data = borrow_data[(borrow_data.index.duplicated() == False) & (borrow_data['coin'] == spot)]

    data = pd.DataFrame()
    data['USD/rate/borrow'] = borrow_data['rate']
    data['USD/rate/size'] = borrow_data['size']*1

    ###for each underlying...
    for (useless,perp) in perps.iterrows():
        spot=perp['underlying']
        ### grab data per batch of 5000
        borrow_data=pd.DataFrame()
        funding_data=pd.DataFrame()
        end_time = end.timestamp()
        start_time = (datetime.fromtimestamp(end_time) - timedelta(hours=max_funding_data)).timestamp()

        while end_time > start.timestamp():
            if start_time<start.timestamp(): start_time=start.timestamp()

            new_borrows=fetch_borrow_rate_history(exchange,spot,start_time,end_time)
            if len(new_borrows)==0: break
            borrow_data= pd.concat([borrow_data,new_borrows], join='outer', axis=0)

            new_funding=fetch_funding_rate_history(exchange,perp,start_time,end_time)
            if len(new_funding)==0: break
            funding_data= pd.concat([funding_data,new_funding], join='outer', axis=0)

            end_time = (datetime.fromtimestamp(start_time) - timedelta(hours=1)).timestamp()
            start_time = (datetime.fromtimestamp(end_time) - timedelta(hours=max_funding_data)).timestamp()

        if len(borrow_data)>0:
            borrow_data = borrow_data.astype(dtype={'time': 'int64'}).set_index(
                'time')[['coin','rate','size']]
            borrow_data = borrow_data[(borrow_data.index.duplicated() == False)&(borrow_data['coin']==spot)]
            new_borrow = pd.DataFrame()
            new_borrow[spot+'/rate/borrow'] = borrow_data['rate']
            new_borrow[spot+'/rate/size'] = borrow_data['size']
        else: new_borrow=pd.DataFrame()

        if len(funding_data)>0:
            funding_data = funding_data.astype(dtype={'time': 'int64'}).set_index(
                'time')[['rate']]
            funding_data = funding_data[(funding_data.index.duplicated() == False)]
            new_funding = pd.DataFrame()
            new_funding[perp['name']+'/rate/funding'] = funding_data['rate']
        else: new_funding=pd.DataFrame()

        data = data.join(new_borrow.join(new_funding, how='outer'), how='outer')

    data.index = [datetime.fromtimestamp(x/ 1000) for x in data.index]

    ### optional debug info
    datatype = "perphistory"
    exchange = "ftx"
    if  params['pickleit']:
        pickleit(data,"C:/Users/david/Dropbox/mobilier/crypto/"+exchange + datatype +".pickle","wb")
    if params['excelit']:
        try:
            data.to_excel("C:/Users/david/Dropbox/mobilier/crypto/" + exchange + datatype + ".xlsx")
        except:###usually because file is open
            data.to_excel("C:/Users/david/Dropbox/mobilier/crypto/" + exchange + datatype + "copy.xlsx")

    return data

def single_perp(future,rates_history,
                   end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                   start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30)):
    rates_history = rates_history[start:end]
    USDborrow = rates_history['USD/rate/borrow']
    pnlHistory=pd.DataFrame()
    pnlHistory['funding'] = (rates_history[future['name']+'/rate/funding']*np.sign(future['maxPos']))/365.25/24
    pnlHistory['borrow'] = (-USDborrow*np.sign(future['maxPos'])-
                rates_history[future['underlying']+'/rate/borrow'] if future['maxPos']> 0 else 0)/365.25/24
    pnlHistory['maxCarry'] = pnlHistory['funding']+pnlHistory['borrow']
    return pnlHistory

def single_future(future,rates_history,
                   end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                   start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30)):
    rates_history=rates_history[start:end]
    USDborrow = rates_history['USD/rate/borrow']
    pnlHistory = pd.DataFrame()
    pnlHistory['funding'] = (rates_history.loc[start,future['name']+'/rate/funding']*np.sign(future['maxPos']))/365.25/24
    pnlHistory['borrow'] = (-USDborrow*np.sign(future['maxPos'])-
                rates_history[future['underlying']+'/rate/borrow'] if future['maxPos']> 0 else 0)/365.25/24
    pnlHistory['maxCarry'] = pnlHistory['funding']+pnlHistory['borrow']
    return pnlHistory

def max_leverage_carry(futures,rates_history,
                       end= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0)),
                       start= (datetime.now(tz=timezone.utc).replace(minute=0,second=0,microsecond=0))-timedelta(days=30)):
    data = pd.concat([single_perp(future,rates_history,start=start,end=end) for (useless,future) in futures[futures['type']=='perpetual'].iterrows()]
                    +[single_future(future,rates_history,start=start,end=end) for (useless,future) in futures[futures['type']=='future'].iterrows()],
                     join='inner', axis=0)

    return data

#def max_leverage moments