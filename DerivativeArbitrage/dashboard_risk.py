import os
from typing import Tuple
import ccxt
import numpy as np
import pandas as pd

# TODO: also defined in config but not read from there :(
NB_BLOWUPS = 3
SHORT_BLOWUP = 0.3
LONG_BLOWUP = 0.15
DELTA_BLOWUP_ALERT = 0.5

def fetch_futures(exchange,includeExpired=False,params={}):
    response = exchange.publicGetFutures(params)
    expired = exchange.publicGetExpiredFutures(params) if includeExpired==True else []

    #### for IM calc
    account_leverage = exchange.privateGetAccount()['result']
    if float(account_leverage['leverage']) >= 50: print("margin rules not implemented for leverage >=50")
    dummy_size = 100000  ## IM is in ^3/2 not linear, but rule typically kicks in at a few M for optimal leverage of 20 so we linearize

    markets = exchange.safe_value(response, 'result', []) + exchange.safe_value(expired, 'result', [])
    result = []
    for i in range(0, len(markets)):
        market = markets[i]
        underlying = exchange.safe_string(market, 'underlying')
        mark = exchange.safe_number(market, 'mark')
        imfFactor = exchange.safe_number(market, 'imfFactor')

        result.append({
            'ask': exchange.safe_number(market, 'ask'),
            'bid': exchange.safe_number(market, 'bid'),
            'change1h': exchange.safe_number(market, 'change1h'),
            'change24h': exchange.safe_number(market, 'change24h'),
            'changeBod': exchange.safe_number(market, 'changeBod'),
            'volumeUsd24h': exchange.safe_number(market, 'volumeUsd24h'),
            'volume': exchange.safe_number(market, 'volume'),
            'symbol': exchange.safe_string(market, 'name'),
            "enabled": exchange.safe_value(market, 'enabled'),
            "expired": exchange.safe_value(market, 'expired'),
            "expiry": exchange.safe_string(market, 'expiry') if exchange.safe_string(market, 'expiry') else 'None',
            'index': exchange.safe_number(market, 'index'),
            'imfFactor': exchange.safe_number(market, 'imfFactor'),
            'last': exchange.safe_number(market, 'last'),
            'lowerBound': exchange.safe_number(market, 'lowerBound'),
            'mark': exchange.safe_number(market, 'mark'),
            'name': exchange.safe_string(market, 'name'),
            "perpetual": exchange.safe_value(market, 'perpetual'),
            'positionLimitWeight': exchange.safe_value(market, 'positionLimitWeight'),
            "postOnly": exchange.safe_value(market, 'postOnly'),
            'priceIncrement': exchange.safe_value(market, 'priceIncrement'),
            'sizeIncrement': exchange.safe_value(market, 'sizeIncrement'),
            'underlying': exchange.safe_string(market, 'underlying'),
            'upperBound': exchange.safe_value(market, 'upperBound'),
            'type': exchange.safe_string(market, 'type'),
        })
    return result

def open_exchange(exchange_name,subaccount=''):
    if exchange_name=='ftx':
        exchange = ccxt.ftx({ ## David personnal
            'enableRateLimit': True,
            'apiKey': 'SRHF4xLeygyOyi4Z_P_qB9FRHH9y73Y9jUk4iWvI',
            'secret': 'NHrASsA9azwQkvu_wOgsDrBFZOExb1E43ECXrZgV',
        })
        if subaccount!='': exchange.headers= {'FTX-SUBACCOUNT': subaccount}
 #   elif exchange_name == 'ftx_auk':
 #       exchange = ccxt.ftx({  ## Benoit personnal
 #           'enableRateLimit': True,
 #           'apiKey': 'nEAyW--EaRBqBJ0yG9H04cQMWD3fCv_jetzaw8Xx',
 #           'secret': 'xp-oPdGBn5I60RZOxv-cbySLUE40rtmAtoI7p95J',
 #       })
 #       if subaccount!='': exchange.headers = {'FTX-SUBACCOUNT': subaccount}
    elif exchange_name == 'binance':
        exchange = ccxt.binance({
        'enableRateLimit': True,
        'apiKey': 'pMaBWUoEVqsRJXZJoQ31JkA13QJHNRZyb6N0uZSAlwJscBMXprjgDQqKAfOLdGPK',
        'secret': 'neVVDD4oOyXbti1Xi5gI3nckEsIWz8BJ7CNd4UsRtK34GsWTMqS2D3xc0wY8mtxY',
    })
    else: print('what exchange?')
    #print('subaccount list: '+ ''.join([r['nickname']+' / ' for r in exchange.privateGetSubaccounts()['result']]))
    exchange.checkRequiredCredentials()  # raises AuthenticationError
    #exchange['secret']='none of your buisness'
    return exchange

def open_all_subaccounts(exchange_name):
    if exchange_name=='ftx':
        exchange = ccxt.ftx({ ## David personnal
            'enableRateLimit': True,
            'apiKey': 'SRHF4xLeygyOyi4Z_P_qB9FRHH9y73Y9jUk4iWvI',
            'secret': 'NHrASsA9azwQkvu_wOgsDrBFZOExb1E43ECXrZgV',
        })
 #   elif exchange_name == 'ftx_auk':
 #       exchange = ccxt.ftx({  ## Benoit personnal
 #           'enableRateLimit': True,
 #           'apiKey': 'nEAyW--EaRBqBJ0yG9H04cQMWD3fCv_jetzaw8Xx',
 #           'secret': 'xp-oPdGBn5I60RZOxv-cbySLUE40rtmAtoI7p95J',
 #       })
    else: print('what exchange?')

    subaccount_list = pd.DataFrame(exchange.privateGetSubaccounts()['result'])['nickname'].values
    return [open_exchange(exchange_name,subaccount) for subaccount in subaccount_list]

def live_risk(stablecoin_list):
    all_subaccounts = open_all_subaccounts('ftx')
    exchange = open_exchange('ftx','')#
    futures = pd.DataFrame(fetch_futures(exchange, includeExpired=False)).set_index('name')
    markets = pd.DataFrame([r['info'] for r in exchange.fetch_markets()]).set_index('name')

    risk_sum = pd.DataFrame()
    metrics_list = pd.DataFrame()
    for exchange in all_subaccounts:
        positions = pd.DataFrame([r['info'] for r in exchange.fetch_positions(params={})],dtype=float)#'showAvgPrice':True})
        positions['coin'] = positions['future'].apply(lambda f: f.split('-')[0])
        positions = positions[positions['netSize'] != 0.0].set_index('coin').fillna(0.0)

        balances=pd.DataFrame(exchange.fetch_balance(params={})['info']['result'],dtype=float)#'showAvgPrice':True})
        balances=balances[balances['total']!=0.0].set_index('coin').fillna(0.0)

        greeks=balances.join(positions,how='outer')
        greeks['futureDelta'] = 0 if positions.empty else positions.apply(lambda f: f['netSize'] * futures.loc[f['future'], 'mark'], axis=1)
        greeks['spotDelta'] = balances.apply(lambda f: f['total'] * (1.0 if f.name=='USD' else float(markets.loc[f.name+'/USD', 'price'])), axis=1)
        result = greeks[['futureDelta','spotDelta']].fillna(0.0)
        result['netDelta'] = result['futureDelta'] + result['spotDelta']
        result['futureMark'] = 0 if positions.empty else positions.apply(lambda f: futures.loc[f['future'], 'mark'], axis=1)
        result['futureIndex'] = 0 if positions.empty else positions.apply(lambda f: futures.loc[f['future'], 'index'], axis=1)
        result['spotMark'] = balances.apply(lambda f: (1.0 if f.name=='USD' else float(markets.loc[f.name+'/USD', 'price'])), axis=1)

        risk_sum = risk_sum.add(result[['futureDelta','spotDelta','netDelta']],fill_value=0.0)

        metrics=pd.DataFrame(index=[exchange.headers['FTX-SUBACCOUNT']])
        metrics['PV'] = (balances['total']*result['spotMark']).sum()+ (0 if positions.empty else positions['unrealizedPnl'].sum())
        account_info = pd.DataFrame(exchange.privateGetAccount()['result']).iloc[-1][
            ['totalAccountValue', 'totalPositionSize', 'marginFraction', 'maintenanceMarginRequirement','initialMarginRequirement']].astype(float)
        metrics['GrossPosition'] = account_info['totalPositionSize']
        metrics['excess IM'] = (account_info['marginFraction']-account_info['initialMarginRequirement'])*account_info['totalPositionSize']
        metrics['excess MM'] = (account_info['marginFraction']-account_info['maintenanceMarginRequirement'])*account_info['totalPositionSize']

        # trigger alert using DELTA_BLOWUP_ALERT and long/short/nb_blowup just like in the MM constraint. excl stablecoins.
        premium_blowup_futures = result[~result.index.isin(stablecoin_list)].sort_values(by='futureDelta',key=np.abs,ascending=False).head(NB_BLOWUPS)
        premium_blowup = premium_blowup_futures.apply(lambda f: f['futureDelta']*(-LONG_BLOWUP if f['futureDelta']<0 else SHORT_BLOWUP),axis=1).sum()
        delta_blowup = DELTA_BLOWUP_ALERT * result.loc[~result.index.isin(stablecoin_list),'netDelta'].apply(abs).sum()
        metrics['stop adding'] = (metrics['excess MM'] < premium_blowup)|(metrics['excess IM']<0)
        metrics['liquidation on shock'] = (metrics['excess MM'] < delta_blowup + premium_blowup)

        metrics_list = metrics_list.append(metrics)

    return (risk_sum,metrics_list)

print(live_risk(['USD']))