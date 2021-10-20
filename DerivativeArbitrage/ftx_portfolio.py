# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import numpy
import os
import sys
from ftx_utilities import *
from ftx_ftx import *

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')

import ccxt  # noqa: E402
from datetime import datetime,timezone,timedelta,date
#import time
import pickle
import dateutil

### list of dicts positions (resp. balances) assume unique 'future' (resp. 'coin')
### positions need netSize, future, initialMarginRequirement, maintenanceMarginRequirement, realizedPnl, unrealizedPnl
### balances need coin, total
### careful: carry on balances cannot be overal positive.
def portfolio_greeks(exchange,positions,balances,params={'positive_carry_on_balances':False}):
    markets = exchange.fetch_markets()
    coin_details = fetch_coin_details(exchange)  ### * (1+500*taker fee)
    futures = fetch_futures(exchange)

    greeks = pd.DataFrame(columns=pd.MultiIndex.from_tuples([], names=['underlyingType',"underlying", "margining", "expiry","name","contractType"]))
    updated=str(datetime.now())
    rho=0.4

    for x in positions:
        if float(x['netSize']) !=0.0:

            future_item=next(item for item in futures if item['name'] == x['future'])
            coin = future_item['underlying']
            underlyingType=getUnderlyingType(coin_details.loc[coin]) if coin in coin_details.index else 'index'
            funding_stats =fetch_funding_rates(exchange,future_item['name'])['result']

            size = float(x['netSize'])
            chg = float(future_item['change24h'])
            f=float(future_item['mark'])
            s = float(future_item['index'])
            if future_item['type']=='perpetual':
                t=0.0
                carry= - size*s*float(funding_stats['nextFundingRate'])*24*365.25
            else:
                days_diff = (dateutil.parser.isoparse(future_item['expiry']) - datetime.now(tz=timezone.utc))
                t=days_diff.days/365.25
                carry = - size*f * numpy.log(f / s) / t

            margin_coin = 'USD'  ## always USD on FTX
            if margin_coin == 'USD':
                greeks[(underlyingType,
                    str(coin),
                    margin_coin,
                    future_item['expiry'],
                    future_item['name'],
                    future_item['type'])]= pd.Series({
                        (updated,'PV'):0,
                        (updated, 'ref'): f,
                        (updated,'Delta'):size*f,
                        (updated,'ShadowDelta'):size*f*(1+rho*t),
                        (updated,'Gamma'):size*f*rho*t*(1+rho*t),
                        (updated,'IR01'):size*t*f/10000,
                        (updated,'Carry'):carry,
                        (updated,'collateralValue'):0,
                        (updated,'IM'): float(x['initialMarginRequirement'])*numpy.abs(size)*f,
                        (updated,'MM'): float(x['maintenanceMarginRequirement'])*numpy.abs(size)*f,
                            })
            else:
                greeks[(underlyingType,
                    str(coin),
                    margin_coin,
                    future_item['expiry'],
                    future_item['name'],
                    future_item['type'])] = pd.Series({
                    (updated, 'PV'): 0,
                    (updated, 'ref'): f,
                    (updated, 'Delta'): size / f*s,
                    (updated, 'ShadowDelta'): size / f * s * (1 + rho * t),
                    (updated, 'Gamma'): size / f *s* rho * t * (1 + rho * t),
                    (updated, 'IR01'): size*t*s/f/10000,
                    (updated, 'Carry'): carry,
                    (updated, 'collateralValue'): 0,
                    (updated, 'IM'): float(x['collateralUsed']),
                    (updated, 'MM'): float(x['maintenanceMarginRequirement']) * size ,
                })

            margin_cash=float(x['realizedPnl'])+float(x['unrealizedPnl'])
            try:
                for item in balances:
                    if item['coin'] == margin_coin: item['total']=float(item['total'])+margin_cash
            except:
                balances.append({'total':margin_cash,'coin':margin_coin})

#        margin_greeks=pd.Series(index=list(zip([updated]*10,['PV','ref','Delta','ShadowDelta','Gamma','IR01','Carry','collateralValue','IM','MM'])),
#                   data=[margin_cash,1.0,0.0,0.0,0,0,0,margin_cash,0,0])# + float(x['realizedPnl'])
#        if (margin_coin, 'USD', None, margin_coin, 'spot') in greeks.columns:
#            greeks[('usdFungible',margin_coin, 'USD', None, margin_coin, 'spot')]=margin_greeks.add(greeks[(margin_coin, 'USD', None, margin_coin, 'spot')],fill_value=0.0)
#        else:
#            greeks[('usdFungible',margin_coin, 'USD', None, margin_coin, 'spot')]=margin_greeks ### 'usdFungible' for now...

    stakes = pd.DataFrame(exchange.privateGetStakingBalances()['result']).set_index('coin')
    for x in balances:
        try:
            market_item = next(item for item in markets if item['id'] == x['coin']+'/USD')
            s = float(market_item['info']['price'])
            chg = float(market_item['info']['change24h'])
        except: ## fails for USD
            s = 1.0
            chg = 0.0

        coin = x['coin']
        underlyingType=getUnderlyingType(coin_details.loc[coin])

        size=float(x['total'])
        if size!=0:
            staked = float(stakes.loc[coin,'staked']) if coin in stakes.index else 0
            collateralValue=size*s*(coin_details.loc[coin,'collateralWeight'] if size>0 else 1)-staked*s
            ### weight(initial)=weight(total)-5% for all but stablecoins/ftt(0) and BTC (2.5)
            im=(1.1 / (coin_details.loc[coin,'collateralWeight']-0.05) - 1) * s * -size if (size<0) else 0.0
            mm=(1.03 / (coin_details.loc[coin,'collateralWeight']-0.05) - 1) * s * -size if (size<0) else 0.0
            ## prevent positive carry on balances (no implicit lending/staking)
            carry=size*s* (float(coin_details.loc[coin,('borrow')]) if (size<0) else 0)
            delta = size*s if coin!='USD' else 0

            newgreeks=pd.Series({
                    (updated,'PV'):size*s,
                    (updated, 'ref'): s,
                    (updated,'Delta'):delta,
                    (updated,'ShadowDelta'):delta,
                    (updated,'Gamma'):0,
                    (updated,'IR01'):0,
                    (updated,'Carry'):carry,
                    (updated,'collateralValue'): collateralValue,
                    (updated,'IM'): im,
                    (updated,'MM'): mm})
            if (underlyingType,coin,'USD',None,coin,'spot') in greeks.columns:
                greeks[(underlyingType,
                        coin,
                        'USD',
                        None,
                        coin,
                        'spot')] = greeks[(underlyingType,
                        coin,
                        'USD',
                        None,
                        coin,
                        'spot')] + newgreeks
            else:
                greeks[(underlyingType,
                coin,
                'USD',
                None,
                coin,
                'spot')]=newgreeks

    ## add a sum column
    greeks.sort_index(axis=1, level=[0, 1, 3, 5], ascending=[True, True, True, True],inplace=True)
    greeks[('sum',
            None,
            None,
            None,
            None,
            None)] = greeks.sum(axis=1)
    return greeks

### account starts with usdcash and enters positions X, incurring X*slippage
### X are cash delta, -X*hedgeratio are futures deltas
### usd cash is then usdcash-sum(X)-txCost, incurring usdborrow
def carry_vs_excesscollateral(X,collateral_weights,shortIM,futIM,slippage,basis,borrow,lend,usdcash,usdborrow,hedgeRatio=[]):
    X_plus = X.apply(lambda x: np.max(0,x))
    X_minus = X.apply(lambda x: np.max(0,-x))

    txCost=X_plus*slippage['spot']['ask']+X_minus*slippage['spot']['bid']+\
           X_plus * hedgeRatio * slippage['future']['bid'] + X_minus * hedgeRatio * slippage['spot']['ask']

    cash = usdcash - sum(X) - txCost

    excessCollateral = cash + sum(X_plus * collateral_weights - X_minus)-\
                       sum( X_minus*shortIM - (X_plus + X_minus) * futIM)

    carry = min(0,cash) * usdborrow + sum( -X_minus*borrow + X_plus*lend + X*basis)

    return [carry,excessCollateral,txCost]

### optimizer using https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp
#eq_cons = {'type': 'eq',
#           'fun' : lambda x: np.array([2*x[0] + x[1] - 1]),
#           'jac' : lambda x: np.array([2.0, 1.0])}
#res = minimize(rosen, x0, method='SLSQP', jac=rosen_der,
#               constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True},
#               bounds=bounds)


def live_risk():
    exchange = open_exchange('ftx')
    coin_details = fetch_coin_details(exchange)
    positions=exchange.fetch_positions(params={})#'showAvgPrice':True})
    balances=exchange.fetch_balance(params={})['info']['result']#'showAvgPrice':True})

    greeks = portfolio_greeks(exchange,positions,balances)

    ## 'actual' column
    account_info = exchange.privateGetAccount()['result']
    updated=greeks.index[0][0]
    greeks[('actual',
            None,
            None,
            None,
            None,
            None)] = pd.Series({
                    (updated,'PV'):float(account_info['totalAccountValue']),
                    (updated, 'ref'): float(account_info['totalPositionSize']), ## gross notional in lieu of ref
                    (updated,'Delta'):None,
                    (updated,'ShadowDelta'):None,
                    (updated,'Gamma'):None,
                    (updated,'IR01'):None,
                    (updated,'Carry'):None,
                    (updated,'collateralValue'): float(account_info['collateral']),
                    (updated,'IM'): float(account_info['initialMarginRequirement']),
                    (updated,'MM'): float(account_info['maintenanceMarginRequirement'])})
    return greeks