import numpy as np
import pandas as pd
from ftx_snap_basis import *
from ftx_portfolio import *
from ftx_ftx import *
import seaborn as sns
import ccxt
#from sklearn import *

### screening: dated futures on borrowable
### weight: linear on nb_pos biggest abs(basis)
### future deltas = target_leverage*1
### assumes account contains 1 usd to start with
def strategy1(nb_pos=1,account_equity=10000,params={'type':'perpetual','excelit':False,'pickleit':False}):
    exchange = open_exchange('ftx')

    #### screening
    futures = pd.DataFrame(fetch_futures(exchange))
    coin_details = fetch_coin_details(exchange)
    futures = pd.merge(futures, coin_details, how='left', left_on='underlying', right_index=True)
    futures['quote_borrow'] = coin_details.loc['USD','borrow']
    futures['quote_lend'] = coin_details.loc['USD', 'lend']
    futures = futures[futures['type'] == params['type']]
    futures = futures[futures['funding_volume'] > 1000]

    ### enrich with rates and slippages
    account_equity=max(1,account_equity)
    size = int(account_equity / nb_pos)  ## assume equidistributed for tx costs and 5 leverage (initial)
    scanner,history=basis_scanner(exchange,futures,[size])
    if scanner.shape[0] == 0: return pd.DataFrame()

    ####scoring
    futures=scanner.sort_values(by='adjBasis',axis=0,ascending=False,key=np.abs).head(nb_pos)

    #### convert to size and rescale to target leverage
    #target = .05 ## keep a buffer
    #collateral=1
    #IM=0
    futures['weights'] = max(size,1) * futures['adjBasis'] / futures['adjBasis'].apply(np.abs).sum()
    target_excess_collateral = 0.1  ## to solve later
    i = 0
    while i<3:### disgusting and stupid fixed point method.
        futures = futures.astype({'index': float, 'mark': float})
        positions=[{'netSize':-f['weights']/f['mark'],
                   'future':f['name'],
                   'initialMarginRequirement':0.05,
                   'maintenanceMarginRequirement':0.03,
                   'realizedPnl':f['weights']*(f['future_bid_in_'+str(size)] if f['weights']>0 else -f['future_ask_in_'+str(size)]),
                   'unrealizedPnl':0} for (i,f) in futures.iterrows()]

        ##### cash and funding leg
        #cash_balances=futures.apply(lambda f: f['weights']/f['mark']*f['index'],axis=1)
        balances=[{'total':f['weights']/f['index'],
                   'coin':f['underlying']} for (i,f) in futures.iterrows()]
        balances.append({'total': account_equity-futures.apply(lambda f: f['weights']*
            (1+f['spot_'+ ('ask' if f['weights']>0 else 'bid') +'_in_'+str(size)]),axis=1).sum(),
                   'coin': 'USD'})

        #### margin calc->scale again
        greeks=portfolio_greeks(exchange,positions,balances)
        excess_collateral=greeks.loc[(slice(None),'collateralValue'),('sum')].iloc[0,0]-greeks.loc[(slice(None),'IM'),('sum')].iloc[0,0]
        futures['weights'] = (1 - target_excess_collateral) / (1 - excess_collateral / account_equity) * futures['weights']
        i+=1

    outputit(greeks,"trade","ftx",params=params)
    outputit(scanner, "basis", "ftx", params=params)
    #outputit(scanner[1], "smallhistory", "ftx", params=params)
    return greeks

if True:
    exchange=open_exchange('ftx')
    futures = pd.DataFrame(fetch_futures(exchange,includeExpired=False))

    enriched=enricher(exchange, futures)
    pre_filtered = enriched[
        (enriched['expired'] == False)
        & (enriched['funding_volume'] * enriched['mark'] > 1e4)
        & (enriched['volumeUsd24h'] > 1e5)
        & (enriched['tokenizedEquity']!=True)
        & (enriched['type']=='perpetual')]

    #### get history ( this is sloooow)
    hy_history = build_history(pre_filtered,exchange)

    scanned=basis_scanner(exchange,pre_filtered,hy_history,depths=[0],slippage_scaler=0.5)
    print(scanned[['symbol','maxPos','maxCarry']])
    backtest = max_leverage_carry(scanned,hy_history)
    #a = fine_history(filtered,exchange,'1h',start=datetime.strptime('Oct 19 2021', '%b %d %Y'),params={'excelit':False,'pickleit':True}) ## critical size 26 oct 19 (2019,42,6)
    #    sns.histplot(data=pd.DataFrame([(a.loc[1:,'BTC-PERP/mark/c']-a.loc[:-2,'BTC-PERP/mark/c']),a['BTC-PERP/rate/c'],a['BTC-PERP/rate/h']-a['BTC-PERP/rate/c'],a['BTC-PERP/rate/l']-a['BTC-PERP/rate/c']]))
    outputit(scanned,'maxCarry','ftx',{'excelit':True,'pickleit':False})
    outputit(backtest,'backtest','ftx',{'excelit':True,'pickleit':False})