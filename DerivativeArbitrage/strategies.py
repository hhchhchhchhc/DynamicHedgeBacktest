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

def strategy2():
    exchange=open_exchange('ftx')
    futures = pd.DataFrame(fetch_futures(exchange,includeExpired=False))

    funding_threshold = 1e4
    volume_threshold = 1e6
    type_allowed='perpetual'
    max_nb_coins = 10
    carry_floor = 0.4
    slippage_override=2e-4
#    slippage_scaler=0.5
#    slippage_orderbook_depth=0
    signal_horizon=timedelta(days=3)
    backtest_window=timedelta(days=180)
    holding_period_for_slippage=timedelta(days=3)

    enriched=enricher(exchange, futures)
    pre_filtered = enriched[
        (enriched['expired'] == False)
        & (enriched['funding_volume'] * enriched['mark'] > funding_threshold)
        & (enriched['volumeUsd24h'] > volume_threshold)
        & (enriched['tokenizedEquity']!=True)
        & (enriched['type']==type_allowed)]

    #### get history ( this is sloooow)
    try:
        hy_history = from_parquet("history.parquet")
        existing_futures = [name.split('/')[0] for name in hy_history.columns]
        new_futures = pre_filtered[pre_filtered['symbol'].isin(existing_futures)==False]
        if new_futures.empty==False:
            hy_history=pd.concat([hy_history,
                    build_history(new_futures,exchange,timeframe='1h',end=asofdate,start=asofdate-backtest_window)],
                    join='outer',axis=1)
            to_parquet(hy_history, "history.parquet")
        asofdate = np.max(hy_history.index)
    except:
        asofdate = datetime.today()
        hy_history = build_history(pre_filtered,exchange,timeframe='1h',end=asofdate,start=asofdate-backtest_window)
        to_parquet(hy_history,"history.parquet")

    point_in_time=asofdate-holding_period_for_slippage
    scanned=basis_scanner(exchange,pre_filtered,hy_history,point_in_time=point_in_time,
                            slippage_override=slippage_override,
                            holding_period__for_slippage = holding_period_for_slippage,signal_horizon=signal_horizon,
                            risk_aversion=1).sort_values(by='optimalWeight')
    print(scanned[['symbol','optimalWeight','optimalCarry']])
    floored=scanned[scanned['optimalCarry']>carry_floor].tail(max_nb_coins)
    static_backtest = max_leverage_carry(floored,hy_history,end=point_in_time,start=asofdate-backtest_window)

    #    sns.histplot(data=pd.DataFrame([(a.loc[1:,'BTC-PERP/mark/c']-a.loc[:-2,'BTC-PERP/mark/c']),a['BTC-PERP/rate/c'],a['BTC-PERP/rate/h']-a['BTC-PERP/rate/c'],a['BTC-PERP/rate/l']-a['BTC-PERP/rate/c']]))
    outputit(scanned,'maxCarry','ftx',{'excelit':True,'pickleit':False})
    outputit(static_backtest.T.describe([.1,.25, .5,.75, .9])*24*365.25,'backtest','ftx',{'excelit':True,'pickleit':False})


strategy2()

#hy_history = from_parquet("fullhistory.parquet")
hy_history=[]