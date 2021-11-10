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
    markets = exchange.fetch_markets()
    futures = pd.DataFrame(fetch_futures(exchange,includeExpired=False))

    # filtering params
    funding_volume_threshold = 5e5
    spot_volume_threshold = 5e4
    borrow_volume_threshold = 5e5
    type_allowed='perpetual'
    max_nb_coins = 15
    carry_floor = 0.4

    # fee estimation params
    slippage_override=2e-4  #### this is given by mktmaker
    slippage_scaler=1
    slippage_orderbook_depth=1000

    # startegy params
    signal_horizon=timedelta(days=7)
    backtest_window=timedelta(days=90)
    holding_period=timedelta(days=7)
    concentration_limit = 99 ## no limit...
    loss_tolerance= 0.99 ## no limit..
    marginal_coin_penalty = 0.05 ## TODO: not used

    enriched=enricher(exchange, futures, holding_period,
                    slippage_override=slippage_override, slippage_orderbook_depth=slippage_orderbook_depth,
                    slippage_scaler=slippage_scaler,
                    params={'override_slippage': True,'type_allowed':type_allowed,'fee_mode':'retail'})

    #### get history ( this is sloooow)
    try:
        hy_history = from_parquet("temporary_parquets/history.parquet")
        asofdate = np.max(hy_history.index)
        existing_futures = [name.split('/')[0] for name in hy_history.columns]
        new_futures = enriched[enriched['symbol'].isin(existing_futures)==False]
        if new_futures.empty==False:
            hy_history=pd.concat([hy_history,
                    build_history(new_futures,exchange,timeframe='1h',end=asofdate,start=asofdate-backtest_window)],
                    join='outer',axis=1)
            to_parquet(hy_history, "temporary_parquets/history.parquet")
    except FileNotFoundError:
        asofdate = datetime.today()
        hy_history = build_history(enriched,exchange,timeframe='1h',end=asofdate,start=asofdate-backtest_window)
        to_parquet(hy_history,"temporary_parquets/history.parquet")

    filter_window=hy_history[datetime(2021,6,1):datetime(2021,11,1)].index
    enriched['borrow_volume_avg'] = enriched.apply(lambda f:
                            (hy_history.loc[filter_window,f['underlying']+'/rate/size']*hy_history.loc[filter_window,f['name']+'/mark/o']).mean(),axis=1)
    enriched['spot_volume_avg'] = enriched.apply(lambda f:
                            (hy_history.loc[filter_window,f['underlying'] + '/price/volume'] ).mean(),axis=1)
    enriched['future_volume_avg'] = enriched.apply(lambda f:
                            (hy_history.loc[filter_window,f['name'] + '/price/volume']).mean(),axis=1)
    pre_filtered=enriched[
              (enriched['borrow_volume_avg'] >borrow_volume_threshold)
            & (enriched['spot_volume_avg'] >spot_volume_threshold)
            & (enriched['spot_volume_avg'] >spot_volume_threshold)]

    point_in_time=asofdate-holding_period
    scanned=basis_scanner(exchange,pre_filtered,hy_history,
                            point_in_time=point_in_time,
                            previous_weights=np.array(0),
                            holding_period = holding_period,
                            signal_horizon=signal_horizon,
                            concentration_limit=concentration_limit,
                            loss_tolerance=loss_tolerance,
                            marginal_coin_penalty=marginal_coin_penalty
                          ).sort_values(by='optimalWeight')
    return

    floored=scanned[scanned['ExpectedCarry']>carry_floor].tail(max_nb_coins)

    ladder = pd.Series(np.linspace(0.1, 1, 5))
    with pd.ExcelWriter('optimal.xlsx', engine='xlsxwriter') as writer:
        for c in ladder:
            futures = basis_scanner(exchange,pre_filtered,hy_history,
                            point_in_time=point_in_time,
                            holding_period = holding_period,
                            signal_horizon=signal_horizon,
                            concentration_limit=c,
                            loss_tolerance=loss_tolerance,
                            marginal_coin_penalty=marginal_coin_penalty
                          ).sort_values(by='optimalWeight')
            futures[['symbol', 'borrow', 'quote_borrow', 'basis_mid', 'spotCarry','medianCarryInt',
             'MaxLongWeight', 'MaxShortWeight', 'direction', 'optimalWeight',
             'ExpectedCarry','RealizedCarry','lossProbability','excessIM','excessMM']]\
                .to_excel(writer,sheet_name='concentration' + str(int(c * 100)))

    return

    static_backtest = max_leverage_carry(floored,hy_history,end=point_in_time,start=asofdate-backtest_window)

    #    sns.histplot(data=pd.DataFrame([(a.loc[1:,'BTC-PERP/mark/c']-a.loc[:-2,'BTC-PERP/mark/c']),a['BTC-PERP/rate/c'],a['BTC-PERP/rate/h']-a['BTC-PERP/rate/c'],a['BTC-PERP/rate/l']-a['BTC-PERP/rate/c']]))
    outputit(scanned,'maxCarry','ftx',{'excelit':True,'pickleit':False})
    outputit(static_backtest.T.describe([.1,.25, .5,.75, .9])*24*365.25,'backtest','ftx',{'excelit':True,'pickleit':False})

strategy2()
