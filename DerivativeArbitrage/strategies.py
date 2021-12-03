import os

import numpy as np
import pandas as pd
from ftx_snap_basis import *
from ftx_portfolio import *
from ftx_ftx import *
import seaborn as sns
import ccxt
#from sklearn import *

def refresh_universe(exchange_name,universe_size):
    filename = 'Runtime/Configs/universe.xlsx'
    if os.path.isfile(filename):
        try:
            return pd.read_excel(filename,sheet_name=universe_size,index_col=0)
        except:
            print('refreshing universe')
    exchange=open_exchange(exchange_name)
    futures = pd.DataFrame(fetch_futures(exchange, includeExpired=False)).set_index('name')
    markets=exchange.fetch_markets()

    universe_start = datetime(2021, 6, 1)
    universe_end = datetime(2021, 11, 1)
    borrow_decile = 0.1
    #type_allowed=['perpetual']
    screening_params=pd.DataFrame(
        index=['future_volume_threshold','spot_volume_threshold','borrow_volume_threshold'],
        data={'max':[5e4,5e4,5e4],
              'wide':[2e5,2e5,2e5],# important that wide is first :(
              'tight':[5e5,5e5,5e5]})

   # qualitative screening
    futures = futures[
        (futures['expired'] == False) & (futures['enabled'] == True) & (futures['type'] != "move")
        & (futures.apply(lambda f: float(find_spot_ticker(markets, f, 'ask')), axis=1) > 0.0)
        & (futures['tokenizedEquity'] != True)
        & (futures['spotMargin'] == True)]

    # volume screening
    hy_history = build_history(futures, exchange,
                               timeframe='1h', end=universe_end, start=universe_start,
                               dirname='Runtime/Configs/universe_history')
    universe_filter_window=hy_history[universe_start:universe_end].index
    futures['borrow_volume_decile'] = futures.apply(lambda f:
                            (hy_history.loc[universe_filter_window,f['underlying']+'/rate/size']*hy_history.loc[universe_filter_window,f.name+'/mark/o']).quantile(q=borrow_decile),axis=1)
    futures['spot_volume_avg'] = futures.apply(lambda f:
                            (hy_history.loc[universe_filter_window,f['underlying'] + '/price/volume'] ).mean(),axis=1)
    futures['future_volume_avg'] = futures.apply(lambda f:
                            (hy_history.loc[universe_filter_window,f.name + '/mark/volume']).mean(),axis=1)

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for c in screening_params:# important that wide is first :(
            population = futures.loc[
                  (futures['borrow_volume_decile'] > screening_params.loc['borrow_volume_threshold',c])
                & (futures['spot_volume_avg'] > screening_params.loc['spot_volume_threshold',c])
                & (futures['future_volume_avg'] > screening_params.loc['future_volume_threshold',c])]
            population.to_excel(writer, sheet_name=c)

        parameters = pd.Series(name='run_params',data={
            'run_date':datetime.today(),
            'universe_start': universe_start,
            'universe_end': universe_end,
            'borrow_decile': borrow_decile}).to_excel(writer,sheet_name='parameters')
        screening_params.to_excel(writer, sheet_name='screening_params')
        #TODO: s3_upload_file(filename, 'gof.crypto.shared', 'ftx_universe_'+str(datetime.now())+'.xlsx')

    return futures

def perp_vs_cash_live(
                signal_horizon,
                holding_period,
                slippage_override,
                concentration_limit,
                equity=EQUITY,
                exclusion_list=EXCLUSION_LIST,
                run_dir=''):
    try:
        first_history=pd.read_parquet(run_dir+'/'+os.listdir(run_dir)[0])
        if max(first_history.index)<datetime.now().replace(minute=0,second=0,microsecond=0):
            for file in os.listdir(run_dir): os.remove(run_dir+'/'+file)
        else: pass # otherwise do nothing and build_history will use what's there
    except: pass

    exchange = open_exchange('ftx')
    markets = exchange.fetch_markets()
    futures = pd.DataFrame(fetch_futures(exchange, includeExpired=False)).set_index('name')

    point_in_time = (datetime.now()-timedelta(hours=0)).replace(minute=0,second=0,microsecond=0)

    # filtering params
    universe=refresh_universe('ftx','wide')
    universe=universe[~universe['underlying'].isin(exclusion_list)]
    type_allowed = ['perpetual']
    max_nb_coins = 99
    carry_floor = 0.4

    # fee estimation params
    slippage_scaler = 1
    slippage_orderbook_depth = 10000

    for (holding_period,signal_horizon,slippage_override,concentration_limit) in [(hp,sh,sl,c) for hp in holding_period for sh in signal_horizon for sl in slippage_override for c in concentration_limit]:

        ## ----------- enrich, get history, filter
        enriched = enricher(exchange, futures, holding_period, equity=equity,
                            slippage_override=slippage_override, slippage_orderbook_depth=slippage_orderbook_depth,
                            slippage_scaler=slippage_scaler,
                            params={'override_slippage': True, 'type_allowed': type_allowed, 'fee_mode': 'retail'})

        #### get history ( this is sloooow)
        hy_history = build_history(enriched, exchange,
                                   timeframe='1h', end=point_in_time, start=point_in_time-signal_horizon-holding_period,
                                   dirname='Runtime/Live_parquets')

        # ------- build derived data history
        (intLongCarry, intShortCarry, intUSDborrow, intBorrow, E_long, E_short, E_intUSDborrow,E_intBorrow) = forecast(
            exchange, enriched, hy_history,
            holding_period,  # to convert slippage into rate
            signal_horizon,filename='history')  # historical window for expectations)
        updated, marginFunc = update(enriched, point_in_time, hy_history, equity,
                                     intLongCarry, intShortCarry, intUSDborrow, intBorrow, E_long, E_short, E_intUSDborrow,E_intBorrow)
        # final filter, needs some history and good avg volumes
        pre_filtered = updated[
            (~np.isnan(updated['E_intCarry']))
            & (updated['type'].isin(type_allowed))
            & (updated['symbol'].isin(universe.index))]
        pre_filtered = pre_filtered.sort_values(by='E_intCarry', ascending=False).head(max_nb_coins)  # ,key=abs

        # run a trajectory
        optimized = pre_filtered
        updated, marginFunc = update(optimized, point_in_time, hy_history, equity,
                                     intLongCarry, intShortCarry, intUSDborrow, intBorrow, E_long, E_short, E_intUSDborrow,E_intBorrow)
        previous_weights_df = pd.read_excel('Runtime/ApprovedRuns/current_weights.xlsx', sheet_name='optimized', index_col=0)['optimalWeight']

        optimized=cash_carry_optimizer(exchange,updated,marginFunc,
                                    previous_weights_df=previous_weights_df,
                                    holding_period = holding_period,
                                    signal_horizon=signal_horizon,
                                    concentration_limit=concentration_limit,
                                    equity=equity,
                                    optional_params= ['verbose']
                                  )

    with pd.ExcelWriter('Runtime/ApprovedRuns/ftx_optimal_cash_carry_'+datetime.utcnow().strftime("%Y-%m-%d-%Hh")+'.xlsx', engine='xlsxwriter') as writer:
        parameters = pd.Series({
            'run_date':datetime.today(),
            'exclusion_list': exclusion_list,
            'type_allowed': type_allowed,
            'max_nb_coins': max_nb_coins,
            'carry_floor': carry_floor,
            'slippage_override': slippage_override,
            'slippage_scaler': slippage_scaler,
            'slippage_orderbook_depth': slippage_orderbook_depth})
        optimized.to_excel(writer,sheet_name='optimized')
        parameters.to_excel(writer,sheet_name='parameters')

    display=optimized[['optimalWeight','ExpectedCarry','transactionCost']]
    display['absWeight']=display['optimalWeight'].apply(abs)
    display.loc['total','absWeight']=display.drop(index='total')['absWeight'].sum()
    display=display.sort_values(by='absWeight',ascending=True)
    display= display[display['absWeight'].cumsum()>display.loc['total','absWeight']*.1]
    print(display)

    return optimized


def perp_vs_cash_backtest(
                signal_horizon,
                holding_period,
                slippage_override,
                concentration_limit,
                equity=EQUITY,
                exclusion_list=EXCLUSION_LIST,
                filename='',
                optional_params=[]):
    exchange=open_exchange('ftx')
    markets = exchange.fetch_markets()
    futures = pd.DataFrame(fetch_futures(exchange,includeExpired=False)).set_index('name')

    # filtering params
    universe = refresh_universe('ftx', 'tight')
    universe = universe[~universe['underlying'].isin(exclusion_list)]
    type_allowed=['perpetual']
    max_nb_coins = 99
    pre_filtered=futures[
                (futures['type'].isin(type_allowed))
              & (futures['symbol'].isin(universe.index))]

    # fee estimation params
    slippage_scaler=1
    slippage_orderbook_depth=1000

    # backtest params
    backtest_start = datetime(2021, 7, 1)
    backtest_end = datetime(2021, 10, 1)

    ## ----------- enrich, get history, filter
    enriched=enricher(exchange, pre_filtered, holding_period,equity=float(equity),
                    slippage_override=slippage_override, slippage_orderbook_depth=slippage_orderbook_depth,
                    slippage_scaler=slippage_scaler,
                    params={'override_slippage': True,'type_allowed':type_allowed,'fee_mode':'retail'})

    #### get history ( this is sloooow)
    try:
        hy_history = from_parquet("Runtime/temporary_parquets/history.parquet")
        existing_futures = [name.split('/')[0] for name in hy_history.columns]
        new_futures = enriched[enriched['symbol'].isin(existing_futures)==False]
        if new_futures.empty==False:
            hy_history=pd.concat([hy_history,
                    build_history(new_futures,exchange,timeframe='1h',end=backtest_end,start=backtest_start-signal_horizon-holding_period)],
                    join='outer',axis=1)
            to_parquet(hy_history, "Runtime/temporary_parquets/history.parquet")
    except FileNotFoundError:
        hy_history = build_history(enriched,exchange,timeframe='1h',end=backtest_end,start=backtest_start-signal_horizon-holding_period)
        to_parquet(hy_history,"Runtime/temporary_parquets/history.parquet")

    # ------- build derived data history
    (intLongCarry, intShortCarry, intUSDborrow, E_long, E_short, E_intUSDborrow)=forecast(
        exchange, enriched, hy_history,
        holding_period,  # to convert slippage into rate
        signal_horizon,filename)  # historical window for expectations)
    updated, marginFunc = update(enriched, backtest_end, hy_history, equity,
                                 intLongCarry, intShortCarry, intUSDborrow, E_long, E_short, E_intUSDborrow)
    # final filter, needs some history and good avg volumes
    post_filtered=updated[~np.isnan(updated['E_intCarry'])]
    post_filtered = post_filtered.sort_values(by='E_intCarry',key=abs,ascending=False).head(max_nb_coins)#,key=abs

    # run a trajectory
    point_in_time=backtest_start+signal_horizon+holding_period # integrals not defined before that
    updated, marginFunc = update(post_filtered, point_in_time, hy_history, equity,
                                 intLongCarry, intShortCarry, intUSDborrow, E_long, E_short, E_intUSDborrow)
    previous_weights = pd.DataFrame()
    previous_weights['optimalWeight'] = equity * updated['E_intCarry'] \
                       / (updated['E_intCarry'].sum() if np.abs(updated['E_intCarry'].sum()) > 0.1 else 0.1)
    previous_time=point_in_time
    trajectory=pd.DataFrame()

    while point_in_time<backtest_end:
        updated,excess_margin=update(post_filtered,point_in_time,hy_history,equity,
                       intLongCarry, intShortCarry, intUSDborrow, E_long, E_short, E_intUSDborrow)
        optimized=cash_carry_optimizer(exchange,updated,excess_margin,
                                previous_weights_df=previous_weights,
                                holding_period = holding_period,
                                signal_horizon=signal_horizon,
                                concentration_limit=concentration_limit,
                                equity=equity,
                                optional_params= optional_params
                              )
        # need to assign RealizedCarry to previous_time
        if not trajectory.empty: trajectory.loc[trajectory['time']==previous_time,'RealizedCarry']=optimized['RealizedCarry'].values
        optimized['time'] = point_in_time

        # increment
        trajectory=trajectory.append(optimized.reset_index().rename({'name':'symbol'}),ignore_index=True)
        previous_weights = optimized['optimalWeight'].drop(index=['USD', 'total'])
        previous_time=point_in_time
        point_in_time += holding_period

    # remove last line because RealizedCarry is wrong there
    trajectory=trajectory.drop(trajectory[trajectory['time']==previous_time].index)

    #trajectory.xs('ask',level='field',axis=1)

#    with pd.ExcelWriter('summary_'+run_name+'.xlsx', engine='xlsxwriter') as writer:
#        trajectory.reorder_levels(['field','time'],axis='columns').to_excel(writer,'summary.xlsx')

    return trajectory

def run_ladder( concentration_limit_list,
                holding_period_list,
                signal_horizon_list,
                slippage_override,
                run_dir):
    if not os.path.isdir(run_dir): os.mkdir(run_dir)

    ladder = pd.DataFrame()
    for c in concentration_limit_list:
        for h in holding_period_list:
            for s in signal_horizon_list:
                if s < h: continue
                for txcost in slippage_override:
                    run_name = 'concentration_limit_'+ str(c)\
                               +'_holding_period_' + timedeltatostring(h) \
                               + '_signal_horizon_' + timedeltatostring(s) \
                               + '_slippage_override_' + str(txcost)
                    trajectory=perp_vs_cash_backtest(
                                      signal_horizon=s,
                                      holding_period=h,
                                      slippage_override=txcost,
                                      concentration_limit=c,
                                      filename='')#non verbose
                    #accrual[(c, h, s,)] = trajectory  # [(t,f)]
                    #for t in trajectory.columns.get_level_values('time').unique():
                    #    for f in trajectory.columns.get_level_values('field').unique():
                    #        accrual[(c,h,s,t,f)]=trajectory[(t,f)]
                    trajectory['slippage_override']=txcost
                    trajectory['concentration_limit'] = c
                    trajectory['signal_horizon'] = s
                    trajectory['holding_period'] = h
                    trajectory.to_excel(run_dir + '/' + run_name + '.xlsx')
                    ladder=ladder.append(trajectory,ignore_index=True)

    ladder.to_pickle(run_dir + '/ladder.pickle')

def run_benchmark_ladder(
                concentration_limit_list,
                holding_period_list,
                signal_horizon_list,
                slippage_override_list,
                run_dir):
    ladder = pd.DataFrame()
    #### first, pick best basket every hour, ignoring tx costs
    for c in concentration_limit_list:
            for txcost in slippage_override_list:
                trajectory = perp_vs_cash_backtest(
                                                   signal_horizon=timedelta(hours=1),
                                                   holding_period=timedelta(hours=1),
                                                   slippage_override=txcost,
                                                   concentration_limit=c,
                                                   filename='cost_blind',
                                                   optional_params=['cost_blind'])#,'verbose'])
                trajectory['slippage_override'] = txcost
                trajectory['concentration_limit'] = c
                ladder = ladder.append(trajectory,ignore_index=True)

    ladder.to_pickle(run_dir + '/ladder.pickle')

def run(command_list):
    print(static_params)
    if 'live' in command_list:
        perp_vs_cash_live(
                    concentration_limit=[CONCENTRATION_LIMIT],
                    signal_horizon = [HOLDING_PERIOD],
                    holding_period = [SIGNAL_HORIZON],
                    slippage_override = [SLIPPAGE_OVERRIDE],
                    run_dir='Runtime/Live_parquets')
        #s3_upload_file('Runtime/live_parquets/optimal_live.xlsx', 'gof.crypto.shared', 'ftx_optimal_cash_carry_'+datetime.utcnow().strftime("%Y-%m-%d-%Hh")+'.xlsx')
    if 'benchmark' in command_list:
        run_benchmark_ladder(
                    concentration_limit_list=[.5],
                    holding_period_list=[HOLDING_PERIOD],
                    signal_horizon_list=[SIGNAL_HORIZON],
                    slippage_override_list=[SLIPPAGE_OVERRIDE],
                    run_dir='Runtime/cost_blind')
    if 'ladder' in command_list:
        run_ladder( concentration_limit_list=[9,1,0.5],
                    holding_period_list = [timedelta(hours=d) for d in [6,12]] + [timedelta(days=d) for d in [1,2,3,4,5]],
                    signal_horizon_list = [timedelta(hours=d) for d in [12]] + [timedelta(days=d) for d in [1,2,3,4,5,7,10,30]],
                    slippage_override = [2e-4,5e-4],
                    run_dir='Runtime/runs')

    #run_ladder( concentration_limit_list=[9, 1, .5],
    #            holding_period_list = [timedelta(hours=h) for h in [6,12]] + [timedelta(days=d) for d in [1, 2, 3, 4,5]],
    #            signal_horizon_list = [timedelta(hours=h) for h in [12]] + [timedelta(days=d) for d in [1, 2,3,4,5,7,10,30]],
    #            slippage_override = [2e-4,5e-4],
    #            run_dir='Runtime/runs')

run(['live'])