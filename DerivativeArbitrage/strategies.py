import numpy as np
import pandas as pd
from ftx_snap_basis import *
from ftx_portfolio import *
from ftx_ftx import *
import seaborn as sns
import ccxt
#from sklearn import *

def perp_vs_cash_live(equity=1,
                signal_horizon = timedelta(days=7),
                holding_period = timedelta(days=7),
                concentration_limit = 99,
                loss_tolerance = 0.05,
                marginal_coin_penalty = 0.05,
                run_name=''):
    exchange = open_exchange('ftx')
    markets = exchange.fetch_markets()
    futures = pd.DataFrame(fetch_futures(exchange, includeExpired=False)).set_index('name')

    point_in_time = datetime.now().replace(minute=0,second=0,microsecond=0)

    # filtering params
    funding_volume_threshold = 1e5  # 5e5
    spot_volume_threshold = 1e4  # 5e4
    borrow_volume_threshold = 1e5  # 5e5
    type_allowed = 'perpetual'
    max_nb_coins = 99
    carry_floor = 0.4

    # fee estimation params
    slippage_override = 0  # TODO: 2e-4  #### this is given by mktmaker
    slippage_scaler = 1
    slippage_orderbook_depth = 1000

    ## ----------- enrich, get history, filter
    enriched = enricher(exchange, futures, holding_period, equity=equity,
                        slippage_override=slippage_override, slippage_orderbook_depth=slippage_orderbook_depth,
                        slippage_scaler=slippage_scaler,
                        params={'override_slippage': True, 'type_allowed': type_allowed, 'fee_mode': 'retail'})

    #### get history ( this is sloooow)
    hy_history = build_history(enriched, exchange,
                               timeframe='1h', end=point_in_time, start=point_in_time-signal_horizon-holding_period,
                               dirname='live_parquets')

    universe_filter_window = hy_history.index
    enriched['borrow_volume_avg'] = enriched.apply(lambda f:
                                                   (hy_history.loc[
                                                        universe_filter_window, f['underlying'] + '/rate/size'] *
                                                    hy_history.loc[universe_filter_window, f.name + '/mark/o']).mean(),
                                                   axis=1)
    enriched['spot_volume_avg'] = enriched.apply(lambda f:
                                                 (hy_history.loc[
                                                     universe_filter_window, f['underlying'] + '/price/volume']).mean(),
                                                 axis=1)
    enriched['future_volume_avg'] = enriched.apply(lambda f:
                                                   (hy_history.loc[
                                                       universe_filter_window, f.name + '/price/volume']).mean(),
                                                   axis=1)

    # ------- build derived data history
    (intLongCarry, intShortCarry, intUSDborrow, E_long, E_short, E_intUSDborrow) = build_derived_history(
        exchange, enriched, hy_history,
        holding_period,  # to convert slippage into rate
        signal_horizon)  # historical window for expectations)
    updated, marginFunc = update(enriched, point_in_time, hy_history, equity,
                                 intLongCarry, intShortCarry, intUSDborrow, E_long, E_short, E_intUSDborrow)
    # final filter, needs some history and good avg volumes
    pre_filtered = updated[
        (~np.isnan(updated['E_intCarry']))
        & (enriched['borrow_volume_avg'] > borrow_volume_threshold)
        & (enriched['spot_volume_avg'] > spot_volume_threshold)
        & (enriched['spot_volume_avg'] > spot_volume_threshold)]
    pre_filtered = pre_filtered.sort_values(by='E_intCarry', ascending=False).head(max_nb_coins)  # ,key=abs

    # run a trajectory
    optimized = pre_filtered
    updated, marginFunc = update(optimized, point_in_time, hy_history, equity,
                                 intLongCarry, intShortCarry, intUSDborrow, E_long, E_short, E_intUSDborrow)
    previous_weights = optimized['E_intCarry'] \
                       / (optimized['E_intCarry'].sum() if np.abs(optimized['E_intCarry'].sum()) > 0.1 else 0.1)

    optimized=cash_carry_optimizer(exchange,updated,marginFunc,
                                previous_weights=previous_weights,
                                holding_period = holding_period,
                                signal_horizon=signal_horizon,
                                concentration_limit=concentration_limit,
                                loss_tolerance=loss_tolerance,
                                equity=equity,
                                verbose=True
                              )

    optimized.to_excel('optimal_live.xlsx')
    return optimized


def perp_vs_cash_backtest(  equity=1,
                signal_horizon = timedelta(days=7),
                holding_period = timedelta(days=7),
                concentration_limit = 99,
                loss_tolerance = 0.05,
                marginal_coin_penalty = 0.05,
                run_name=''):
    exchange=open_exchange('ftx')
    markets = exchange.fetch_markets()
    futures = pd.DataFrame(fetch_futures(exchange,includeExpired=False)).set_index('name')

    # filtering params
    funding_volume_threshold = 1e5#5e5
    spot_volume_threshold = 1e4#5e4
    borrow_volume_threshold = 1e5#5e5
    type_allowed='perpetual'
    max_nb_coins = 99
    carry_floor = 0.4

    # fee estimation params
    slippage_override=0# TODO: 2e-4  #### this is given by mktmaker
    slippage_scaler=1
    slippage_orderbook_depth=1000

    # backtest params
    backtest_start = datetime(2021, 7, 1)
    backtest_end = datetime(2021, 10, 1)

    ## ----------- enrich, get history, filter
    enriched=enricher(exchange, futures, holding_period,equity=equity,
                    slippage_override=slippage_override, slippage_orderbook_depth=slippage_orderbook_depth,
                    slippage_scaler=slippage_scaler,
                    params={'override_slippage': True,'type_allowed':type_allowed,'fee_mode':'retail'})

    #### get history ( this is sloooow)
    try:
        hy_history = from_parquet("temporary_parquets/history.parquet")
        existing_futures = [name.split('/')[0] for name in hy_history.columns]
        new_futures = enriched[enriched['symbol'].isin(existing_futures)==False]
        if new_futures.empty==False:
            hy_history=pd.concat([hy_history,
                    build_history(new_futures,exchange,timeframe='1h',end=backtest_end,start=backtest_start-signal_horizon-holding_period)],
                    join='outer',axis=1)
            to_parquet(hy_history, "temporary_parquets/history.parquet")
    except FileNotFoundError:
        hy_history = build_history(enriched,exchange,timeframe='1h',end=backtest_end,start=backtest_start-signal_horizon-holding_period)
        to_parquet(hy_history,"temporary_parquets/history.parquet")

    universe_filter_window=hy_history[datetime(2021,6,1):datetime(2021,11,1)].index
    enriched['borrow_volume_avg'] = enriched.apply(lambda f:
                            (hy_history.loc[universe_filter_window,f['underlying']+'/rate/size']*hy_history.loc[universe_filter_window,f.name+'/mark/o']).mean(),axis=1)
    enriched['spot_volume_avg'] = enriched.apply(lambda f:
                            (hy_history.loc[universe_filter_window,f['underlying'] + '/price/volume'] ).mean(),axis=1)
    enriched['future_volume_avg'] = enriched.apply(lambda f:
                            (hy_history.loc[universe_filter_window,f.name + '/price/volume']).mean(),axis=1)

    # ------- build derived data history
    (intLongCarry, intShortCarry, intUSDborrow, E_long, E_short, E_intUSDborrow)=build_derived_history(
        exchange, enriched, hy_history,
        holding_period,  # to convert slippage into rate
        signal_horizon)  # historical window for expectations)
    updated, marginFunc = update(enriched, backtest_end, hy_history, equity,
                                 intLongCarry, intShortCarry, intUSDborrow, E_long, E_short, E_intUSDborrow)
    # final filter, needs some history and good avg volumes
    pre_filtered=updated[
              (~np.isnan(updated['E_intCarry']))
            & (enriched['borrow_volume_avg'] >borrow_volume_threshold)
            & (enriched['spot_volume_avg'] >spot_volume_threshold)
            & (enriched['spot_volume_avg'] >spot_volume_threshold)]
    pre_filtered = pre_filtered.sort_values(by='E_intCarry',ascending=False).head(max_nb_coins)#,key=abs

    # run a trajectory
    optimized=pre_filtered
    point_in_time=backtest_start+signal_horizon+holding_period # integrals not defined before that
    updated, marginFunc = update(optimized, backtest_start + signal_horizon + holding_period, hy_history, equity,
                                 intLongCarry, intShortCarry, intUSDborrow, E_long, E_short, E_intUSDborrow)
    previous_weights = optimized['E_intCarry'] \
                       / (optimized['E_intCarry'].sum() if np.abs(optimized['E_intCarry'].sum()) > 0.1 else 0.1)

    trajectory=pd.DataFrame()
    while point_in_time<backtest_end:
        updated,excess_margin=update(pre_filtered,point_in_time,hy_history,equity,
                       intLongCarry, intShortCarry, intUSDborrow, E_long, E_short, E_intUSDborrow)
        optimized=cash_carry_optimizer(exchange,updated,excess_margin,
                                previous_weights=previous_weights,
                                holding_period = holding_period,
                                signal_horizon=signal_horizon,
                                concentration_limit=concentration_limit,
                                loss_tolerance=loss_tolerance,
                                equity=equity,
                                verbose=True
                              )

        previous_weights=optimized['optimalWeight'].drop(['USD','total'])
        point_in_time+=holding_period
        trajectory=diagnosis_checkpoint(trajectory, optimized, 'time',point_in_time)

    #trajectory.xs('ask',level='field',axis=1)

#    with pd.ExcelWriter('summary_'+run_name+'.xlsx', engine='xlsxwriter') as writer:
#        trajectory.reorder_levels(['field','time'],axis='columns').to_excel(writer,'summary.xlsx')

    return trajectory

def timedeltatostring(dt):
    return str(dt.days)+'d'+str(int(dt.seconds/3600))+'h'
def run_ladder(dirname='runs'):
    ladder = pd.DataFrame()
    concentration_limit = [9,1,.5,.25]
    holding_period = [timedelta(hours=h) for h in [1, 3, 6, 12]] + [timedelta(days=d) for d in [1, 2, 3, 4, 5, 6, 7]]
    signal_horizon = [timedelta(hours=h) for h in [1, 12]] + [timedelta(days=d) for d in [1, 2, 3, 7, 30]]
    for c in concentration_limit:
        for hp in holding_period:
            for sh in signal_horizon:
                if sh < hp: continue
                run_name = 'hold' + timedeltatostring(hp) + 'signal' + timedeltatostring(sh)
                trajectory=perp_vs_cash_backtest(equity=1,
                                  signal_horizon=sh,
                                  holding_period=hp,
                                  concentration_limit=c,
                                  loss_tolerance=0.05,
                                  marginal_coin_penalty=0.05,
                                  run_name=run_name)
                ladder=diagnosis_checkpoint(ladder,trajectory, 'holding_period_signal_horizon',timedeltatostring(hp)+'_'+timedeltatostring(sh))

    ladder.to_pickle(dirname + '/runs.pickle')

perp_vs_cash_live(equity=1,
                signal_horizon = timedelta(days=2),
                holding_period = timedelta(days=1),
                concentration_limit = 0.5,
                loss_tolerance = 0.05,
                marginal_coin_penalty = 0.05,
                run_name='live')
run_ladder()