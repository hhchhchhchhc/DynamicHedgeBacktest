from ftx_snap_basis import *
from ftx_portfolio import *
from ftx_ftx import *
#import seaborn as sns
#from sklearn import *

#ignores borrow decile since depends on direction
def populate_concentration_limit(futures, hy_history, universe_filter_window=[]):
    if len(universe_filter_window) == 0:
        universe_filter_window = hy_history.index
    borrow_decile=0.5
    futures['borrow_volume_decile'] =0
    futures.loc[futures['spotMargin']==True,'borrow_volume_decile'] = futures[futures['spotMargin']==True].apply(lambda f:
                                                                                                                 (hy_history.loc[universe_filter_window,f['underlying']+'/rate/size']).quantile(q=borrow_decile)
                                                                                                                 ,axis=1)
    futures['spot_volume_avg'] = futures.apply(lambda f:
                                               (hy_history.loc[universe_filter_window,f['underlying'] + '/price/volume'] ).mean()
                                               ,axis=1)
    futures['future_volume_avg'] = futures.apply(lambda f:
                                                 (hy_history.loc[universe_filter_window,f.name + '/mark/volume']).mean()
                                                 ,axis=1)
    futures['concentration_limit']=futures.apply(lambda f:
                                                 min([f['borrow_volume_decile'] if f['spotMargin'] else f['openInterestUsd'],f['openInterestUsd'],f['spot_volume_avg']/24,f['future_volume_avg']/24])
                                                 ,axis=1)

    return futures

async def refresh_universe(exchange,universe_size):
    filename = 'Runtime/configs/universe.xlsx'
    if os.path.isfile(filename):
        try:
            return pd.read_excel(filename,sheet_name=universe_size,index_col=0)
        except:
            raise Exception('invalid Runtime/configs/universe.xlsx')

    futures = pd.DataFrame(await fetch_futures(exchange, includeExpired=False)).set_index('name')
    markets=await exchange.fetch_markets()

    universe_start = datetime(2021, 10, 1)
    universe_end = datetime(2022, 1, 1)
    borrow_decile = 0.25
    #type_allowed=['perpetual']
    screening_params=pd.DataFrame(
        index=['future_volume_threshold','spot_volume_threshold','borrow_volume_threshold','open_interest_threshold'],
        data={'max':[5e4,5e4,-1,5e4],# important that sets are decreasing  :(
              'wide':[1e5,1e5,2e5,5e6],# to run say 1M after leverage
              'institutional':[5e6,5e6,-1,1e7]})# instiutionals borrow OTC

    # qualitative screening
    futures = futures[
        (futures['expired'] == False) & (futures['enabled'] == True) & (futures['type'] != "move")
        & (futures.apply(lambda f: float(find_spot_ticker(markets, f, 'ask')), axis=1) > 0.0)
        & (futures['tokenizedEquity'] != True)]
    #& (futures['spotMargin'] == True)]

    # volume screening
    hy_history = await build_history(futures, exchange,
                                     timeframe='1h', end=universe_end, start=universe_start,
                                     dirname='Runtime/configs/universe_history_cache')
    futures = populate_concentration_limit(futures, hy_history, universe_filter_window=hy_history[universe_start:universe_end].index)

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for c in screening_params:# important that wide is first :(
            population = futures.loc[
                (futures['borrow_volume_decile'] > screening_params.loc['borrow_volume_threshold',c])
                & (futures['spot_volume_avg'] > screening_params.loc['spot_volume_threshold',c])
                & (futures['future_volume_avg'] > screening_params.loc['future_volume_threshold',c])
                & (futures['openInterestUsd'] > screening_params.loc['open_interest_threshold',c])]
            population.to_excel(writer, sheet_name=c)

        parameters = pd.Series(name='run_params',data={
            'run_date':datetime.today(),
            'universe_start': universe_start,
            'universe_end': universe_end,
            'borrow_decile': borrow_decile}).to_excel(writer,sheet_name='parameters')
        screening_params.to_excel(writer, sheet_name='screening_params')
        #TODO: s3_upload_file(filename, 'gof.crypto.shared', 'ftx_universe_'+str(datetime.now())+'.xlsx')

    print('refreshed universe')
    return futures

# runs optimization * [time, params]
async def perp_vs_cash(
        exchange,
        signal_horizon,
        holding_period,
        slippage_override,
        concentration_limit,
        exclusion_list=EXCLUSION_LIST+['AVAX','OMG'],
        run_dir='',
        backtest_start = None,# None means live-only
        backtest_end = None):
    try:
        first_history=pd.read_parquet(run_dir+'/'+os.listdir(run_dir)[0])
        if max(first_history.index)==datetime.now().replace(minute=0,second=0,microsecond=0):
            pass# if fetched less on this hour
        else:
            for file in os.listdir(run_dir): os.remove(run_dir+'/'+file)# otherwise do nothing and build_history will use what's there
    except:
        for file in os.listdir(run_dir): os.remove(run_dir + '/' + file)

    markets = await exchange.fetch_markets()
    futures = pd.DataFrame(await fetch_futures(exchange, includeExpired=False)).set_index('name')
    now_time = datetime.now()

    # filtering
    universe=await refresh_universe(exchange,UNIVERSE)
    universe=universe[~universe['underlying'].isin(exclusion_list)]
    type_allowed = TYPE_ALLOWED
    max_nb_coins = 99
    carry_floor = 0.4
    filtered = futures[(futures['type'].isin(type_allowed))
                       & (futures['symbol'].isin(universe.index))]

    # fee estimation params
    slippage_scaler = 1
    slippage_orderbook_depth = 10000

    # previous book
    if EQUITY.isnumeric():
        previous_weights_df = pd.DataFrame(index=futures.index,columns=['optimalWeight'],data=0.0)
        equity = float(EQUITY)
    elif '.xlsx' in EQUITY:
        previous_weights_df = pd.read_excel(EQUITY, sheet_name='optimized', index_col=0)['optimalWeight']
        equity = previous_weights_df.loc['total']
        previous_weights_df = previous_weights_df.drop(['USD', 'total'])
    else:
        start_portfolio = await fetch_portfolio(exchange, now_time)
        previous_weights_df = -start_portfolio.loc[
            start_portfolio['attribution'].isin(futures.index), ['attribution', 'usdAmt']
        ].set_index('attribution').rename(columns={'usdAmt': 'optimalWeight'})
        equity = start_portfolio.loc[start_portfolio['event_type'] == 'PV', 'usdAmt'].values[0]

    # run a trajectory
    if backtest_start == backtest_end:
        point_in_time = now_time.replace(minute=0, second=0, microsecond=0)
        backtest_start = point_in_time
        backtest_end = point_in_time
    else:
        point_in_time = backtest_start + signal_horizon + holding_period

    ## ----------- enrich/filter, get history, populate concentration limit
    enriched = await enricher(exchange, filtered, holding_period, equity=equity,
                              slippage_override=slippage_override, slippage_orderbook_depth=slippage_orderbook_depth,
                              slippage_scaler=slippage_scaler,
                              params={'override_slippage': True, 'type_allowed': type_allowed, 'fee_mode': 'retail'})
    hy_history = await build_history(enriched, exchange,
                                     timeframe='1h', end=backtest_end, start=backtest_start-signal_horizon-holding_period,
                                     dirname=run_dir)
    enriched = populate_concentration_limit(enriched, hy_history)

    # ------- build derived data history
    (intLongCarry, intShortCarry, intUSDborrow, intBorrow, E_long, E_short, E_intUSDborrow,E_intBorrow) = forecast(
        exchange, enriched, hy_history,
        holding_period,  # to convert slippage into rate
        signal_horizon,filename='history')  # historical window for expectations)
    updated, marginFunc = update(enriched, point_in_time, hy_history, equity,
                                 intLongCarry, intShortCarry, intUSDborrow, intBorrow, E_long, E_short, E_intUSDborrow,E_intBorrow)
    # final filter, needs some history and good avg volumes
    filtered = updated[~np.isnan(updated['E_intCarry'])]
    filtered = filtered.sort_values(by='E_intCarry', ascending=False).head(max_nb_coins)  # ,key=abs

    # run a trajectory
    optimized = filtered
    previous_time = point_in_time
    trajectory = pd.DataFrame()

    if backtest_start == backtest_end:
        previous_weights = previous_weights_df
    else:
        # set initial weights at non-silly values
        updated, marginFunc = update(optimized, point_in_time, hy_history, equity,
                                     intLongCarry, intShortCarry, intUSDborrow, intBorrow, E_long, E_short,
                                     E_intUSDborrow, E_intBorrow)
        previous_weights = pd.DataFrame()
        previous_weights['optimalWeight'] = equity * updated['E_intCarry'] \
                                            / (updated['E_intCarry'].sum() if np.abs(
            updated['E_intCarry'].sum()) > 0.1 else 0.1)


    while point_in_time <= backtest_end:
        updated, excess_margin = update(filtered, point_in_time, hy_history, equity,
                                        intLongCarry, intShortCarry, intUSDborrow, intBorrow, E_long, E_short, E_intUSDborrow,E_intBorrow)

        optimized = cash_carry_optimizer(exchange, updated, marginFunc,
                                         previous_weights_df=previous_weights[
                                             previous_weights.index.isin(filtered.index)],
                                         holding_period=holding_period,
                                         signal_horizon=signal_horizon,
                                         concentration_limit=concentration_limit,
                                         equity=equity,
                                         optional_params=['verbose']
                                         )
        # need to assign RealizedCarry to previous_time
        if not trajectory.empty: trajectory.loc[trajectory['time'] == previous_time, 'RealizedCarry'] = optimized[
            'RealizedCarry'].values
        optimized['time'] = point_in_time

        # increment
        trajectory = trajectory.append(optimized.reset_index().rename({'name': 'symbol'}), ignore_index=True)
        previous_weights = optimized['optimalWeight'].drop(index=['USD', 'total'])
        previous_time = point_in_time
        point_in_time += holding_period

    # for live, just send last optimized
    if backtest_start==backtest_end:
        filename = 'Runtime/ApprovedRuns/ftx_optimal_cash_carry_'+datetime.utcnow().strftime("%Y-%m-%d-%Hh")+'.xlsx'
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            parameters = pd.Series({
                'run_date':datetime.today(),
                'universe':UNIVERSE,
                'exclusion_list': exclusion_list,
                'type_allowed': type_allowed,
                'signal_horizon': signal_horizon,
                'holding_period': holding_period,
                'slippage_override':slippage_override,
                'concentration_limit': concentration_limit,
                'equity':equity,
                'max_nb_coins': max_nb_coins,
                'carry_floor': carry_floor,
                'slippage_scaler': slippage_scaler,
                'slippage_orderbook_depth': slippage_orderbook_depth})
            optimized.to_excel(writer,sheet_name='optimized')
            parameters.to_excel(writer,sheet_name='parameters')
            updated.to_excel(writer, sheet_name='snapshot')

        shutil.copy2(filename,'Runtime/ApprovedRuns/current_weights.xlsx')

        display=optimized[['optimalWeight','ExpectedCarry','transactionCost']]
        display['absWeight']=display['optimalWeight'].apply(abs)
        display.loc['total','absWeight']=display.drop(index='total')['absWeight'].sum()
        display=display.sort_values(by='absWeight',ascending=True)
        display= display[display['absWeight'].cumsum()>display.loc['total','absWeight']*.1]
        print(display)

        return optimized
    # for backtest, remove last line because RealizedCarry is wrong there
    else:
        trajectory = trajectory.drop(trajectory[trajectory['time'] == previous_time].index)
        trajectory['slippage_override'] = slippage_override
        trajectory['concentration_limit'] = concentration_limit
        trajectory['signal_horizon'] = signal_horizon
        trajectory['holding_period'] = holding_period

        filename = 'Runtime/runs/run_'+datetime.utcnow().strftime("%Y-%m-%d-%Hh")+'.xlsx'
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            parameters = pd.Series({
                'run_date':datetime.today(),
                'universe':UNIVERSE,
                'exclusion_list': exclusion_list,
                'type_allowed': type_allowed,
                'signal_horizon': signal_horizon,
                'holding_period': holding_period,
                'slippage_override':slippage_override,
                'concentration_limit': concentration_limit,
                'equity':equity,
                'max_nb_coins': max_nb_coins,
                'carry_floor': carry_floor,
                'slippage_scaler': slippage_scaler,
                'slippage_orderbook_depth': slippage_orderbook_depth,
                'backtest_start':backtest_start,
                'backtest_end':backtest_end,})
            trajectory.to_excel(writer,sheet_name='trajectory')
            parameters.to_excel(writer,sheet_name='parameters')

        return trajectory

async def strategy_wrapper(**kwargs):

    if EQUITY.isnumeric() or '.xlsx' in EQUITY:
        exchange = await open_exchange(kwargs['exchange'], '')
    else:
        exchange = await open_exchange(kwargs['exchange'],EQUITY, config={'asyncio_loop':asyncio.get_running_loop()})
    await exchange.load_markets()

    result = await asyncio.gather(*[perp_vs_cash(
        exchange=exchange,
        concentration_limit=concentration_limit,
        signal_horizon=signal_horizon,
        holding_period=holding_period,
        slippage_override=slippage_override,
        run_dir=kwargs['run_dir'],
        backtest_start=kwargs['backtest_start'],
        backtest_end=kwargs['backtest_end'])
        for concentration_limit in kwargs['concentration_limit']
        for signal_horizon in kwargs['signal_horizon']
        for holding_period in kwargs['holding_period']
        for slippage_override in kwargs['slippage_override']])
    await exchange.close()

    return None

def strategies_main(*argv):
    argv=list(argv)
    if len(argv) == 0:
        argv.extend(['sysperp'])
    if len(argv) < 3:
        argv.extend([HOLDING_PERIOD, SIGNAL_HORIZON])
    print(f'running {argv}')
    if argv[0] == 'sysperp':
        return asyncio.run(strategy_wrapper(
            exchange='ftx',
            concentration_limit=[CONCENTRATION_LIMIT],
            signal_horizon=[argv[1]],
            holding_period=[argv[2]],
            slippage_override=[SLIPPAGE_OVERRIDE],
            run_dir='Runtime/Live_parquets',
            backtest_start=None,backtest_end=None))
    elif argv[0] == 'backtest':
        concentration_limit=[1]
        signal_horizon=[timedelta(hours=h) for h in [12,48,60]]
        holding_period=[timedelta(hours=h) for h in [24]]
        slippage_override=[0.0005]
        return asyncio.run(strategy_wrapper(
            exchange='ftx',
            concentration_limit=concentration_limit,
            signal_horizon=signal_horizon,
            holding_period=holding_period,
            slippage_override=slippage_override,
            run_dir='Runtime/runs',
            backtest_start=datetime(2021,11,1),
            backtest_end=datetime(2022,2,1)))
    else:
        print(f'commands: sysperp [signal_horizon] [holding_period], backtest')

if __name__ == "__main__":
    strategies_main(*sys.argv[1:])





