from time import sleep
from ftx_history import *
from ftx_utilities import *
from ftx_snap_basis import enricher

def ftx_history(dirname='',
                       start=datetime(2021, 9, 20),
                       end = datetime(2021, 11, 16),
                       timeframe='5m',
                       coin_list=[]
                        ):
    exchange = open_exchange('ftx')
    markets = exchange.fetch_markets()
    futures = pd.DataFrame(fetch_futures(exchange, includeExpired=False)).set_index('name')
    if coin_list!=[]: futures=futures[futures['underlying'].isin(coin_list)]

    # filtering params
    type_allowed = ['future','perpetual']
    max_nb_coins = 15
    carry_floor = 0.4

    # fee estimation params
    slippage_override = 2e-4  #### this is given by mktmaker
    slippage_scaler = 1
    slippage_orderbook_depth = 1000
    equity = 20000
    holding_period=timedelta(days=7)

    ## ----------- enrich, get history, filter
    enriched = enricher(exchange, futures, holding_period, equity=equity,
                        slippage_override=slippage_override, slippage_orderbook_depth=slippage_orderbook_depth,
                        slippage_scaler=slippage_scaler,
                        params={'override_slippage': True, 'type_allowed': type_allowed, 'fee_mode': 'retail'})

    #### get history ( this is sloooow)
    hy_history = build_history(enriched, exchange, timeframe=timeframe, end=end, start=start,dirname=dirname)

    universe_filter_window = hy_history[datetime(2021, 9, 1):datetime(2021, 11, 15)].index
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
                                                       universe_filter_window, f.name + '/mark/volume']).mean(),
                                                   axis=1)

    if dirname!='':
        hy_history.to_parquet(dirname+'/history.parquet')
        enriched.to_excel(dirname+'/history.xlsx')

    return hy_history

i=1
while i<1:
    try:
        end_time = datetime(2021, 7, 8)  # datetime.today().replace(minute=0,second=0,microsecond=0)
        start_time = datetime(2021, 7, 7)
        coins = ['OKB']
        ftx_history(dirname='',start=start_time,end=end_time,timeframe='5m',coin_list=coins)
    except:
        sleep(5)
    i=i+1