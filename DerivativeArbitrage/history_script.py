from time import sleep
from ftx_history import *
from ftx_utilities import *
from ftx_snap_basis import enricher
from s3 import *

def build_fine_history(dirname):
    exchange = open_exchange('ftx')
    markets = exchange.fetch_markets()
    futures = pd.DataFrame(fetch_futures(exchange, includeExpired=False)).set_index('name')

    # filtering params
    type_allowed = 'future'
    max_nb_coins = 15
    carry_floor = 0.4

    # fee estimation params
    slippage_override = 2e-4  #### this is given by mktmaker
    slippage_scaler = 1
    slippage_orderbook_depth = 1000
    equity = 20000
    holding_period=timedelta(days=7)

    # backtest params
    backtest_start = datetime(2021, 9, 20)
    backtest_end = datetime(2021, 11, 16)

    ## ----------- enrich, get history, filter
    enriched = enricher(exchange, futures, holding_period, equity=equity,
                        slippage_override=slippage_override, slippage_orderbook_depth=slippage_orderbook_depth,
                        slippage_scaler=slippage_scaler,
                        params={'override_slippage': True, 'type_allowed': type_allowed, 'fee_mode': 'retail'})

    #### get history ( this is sloooow)
    hy_history = build_history(enriched, exchange, timeframe='15s', end=backtest_end, start=backtest_start,dirname=dirname)

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
                                                       universe_filter_window, f.name + '/price/volume']).mean(),
                                                   axis=1)

    hy_history.to_parquet(dirname+'/history.parquet')
    enriched.to_excel(dirname+'/history.xlsx')
    return None

build_fine_history('archived data/omg')
i=0
while i<0:
    try:
        build_fine_history('archived data/omg')
    except:
        sleep(60)
    i=i+1