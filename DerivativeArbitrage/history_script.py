from time import sleep
from ftx_history import *
from ftx_utilities import *
from ftx_snap_basis import enricher,forecast

def ftx_read_history(dirname='',coin_list=[]):
    exchange = open_exchange('ftx')
    futures = pd.DataFrame(fetch_futures(exchange, includeExpired=False)).set_index('name')
    if coin_list!=[]: futures=futures[futures['underlying'].isin(coin_list)]

    # pass timeframe=np.NaN to avoid recreating history....
    return build_history(futures, exchange, dirname=dirname, timeframe=np.NaN).dropna()

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
    signal_horizon = timedelta(days=7)

    ## ----------- enrich, get history, filter
    enriched = enricher(exchange, futures, holding_period, equity=equity,
                        slippage_override=slippage_override, slippage_orderbook_depth=slippage_orderbook_depth,
                        slippage_scaler=slippage_scaler,
                        params={'override_slippage': True, 'type_allowed': type_allowed, 'fee_mode': 'retail'})

    #### get history ( this is sloooow)
    hy_history = build_history(enriched, exchange, timeframe=timeframe, end=end, start=start,dirname=dirname)
    for (i,f) in enriched[enriched['type']=='perpetual'].iterrows():
        hy_history[f['underlying']+'/CarryLong'] = - hy_history['USD/rate/borrow'] + hy_history[f.name+'/rate/funding']
        hy_history[f['underlying'] + '/CarryShort'] = -hy_history[f['underlying']+'/rate/borrow'] - hy_history[f.name + '/rate/funding']

    if dirname!='':
        hy_history.to_parquet(dirname+'/history.parquet')
        enriched.to_excel(dirname+'/history.xlsx')

    return hy_history

i=1
while i<1:
    try:
        end_time = datetime(2021, 11, 23)  # datetime.today().replace(minute=0,second=0,microsecond=0)
        start_time = datetime(2021, 11, 18)
        coins = ['OKB']
        ftx_history(dirname='',start=start_time,end=end_time,timeframe='5m',coin_list=coins)
    except:
        sleep(5)
    i=i+1