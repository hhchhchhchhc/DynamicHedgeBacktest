from time import sleep,perf_counter
import asyncio
import dateutil.parser
import numpy
import pandas as pd
import functools

from ftx_ftx import *
from ftx_utilities import *
from ftx_portfolio import diff_portoflio

placement_latency = 0.1# in sec
amend_speed = 10.# in sec
amend_trigger = 0.0005
taker_trigger = 0.002
slice_factor = 10.# in USD
audit = []

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = perf_counter()      # 2
        run_time = end_time - start_time    # 3

        global audit
        #audit += [{'audit_type': 'timer', 'func': func.__name__, 'start_time': start_time, 'run_time': run_time}]
        return value
    return wrapper_timer

@timer
async def sure_cancel(exchange: Exchange,order: str) -> dict:
    order_status = exchange.fetch_order(order['id'])
    attempt = 0
    while order_status['status'] == 'open':
        exchange.cancel_order(order['id'])
        await asyncio.sleep(placement_latency)
        order_status = exchange.fetch_order(order['id'])
        attempt += 1
        global audit
        audit += [{'audit_type': 'cancel', 'id': order['id'], 'cancel_attempt': attempt}|order]

    return order

@timer
async def monitored_market_order(exchange: Exchange,
                              ticker: str,
                              side: str,
                              size: float  # in coin
                              ) -> dict:
    mkt_order = exchange.createOrder(ticker, 'market', side, size)
    await asyncio.sleep(placement_latency)
    order_status = exchange.fetch_order(mkt_order['id'])
    remaining = order_status['remaining']

    global audit
    audit += [{'audit_type': 'market', 'id': mkt_order['id'], 'remaining': remaining}|mkt_order]

    return mkt_order

class PlacementStrategy:
    def __init__(self,exchange: Exchange,
                              ticker: str,
                              side: str,
                              size: float  # in coin
                              ):
        self._ticker = ticker
        self._side = side
        self._size = size
    def update(self)->None:
        return None

class AggressivePost:
    def __init__(self, exchange: Exchange,
                 ticker: str,
                 side: str,
                 size: float,  # in coin
                 increments_from_opposite_tob: int):
        self.__init__(exchange, ticker, side, size)
        self._increments_from_opposite_tob = increments_from_opposite_tob
    def update(self)->None:
        return None
# ensures postOnly
# places limit right before top of book
# if canceled (because through mid, typically), try farther and farther to fight momentum
@timer
async def sure_postAggressive(exchange: Exchange,
                              ticker: str,
                              side: str,
                              size: float  # in coin
                              ) -> dict:
    sizeIncrement=float(exchange.fetch_ticker(ticker)['info']['sizeIncrement'])
    if (size<sizeIncrement):
        print(f'skip {size} < {sizeIncrement}')
        return dict()
    px_increment = float(exchange.fetch_ticker(ticker)['info']['priceIncrement'])
    attempt = 0 # start one increment in front of tob
    status = 'canceled'
    while status == 'canceled':
        top_of_book = exchange.fetch_ticker(ticker)['bid' if side == 'buy' else 'ask']
        # try farther and farther to fight momentum
        limit_price = top_of_book - (1 if side=='buy' else -1)*(attempt-1)*px_increment
        order = exchange.createOrder(ticker, 'limit', side, size,
                                     price=limit_price,
                                     params={'postOnly': True})

        await asyncio.sleep(placement_latency)

        order_status = exchange.fetch_order(order['id'])
        status = order_status['status']
        global audit

        audit+=[{'audit_type': 'postAggressive',
                 'id': order['id'],
                 'place_attempt': attempt,
                 'status': order_status['status'],
                 'top_of_book': top_of_book}|order]
        attempt+=1
    return order

# doesn't stop on partial fills
async def chase(exchange: Exchange,order: str, taker_trigger: float) -> dict:
    order_status = exchange.fetch_order(order['id'])
    ticker=order_status['symbol']
    side=order_status['side']
    trigger_level = order['price'] * (1 + taker_trigger * (1 if order['side'] == 'buy' else -1))
    attempt=0
    while order_status['status'] == 'open':
        top_of_book = exchange.fetch_ticker(ticker)['bid' if side=='buy' else 'ask']

        # if loss too big vs initial level, stop out
        if (1 if side=='buy' else -1)*(top_of_book-trigger_level)>0:
            await sure_cancel(exchange, order)
            stop_order = await monitored_market_order(exchange,ticker, side, order_status['remaining']) ## TODO: what if fail ?
            return stop_order

        # if move is material, chase

        # calibrate amend threshold
        #end_time = datetime.now().timestamp() * 1000
        #start_time = end_time - 100 * amend_speed * 1000
        #volatility = underlying_vol(exchange,symbol, start_time, end_time)

        if np.abs(top_of_book/order_status['price']-1)>amend_trigger:
            # ftx modify order isn't a good deal. cancel.
            await sure_cancel(exchange,order)
            order = await sure_postAggressive(exchange, ticker, side, order_status['remaining'])

        await asyncio.sleep(amend_speed)
        order_status = exchange.fetch_order(order['id'])

    return order

def most_illiquid(exchange,ticker_buy,ticker_sell):
    ## get 5m histories
    nowtime = datetime.now().timestamp() * 1000
    since = nowtime - 5 * 60 * 1000
    try:
        history1 = pd.DataFrame(exchange.fetch_trades(ticker_buy, since=since)).set_index('timestamp')
        history2 = pd.DataFrame(exchange.fetch_trades(ticker_sell, since=since)).set_index('timestamp')
    except Exception as e:
        print(e)
        print(f"for either {ticker_buy} or {ticker_sell}")
        return ticker_buy

    ## least active first
    volume1 = history1['amount'].sum()
    volume2 = history2['amount'].sum()

    result = ticker_buy if volume1<volume2 else ticker_sell
    return result

# size >0, in coin
@timer
async def execute_spread(exchange: Exchange, ticker_buy: str, ticker_sell: str, size_buy: float,size_sell: float) -> None:
    if most_illiquid(exchange, ticker_buy, ticker_sell)==ticker_buy:
        (first_ticker,second_ticker,first_side,second_side,first_size,second_size) = \
            (ticker_buy,ticker_sell,
            'buy','sell',
             size_buy,size_sell)
    else:
        (first_ticker, second_ticker, first_side, second_side,first_size,second_size) = \
            (ticker_sell,ticker_buy,
             'sell','buy',
             size_sell,size_buy)

    try:
        order = await sure_postAggressive(exchange, first_ticker, first_side, first_size)
        if order:
            await chase(exchange,order,taker_trigger=999) # no trigger for first leg

        order = await sure_postAggressive(exchange, second_ticker, second_side, second_size)
        if order:
            await chase(exchange,order,taker_trigger=taker_trigger)
    except Exception as e:
        print(e)

    return None

# size >0, in coin
@timer
async def execute_single(exchange, ticker, size):
    order = await sure_postAggressive(exchange, ticker, 'buy' if size > 0 else 'sell',np.abs(size))
    if order:
        await chase(exchange, order, taker_trigger=taker_trigger)

# size are + or -, in USD
async def slice_spread_order(exchange: Exchange, ticker1: str, ticker2: str, size1: float, size2: float)->None:
    fetched1 = exchange.fetch_ticker(ticker1)['info']
    increment1 = float(fetched1['sizeIncrement']) if ticker1!='BTC-PERP' else 0.01
    price1 = float(fetched1['price'])

    fetched2 = exchange.fetch_ticker(ticker2)['info']
    increment2 = float(fetched2['sizeIncrement']) if ticker2!='BTC-PERP' else 0.01
    price2 = float(fetched2['price'])

    slice_size = max(increment1*price1, increment2*price2,slice_factor)

    # if same side, single order
    if (size2 * size1 >= 0):
        await asyncio.gather(*[slice_single_order(exchange,ticker1,size1),
                               slice_single_order(exchange, ticker2, size2)])
        return None
    # size too small
    if ((np.abs(size1)<=increment1)|(np.abs(size2)<=increment2)):
        return None

    # slice the spread
    (spread_ticker_buy, spread_ticker_sell) = (ticker1, ticker2) if size1 > 0 else (ticker2, ticker1)
    spread_size = min(np.abs(size1),np.abs(size2))
    amount_sliced = 0
    execution_report = pd.DataFrame()
    while amount_sliced + 2 * slice_size < spread_size:
        price1 = float(exchange.fetch_ticker(ticker1)['info']['price'])
        price2 = float(exchange.fetch_ticker(ticker2)['info']['price'])
        await execute_spread(exchange, spread_ticker_buy, spread_ticker_sell,
                             slice_size/price1, slice_size/price2)
        amount_sliced += slice_size
    print(f'spread {ticker1}/{ticker2} done in {amount_sliced}')
    if ((spread_size-amount_sliced)/price1>increment1)&((spread_size-amount_sliced)/price2>increment2):
        await execute_spread(exchange, spread_ticker_buy, spread_ticker_sell,
                         (spread_size-amount_sliced)/price1,(spread_size-amount_sliced)/price2)
    print(f'spread residual {ticker1}/{ticker2} done in {spread_size-amount_sliced}')

    # residual, from book
    diff=diff_portoflio(exchange)
    residual_ticker=diff.loc[diff['name'].isin([ticker1,ticker2]),'name'].values
    residual_size = diff.loc[diff['name'].isin([ticker1, ticker2]), 'diff'].values
    await slice_single_order(exchange, residual_ticker[0], residual_size[0])
    await slice_single_order(exchange, residual_ticker[1], residual_size[1])
    print('spread:single residual {} done in {}'.format(residual_ticker[0], residual_size[0]))
    print('spread:single residual {} done in {}'.format(residual_ticker[1], residual_size[1]))

    return None

# size are + or -, in USD
async def slice_single_order(exchange: Exchange, ticker: str, size: float)->None:
    fetched = exchange.fetch_ticker(ticker)['info']
    increment = (float(fetched['sizeIncrement']) if ticker!='BTC-PERP' else 0.01)
    price=float(fetched['price'])
    slice_size = max(increment*price,slice_factor)

    # split order into slice_size
    amount_sliced = 0
    while amount_sliced + 2 * slice_size < np.abs(size):
        price=float(exchange.fetch_ticker(ticker)['info']['price'])
        await execute_single(exchange, ticker, np.sign(size)*slice_size/price)
        amount_sliced += slice_size
    print(f'single {ticker} done in {np.sign(size)*amount_sliced}')

    # residual, from book
    diff=diff_portoflio(exchange)
    residual_size=diff.loc[diff['name']==ticker,'diff'].values[0]
    if np.abs(residual_size)>=2*slice_size:
        print(f'residual {residual_size} > 2*slice {slice_size}')
    if np.abs(residual_size)/price > 1.1*increment:
        price=float(exchange.fetch_ticker(ticker)['info']['price'])
        await monitored_market_order(exchange, ticker,'buy' if residual_size>0 else 'sell', np.abs(residual_size)/price)
        print(f'single residual {ticker} done in {residual_size}')

    return None

async def execute_weights(exchange,weights):
    orders_by_underlying = [r[1] for r in weights.groupby('underlying')]
    single_orders = [{'ticker':r.head(1)['name'].values[0], 'size':r.head(1)['diff'].values[0]}
                     for r in orders_by_underlying if r.shape[0]==1]
    await asyncio.gather(*[slice_single_order(exchange,
                                    r['ticker'],
                                    r['size'])
                                         for r in single_orders])

    spread_orders = [{'ticker1':r.head(1)['name'].values[0], 'size1':r.head(1)['diff'].values[0],
                      'ticker2':r.tail(1)['name'].values[0], 'size2':r.tail(1)['diff'].values[0]}
                     for r in orders_by_underlying
                     if (r.shape[0]==2)]
    await asyncio.gather(*[slice_spread_order(exchange,
                                           r['ticker1'],
                                           r['ticker2'],
                                           r['size1'],
                                           r['size2'])
                        for r in spread_orders])

    n_orders = [None for r in orders_by_underlying if r.shape[0]>2]
    assert(len(n_orders)==0)

        #leftover_delta =
        #if leftover_delta>slice_sizeUSD/slice_factor:
        #    exchange.createOrder(future, 'market', order_status['side'], order_status['remaining'])

if True:
    exchange = open_exchange('ftx', 'SysPerp')
    weights = diff_portoflio(exchange, 'Runtime/ApprovedRuns/current_weights.xlsx')
    weights = weights[weights['diff'].apply(np.abs) > slice_factor]
    print(weights)
#    weights = weights[weights['name'].isin(['LINK/USD','LINK-PERP'])]

    start_time = datetime.now().timestamp()
    try:
        asyncio.run(execute_weights(exchange,weights))
        #asyncio.run(clean_dust(exchange))
    except Exception as e:
        print(e)
    finally:
        exchange.cancel_all_orders()
        end_time = datetime.now().timestamp()
        with pd.ExcelWriter('Runtime/execution_diagnosis.xlsx', engine='xlsxwriter') as writer:
            pd.DataFrame(exchange.fetch_orders(params={'start_time':start_time,'end_time':end_time})).to_excel(writer,sheet_name='fills')
            pd.DataFrame(audit).to_excel(writer,sheet_name='audit')

        stats = fetch_latencyStats(exchange,days=1,subaccount_nickname='SysPerp')
        print(stats)