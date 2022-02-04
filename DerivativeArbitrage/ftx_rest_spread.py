import copy
from time import sleep,perf_counter
import functools

import pandas as pd

from ftx_ftx import *
from ftx_utilities import *
from ftx_portfolio import diff_portoflio
from ftx_history import fetch_trades_history

placement_latency = 0.25# in sec
amend_speed = 10.# in sec
amend_trigger = 0.005
taker_trigger = 0.0025
max_slice_factor = 50.# in USD
global_log = []
audit = []

special_maker_minimum_size = {'BTC-PERP': 0.01, 'RUNE-PERP': 1.0}

#################### various helpers ##############################

#async def timer(func):
#    """Print the runtime of the decorated function"""
#    @functools.wraps(func)
#    async def wrapper_timer(*args, **kwargs):
#        start_time = perf_counter()    # 1
#        value = await func(*args, **kwargs)
#        end_time = perf_counter()      # 2
#        run_time = end_time - start_time    # 3####
#
#        global audit
#        #audit += [{'audit_type': 'timer', 'func': func.__name__, 'start_time': start_time, 'run_time': run_time}]
#        return value
#    return wrapper_timer

class PlacementStrategy:
    async def __init__(self,exchange: ccxt.Exchange,
                              symbol: str,
                              side: str,
                              size: float  # in coin
                              ):
        self._symbol = symbol
        self._side = side
        self._size = size
    async def update(self)->None:
        return None

class AggressivePost:
    async def __init__(self, exchange: ccxt.Exchange,
                 symbol: str,
                 side: str,
                 size: float,  # in coin
                 increments_from_opposite_tob: int):
        self.__init__(exchange, symbol, side, size)
        self._increments_from_opposite_tob = increments_from_opposite_tob
    async def update(self)->None:
        return None


async def symbol_ordering(exchange: ccxt.Exchange,symbol1: str,symbol2: str) -> Tuple:
    ## get 5m histories
    nowtime = datetime.now().timestamp()
    since = nowtime - 5 * 60
    try:
        history1 = pd.DataFrame(await exchange.fetch_trades(symbol1, since=since*1000)).set_index('timestamp')
        history2 = pd.DataFrame(await exchange.fetch_trades(symbol2, since=since*1000)).set_index('timestamp')
    except Exception as e:
        warnings.warn(f"bad history for either {symbol1} or {symbol2}. Passing {symbol2}",RuntimeWarning)
        return (symbol2,symbol1)

    ## least active first
    volume1 = history1['amount'].sum()
    volume2 = history2['amount'].sum()

    return (symbol1,symbol2) if volume1<volume2 else (symbol2,symbol1)

#################### low level orders, with checks and resend ##############################

#@timer
async def sure_cancel(exchange: ccxt.Exchange,id: str) -> dict:
    order_status = await exchange.fetch_order(id)
    attempt = 0
    while order_status['status'] == 'open':
        await exchange.cancel_order(id)
        #await asyncio.sleep(placement_latency)
        order_status = await exchange.fetch_order(id)
        attempt += 1

    return attempt

#@timer
async def monitored_market_order(exchange: ccxt.Exchange,
                              symbol: str,
                              size: float  # in coin
                              ) -> ExecutionLog:
    log=ExecutionLog(sys._getframe().f_code.co_name, [{'symbol': symbol, 'size': size}])
    await log.populate_context(exchange)
    order = await exchange.createOrder(symbol, 'market', 'buy' if size>0 else 'sell', np.abs(size))
    await asyncio.sleep(placement_latency)
    log._id = order['id']

    return log

# ensures postOnly
# places limit right before top of book
# if canceled (because through mid, typically), try farther and farther to fight momentum
#@timer
async def sure_post(exchange: ccxt.Exchange,
                              symbol: str,
                              size: float,  # in coin
                              px_increment: float,
                              mode: str = 'passive') -> ExecutionLog:# shift pushes towards aggressive
    sizeIncrement=float((await exchange.fetch_ticker(symbol))['info']['sizeIncrement'])
    if (np.abs(size)>sizeIncrement/2)&(np.abs(size)<sizeIncrement):
        size = sizeIncrement*np.sign(size)
        warnings.warn(f'rounded up {symbol} in {size} to {sizeIncrement}',RuntimeWarning)

    log = ExecutionLog(sys._getframe().f_code.co_name, [{'symbol': symbol, 'size': size}])
    await log.populate_context(exchange)

    attempt = 1 # start one increment in front of tob
    status = 'canceled'
    while status == 'canceled':
        # try farther and farther to fight momentum
        if mode=='aggressive':
            top_of_book = (await exchange.fetch_ticker(symbol))['ask' if size>0 else 'bid']
            limit_price = top_of_book - (1 if size>0 else -1) * attempt * px_increment
        elif mode=='passive':
            top_of_book = (await exchange.fetch_ticker(symbol))['bid' if size>0 else 'ask']
            limit_price = top_of_book + (1 if size>0 else -1) * (2-attempt) * px_increment

        order = await exchange.createOrder(symbol, 'limit', 'buy' if size>0 else 'sell', np.abs(size),
                                     price=limit_price,
                                     params={'postOnly': True})
        log._id = order['id']

        await asyncio.sleep(placement_latency)
        order_status = await exchange.fetch_order(order['id'])
        status = order_status['status']
        attempt+=1
    # put everything in log

    return log

#################### complex order: post_chase_trigger ##############################

# doesn't stop on partial fills
# size in coin
async def post_chase_trigger(exchange: ccxt.Exchange,
                            symbol: str, size: float,
                            taker_trigger: float) -> ExecutionLog:
    fetched = (await exchange.fetch_ticker(symbol))['info']
    increment = float(fetched['priceIncrement'])
    mode='passive' if taker_trigger>1 else 'aggressive'
    trigger_level = float(fetched['price']) * (1 + taker_trigger * (1 if size > 0 else -1))

    log = ExecutionLog(sys._getframe().f_code.co_name, [{'symbol': symbol, 'size': size}])
    await log.populate_context(exchange)

    # post
    from_tob=increment
    log.children += [await sure_post(exchange,symbol,
                                    size,from_tob,
                                    mode=mode)]
    await asyncio.sleep(amend_speed)
    current_id=log.children[-1]._id
    order_status = await exchange.fetch_order(current_id)
    while float(order_status['remaining']) > 0:
        opposite_tob = (await exchange.fetch_ticker(symbol))['ask' if size>0 else 'bid']

        # if loss too big vs initial level, stop out
        if (1 if size>0 else -1)*(opposite_tob-trigger_level)>0:
            await sure_cancel(exchange, current_id)
            log.children+= [await monitored_market_order(exchange,symbol,
                                order_status['remaining']*(1 if size>0 else -1))]
            return log

        # if move is material, chase 1 increment from opposite tob
        if np.abs(opposite_tob/order_status['price']-1)>amend_trigger:
            # ftx modify order isn't a good deal. cancel.
            await sure_cancel(exchange,current_id)
            log.children += [await sure_post(exchange, symbol, order_status['remaining']*(1 if size>0 else -1),increment,mode='aggressive')]

        if False:
            # if nothing happens, get a little closer to mid
            from_tob+=increment
            await sure_cancel(exchange, order)
            log.children += [await sure_post(exchange, symbol, order_status['remaining']*(1 if size>0 else -1), from_tob, mode=mode)]

        await asyncio.sleep(amend_speed)
        current_id = log.children[-1]._id
        order_status = await exchange.fetch_order(current_id)

    return log

#################### slicers: chose sequence and size for orders  ##############################

# size are + or -, in USD
async def slice_spread_order(exchange: ccxt.Exchange, symbol_1: str, symbol_2: str, size_1: float, size_2: float, min_volume: float)->ExecutionLog:
    log = ExecutionLog(sys._getframe().f_code.co_name, [{'symbol': symbol_1, 'size': size_1},
                                                        {'symbol': symbol_2, 'size': size_2}])
    await log.populate_context(exchange)

    (symbol1,symbol2) = await symbol_ordering(exchange, symbol_1, symbol_2)
    (size1,size2) = (size_1,size_2) if symbol1==symbol_1 else (size_2,size_1)

    fetched1 = (await exchange.fetch_ticker(symbol1))['info']
    increment1 = special_maker_minimum_size[symbol1] \
        if symbol1 in special_maker_minimum_size.keys() \
        else float(fetched1['sizeIncrement'])
    price1 = float(fetched1['price'])


    fetched2 = (await exchange.fetch_ticker(symbol2))['info']
    increment2 = special_maker_minimum_size[symbol2] \
        if symbol2 in special_maker_minimum_size.keys() \
        else float(fetched2['sizeIncrement'])
    price2 = float(fetched2['price'])

    slice_factor = amend_speed * min_volume/price1
    slice_size = max(increment1, increment2,min(max_slice_factor/price1,slice_factor))

    # if same side, single order
    if (size2 * size1 >= 0):
        log.children += [await slice_single_order(exchange, symbol1, size1,min_volume),
                                               slice_single_order(exchange, symbol2, size2,min_volume)]
        return log
    # size too small
    if ((np.abs(size1)<=increment1)|(np.abs(size2)<=increment2)):
        return log

    # slice the spread
    spread_size = min(np.abs(size1),np.abs(size2))
    amount_sliced = 0
    while amount_sliced + slice_size < spread_size:
        price1 = float((await exchange.fetch_ticker(symbol1))['info']['price'])
        log.children += [await post_chase_trigger(exchange, symbol1, np.sign(size1) * slice_size, taker_trigger=999)]

        price2 = float((await exchange.fetch_ticker(symbol2))['info']['price'])
        log.children +=[await post_chase_trigger(exchange, symbol2, np.sign(size2) * slice_size, taker_trigger=taker_trigger)]

        amount_sliced += slice_size
    print(f'spread {symbol1}/{symbol2} done in {amount_sliced*price1}')

#    if ((spread_size-amount_sliced)>max(increment1,increment2)):
#        print('residual:')
#        log.children +=[await slice_spread_order(exchange,
#                                                symbol1, symbol2,
#                                                np.sign(size1) * (spread_size-amount_sliced) ,
#                                                np.sign(size2) * (spread_size - amount_sliced) )]

    # residual, from book
    diff=await diff_portoflio(exchange)
    residual_symbol=diff.loc[diff['name'].isin([symbol1,symbol2]),'name'].values
    residual_size = diff.loc[diff['name'].isin([symbol1, symbol2]), 'diff'].values
    residual_price = diff.loc[diff['name'].isin([symbol1, symbol2]), 'price'].values
    log.children += [await slice_single_order(exchange, residual_symbol[0], residual_size[0],min_volume)]
    print('spread:single residual {} done in {}'.format(residual_symbol[0], residual_size[0]*residual_price[0]))
    log.children += [await slice_single_order(exchange, residual_symbol[1], residual_size[1],min_volume)]
    print('spread:single residual {} done in {}'.format(residual_symbol[1], residual_size[1]*residual_price[1]))

    return log

# size are + or -, in coin
async def slice_single_order(exchange: ccxt.Exchange, symbol: str, size: float,min_volume: float) -> ExecutionLog:
    fetched = (await exchange.fetch_ticker(symbol))['info']
    increment = (float(fetched['sizeIncrement']) if symbol != 'BTC-PERP' else 0.01)
    price = float(fetched['price'])

    slice_factor = amend_speed * min_volume/price
    slice_size = max(increment, min(max_slice_factor,slice_factor)/price)

    log = ExecutionLog(sys._getframe().f_code.co_name, [{'symbol': symbol, 'size': size}])
    await log.populate_context(exchange)

    # split order into slice_size
    amount_sliced = 0
    while amount_sliced + slice_size < np.abs(size):
        price = float((await exchange.fetch_ticker(symbol))['info']['price'])
        log.children += [await post_chase_trigger(exchange, symbol, np.sign(size) * slice_size,taker_trigger=taker_trigger)]
        amount_sliced += slice_size
    print(f'single {symbol} done in {np.sign(size)*amount_sliced*price}')

    # residual, from book
    diff=await diff_portoflio(exchange)
    residual_size=diff.loc[diff['name']==symbol,'diff'].values[0]
    if np.abs(residual_size)>=slice_size:
        warnings.warn(f'residual {residual_size*price} > slice {slice_size*price}',RuntimeWarning)
    if np.abs(residual_size) > 1.1*increment:
        price=float((await exchange.fetch_ticker(symbol))['info']['price'])
        log.children += [await post_chase_trigger(exchange, symbol,residual_size,taker_trigger=taker_trigger)]
        print(f'single residual {symbol} done in {residual_size*price}')

    return log

########### executer: calls slicers in parallel

async def executer_sysperp(exchange: ccxt.Exchange,weights: pd.DataFrame) -> ExecutionLog:
    log=ExecutionLog(sys._getframe().f_code.co_name,[
        {'symbol':r['name'],
         'size':r['diff']}
        for (i,r) in weights.iterrows()])
    await log.populate_context(exchange)

    orders_by_underlying = [r[1] for r in weights.groupby('underlying')]
    single_orders = [{'symbol':r.head(1)['name'].values[0], 'size':r.head(1)['diff'].values[0], 'volume':r.head(1)['volume'].values[0]}
                     for r in orders_by_underlying if r.shape[0]==1]
    log.children += await asyncio.gather(*[slice_single_order(exchange,
                                                              r['symbol'],
                                                              r['size'],r['volume'])
                                           for r in single_orders])

    spread_orders = [{'symbol1':r.head(1)['name'].values[0], 'size1':r.head(1)['diff'].values[0],
                      'symbol2':r.tail(1)['name'].values[0], 'size2':r.tail(1)['diff'].values[0], 'volume':r['volume'].min()}
                     for r in orders_by_underlying
                     if (r.shape[0]==2)]
    log.children += await asyncio.gather(*[slice_spread_order(exchange,
                                                              r['symbol1'],
                                                              r['symbol2'],
                                                              r['size1'],
                                                              r['size2'],r['volume'])
                                           for r in spread_orders])

    n_orders = [None for r in orders_by_underlying if r.shape[0]>2]
    assert(len(n_orders)==0)

    return log
        #leftover_delta =
        #if leftover_delta>slice_sizeUSD/slice_factor:
        #    await exchange.createOrder(future, 'market', order_status['side'], order_status['remaining'])


async def ftx_rest_spread_main_wrapper(*argv):
    exchange = await open_exchange(argv[1], argv[2])
    weights = await diff_portoflio(exchange, argv[3])

    trades_history_list=await asyncio.gather(*[fetch_trades_history(
       s,exchange,datetime.now()-timedelta(weeks=1),datetime.now(),'1s')
                      for s in weights['name'].values])
    weights['volume'] = [th.mean()[s.split('/USD')[0] + '/trades/volume'] for th,s in zip(trades_history_list,weights['name'].values)]
    weights['time2do'] = weights['price'] * weights['diff'].apply(np.abs) / weights['volume']
    too_slow = weights.loc[weights['time2do'] > time_budget, 'underlying'].unique()
    weights = weights[(weights['price'] * weights['diff'].apply(np.abs) > max_slice_factor)
                      & (weights['underlying'].isin(too_slow) == False)]
    print(too_slow + ' were too slow. Only doing:')
    print(weights)
    #coin='OMG'
    #weights = weights[weights['name'].isin([coin+'/USD',coin+'-PERP'])]

    start_time = datetime.now().timestamp()
    log = ExecutionLog('dummy', [])
    try:
        if weights.empty: raise Exception('nothing to execute')
        log = await executer_sysperp(exchange, weights)
        # asyncio.run(clean_dust(exchange))
        end_time = datetime.now().timestamp()
        with pd.ExcelWriter('Runtime/execution_diagnosis.xlsx', engine='xlsxwriter') as writer:
            pd.DataFrame(await exchange.fetch_orders(params={'start_time': start_time, 'end_time': end_time})).to_excel(
                writer, sheet_name='fills')
            pd.DataFrame(audit).to_excel(writer, sheet_name='audit')
        print(log.bpCost())

    except Exception as e:
        print(e)
    finally:
        # pickleit(log, "Runtime/ExecutionLog.pickle")
        await exchange.cancel_all_orders()
        stats = await fetch_latencyStats(exchange, days=1, subaccount_nickname='SysPerp')
        print(f'latencystats:{stats}')
        await log.populateFill(exchange)
        await exchange.close()

    return log.to_df()

def ftx_rest_spread_main(*argv):
    argv=list(argv)
    if len(argv) == 0:
        argv.extend(['execute'])
    if len(argv) < 4:
        argv.extend(['ftx', 'SysPerp', 'Runtime/ApprovedRuns/current_weights.xlsx'])
    print(f'running {argv}')
    if argv[0] == 'execute':
        return asyncio.run(ftx_rest_spread_main_wrapper(*argv))
    else:
        print(f'commands: execute')

if __name__ == "__main__":
    ftx_rest_spread_main(*sys.argv[1:])
