import copy
from time import sleep,perf_counter
import functools

from ftx_ftx import *
from ftx_utilities import *
from ftx_portfolio import diff_portoflio
from ftx_history import fetch_trades_history

placement_latency = 0.25# in sec
amend_speed = 10.# in sec
amend_trigger = 0.005
taker_trigger = 0.0025
max_slice_factor = 50.# in USD
time_budget = 60
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
    async def __init__(self,exchange: Exchange,
                              ticker: str,
                              side: str,
                              size: float  # in coin
                              ):
        self._ticker = ticker
        self._side = side
        self._size = size
    async def update(self)->None:
        return None

class AggressivePost:
    async def __init__(self, exchange: Exchange,
                 ticker: str,
                 side: str,
                 size: float,  # in coin
                 increments_from_opposite_tob: int):
        self.__init__(exchange, ticker, side, size)
        self._increments_from_opposite_tob = increments_from_opposite_tob
    async def update(self)->None:
        return None

class ExecutionLog:
    _logCounter = 0
    def __init__(self, order_type: str,
                 legs: list): # len is 1 or 2, needs 'ticker' and 'size'
        self._id = ExecutionLog._logCounter # orverriden to orderID for leafs
        ExecutionLog._logCounter +=1
        self._type = order_type
        self._legs = legs # len is 1 or 2, 'ticker' and 'size', later benchmarks and fills(average,filled,status)
        self._receivedAt = None # temporary
        self.children = []# list of ExecutionLog

    # fetches exchange to populate benchmarks
    async def initializeBenchmark(self,exchange: Exchange) -> None:
        self._receivedAt=datetime.now()
        fetched = await asyncio.gather(*[mkt_at_size(exchange,leg['ticker'],'asks' if leg['size']>0 else 'bids',np.abs(leg['size'])) for leg in self._legs])
        if datetime.now()>self._receivedAt + timedelta(milliseconds=100): # 15ms is approx throttle..
            warnings.warn('mkt_at_size was slow',RuntimeWarning)
        self._legs = [leg | {'benchmark':
                                 {'initial_mkt': f['mid'] * (1 + f['slippage']),
                                  'initial_mid':f['mid']}}
                      for f,leg in zip(fetched,self._legs)]

    # only native orders receive values. Otherwise recurse on children.
    # assigns and returns legs
    # fillSize is just an assert
    async def populateFill(self,exchange: Exchange) -> list:
        if self.children:
            children_list = await asyncio.gather(*[child.populateFill(exchange) for child in self.children])
            children_fills =[[l for l in children_list if l['ticker'] == leg['ticker']]
                             for leg in self._legs]

            self._legs = [leg | {'average': sum(l['filled']*l['average'] for l in fills if l['filled']!=0.0)/
                                            sum(l['filled'] for l in fills),
                                 'filled':sum(l['filled'] for l in fills),
                                 'status': 'closed' if
                                 all(l['status']=='closed' for l in fills)
                                                    else 'unclosed'}
                            for leg,fills in zip(self._legs,children_fills)]
        else:
            try:
                fills = await exchange.fetch_order(self._id)
            except Exception as e:
                if self._legs[0]['size'] < float(
                        (await exchange.fetch_ticker(self._legs[0]['ticker']))['info']['sizeIncrement']):
                    self._legs[0] = self._legs[0] | {'average': -1., 'filled': 0., 'status': 'closed'}
                    return self._legs

            if fills['status']=='open':
                warnings.warn('leaf still open')
            self._legs = [self._legs[0] | {'average':fills['average'],
                                       'filled':fills['filled']*(1 if fills['side']=='buy' else -1),
                                       'status':fills['status']}]
        return self._legs

    def bpCost(self)->str:
        vs_initial_mkt = sum(leg['filled']*(leg['benchmark']['initial_mkt']/leg['average']-1)*10000 for leg in self._legs)
        vs_initial_mid = sum(leg['filled'] * (leg['benchmark']['initial_mid']/leg['average']-1)*10000 for leg in self._legs)
        size_done = sum(leg['filled'] for leg in self._legs)
        return {'vs_initial_mkt':vs_initial_mkt/size_done,'vs_initial_mid':vs_initial_mid/size_done}

async def symbol_ordering(exchange: Exchange,ticker1: str,ticker2: str) -> Tuple:
    ## get 5m histories
    nowtime = datetime.now().timestamp()
    since = nowtime - 5 * 60
    try:
        history1 = pd.DataFrame(await exchange.fetch_trades(ticker1, since=since*1000)).set_index('timestamp')
        history2 = pd.DataFrame(await exchange.fetch_trades(ticker2, since=since*1000)).set_index('timestamp')
    except Exception as e:
        warnings.warn(f"bad history for either {ticker1} or {ticker2}. Passing {ticker2}",RuntimeWarning)
        return (ticker2,ticker1)

    ## least active first
    volume1 = history1['amount'].sum()
    volume2 = history2['amount'].sum()

    return (ticker1,ticker2) if volume1<volume2 else (ticker2,ticker1)

#################### low level orders, with checks and resend ##############################

#@timer
async def sure_cancel(exchange: Exchange,id: str) -> dict:
    order_status = await exchange.fetch_order(id)
    attempt = 0
    while order_status['status'] == 'open':
        await exchange.cancel_order(id)
        #await asyncio.sleep(placement_latency)
        order_status = await exchange.fetch_order(id)
        attempt += 1

    return attempt

#@timer
async def monitored_market_order(exchange: Exchange,
                              ticker: str,
                              side: str,
                              size: float  # in coin
                              ) -> ExecutionLog:
    log=ExecutionLog(sys._getframe().f_code.co_name, [{'ticker': ticker, 'size': size}])
    await log.initializeBenchmark(exchange)
    order = await exchange.createOrder(ticker, 'market', side, size)
    await asyncio.sleep(placement_latency)
    log._id = order['id']

    return log

# ensures postOnly
# places limit right before top of book
# if canceled (because through mid, typically), try farther and farther to fight momentum
#@timer
async def sure_post(exchange: Exchange,
                              ticker: str,
                              side: str,
                              size: float,  # in coin
                              px_increment: float,
                              mode: str = 'passive') -> ExecutionLog:# shift pushes towards aggressive
    sizeIncrement=float((await exchange.fetch_ticker(ticker))['info']['sizeIncrement'])
    if (size>sizeIncrement/2)&(size<sizeIncrement):
        size = sizeIncrement
        warnings.warn(f'rounded up {ticker} in {size} to {sizeIncrement}',RuntimeWarning)

    log = ExecutionLog(sys._getframe().f_code.co_name, [{'ticker': ticker, 'size': size}])
    await log.initializeBenchmark(exchange)

    attempt = 1 # start one increment in front of tob
    status = 'canceled'
    while status == 'canceled':
        # try farther and farther to fight momentum
        if mode=='aggressive':
            top_of_book = (await exchange.fetch_ticker(ticker))['ask' if side == 'buy' else 'bid']
            limit_price = top_of_book - (1 if side == 'buy' else -1) * attempt * px_increment
        elif mode=='passive':
            top_of_book = (await exchange.fetch_ticker(ticker))['bid' if side == 'buy' else 'ask']
            limit_price = top_of_book + (1 if side == 'buy' else -1) * (2-attempt) * px_increment

        order = await exchange.createOrder(ticker, 'limit', side, size,
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
async def post_chase_trigger(exchange: Exchange,
                            ticker: str, size: float,
                            taker_trigger: float) -> ExecutionLog:
    fetched = (await exchange.fetch_ticker(ticker))['info']
    increment = float(fetched['priceIncrement'])
    side ='buy' if size>0 else 'sell'
    mode='passive' if taker_trigger>1 else 'aggressive'
    trigger_level = float(fetched['price']) * (1 + taker_trigger * (1 if size > 0 else -1))

    log = ExecutionLog(sys._getframe().f_code.co_name, [{'ticker': ticker, 'size': size}])
    await log.initializeBenchmark(exchange)

    # post
    from_tob=increment
    log.children += await asyncio.gather(*[sure_post(exchange,ticker,
                                    side,np.abs(size),from_tob,
                                    mode=mode)])
    await asyncio.sleep(amend_speed)
    current_id=log.children[-1]._id
    order_status = await exchange.fetch_order(current_id)
    while float(order_status['remaining']) > 0:
        opposite_tob = (await exchange.fetch_ticker(ticker))['ask' if size>0 else 'bid']

        # if loss too big vs initial level, stop out
        if (1 if side=='buy' else -1)*(opposite_tob-trigger_level)>0:
            await sure_cancel(exchange, current_id)
            log.children+= await asyncio.gather(*[monitored_market_order(exchange,ticker,
                                side,
                                order_status['remaining'])])
            return log

        # if move is material, chase 1 increment from opposite tob
        if np.abs(opposite_tob/order_status['price']-1)>amend_trigger:
            # ftx modify order isn't a good deal. cancel.
            await sure_cancel(exchange,current_id)
            log.children += await asyncio.gather(*[sure_post(exchange, ticker, side, order_status['remaining'],increment,mode='aggressive')])

        if False:
            # if nothing happens, get a little closer to mid
            from_tob+=increment
            await sure_cancel(exchange, order)
            log.children += await asyncio.gather(*[sure_post(exchange, ticker, side, order_status['remaining'], from_tob, mode=mode)])

        await asyncio.sleep(amend_speed)
        current_id = log.children[-1]._id
        order_status = await exchange.fetch_order(current_id)

    await log.populateFill(exchange)
    return log

#################### slicers: chose sequence and size for orders  ##############################

# size are + or -, in USD
async def slice_spread_order(exchange: Exchange, ticker_1: str, ticker_2: str, size_1: float, size_2: float, min_volume: float)->ExecutionLog:
    log = ExecutionLog(sys._getframe().f_code.co_name, [{'ticker': ticker_1, 'size': size_1},
                                                        {'ticker': ticker_2, 'size': size_2}])
    await log.initializeBenchmark(exchange)

    (ticker1,ticker2) = await symbol_ordering(exchange, ticker_1, ticker_2)
    (size1,size2) = (size_1,size_2) if ticker1==ticker_1 else (size_2,size_1)

    fetched1 = (await exchange.fetch_ticker(ticker1))['info']
    increment1 = special_maker_minimum_size[ticker1] \
        if ticker1 in special_maker_minimum_size.keys() \
        else float(fetched1['sizeIncrement'])
    price1 = float(fetched1['price'])


    fetched2 = (await exchange.fetch_ticker(ticker2))['info']
    increment2 = special_maker_minimum_size[ticker2] \
        if ticker2 in special_maker_minimum_size.keys() \
        else float(fetched2['sizeIncrement'])
    price2 = float(fetched2['price'])

    slice_factor = amend_speed * min_volume/price1
    slice_size = max(increment1, increment2,min(max_slice_factor/price1,slice_factor))

    # if same side, single order
    if (size2 * size1 >= 0):
        log.children += await asyncio.gather(*[slice_single_order(exchange, ticker1, size1,min_volume),
                                               slice_single_order(exchange, ticker2, size2,min_volume)])
        return log
    # size too small
    if ((np.abs(size1)<=increment1)|(np.abs(size2)<=increment2)):
        return log

    # slice the spread
    spread_size = min(np.abs(size1),np.abs(size2))
    amount_sliced = 0
    while amount_sliced + slice_size < spread_size:
        price1 = float((await exchange.fetch_ticker(ticker1))['info']['price'])
        log.children +=await asyncio.gather(*[post_chase_trigger(exchange, ticker1, np.sign(size1) * slice_size, taker_trigger=999)])

        price2 = float((await exchange.fetch_ticker(ticker2))['info']['price'])
        log.children +=await asyncio.gather(*[post_chase_trigger(exchange, ticker2, np.sign(size2) * slice_size, taker_trigger=taker_trigger)])

        amount_sliced += slice_size
    print(f'spread {ticker1}/{ticker2} done in {amount_sliced*price1}')

#    if ((spread_size-amount_sliced)>max(increment1,increment2)):
#        print('residual:')
#        log.children +=[await slice_spread_order(exchange,
#                                                ticker1, ticker2,
#                                                np.sign(size1) * (spread_size-amount_sliced) ,
#                                                np.sign(size2) * (spread_size - amount_sliced) )]

    # residual, from book
    diff=diff_portoflio(exchange)
    residual_ticker=diff.loc[diff['name'].isin([ticker1,ticker2]),'name'].values
    residual_size = diff.loc[diff['name'].isin([ticker1, ticker2]), 'diff'].values
    residual_price = diff.loc[diff['name'].isin([ticker1, ticker2]), 'price'].values
    log.children +=await asyncio.gather(*[slice_single_order(exchange, residual_ticker[0], residual_size[0],min_volume)])
    print('spread:single residual {} done in {}'.format(residual_ticker[0], residual_size[0]*residual_price[0]))
    log.children +=await asyncio.gather(*[slice_single_order(exchange, residual_ticker[1], residual_size[1],min_volume)])
    print('spread:single residual {} done in {}'.format(residual_ticker[1], residual_size[1]*residual_price[1]))

    await log.populateFill(exchange)
    pickleit(global_log, "Runtime/ExecutionLog.pickle")
    return log

# size are + or -, in coin
async def slice_single_order(exchange: Exchange, ticker: str, size: float,min_volume: float) -> ExecutionLog:
    fetched = (await exchange.fetch_ticker(ticker))['info']
    increment = (float(fetched['sizeIncrement']) if ticker != 'BTC-PERP' else 0.01)
    price = float(fetched['price'])

    slice_factor = amend_speed * min_volume/price
    slice_size = max(increment, min(max_slice_factor,slice_factor)/price)

    log = ExecutionLog(sys._getframe().f_code.co_name, [{'ticker': ticker, 'size': size}])
    await log.initializeBenchmark(exchange)

    # split order into slice_size
    amount_sliced = 0
    while amount_sliced + slice_size < np.abs(size):
        price = float((await exchange.fetch_ticker(ticker))['info']['price'])
        log.children += await asyncio.gather(*[post_chase_trigger(exchange, ticker, np.sign(size) * slice_size,taker_trigger=taker_trigger)])
        amount_sliced += slice_size
    print(f'single {ticker} done in {np.sign(size)*amount_sliced*price}')

    # residual, from book
    diff=diff_portoflio(exchange)
    residual_size=diff.loc[diff['name']==ticker,'diff'].values[0]
    if np.abs(residual_size)>=slice_size:
        warnings.warn(f'residual {residual_size*price} > slice {slice_size*price}',RuntimeWarning)
    if np.abs(residual_size) > 1.1*increment:
        price=float((await exchange.fetch_ticker(ticker))['info']['price'])
        log.children += await asyncio.gather(*[post_chase_trigger(exchange, ticker,residual_size,taker_trigger=taker_trigger)])
        print(f'single residual {ticker} done in {residual_size*price}')

    await log.populateFill(exchange)
    pickleit(log, "Runtime/ExecutionLog.pickle")
    return log

########### executer: calls slicers in parallel

async def executer_sysperp(exchange: Exchange,weights: pd.DataFrame) -> ExecutionLog:
    log=ExecutionLog(sys._getframe().f_code.co_name,[
        {'ticker':r['name'],
         'size':r['diff']}
        for (i,r) in weights.iterrows()])
    await log.initializeBenchmark(exchange)

    orders_by_underlying = [r[1] for r in weights.groupby('underlying')]
    single_orders = [{'ticker':r.head(1)['name'].values[0], 'size':r.head(1)['diff'].values[0], 'volume':r.head(1)['volume'].values[0]}
                     for r in orders_by_underlying if r.shape[0]==1]
    log.children += await asyncio.gather(*[slice_single_order(exchange,
                                                              r['ticker'],
                                                              r['size'],r['volume'])
                                           for r in single_orders])

    spread_orders = [{'ticker1':r.head(1)['name'].values[0], 'size1':r.head(1)['diff'].values[0],
                      'ticker2':r.tail(1)['name'].values[0], 'size2':r.tail(1)['diff'].values[0], 'volume':r['volume'].min()}
                     for r in orders_by_underlying
                     if (r.shape[0]==2)]
    log.children += await asyncio.gather(*[slice_spread_order(exchange,
                                                              r['ticker1'],
                                                              r['ticker2'],
                                                              r['size1'],
                                                              r['size2'],r['volume'])
                                           for r in spread_orders])

    n_orders = [None for r in orders_by_underlying if r.shape[0]>2]
    assert(len(n_orders)==0)

    await log.populateFill(exchange)
    return log
        #leftover_delta =
        #if leftover_delta>slice_sizeUSD/slice_factor:
        #    await exchange.createOrder(future, 'market', order_status['side'], order_status['remaining'])


async def ftx_rest_spread_main_wrapper(*argv):
    exchange = open_exchange(argv[1], argv[2])
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
    #        coin='MATIC'
    #        weights = weights[weights['name'].isin([coin+'/USD',coin+'-PERP'])]

    start_time = datetime.now().timestamp()
    log = ExecutionLog('dummy', [])
    try:
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
        await exchange.close()

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

if False:
    data=pd.DataFrame()
    with open('Runtime/ExecutionLog.pickle', 'rb') as file:
        try:
            while True:
                df=pickle.load(file)
                data=data.append(df)
        except EOFError:
            print('l')

