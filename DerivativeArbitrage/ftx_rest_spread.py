from time import sleep,perf_counter
import asyncio
import dateutil.parser
import numpy
import pandas as pd
import functools

from ftx_ftx import *
from ftx_utilities import *

placement_latency = 0.1
amend_speed = 10.
amend_tolerance = 5.
taker_trigger = 0.002
slice_factor = int(10) # integer>1
from_tob = 1.001
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
        audit += {'audit_type':'timer','func':func.__name__,'start_time':start_time,'run_time':run_time}
        return value
    return wrapper_timer

@timer
async def sure_cancel(exchange: Exchange,order: str) -> int:
    order_status = exchange.fetch_order(order['id'])
    attempt = 0
    while order_status['status'] == 'open':
        exchange.cancel_order(order['id'])
        await asyncio.sleep(placement_latency)
        order_status = exchange.fetch_order(order['id'])
        attempt += 1
        global audit
        audit += {'audit_type':'cancel','id': order['id'], 'cancel_attempt': attempt}

    return attempt

@timer
async def sure_postOnly(exchange: Exchange,
                        ticker: str,
                        side: str,
                        size: float  # in coin
                        ) -> dict:
    attempt = 0
    remaining_size = size
    while remaining_size>0:
        top_of_book = exchange.fetch_ticker(ticker)['bid' if side == 'buy' else 'ask']
        order = exchange.createOrder(ticker, 'limit', side, remaining_size, price=top_of_book*from_tob,
                                     params={'postOnly': True})
        await asyncio.sleep(placement_latency)
        order_status = exchange.fetch_order(order['id'])
        remaining_size = order_status['remaining']
        global audit
        audit+={'audit_type':'postOnly',
                'id':order['id'],
                'place_attempt': attempt,
                'status':order_status['status'],
                'top_of_book': top_of_book}
        attempt+=1
    return order

# doesn't stop on partial fills
async def chase(exchange: Exchange,order: str, taker_trigger: float) -> dict:
    order_status = exchange.fetch_order(order['id'])
    symbol=order_status['symbol']
    side=order_status['side']
    attempt=0
    while order_status['status'] == 'open':
        top_of_book = exchange.fetch_ticker(symbol)['bid' if side=='buy' else 'ask']

        # if loss too big, stop out
        trigger_level = order['price'] * (1 + taker_trigger * (1 if order['side'] == 'buy' else -1))
        if (1 if side=='buy' else -1)*(top_of_book-trigger_level)>0:
            if await sure_cancel(exchange, order)>0:# if it was somehow gone
                stop_order = exchange.createOrder(symbol, 'market', side, order_status['remaining']) ## TODO: what if fail ?
                global audit
                audit+={'audit_type':'market','id':order['id'],'top_of_book': top_of_book}
            return stop_order

        # if move is material, chase
        # read recent history
        end_time = datetime.now().timestamp() * 1000
        start_time = end_time - 100 * amend_speed * 1000
        try:
            volatility = underlying_vol(exchange,symbol, start_time, end_time)
        except Exception as e:
            print(e)
            volatility = top_of_book*0.002/amend_tolerance
        #not really necessary: top_of_book = exchange.fetch_ticker(symbol)['bid' if side=='buy' else 'ask']
        if np.abs(top_of_book-order_status['price'])>amend_tolerance*volatility:
            # ftx modify order isn't a good deal. cancel.
            await sure_cancel(exchange,order)
            order = await sure_postOnly(exchange, symbol, side, order_status['remaining'])

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
        return ticker_buy

    ## least active first
    volume1 = history1['amount'].sum()
    volume2 = history2['amount'].sum()

    result = ticker_buy if volume1<volume2 else ticker_sell
    return result

# size are + or -, in USD
@timer
async def execute_spread(exchange: Exchange, ticker_buy: str, ticker_sell: str, size: float) -> pd.DataFrame:# size in USD
    assert (float(exchange.fetch_ticker(ticker_buy)['info']['sizeIncrement']))
    assert (float(exchange.fetch_ticker(ticker_sell)['info']['sizeIncrement']))

    if most_illiquid(exchange, ticker_buy, ticker_sell)==ticker_buy:
        (first_ticker,second_ticker,first_side,second_side) = \
            (ticker_buy,ticker_sell,
            'buy','sell')
    else:
        (first_ticker, second_ticker, first_side, second_side) = \
            (ticker_sell,ticker_buy,
             'sell','buy')

    try:
        order = await sure_postOnly(exchange, first_ticker, first_side, size / exchange.fetch_ticker(first_ticker)['close'])
        order = await chase(exchange,order,taker_trigger=999) # no trigger for first leg

        order = await sure_postOnly(exchange, second_ticker, second_side, size / exchange.fetch_ticker(second_ticker)['close'])
        order = await chase(exchange,order,taker_trigger=taker_trigger)
    except Exception as e:
        print(e)

    ##TODO: also cancel pre-existing ones :(
#    exchange.cancel_all_orders(symbol=first_ticker)
#    exchange.cancel_all_orders(symbol=second_ticker)

    first_recap = pd.DataFrame(exchange.fetch_orders(first_ticker))
    #first_recap['createdAt']=first_recap['info'].apply(lambda x: x['createdAt'])
    second_recap = pd.DataFrame(exchange.fetch_orders(second_ticker))
    #second_recap['createdAt'] = second_recap['info'].apply(lambda x: x['createdAt'])
    return first_recap.append(second_recap)

# size are + or -, in USD
async def slice_spread_trade(exchange: Exchange, ticker1: str, ticker2: str, size1: float, size2: float)->dict:
    fetched1 = exchange.fetch_ticker(ticker1)['info']
    increment1 = float(fetched1['sizeIncrement']) * float(fetched1['price'])
    fetched2 = exchange.fetch_ticker(ticker2)['info']
    increment2 = float(fetched2['sizeIncrement']) * float(fetched2['price'])
    slice_size = max(increment1, increment2) * slice_factor

    # split order into a spread and a residual
    assert(size2 * size1 <= -increment1*increment2) # needs to be a spread
    (spread_ticker_buy, spread_ticker_sell) = (ticker1, ticker2) if size1 > 0 else (ticker2, ticker1)
    if np.abs(size1) < np.abs(size2):
        spread_size = np.abs(size1)
        residual_ticker = ticker2
    else:
        spread_size = np.abs(size2)
        residual_ticker = ticker1
    residual_size = size2+size1
    residual_side = 'buy' if residual_size > 0 else 'sell'

    # slice the spread
    amount_sliced = 0
    execution_report = pd.DataFrame()
    while amount_sliced + 2 * slice_size < spread_size:
        temp_report = await execute_spread(exchange, spread_ticker_buy, spread_ticker_sell, slice_size)
        execution_report = execution_report.append(temp_report)
        amount_sliced += slice_size

    temp_report = await execute_spread(exchange, spread_ticker_buy, spread_ticker_sell, spread_size-amount_sliced)
    execution_report = execution_report.append(temp_report)

    # do the residual
    increment = increment1 if residual_ticker==ticker1 else increment2
    if residual_size>increment:
        order = await sure_postOnly(exchange, residual_ticker, residual_side, residual_size / exchange.fetch_ticker(residual_ticker)['close'])
        execution_report = execution_report.append(await
            chase(exchange, order, taker_trigger=taker_trigger))  # no trigger for first leg

    return execution_report

def diff_portoflio(exchange,filename = 'Runtime/ApprovedRuns/current_weights.xlsx'):
    # open file
    future_weights = pd.read_excel('Runtime/ApprovedRuns/current_weights.xlsx')
    future_weights = future_weights[(future_weights['name'] != 'USD') & (future_weights['name'] != 'total')]
    cash_weights = future_weights.copy()
    cash_weights['name']=cash_weights['name'].apply(lambda x: x.split('-')[0]+'/USD')
    cash_weights['optimalWeight'] *= -1
    target = future_weights.append(cash_weights)

    # get portfolio in USD
    markets = pd.DataFrame([r['info'] for r in exchange.fetch_markets()]).set_index('name')
    positions = pd.DataFrame([r['info'] for r in exchange.fetch_positions(params={})],
                             dtype=float).rename(columns={'future':'name','netSize':'total'})  # 'showAvgPrice':True})
    balances=pd.DataFrame(exchange.fetch_balance(params={})['info']['result'],dtype=float)#'showAvgPrice':True})
    balances=balances[balances['coin']!='USD']
    balances['name']=balances['coin']+'/USD'

    current=positions.append(balances)[['name','total']]
    current['current'] = current.apply(lambda f:
                    f['total'] * float(markets.loc[f['name'], 'price']), axis=1)

    # join, diff, coin
    diffs = target.set_index('name')[['optimalWeight']].join(current.set_index('name')[['current']],how='outer')
    diffs=diffs.fillna(0.0).reset_index()
    diffs['diff']=diffs['optimalWeight']-diffs['current']
    diffs['underlying'] = diffs['name'].apply(lambda x: x.split('-')[0].split('/USD')[0])

    return diffs

async def execute_weights(filename = 'Runtime/ApprovedRuns/current_weights.xlsx'):
    exchange = open_exchange('ftx','SysPerp')
    weights = diff_portoflio(exchange,filename)
    weights=weights[weights['diff'].apply(np.abs)>500]

    exec_report = await asyncio.gather(*[slice_spread_trade(exchange,
                                    r[1].head(1)['name'].values[0],
                                    r[1].tail(1)['name'].values[0],
                                    r[1].head(1)['diff'].values[0],
                                    r[1].tail(1)['diff'].values[0])
                                         for r in weights.groupby('underlying')])

    print(audit)
        #leftover_delta =
        #if leftover_delta>slice_sizeUSD/slice_factor:
        #    exchange.createOrder(future, 'market', order_status['side'], order_status['remaining'])

if True:
    asyncio.run(execute_weights())