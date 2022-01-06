from time import sleep
import asyncio
import dateutil.parser
import numpy
import pandas as pd

from ftx_ftx import *
from ftx_utilities import *

placement_latency = 1
amend_speed = 10
amend_tolerance = 5
taker_trigger = 0.005
slice_factor = 10.0 # stay away from 1

async def place_postOnly(exchange: Exchange,
                   ticker: str,
                   side: str,
                   size: float# in coin
                    ) -> dict:
    status = 'new'
    while status != 'open':
        top_of_book = exchange.fetch_ticker(ticker)['bid' if side == 'buy' else 'ask']
        order = exchange.createOrder(ticker, 'limit', side, size, price=top_of_book,
                                     params={'postOnly': True})
        await asyncio.sleep(placement_latency)
        order_status = exchange.fetch_order(order['id'])
        status = order_status['status']
    return order

# doesn't stop on partial fills
async def chase(exchange: Exchange,
          order: str,
          # don't lose more than 20 bp (for second leg)
          taker_trigger: float = 0.002
          ) -> dict:
    trigger_level = order['price']*(1+taker_trigger*(1 if order['side']=='buy' else -1))
    order_status = exchange.fetch_order(order['id'])
    symbol=order_status['symbol']
    side=order_status['side']
    while order_status['status'] == 'open':
        top_of_book = exchange.fetch_ticker(symbol)['bid' if side=='buy' else 'ask']
        if (1 if side=='buy' else -1)*(top_of_book-trigger_level)>0:
            order = exchange.createOrder(symbol, 'market', side, order_status['remaining']) ## TODO: what if fail ?
            return order

        end_time=datetime.now().timestamp() * 1000
        start_time =end_time - 100*amend_speed * 1000
        volatility = underlying_vol(exchange,symbol,start_time,end_time)

        if np.abs(top_of_book-order_status['price'])>amend_tolerance*volatility:
            # ftx modify order isn't a good deal
            order_status = exchange.fetch_order(order['id'])
            while order_status['status'] == 'open':
                exchange.cancel_order(order['id'])
                await asyncio.sleep(placement_latency)
                order_status = exchange.fetch_order(order['id'])
            order = place_postOnly(exchange,symbol, side, order_status['remaining'])

        await  asyncio.sleep(amend_speed)
        order_status = exchange.fetch_order(order['id'])

    return order

async def execute_spread(exchange: Exchange, buy_ticker: str, sell_ticker: str, buy_size: int,sell_size: int) -> pd.DataFrame:# size in USD
    ## get 5m histories
    nowtime = datetime.now().timestamp()*1000
    since = nowtime-5*60*1000
    buy_history=pd.DataFrame(exchange.fetch_trades(buy_ticker,since=since)).set_index('timestamp')
    sell_history = pd.DataFrame(exchange.fetch_trades(sell_ticker, since=since)).set_index('timestamp')

    ## least active first
    buy_volume = buy_history['amount'].sum()
    sell_volume = sell_history['amount'].sum()
    if buy_volume<sell_volume:
        (first_ticker,second_ticker,first_side,second_side,first_size,second_size) = (buy_ticker,sell_ticker,'buy','sell',buy_size,sell_size)
    else:
        (first_ticker, second_ticker, first_side, second_side, first_size,second_size) = (sell_ticker,buy_ticker,'sell','buy',sell_size,buy_size)

    try:
        order = await place_postOnly(exchange, first_ticker, first_side, first_size/exchange.fetch_ticker(first_ticker)['close'])
        order = await chase(exchange,order,taker_trigger=999) # no trigger for first leg
        
        order = await place_postOnly(exchange, second_ticker, second_side, second_size/exchange.fetch_ticker(second_ticker)['close'])
        order = await chase(exchange,order,taker_trigger=taker_trigger)
    except Exception as e:
        print(e)

    ##TODO: also cancel pre-existing ones :(

    exchange.cancel_all_orders(symbol=buy_ticker)
    exchange.cancel_all_orders(symbol=sell_ticker)

    first_recap = pd.DataFrame(exchange.fetch_orders(first_ticker,nowtime))
    #first_recap['createdAt']=first_recap['info'].apply(lambda x: x['createdAt'])
    second_recap = pd.DataFrame(exchange.fetch_orders(second_ticker, nowtime))
    #second_recap['createdAt'] = second_recap['info'].apply(lambda x: x['createdAt'])
    return first_recap.append(second_recap)


async def slice_spread_trade(exchange, buy_ticker, sell_ticker, weight):
    buy_fetched = exchange.fetch_ticker(buy_ticker)['info']
    buy_increment, buy_close = float(buy_fetched['sizeIncrement']), float(buy_fetched['price'])
    sell_fetched = exchange.fetch_ticker(sell_ticker)['info']
    sell_increment, sell_close = float(sell_fetched['sizeIncrement']), float(sell_fetched['price'])
    slice_sizeUSD = min(buy_increment * buy_close, sell_increment * sell_close) * slice_factor

    amount_sliced = 0
    execution_report = pd.DataFrame()
    while amount_sliced + 2 * slice_sizeUSD < weight:
        execution_report = execution_report.append(await
            execute_spread(exchange, buy_ticker, sell_ticker, slice_sizeUSD, slice_sizeUSD))
        amount_sliced += slice_sizeUSD
    execution_report = execution_report.append(await
        execute_spread(exchange, buy_ticker, sell_ticker, weight - amount_sliced, weight - amount_sliced))

    return execution_report

async def execute_weights(filename = 'Runtime/ApprovedRuns/current_weights.xlsx'):
    exchange = open_exchange('ftx','SysPerp')
    weights = pd.read_excel('Runtime/ApprovedRuns/current_weights.xlsx')
    equity = weights.loc[weights['name']=='total','optimalWeight'].values
    weights=weights[(weights['name']!='USD')&(weights['name']!='total')].sort_values(by='ExpectedCarry',ascending=False)

    weights['spot']=weights['name'].apply(lambda x: x.split('-')[0]+'/USD')
    weights['buy_ticker'] = weights['spot']
    weights.loc[weights['optimalWeight']<0,'buy_ticker'] = weights.loc[weights['optimalWeight']<0,'name']
    weights['sell_ticker'] = weights['name']
    weights.loc[weights['optimalWeight']<0,'sell_ticker'] = weights.loc[weights['optimalWeight']<0,'spot']
    weights['optimalWeight'] = weights['optimalWeight'].apply(np.abs)

    exec_report = await asyncio.gather(*[slice_spread_trade(exchange, d['buy_ticker'], d['sell_ticker'], d['optimalWeight']) for (i,d) in weights.iterrows() if d['optimalWeight']>0.01*equity])

        #leftover_delta =
        #if leftover_delta>slice_sizeUSD/slice_factor:
        #    exchange.createOrder(future, 'market', order_status['side'], order_status['remaining'])


if True:
    asyncio.run(execute_weights())