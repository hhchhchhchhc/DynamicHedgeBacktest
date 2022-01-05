from time import sleep

import dateutil.parser
import numpy

from ftx_ftx import *
from ftx_utilities import *

placement_latency = 0.1
amend_speed = 1
amend_tolerance = 2
taker_trigger = 0.002

def place_postOnly(exchange: Exchange,
                   ticker: str,
                   side: str,
                   size: float# in coin
                    ) -> dict:
    status = 'new'
    while status != 'open':
        top_of_book = exchange.fetch_ticker(ticker)['bid' if side == 'buy' else 'ask']
        order = exchange.createOrder(ticker, 'limit', side, size, price=top_of_book,
                                     params={'postOnly': True})
        sleep(placement_latency)
        order_status = exchange.fetch_order(order['id'])
        status = order_status['status']
    return order

# doesn't stop on partial fills
def chase(exchange: Exchange,
          order: str,
          # don't lose more than 20 bp (for second leg)
          taker_trigger: float = 0.002
          ) -> dict:
    trigger_level = order['price']*(1+taker_trigger*(1 if order['side']=='buy' else -1))
    order_status = exchange.fetch_order(order['id'])
    while order_status['status'] == 'open':
        top_of_book = exchange.fetch_ticker(order_status['symbol'])['bid' if order_status['side']=='buy' else 'ask']
        if (1 if order_status['side']=='buy' else -1)*(top_of_book-trigger_level)>0:
            order = exchange.createOrder(order_status['symbol'], 'market', order_status['side'], order_status['remaining']/trigger_level) ## TODO: what if fail ?
            return order

        since = datetime.now().timestamp() * 1000 - 100*amend_speed * 1000
        history = pd.DataFrame(exchange.fetch_trades(order_status['symbol'], since=since))
        volatility = history['price'].diff().std() # TODO: yuky

        if np.abs(top_of_book-order_status['price'])>amend_tolerance*volatility/np.sqrt(10):
            # ftx modify order isn't a good deal
            order_status = exchange.fetch_order(order['id'])
            while order_status['status'] == 'open':
                exchange.cancel_order(order['id'])
                sleep(placement_latency)
                order_status = exchange.fetch_order(order['id'])
            order = place_postOnly(exchange,order_status['symbol'], order_status['side'], order_status['remaining'])

        sleep(amend_speed)
        order_status = exchange.fetch_order(order['id'])

    return order

def execute_spread(exchange: Exchange, buy_ticker: str, sell_ticker: str, size: int) -> None:
    ## get 5m histories
    nowtime = datetime.now().timestamp()*1000
    since = nowtime-5*60*1000
    buy_history=pd.DataFrame(exchange.fetch_trades(buy_ticker,since=since)).set_index('timestamp')
    sell_history = pd.DataFrame(exchange.fetch_trades(sell_ticker, since=since)).set_index('timestamp')

    ## least active first
    buy_volume = buy_history['amount'].sum()
    sell_volume = sell_history['amount'].sum()
    (first_ticker,second_ticker,first_side,second_side) = (buy_ticker,sell_ticker,'buy','sell') if buy_volume<sell_volume else (sell_ticker,buy_ticker,'sell','buy')

    try:
        order = place_postOnly(exchange, first_ticker, first_side, size)
        order = chase(exchange,order,taker_trigger=999) # no trigger for first leg
        order = place_postOnly(exchange, second_ticker, second_side, size)
        order = chase(exchange,order,taker_trigger=taker_trigger)

    # TODO: also cancel pre-existing ones...
    exchange.cancel_all_orders(symbol=buy_ticker)
    exchange.cancel_all_orders(symbol=sell_ticker)

    buy_recap = pd.DataFrame(exchange.fetch_orders(buy_ticker,nowtime))
    sell_recap = pd.DataFrame(exchange.fetch_orders(sell_ticker, nowtime))
    return buy_recap.append(sell_recap)

if True:
    exchange = open_exchange('ftx')
    buy_ticker = 'ETH-PERP'
    sell_ticker = 'ETH/USD'
    size = 10/exchange.fetch_ticker(sell_ticker)['close']
    execute_spread(exchange, buy_ticker, sell_ticker, size)