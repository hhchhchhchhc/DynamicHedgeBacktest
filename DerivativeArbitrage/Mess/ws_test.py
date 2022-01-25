import time
import sys
import os
import ccxtpro
import asyncio
from ftx_utilities import max_iterations,min_iterations
import functools
from copy import deepcopy
import pandas as pd

import logging

class myFtx(ccxtpro.ftx):
    class myLogging(logging.Logger):
        def __init__(self):
            super().__init__(name='ftx')
            logging.basicConfig(level=logging.INFO)
            self._list = []

    def __init__(self,symbol: str, config={}):
        super().__init__(config=config)
        self._symbol = symbol
        self._done = False
        self._localLog = myFtx.myLogging()

    def logList(self,item: dict):
        self._localLog._list += [item]
        self._localLog.info(item)

    def handle_trade(self, client, message):
        message = super().handle_trade(client, message)
        self.logList({'orderbook_on_trade': self.orderbooks[self._symbol]['timestamp'],
                      'trade_on_trade': self.trades[self._symbol][-1]['timestamp'],
                      'exchange_on_trade': self.milliseconds()})
        return message

    def handle_order_book_update(self, client, message):
        message = super().handle_order_book_update(client, message)
        self.logList({'orderbook_on_orderbook': self.orderbooks[self._symbol]['timestamp'],
                      'trade_on_orderbook': self.trades[self._symbol][-1]['timestamp'] if (self.trades and self.trades[self._symbol]) else None,
                      'exchange_on_orderbook': self.milliseconds()})
        return message

def loop_try(func):
    @functools.wraps(func)
    async def wrapper_loop(*args, **kwargs):
        while True:
            try:
                value = await func(*args, **kwargs)
            except Exception as e:
                logging.exception(e)
    return wrapper_loop

async def loop_order_books(exchange):
    while True:
        try:
            res = await exchange.watch_order_book(exchange._symbol)
            if exchange._done:
                break
        except Exception as e:
            logging.exception(e)
    return None

async def loop_my_trades(exchange):
    while True:
        try:
            res = await exchange.watch_my_trades(exchange._symbol)
            balance = await exchange.fetch_balance()
            if abs(balance[exchange.market(exchange._symbol)['base']]['total']) > 50./exchange.trades[exchange._symbol][-1]['price']:
                exchange._done = True
                break
        except Exception as e:
            logging.exception(e)
    return None

async def loop_trades(exchange):
    while True:
        try:
            trades = await exchange.watch_trades(exchange._symbol)

            if exchange._done:
                break
            else:
                # places an order btw last trade and opposite top of book
                last_trade_side = exchange.trades[exchange._symbol][-1]['side']
                last_trade_price = exchange.trades[exchange._symbol][-1]['price']
                level = 0.5 * (last_trade_price +
                               exchange.orderbooks[exchange._symbol]['bids' if last_trade_side == 'buy' else 'asks'][0][0])
                mid = 0.5*(exchange.orderbooks[exchange._symbol]['bids'][0][0]+exchange.orderbooks[exchange._symbol]['asks'][0][0])

                exchange.logList({  'mid_on_trade': mid,
                                    'orderbook_on_trade': exchange.orderbooks[exchange._symbol]['timestamp'],
                                    'trade_on_trade': exchange.trades[exchange._symbol][-1]['timestamp'],
                                    'exchange_on_trade': exchange.milliseconds()})
                if exchange.orders and exchange._symbol in exchange.orders:
                    await exchange.edit_order(exchange.orders[exchange._symbol][-1], exchange._symbol,'limit', 'sell' if last_trade_side == 'sell' else 'buy',
                                                      10. / level, price=level,
                                                      params={'postOnly': True})
                else:
                    await exchange.create_limit_order(exchange._symbol, 'sell' if last_trade_side == 'sell' else 'buy', 10. / level, price=level,
                                        params={'postOnly': True})

            fetch = await exchange.fetch_ticker(exchange._symbol)
            mid = 0.5 * (fetch['bid'] + fetch['ask'])
            exchange.logList({  'mid_on_fetch': mid,
                                'orderbook_on_fetch': exchange.orderbooks[exchange._symbol]['timestamp'],
                                'trade_on_fetch': exchange.trades[exchange._symbol][-1]['timestamp'],
                                'exchange_on_fetch': exchange.milliseconds()})
        except Exception as e:
            logging.exception(e)

    return None

async def main(*argv,**kwargs):
    exchange = myFtx('UNI/USD',config={
        'asyncio_loop': kwargs['loop'],
        'newUpdates': False,
        'enableRateLimit': True,
        'apiKey': min_iterations,
        'secret': max_iterations})
    exchange.headers = {'FTX-SUBACCOUNT': 'debug'}
    exchange.authenticate()

    try:
        await asyncio.gather(*[
            loop_my_trades(exchange),
            loop_order_books(exchange),
            loop_trades(exchange)])
    except Exception as e:
        logging.exception(e)
    finally:
        await exchange.cancel_all_orders()
        pd.DataFrame(exchange._localLog._list).to_excel('logs.xlsx')

        await exchange.close()
        logging.info('done')

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(*sys.argv,loop=loop))