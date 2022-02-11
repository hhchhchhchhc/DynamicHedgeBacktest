import sys
import ccxtpro
import asyncio
from ftx_utilities import secret,apiKey
import functools

import logging
logging.basicConfig(level=logging.INFO)

def loop_try(func):
    @functools.wraps(func)
    async def wrapper_loop(*args, **kwargs):
        while True:
            try:
                value = await func(*args, **kwargs)
            except Exception as e:
                if str(e) == 'done': raise e
            except Exception as e:
                logging.exception(e)

    return wrapper_loop

class myFtx(ccxtpro.ftx):
    def __init__(self,symbol: str, config={}):
        super().__init__(config=config)
        self._symbol = symbol

    @loop_try
    async def loop_order_book(self):
        res = await self.watch_order_book(self._symbol)
    
    @loop_try
    async def loop_orders(self):
        res = await self.watch_orders()
    
    @loop_try
    async def loop_my_trades(self):
        res = await self.watch_my_trades(self._symbol)
        balance = await self.fetch_balance()# better maintain it from myTrades?
        coin = self.market(self._symbol)['base']
        if abs(balance[coin]['total'])*self.trades[self._symbol][-1]['price'] > 50:
            raise Exception('done')

    @loop_try
    async def loop_trades(self):
        trades = await self.watch_trades(self._symbol)

        # places an order btw last trade and top of book
        last_trade_side = self.trades[self._symbol][-1]['side']
        last_trade_price = self.trades[self._symbol][-1]['price']
        level = 0.5 * (last_trade_price +
                       self.orderbooks[self._symbol]['bids' if last_trade_side == 'buy' else 'asks'][0][0])
        mid = 0.5*(self.orderbooks[self._symbol]['bids'][0][0]+self.orderbooks[self._symbol]['asks'][0][0])

        logging.info({  'mid_on_trade': mid,
                        'orderbook_on_trade': self.orderbooks[self._symbol]['timestamp'],
                        'trade_on_trade': self.trades[self._symbol][-1]['timestamp'],
                        'self_on_trade': self.milliseconds()})
        if self.orders and self._symbol in self.orders:
            assert(len(self.orders[self._symbol])==1)
            await self.edit_order(self.orders[self._symbol][-1], self._symbol,'limit', 'sell' if last_trade_side == 'sell' else 'buy',
                                              10. / level, price=level,
                                              params={'postOnly': True})
        else:
            await self.create_limit_order(self._symbol, 'sell' if last_trade_side == 'sell' else 'buy', 10. / level, price=level,
                                params={'postOnly': True})

        fetch = await self.fetch_ticker(self._symbol)
        mid = 0.5 * (fetch['bid'] + fetch['ask'])
        logging.info({  'mid_on_fetch': mid,
                        'orderbook_on_fetch': self.orderbooks[self._symbol]['timestamp'],
                        'orders_on_fetch': self.orders[self._symbol][-1]['timestamp'],
                        'trade_on_fetch': self.trades[self._symbol][-1]['timestamp'],
                        'self_on_fetch': self.milliseconds()})

async def main(*argv,**kwargs):
    exchange = myFtx('UNI/USD',config={
        'asyncio_loop': kwargs['loop'],
        'newUpdates': False,
        'enableRateLimit': True,
        'apiKey': apiKey,
        'secret': secret})
    exchange.headers = {'FTX-SUBACCOUNT': 'debug'}
    exchange.authenticate()

    try:
        await asyncio.gather(*[
            exchange.loop_my_trades(),
            exchange.loop_order_book(),
            exchange.loop_orders(),
            exchange.loop_trades()])
    except Exception as e:
        if str(e)=='done':
            logging.info('done')
        else:
            logging.exception(e)
            raise e
    finally:
        await exchange.cancel_all_orders()
        await exchange.close()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(*sys.argv,loop=loop))