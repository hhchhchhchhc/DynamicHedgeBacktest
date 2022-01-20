import time
import ccxtpro
import asyncio
from ftx_utilities import open_exchange

class myFtx(ccxtpro.ftx):
    def __init__(self,symbol: str,depth: float,config={}):
        super().__init__(config=config)
        self._symbol = symbol
        self._depth = depth

    def handle_my_trade(self, client, message):
        message = super().handle_my_trade(client, message)
        return None



    def handle_order_book_update(self, client, message):
        super().handle_order_book_update(client, message)
        if message['market']==self._symbol:
            print('')

async def place_limit_at_depth(self,symbol):
    depth = 0
    for pair in self.orderbooks[symbol]['bids']:
        depth += pair[0] * pair[1]
        if depth > 10000:
            break
    if self.orders and (symbol in self.orders.keys()):
        order = await self.edit_order(self.orders[symbol][0]['id'], symbol, 'limit', 'buy', 10 / pair[0], pair[0],
                        params={'postOnly': True})
    else:
        order = await self.create_limit_order(symbol, 'buy', 10 / pair[0], pair[0], params={'postOnly': True})
    return order

async def order_book_loop(exchange,symbol):
    while True:
        await asyncio.sleep(1)
#        order_book = await exchange.watch_order_book(symbol)
#        order = await place_limit_at_depth(exchange,symbol)#.create_limit_order(symbol, 'buy', 1, 0.00001, params={'postOnly': True})

async def order_loop(exchange,symbol):
    while True:
        try:
            await asyncio.sleep(1)
            orders = await exchange.watch_orders(symbol)
            print('<---------')
            #print(orders)
        except Exception as e:
            print(e)

async def main(loop):
    dummy=open_exchange('ftx','debug')
    await dummy.close()

    symbol = 'LEO/USD'
    exchange=ccxtpro.ftx(config={
        'asyncio_loop': loop,
        'newUpdates':True,
        'enableRateLimit': True,
        'apiKey': dummy.apiKey,
        'secret': dummy.secret})
    exchange.verbose = True
    exchange.headers = dummy.headers

    exchange.authenticate()
    await exchange.load_markets()

    try:
        await asyncio.gather(*[order_book_loop(exchange,symbol),
                                   order_loop(exchange,symbol)])
    except Exception as e:
        print(type(e).__name__, str(e))
        await exchange.cancel_all_orders()
        await exchange.close()

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main(loop))