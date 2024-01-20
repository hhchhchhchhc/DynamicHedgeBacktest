import ccxtpro
from ccxt_utilities import *
import ccxtpro

from ccxt_utilities import *


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

async def place_limit_at_depth(self,symbol,tolerance=0.001):
    depth = 0
    for pair in self.orderbooks[symbol]['bids']:
        depth += pair[0] * pair[1]
        if depth > 10000:
            break

    if self.orders and symbol in self.orders.hashmap:
        if len(self.orders.hashmap[symbol])>1:
            print('?')
        for item in self.orders.hashmap[symbol].items():
            if (item[1]['status'] == 'open')&(np.abs(item[1]['price']/pair[0]-1)>tolerance):
                order = await self.edit_order(item[0], symbol, 'limit', 'buy', 10 / pair[0],
                                          pair[0],
                                          params={'postOnly': True})
    else:
        order = await self.create_limit_order(symbol, 'buy', 10 / pair[0], pair[0], params={'postOnly': True})
    return None

async def watch_order_book(exchange,symbol):
    while True:
        await asyncio.sleep(1)
        order_book = await exchange.watch_order_book(symbol)
        order = await place_limit_at_depth(exchange,symbol)#.create_limit_order(symbol, 'buy', 1, 0.00001, params={'postOnly': True})

async def watch_orders(exchange,symbol):
    while True:
        try:
            orders = await exchange.watch_orders(symbol)
            print('<---------')
            #print(orders)
        except Exception as e:
            print(e)

async def main(loop):
    symbol = 'GRT/USD'
    exchange=ccxtpro.ftx({
        #'asyncio_loop': loop,
        'newUpdates':False,
        'enableRateLimit': True,
        'apiKey': min_iterations,
        'secret': max_iterations})
    exchange.verbose = True
    exchange.headers= {'FTX-SUBACCOUNT': 'debug'}

    exchange.authenticate()
    await exchange.load_markets()

    while True:
        try:
            await safe_gather([watch_order_book(exchange,symbol),
                                       watch_orders(exchange,symbol)])
        except Exception as e:
            print(type(e).__name__, str(e))

    await exchange.cancel_all_orders()
    await exchange.close()


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main(loop))