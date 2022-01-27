import time
import sys
import ccxtpro
import asyncio
from ftx_utilities import *

class myExchange(ccxtpro.ftx):
    def __init__(self, config={}):
        super().__init__(config=config)

    async def watch_orders(self,symbol):
        while True:
            try:
                await super().watch_orders(symbol)
            except Exception as e:
                print(e)

    async def trade_on_update(self,symbol,size):
        while True:
            try:
                top_of_book = await self.watch_ticker(symbol)
                price,opposite_side = float(self.markets[symbol]['info']['ask' if size<0 else 'bid']),float(self.market(symbol)['info']['ask' if size>0 else 'bid'])

                priceIncrement = float(self.markets[symbol]['info']['priceIncrement'])
                sizeIncrement = float(self.markets[symbol]['info']['sizeIncrement'])
                if np.abs(size) < sizeIncrement:
                    order_size =0
                    return None
                else:
                    order_size = int(np.abs(size)/opposite_side/sizeIncrement)*sizeIncrement

                # triggers are in units of bid/ask at size
                edit_trigger = priceIncrement
                edit_price = price - int(0.0001*price/priceIncrement)*priceIncrement * (1 if size>0 else -1)

                orders = await self.fetch_open_orders(symbol=symbol)
                if not orders:
                    await self.create_limit_order(symbol, 'buy' if size>0 else 'sell', order_size, price=edit_price, params={'postOnly': True})
                else:# chase
                    #if (1 if size>0 else -1)*(price-orders[0]['price'])>edit_trigger:
                    if np.abs(edit_price - orders[0]['price']) >= priceIncrement:
                        await self.edit_order(orders[0]['id'], symbol, 'limit', 'buy' if size>0 else 'sell', None ,price=edit_price,params={'postOnly': True})
                        print(f'chased to {edit_price}')

            except Exception as e:
                print(e)



async def main(*argv,**kwargs):
    exchange = myExchange(config={
        'asyncioLoop': kwargs['loop'] if 'loop' in kwargs else None,
        'newUpdates': True,
        'enableRateLimit': True,
        'apiKey': min_iterations,
        'secret': max_iterations})
    exchange.verbose = False
    exchange.headers =  {'FTX-SUBACCOUNT': 'SysPerp'}
    exchange.authenticate()

    await asyncio.gather(*[exchange.trade_on_update(symbol,size) for symbol,size in zip(kwargs['symbols'],kwargs['size'])]+
                          [exchange.watch_orders(symbol) for symbol in kwargs['symbols']])


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(symbols=['DAWN/USD','DAWN/USD:USD'],size=[10,-10],loop=loop))