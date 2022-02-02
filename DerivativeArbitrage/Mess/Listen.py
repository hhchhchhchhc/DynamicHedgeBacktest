import time
import sys
import ftx_utilities
import ccxtpro
import asyncio

async def main(*argv,**kwargs):
    async_exchanges = [getattr(ccxtpro.ccxt, id)() for id in ccxtpro.ccxt.exchanges if id!='woo']

    method_list = ['fetchBorrowRateHistory','fetchBorrowRate']
    haves={}
    for method in method_list:
        haves[method] = [exchange for exchange in async_exchanges if method in exchange.describe()['has'] and exchange.describe()['has'][method]]

    relevant_exchanges = set(haves['fetchBorrowRates'])#-set(haves['fetchBorrowRateHistory'])

    symbols_all_exchanges = await asyncio.gather(*[exchange.fetch_markets() for exchange in relevant_exchanges])

    symbol_by_exchange = {exchange:symbols
                          for exchange,symbols in zip(relevant_exchanges,symbols_all_exchanges)}
    currencies_by_exchange = {exchange:
                                  {currency:
                                       [symbol for symbol in symbols if symbol['base']==currency]
                                   for currency in set(symbol['base'] for symbol in symbols)
                                   if len([symbol for symbol in symbols if symbol['base']==currency])>1 and
                                   len([symbol for symbol in symbols if symbol['base']==currency and symbol['type']!='spot'])>0}
                              for exchange, symbols in symbol_by_exchange.items()}

    # fetch_borrow_rates
    #accessible_exchanges = [ftx_utilities.open_exchange(exchange,'') for exchange in ['okex5','huobi']]
    #await asyncio.gather(*[exchange.fetch_borrow_rates() for exchange in accessible_exchanges])

    # watch order book
    #pro_exchanges = [getattr(ccxtpro, id)() for id in ccxtpro.exchanges]


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(symbols=['DAWN/USD','DAWN/USD:USD'],size=[10,-10],loop=loop))