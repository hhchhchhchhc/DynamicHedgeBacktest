import time
import sys
import logging
import pandas as pd

import ftx_utilities
import ccxtpro
import asyncio

def main(*argv,**kwargs):
    async_exchanges = [getattr(ccxtpro.ccxt, id)() for id in ccxtpro.ccxt.exchanges]

    method_list = ['fetchTickers','fetchBorrowRateHistory','fetchBorrowRate']
    haves={}
    for method in method_list:
        haves[method] = [exchange for exchange in async_exchanges if method in exchange.describe()['has'] and exchange.describe()['has'][method]]

    relevant_exchanges = set(exchange for exchange in haves['fetchTickers'] if not exchange.id in ['vcc'])#[exchange for exchange in async_exchanges if exchange.id in ['okex5','huobipro','ftx']]

    async def get_all(relevant_exchanges):
        symbols_all_exchanges = await safe_gather([exchange.fetch_markets()
                                                       for exchange in relevant_exchanges])
        async def safe_fetch_ticker(exchange,symbol):
            try:
                return await exchange.fetch_ticker(symbol)
            except Exception as e:
                return {'symbol':symbol,'close':None}

        tickers = {}
        for exchange, symbols in zip(relevant_exchanges, symbols_all_exchanges):
            symbol_df = pd.DataFrame(symbols)
            symbol_df=symbol_df[symbol_df['active']==True]
            symbol_df.loc[symbol_df['type']==True,'index']
            ticker_list = await safe_gather([safe_fetch_ticker(exchange,symbol['symbol']) for symbol in symbols])
            tickers |= {exchange:{ticker['symbol']: {key: ticker[key] for key in ['close']} for ticker in ticker_list}}
            logging.info('done {}'.format(exchange.id))

        return tickers

    loop = asyncio.get_event_loop()
    symbol_by_exchange = loop.run_until_complete(get_all(relevant_exchanges))

    pairs_by_exchange = {exchange:
                                  {currency:
                                       {quote:
                                            [symbol for symbol in symbols if symbol['base']==currency and symbol['quote']==quote]
                                        for quote in set(symbol['quote'] for symbol in symbols if symbol['base']==currency)}
                                   for currency in set(symbol['base'] for symbol in symbols)
                                   if len([symbol for symbol in symbols if symbol['base']==currency])>0 and
                                   len([symbol for symbol in symbols if symbol['base']==currency and symbol['type']!='spot'])>0}
                              for exchange, symbols in symbol_by_exchange.items()}
    arbitrage_by_exchange = {exchange:
                                  {currency:
                                       {quote:
                                            [(exchange.markets[symbol]['info']['price']/exchange.markets[symbol]['index'] -1)*365.25 for symbol in symbols if symbol['type']!='spot']
                                        for quote,symbols in currency_data.items() if len(symbols)>1}
                                   for currency,currency_data in exchange_data.items()}
                              for exchange,exchange_data in pairs_by_exchange.items()}

    # fetch_borrow_rates
    #accessible_exchanges = [ftx_utilities.open_exchange(exchange,'') for exchange in ['okex5','huobi']]
    #await safe_gather([exchange.fetch_borrow_rates() for exchange in accessible_exchanges])
    a=0
    # watch order book
    #pro_exchanges = [getattr(ccxtpro, id)() for id in ccxtpro.exchanges]


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(symbols=['DAWN/USD','DAWN/USD:USD'],size=[10,-10],loop=loop))