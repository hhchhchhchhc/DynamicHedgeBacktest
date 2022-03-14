from time import sleep

import pandas as pd

from ftx_history import *
from ftx_utilities import *
from strategies import refresh_universe

async def ftx_history(coin_list=[]):
    exchange = await open_exchange('ftx','')
    markets = await exchange.fetch_markets()
    futures = pd.DataFrame(await fetch_futures(exchange, includeExpired=True)).set_index('name')
    if coin_list!=[]: futures=futures[futures['underlying'].isin(coin_list)]

    # filtering params
    type_allowed = ['future','perpetual']

    # qualitative filtering
    universe = await refresh_universe(exchange,UNIVERSE)
    universe = universe[~universe['underlying'].isin(EXCLUSION_LIST)]


    futures = futures[(futures['type'].isin(type_allowed))
                       & (futures['symbol'].isin(universe.index))]

    futures = futures[#(futures['expired'] == False) &
        (futures['enabled'] == True) & (futures['type'] != "move")
        & (futures.apply(lambda f: float(find_spot_ticker(markets, f, 'ask')), axis=1) > 0.0)
        & (futures['tokenizedEquity'] != True)
        & (futures['type'].isin(type_allowed)==True)]

    #### get history ( this is sloooow)
    hy_history = await build_history(futures, exchange)
    await exchange.close()

    return hy_history

i=0
while i<1:
    try:
        asyncio.run(ftx_history())
    except:
        sleep(10)
    i=i+1