import time
import ccxtpro
import asyncio

import pandas as pd

from ftx_rest_spread import *
from ftx_portfolio import live_risk,fetch_current_portoflio

previous_delta = pd.DataFrame()
current_delta = pd.DataFrame()

import logging
logging.basicConfig(level=logging.INFO)

def loop_and_callback(func):
    @functools.wraps(func)
    async def wrapper_loop(*args, **kwargs):
        while True:
            try:
                value = await func(*args, **kwargs)
            except myFtx.DoneDeal as a:
                logging.info(a)
                raise myFtx.DoneDeal()
                break
            except myFtx.LimitBreached as a:
                logging.info(a)
                raise myFtx.LimitBreached()
                break
            except Exception as e:
                logging.exception(e)

    return wrapper_loop

class myFtx(ccxtpro.ftx):
    class myLogging(logging.Logger):
        def __init__(self):
            super().__init__(name='ftx')
            logging.basicConfig(level=logging.INFO)
            self._list = []

    class DoneDeal(Exception):
        def __init__(self):
            super().__init__()

    class LimitBreached(Exception):
        def __init__(self,limit=100,check_frequency=60):
            super().__init__()
            self.limit = limit
            self.check_frequency = check_frequency
            
    def __init__(self, config={}):
        super().__init__(config=config)
        self._localLog = myFtx.myLogging()
        self._pv = None #temporary
        self._absolute_risk = None # temporary

    def logList(self,item: dict):
        self._localLog._list += [item]
        self._localLog.info(item)

    # size in usd
    def mkt_at_depth_orderbook(self,symbol,size):
        #make new_symbol resistant
        depth = 0
        previous_side = self.orderbooks[symbol]['bids' if size >= 0 else 'asks'][0][0]
        for pair in self.orderbooks[symbol]['bids' if size>=0 else 'asks']:
            depth += pair[0] * pair[1]
            if depth > size:
                break
            previous_side = pair[0]
        depth=0
        previous_opposite = self.orderbooks[symbol]['bids' if size < 0 else 'asks'][0][0]
        for pair in self.orderbooks[symbol]['bids' if size<0 else 'asks']:
            depth += pair[0] * pair[1]
            if depth > size:
                break
            previous_opposite = pair[0]
        return {'side':previous_side,'opposite':previous_opposite}

    #size in usd
    async def aggressive_order(self,symbol,size,depth,edit_scaler,stop_scaler):
        orders = await self.fetch_open_orders(symbol=symbol)
        if len(orders)>3:
            for item in orders[:-1]:
                await self.cancel_order(item['id'])
            warnings.warn('!!!!!!!!!! duplicates orders removed !!!!!!!!!!')
    
        deep_book=self.mkt_at_depth_orderbook(symbol,depth)
        price = deep_book['opposite']
        opposite_side = deep_book['opposite']
        priceIncrement = float(self.markets[symbol]['info']['priceIncrement'])
        sizeIncrement = float(self.markets[symbol]['info']['sizeIncrement'])
        if np.abs(size) < sizeIncrement * opposite_side * 1.1:
            return orders

        # triggers are in units of bis/ask at size
        stop_trigger = stop_scaler * np.abs(opposite_side/price-1)
        edit_trigger = edit_scaler * np.abs(opposite_side/price-1)
        edit_price = opposite_side - priceIncrement * (1 if size>0 else -1)
    
        if not orders:
            order = await self.create_limit_order(symbol, 'buy' if size>0 else 'sell', np.abs(size) / edit_price, price=edit_price, params={'postOnly': True})

        for item in orders:
            # panic stop. we could rather place a trailing stop: more robust to latency, less generic.
            if (1 if item['side']=='buy' else -1)*(opposite_side/item['price']-1)>stop_trigger:
                order = await self.edit_order(item['id'], symbol, 'market', 'buy' if size>0 else 'sell', np.abs(size) / opposite_side)
            # chase
            if (1 if item['side']=='buy' else -1)*(price/item['price']-1)>edit_trigger:
                order = await self.edit_order(item['id'], symbol, 'limit', 'buy' if size>0 else 'sell', np.abs(size) / edit_price,price=edit_price,params={'postOnly': True})
            #print(str(price/item['price']-1) + ' ' + str(opposite_side / item['price'] - 1))
    
        return orders

    # size in usd
    async def passive_order(self, symbol, size, depth, edit_scaler, stop_scaler):
        orders = await self.fetch_open_orders(symbol=symbol)
        if len(orders) > 3:
            for item in orders[:-1]:
                await self.cancel_order(item['id'])
            warnings.warn('!!!!!!!!!! duplicates orders removed !!!!!!!!!!')

        deep_book = self.mkt_at_depth_orderbook(symbol, depth)
        price = deep_book['side']
        opposite_side = deep_book['opposite']
        priceIncrement = float(self.markets[symbol]['info']['priceIncrement'])
        sizeIncrement = float(self.markets[symbol]['info']['sizeIncrement'])
        if np.abs(size) < sizeIncrement * price * 1.1:
            return orders

        edit_trigger = edit_scaler * np.abs(opposite_side / price - 1)
        edit_price = price + priceIncrement * (1 if size > 0 else -1)

        if not orders:
            order = await self.create_limit_order(symbol, 'buy' if size > 0 else 'sell', np.abs(size) / edit_price,
                                                  price=edit_price, params={'postOnly': True})

        for item in orders:
            # chase
            if (1 if item['side'] == 'buy' else -1) * (price / item['price'] - 1) > edit_trigger:
                order = await self.edit_order(item['id'], symbol, 'limit', 'buy' if size > 0 else 'sell',
                                              np.abs(size) / edit_price, price=edit_price, params={'postOnly': True})
            # print(str(price/item['price']-1) + ' ' + str(opposite_side / item['price'] - 1))

        return orders

    @loop_and_callback
    async def execute_on_update(self,symbol,target):
        top_of_book = await self.watch_ticker(symbol)
        global current_delta
        size = target['size']-current_delta.loc[coin,'spotDelta' if type == 'spot' else 'futureDelta']
        netDelta = current_delta.loc[coin,'netDelta']
        target['spread']
        target['spread_']

        # if increases risk, go passive
        if np.abs(netDelta+size)>np.abs(netDelta):
            depth = 10*size
            edit_scaler=2
            stop_scaler=10
            order = await self.passive_order(symbol,size,depth,edit_scaler,stop_scaler)
        else:
            depth = 10 * size
            edit_scaler = 2
            stop_scaler = 10
            order = await self.aggressive_order(symbol, size, depth, edit_scaler, stop_scaler)

    @loop_and_callback
    async def monitor_fills(self):
        orders = await self.watch_my_trades()
        global current_delta,previous_delta
        current_delta= await fetch_current_portoflio(self)# TODO: don;t use live_risk, use states
        delta_change=pd.DataFrame(current_delta)-pd.DataFrame(previous_delta)
        logging.info(delta_change)
        previous_delta= current_delta
    
        return orders

    ## redundant minutely risk check
    @loop_and_callback
    async def fetch_risk(self):
        current_delta = await fetch_current_portoflio(self)
        self._absolute_risk=sum(abs(delta['netDelta']) for delta in current_delta.values())
        if self._absolute_risk>myFtx.LimitBreached.limit:
            raise myFtx.LimitBreached()
        await asyncio.sleep(myFtx.LimitBreached.check_frequency)
        return None

    async def watch_order_book(self,symbol):
        super().watch_order_book(symbol)

async def executer_ws(exchange, targets):
    while True:
        try:
            await asyncio.gather(*([exchange.monitor_fills(),exchange.fetch_risk()]+
                                   [exchange.execute_on_update(symbol, symbol_data['target'])
                                    for coin,target in targets.items() for symbol,symbol_data in target.items() if symbol in exchange.markets]+
                                   [exchange.watch_order_book(symbol)
                                    for coin, target in targets.items() for symbol, symbol_data in target.items() if symbol in exchange.markets]))
        except myFtx.LimitBreached as e:
            logging.exception(e)
            break
        except Exception as e:
            logging.exception(e)

    return log

async def ftx_ws_spread_main_wrapper(*argv,**kwargs):
    exchange = myFtx({
        'asyncioLoop': kwargs['loop'] if 'loop' in kwargs else None,
        'newUpdates': True,
        'enableRateLimit': True,
        'apiKey': min_iterations,
        'secret': max_iterations}) if argv[1]=='ftx' else None
    exchange.verbose = False
    exchange.headers =  {'FTX-SUBACCOUNT': argv[2]}
    exchange.authenticate()

    #weigths and delta
    target_sub_portfolios = await diff_portoflio(exchange, argv[3])
    global current_delta,previous_delta
    balances = (await exchange.fetch_balance(params={}))['info']['result']
    positions = pd.DataFrame([r['info'] for r in await exchange.fetch_positions(params={})],
                             dtype=float)# 'showAvgPrice':True})
    #prices = (await exchange.fetch_tickers(params={}))[symbol]['info']['result']
    current_delta = await live_risk(exchange,exchange.markets_by_id)
    zero_delta = pd.DataFrame(index=list(target_sub_portfolios.loc[target_sub_portfolios['underlying'].isin(current_delta.index)==False,'underlying'].unique()))
    current_delta = current_delta.append(zero_delta).fillna(0)
    previous_delta = current_delta

    target_sub_portfolios = await entry_level_increment(exchange, target_sub_portfolios)

    #coin='OMG'
    #target_sub_portfolios = target_sub_portfolios[target_sub_portfolios['name'].isin([coin+'/USD',coin+'-PERP'])]

    start_time = datetime.now().timestamp()
    log = ExecutionLog('dummy', [])
    try:
        if len(target_sub_portfolios)==0: warnings.warn('nothing to execute')
        log = await executer_ws(exchange, target_sub_portfolios)
        # asyncio.run(clean_dust(exchange))
        end_time = datetime.now().timestamp()
        with pd.ExcelWriter('Runtime/execution_diagnosis.xlsx', engine='xlsxwriter') as writer:
            pd.DataFrame(await exchange.fetch_orders(params={'start_time': start_time, 'end_time': end_time})).to_excel(
                writer, sheet_name='fills')
            pd.DataFrame(audit).to_excel(writer, sheet_name='audit')
        print(log.bpCost())

    except Exception as e:
        logging.exception(e)
    finally:
        # pickleit(log, "Runtime/ExecutionLog.pickle")
        await exchange.cancel_all_orders()
        stats = await fetch_latencyStats(exchange, days=1, subaccount_nickname='SysPerp')
        print(f'latencystats:{stats}')
        await log.populateFill(exchange)
        await exchange.close()

    return log.to_df()

# target_sub_portfolios = {coin:[{symbol:[{'target_size':target_size,'info1':info1...}]}]
# add info to target_sub_portfolios dictionary. Here only weeds out slow underlyings (should be in strategy)
async def entry_level_increment(exchange, weights):
    frequency=timedelta(minutes=1)
    end=datetime.now()
    start=end - timedelta(hours=1)

    trades_history_list = await asyncio.gather(*[fetch_trades_history(
        symbol, exchange, start, end, frequency=frequency)
        for symbol in weights['name']])

    weights['name']=weights['name'].apply(lambda s: s.split('/USD')[0])
    weights.set_index('name',inplace=True)
    coin_list = weights['underlying'].unique()

    #{coin:{symbol1:{data1,data2...},sumbol2:...}}
    data_dict = {coin: {df.columns[0].split('/')[0]:
                                {'volume':df.filter(like='/trades/volume').mean().values[0]/frequency.total_seconds(),
                                 'approx_price':weights.loc[df.columns[0].split('/')[0],'approx_price'],
                                 'diff':weights.loc[df.columns[0].split('/')[0],'diff'],
                                 'target':weights.loc[df.columns[0].split('/')[0],'target'],
                                 'series':df[df.columns[0].split('/')[0]+'/trades/vwap']}
                           for df in trades_history_list if coin in df.columns[0]}
                   for coin in coin_list}

    # exclude coins with slow symbols
    volume_list = {coin: coin_data for coin,coin_data in data_dict.items()
    if min([data['volume']*time_budget / data['approx_price'] / max(1, np.abs(data['diff']))
         for name, data in coin_data.items()])>1}

    # get times series of target baskets
    weighted_vwap_list = {coin: coin_data|{'series':sum([data['series']*data['diff']
                           for name,data in coin_data.items()])}
                   for coin,coin_data in volume_list.items()}

    # quantile of the increments
    quantiles = {coin: {'entry_level_increment':coin_data['series'].dropna().diff().quantile(1-entry_tolerance)}|
                       {exchange.market(symbol)['symbol'] if symbol in exchange.markets_by_id else symbol+'/USD' : {
                           fields:fields_data for fields,fields_data in symbol_data.items() if fields!='series'
                       } for symbol,symbol_data in coin_data.items() if symbol!='series'}
                   for coin,coin_data in weighted_vwap_list.items()}

    print(' were too slow. Only doing:')
    print(weights)
    return quantiles

def ftx_ws_spread_main(*argv):
    argv=list(argv)
    if len(argv) == 0:
        argv.extend(['execute'])
    if len(argv) < 4:
        argv.extend(['ftx', 'debug', 'Runtime/ApprovedRuns/current_target_sub_portfolios.xlsx'])
    print(f'running {argv}')
    loop = asyncio.new_event_loop()
    if argv[0] == 'execute':
        return loop.run_until_complete(ftx_ws_spread_main_wrapper(*argv,loop=loop))
    else:
        print(f'commands: execute')

if __name__ == "__main__":
    ftx_ws_spread_main(*sys.argv[1:])