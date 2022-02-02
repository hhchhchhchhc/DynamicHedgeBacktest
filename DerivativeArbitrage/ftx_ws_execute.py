import json
import time
import ccxtpro
import asyncio
import functools
import pandas as pd
from ftx_utilities import *
from ftx_portfolio import live_risk,diff_portoflio
from ftx_history import fetch_trades_history
from ftx_ftx import mkt_at_size,fetch_latencyStats
import logging
logging.basicConfig(level=logging.INFO)

entry_tolerance = .6
edit_trigger_tolerance = np.sqrt(1) # chase on 30s stdev
stop_tolerance = np.sqrt(5*60) # stop on 5min stdev
time_budget = 5*60
delta_limit = 1. # delta limit / pv
slice_factor = 0.1 # cut in 10

def loop_and_callback(func):
    @functools.wraps(func)
    async def wrapper_loop(*args, **kwargs):
        while True:
            try:
                value = await func(*args, **kwargs)
            except myFtx.DoneDeal as e:
                logging.info(f'{args[1]} done')
                break
            except myFtx.LimitBreached as e:
                logging.info(e)
                raise e
                break
            except ccxt.base.errors.RequestTimeout as e:
                logging.info('reconnect..')
                continue
            except Exception as e:
                logging.exception(e,exc_info=True)

    return wrapper_loop

class ExecutionLog(dict):
    _logCounter = 0
    def __init__(self, order_type: str,
                 legs: list): # len is 1 or 2, needs 'symbol' and 'size'
        super().__init__()
        self._id = ExecutionLog._logCounter # orverriden to orderID for leafs
        ExecutionLog._logCounter +=1
        self.order_type = order_type #{type,delta,netDelta}
        self._legs = legs # len is 1 or 2, 'symbol' and 'size', later benchmarks and fills(average,filled,status) and risk
        self._receivedAt = None # temporary
        self.children = []# list of ExecutionLog

    # fetches exchange to populate benchmarks
    async def initializeBenchmark(self,exchange: ccxt.Exchange) -> None:
        self._receivedAt=exchange.milliseconds()/1000
        fetched = await asyncio.gather(*[mkt_at_size(exchange,leg['symbol'],'asks' if leg['size']>0 else 'bids',np.abs(leg['size'])) for leg in self._legs])
        time_spent = exchange.milliseconds() - self._receivedAt*1000
        if time_spent> 1000: # 15ms is approx throttle, so 200 is a lot
            logging.info(f'mkt_at_size was {time_spent} ms')

        self._legs = [leg | {'benchmark':
                                 {'initial_mkt': f['side'],
                                  'initial_mid':f['mid']}} |
                      {'delta':exchange.risk.delta[exchange.markets[leg['symbol']]['base']][leg['symbol']]['delta']
                          if exchange.markets[leg['symbol']]['base'] in exchange.risk.delta.keys() and
                          leg['symbol'] in exchange.risk.delta[exchange.markets[leg['symbol']]['base']].keys() else 0.0,
                     'netDelta':exchange.risk.delta[exchange.markets[leg['symbol']]['base']]['netDelta']
                            if exchange.markets[leg['symbol']]['base'] in exchange.risk.delta.keys() else 0.0}
                      for f,leg in zip(fetched,self._legs)]

    # only native orders receive values. Otherwise recurse on children.
    # assigns and returns legs
    # fillSize is just an assert
    async def populateFill(self,exchange: ccxt.Exchange) -> list:
        if self.children:
            children_list = await asyncio.gather(*[child.populateFill(exchange) for child in self.children])
            children_fills =[[child_leg for child in children_list for child_leg in child if child_leg['symbol'] == leg['symbol']]
                             for leg in self._legs]

            # only populate parent if all children orders are closed.
            if all(l['status']=='closed' for fills in children_fills for l in fills):
                self._legs = [leg | {'average': sum(l['filled']*l['average'] for l in fills if l['filled']!=0.0)/
                                                sum(l['filled'] for l in fills),
                                     'filled':sum(l['filled'] for l in fills),
                                     'status': 'closed' if
                                     all(l['status']=='closed' for l in fills)
                                                        else 'unclosed'}
                                for leg,fills in zip(self._legs,children_fills)]
        else:
            try:
                fills = await exchange.fetch_order(self._id,self._legs[0]['symbol'],params={'start_time':int(self._receivedAt)})
            # if a leg is too small, don't complain. Note: Leaf orders have only one leg.
            except Exception as e:
                if self._legs[0]['size'] < float(exchange.market(self._legs[0]['symbol'])['info']['sizeIncrement']):
                    self._legs[0] = self._legs[0] | {'average': -1., 'filled': 0., 'status': 'closed'}
                    return self._legs

            if fills['status']=='open':
                logging.debug('leaf still open, populateFill too early?')

            if fills['side']=='sell': fills['filled']*=-1
            self._legs = [self._legs[0] | fills]
            #{'average':fills['average'],
            #'filled':fills['filled']*(1 if fills['side']=='buy' else -1),
            #'status':fills['status']}]
        return self._legs

    # usd cost / avg|size|, so for a spread it's the premium vs benchmark
    def bpCost(self)->str:
        vs_initial_mkt = sum(leg['filled']*(leg['average']-leg['benchmark']['initial_mkt'])*10000 for leg in self._legs)
        vs_initial_mid = sum(leg['filled'] * (leg['average']-leg['benchmark']['initial_mid'])*10000 for leg in self._legs)
        size_done = sum(np.abs(leg['filled'])*leg['average'] for leg in self._legs)/len(self._legs)
        return {'vs_initial_mkt':vs_initial_mkt/size_done,'vs_initial_mid':vs_initial_mid/size_done}

    def to_json(self,file_name):
        with open(file_name, "w+") as file:
            if os.path.getsize('exec.json')>0: file.write(',')
            else: file.write ('[')
            json.dump(self.flatten(),file)

    def flatten(self):
        return flatten({
            'id':self._id,
            'order_type':self.order_type,
            'legs':self._legs,
            'receivedAt':self._receivedAt,
            'children':self.children})

class myFtx(ccxtpro.ftx):
    class DoneDeal(Exception):
        def __init__(self):
            super().__init__()

    class LimitBreached(Exception):
        def __init__(self,limit=None,check_frequency=60):
            super().__init__()
            self.limit = limit
            self.check_frequency = check_frequency
            
    class Risk:
        def __init__(self):
            self.delta={}
            self.pv=0
            self.timestamp=(0,0)
        def flatten(self):
            return flatten(self.delta|{'pv':self.pv,'timestamp':self.timestamp})

    def __init__(self, config={}):
        super().__init__(config=config)
        self._localLog = ExecutionLog('dummy',[])
        self.limit = myFtx.LimitBreached()
        self.risk = myFtx.Risk()

    def __del__(self):
        asyncio.run(self.cancel_all_orders())
        asyncio.run(self._localLog.populateFill(self))
        self._localLog.to_json("exec.json")

    # updates risk. Should be handle_my_trade but needs a coroutine :(
    async def fetch_risk(self, params={}):
        # {coin:{'netDelta':netDelta,symbol1:{'volume','spot_price','diff','target'}]}]
        # [coin:{'netDelta':netDelta,'legs':{symbol:delta}]]
        start = self.milliseconds() / 1000
        positions = await self.fetch_positions()
        mid = self.milliseconds() / 1000
        balances = await self.fetch_balance()
        end = self.milliseconds() / 1000

        time_spent = end - mid
        if time_spent > 1000:  # 15ms is approx throttle, so 200 is a lot
            logging.info(f'balances vs positions: {time_spent} ms')

        # delta is noisy for perps, so override to delta 1.
        positions = {self.market(position['symbol'])['base']:
                         {position['symbol']:
                              {'delta':
                                   (float(position['notional'])
                                    if self.market(position['symbol'])['type'] == 'future'
                                    else float(position['info']['size']) * float(
                                       self.market(self.market(position['symbol'])['base'] + '/USD')[
                                           'info']['price']))
                                   * (1 if position['side'] == 'long' else -1)
                               }
                          }
                     for position in positions}


        self.risk.pv = sum(balance['total'] * (float(self.market(coin + '/USD')['info']['price']) if coin !='USD' else 1) for coin,balance in balances.items() if coin in self.currencies)

        balances = {key:
                        {key + '/USD':
                             {'delta':
                                  float(balance['total']) * float(self.market(key + '/USD')['info']['price'])
                              }
                         }
                    for key, balance in balances.items() if key in self.currencies.keys() and key != 'USD'}

        self.risk.delta = {key:
                         (positions[key] if key in positions.keys() else {})
                         | (balances[key] if key in balances.keys() else {})
                         | {'netDelta': sum(positions[key][symbol]['delta'] for symbol in
                                            (positions[key].keys() if key in positions.keys() else {}))
                                        + (balances[key][key + '/USD']['delta'] if key in balances.keys() else 0)}
                     for key in positions.keys() | balances.keys()}

    def logList(self,item: dict):
        self._localLog._list += [item]
        self._localLog.info(item)

    # size in usd
    def mkt_at_depth_orderbook(self,symbol,size):
        #make symbol resistant
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

    #size in coin, already filtered
    #scalers are in number of bid asks
    async def update_orders(self,symbol,size,depth,edit_trigger_depth,edit_price_depth,stop_depth=None):
        orders = await self.fetch_open_orders(symbol=symbol)
        if len(orders)>2:
            for item in orders[:-1]:
                await self.cancel_order(item['id'])
            logging.warning('!!!!!!!!!! duplicates orders removed !!!!!!!!!!')
    
        price,opposite_side = float(self.markets[symbol]['info']['ask' if size<0 else 'bid']),float(self.market(symbol)['info']['ask' if size>0 else 'bid'])
        mid = 0.5*(price+opposite_side)
        priceIncrement = float(self.markets[symbol]['info']['priceIncrement'])

        # triggers are in units of bid/ask at size
        if stop_depth:
            stop_trigger = max(1,int(stop_depth / priceIncrement))*priceIncrement
        edit_trigger = max(1,int(edit_trigger_depth / priceIncrement))*priceIncrement
        edit_price = opposite_side - (1 if size>0 else -1)*max(1,int(edit_price_depth/priceIncrement))*priceIncrement

        order = None
        if not orders:
            order = await self.create_limit_order(symbol, 'buy' if size>0 else 'sell', np.abs(size), price=edit_price, params={'postOnly': True})
        else:
            # panic stop. we could rather place a trailing stop: more robust to latency, less generic.
            if stop_depth and (1 if orders[0]['side']=='buy' else -1)*(opposite_side-orders[0]['price'])>stop_trigger:
                order = await self.edit_order(orders[0]['id'], symbol, 'market', 'buy' if size>0 else 'sell',None)
            # chase
            if (1 if orders[0]['side']=='buy' else -1)*(price-orders[0]['price'])>edit_trigger and np.abs(edit_price-orders[0]['price'])>=priceIncrement:
                order = await self.edit_order(orders[0]['id'], symbol, 'limit', 'buy' if size>0 else 'sell', None ,price=edit_price,params={'postOnly': True})
            #print(str(price/orders[0]['price']-1) + ' ' + str(opposite_side / orders[0]['price'] - 1))

        return order

    # a mix of watch_ticker and handle_ticker. Can't inherit since handle needs coroutine.
    @loop_and_callback
    async def execute_on_update(self,symbol,coin_request):
        top_of_book = await self.watch_ticker(symbol)
        if self.risk.delta=={}: await asyncio.sleep(0.1)  # not ready, need to refresh risk first
        coin = self.markets[symbol]['base']
        symbol_request = coin_request[symbol]

        # size to do: aim at target and don't overshoot
        size = symbol_request['target']
        netDelta = 0
        if coin in self.risk.delta:
            netDelta = self.risk.delta[coin]['netDelta']
            if symbol in self.risk.delta[coin]:
                size = symbol_request['target'] - self.risk.delta[coin][symbol]['delta']

        size = np.sign(size)*min([np.abs(size),symbol_request['exec_parameters']['slice_size']])
        sizeIncrement = float(self.markets[symbol]['info']['minProvideSize'])
        if np.abs(size) < sizeIncrement:
            size =0
            raise myFtx.DoneDeal()
        else:
            size = np.sign(size)*int(np.abs(size)/sizeIncrement)*sizeIncrement
        assert(np.abs(size)>=sizeIncrement)
        # if increases risk, go passive
        if np.abs(netDelta+size)>np.abs(netDelta):
            # set limit at target quantile
            current_basket_price = sum(float(self.markets[symbol]['info']['price'])*symbol_data['diff'] for symbol,symbol_data in coin_request.items() if symbol in self.markets)
            edit_price_depth = max([0,(current_basket_price-coin_request['entry_level'])/symbol_request['diff']])
            edit_trigger_depth=symbol_request['exec_parameters']['edit_trigger_depth']

            log = ExecutionLog('passive', [{'symbol': symbol, 'size': size}])
            await log.initializeBenchmark(self)
            order = await self.update_orders(symbol,size,0,edit_trigger_depth,edit_price_depth,None)
            if order != None: log._id = order['id']
        else:
            edit_trigger_depth=symbol_request['exec_parameters']['edit_trigger_depth']
            edit_price_depth=symbol_request['exec_parameters']['edit_price_depth']
            stop_depth=symbol_request['exec_parameters']['edit_price_depth']

            log = ExecutionLog('aggressive', [{'symbol': symbol, 'size': size}])
            await log.initializeBenchmark(self)
            order = await self.update_orders(symbol,size,0,edit_trigger_depth,edit_price_depth,stop_depth)
            if order != None: log._id = order['id']

        if order != None:
            self._localLog.children+=[log]
            self._localLog.to_json("exec.json")

    @loop_and_callback
    async def monitor_fills(self):
        orders = await self.watch_my_trades()
        previous_risk = self.risk
        await self.fetch_risk()
        await self._localLog.populateFill(self)
        self._localLog.to_json("exec.json")
        #risk_change=self.risk - previous_risk
        #logging.info(risk_change)

    ## redundant minutely risk check
    @loop_and_callback
    async def monitor_risk(self):
        await self.fetch_risk()

        self.limit.limit = self.risk.pv * delta_limit
        absolute_risk = sum(abs(data['netDelta']) for data in self.risk.delta.values())
        if absolute_risk > self.limit.limit:
            logging.warning(f'limit {self.limit}')

        await asyncio.sleep(self.limit.check_frequency)

    async def watch_order_book(self,symbol):
        super().watch_order_book(symbol)

# target_sub_portfolios = {coin:{'entry_level_increment':entry_level_increment,symbol1:{'volume','spot_price','diff','target'}]}]
# add info to target_sub_portfolios dictionary. Here only weeds out slow underlyings (should be in strategy)
async def execution_request(exchange, weights):
    frequency = timedelta(minutes=1)
    end = datetime.now()
    start = end - timedelta(hours=1)

    trades_history_list = await asyncio.gather(*[fetch_trades_history(
        exchange.market(symbol)['id'], exchange, start, end, frequency=frequency)
        for symbol in weights['name']])

    weights['name'] = weights['name'].apply(lambda s: exchange.market(s)['symbol'])
    weights.set_index('name', inplace=True)
    coin_list = weights['underlying'].unique()

    # {coin:{symbol1:{data1,data2...},sumbol2:...}}
    data_dict = {coin:
                     {df['symbol']:
                          {'volume': df['vwap'].filter(like='/trades/volume').mean().values[0] / frequency.total_seconds(),# in coin per second
                           'spot_price': weights.loc[df['symbol'], 'spot_price'],
                           'diff': weights.loc[df['symbol'], 'diff'],
                           'target': weights.loc[df['symbol'], 'target'],
                           'exec_parameters': {  # depend on risk tolerance other execution parameters...
                               'slice_size': max([float(exchange.market(df['symbol'])['info']['minProvideSize']),
                                                  0.1 * np.abs(weights.loc[df['symbol'], 'diff'])]),  # in usd
                               'edit_trigger_depth': None, # will be quantile
                               'edit_price_depth': 0, # only used for agressive
                               'stop_depth': None},# will be quantile
                           'series': df['vwap'].filter(like='/trades/vwap')}
                      for df in trades_history_list if df['coin']==coin}
                 for coin in coin_list}

    # exclude coins with slow symbols
    volume_list = {coin:
                       coin_data for coin, coin_data in data_dict.items()
                   if all(data['volume'] * time_budget > max(1, np.abs(data['diff']))
                           for data in coin_data.values())}

    def basket_vwap_quantile(series_list,diff_list,quantile):
        series = pd.concat(series_list,axis=1).dropna(axis=0)-diff_list
        return series.sum(axis=1).diff().quantile(quantile)
    def move_quantile(series,quantile):# stdev of 1s prices * quantile
        series = series.dropna(axis=0)
        return series.std().values[0]*quantile

    # get times series of target baskets, compute quantile of increments and add to last price
    # remove series
    light_list = {coin:
                      {'entry_level':
                           sum([float(exchange.market(name)['info']['price']) * data['diff']
                                for name, data in coin_data.items()])
                           + basket_vwap_quantile([data['series'] for data in coin_data.values()],
                                                  [data['diff'] for data in coin_data.values()],
                                                  entry_tolerance)}
                      | {name:
                             {field: field_data for field, field_data in data.items() if field != 'series' and field != 'exec_parameters'}
                             |{'exec_parameters': {  # depend on risk tolerance other execution parameters...
                                 'slice_size': max([float(exchange.market(name)['info']['minProvideSize']),
                                                    slice_factor * np.abs(data['diff'])]),  # in usd
                                 'edit_trigger_depth': move_quantile(data['series'],edit_trigger_tolerance),
                                 'edit_price_depth': 0, # only used for aggressive
                                 'stop_depth': move_quantile(data['series'],stop_tolerance)}}# will be quantile
                         for name, data in coin_data.items()}
                  for coin, coin_data in volume_list.items()}

    return {'LTC':light_list['LTC']}

async def executer_ws(exchange, targets):
    try:
        exchange._localLog = ExecutionLog('basket',[{'symbol':symbol,'size':data['diff']}
                                                    for coin in targets.values() for symbol,data in coin.items() if symbol in exchange.markets])
        await exchange.fetch_risk()
        await exchange._localLog.initializeBenchmark(exchange)
        await asyncio.gather(*([exchange.monitor_fills(),exchange.monitor_risk()]+
                               [exchange.execute_on_update(symbol, target)
                                for coin,target in targets.items()
                                for symbol,symbol_data in target.items() if symbol in exchange.markets.keys()]
                               ))
    except myFtx.LimitBreached as e:
        logging.exception(e,exc_info=True)
        #break
    except ccxt.base.errors.RequestTimeout as e:
        logging.info('reconnect..')
        #continue
    except Exception as e:
        logging.exception(e,exc_info=True)

    return exchange._localLog

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
    await exchange.load_markets()

    #weigths and delta
    target_sub_portfolios = (await diff_portoflio(exchange))
    target_sub_portfolios=target_sub_portfolios[target_sub_portfolios['diff'].apply(np.abs)>=target_sub_portfolios['name'].apply(lambda s: float(exchange.market(s)['info']['minProvideSize']))]
    request = await execution_request(exchange, target_sub_portfolios)
    with open("request.json", "w") as file:
            json.dump(flatten(request),file)

    if os.path.isfile("exec.json"): os.remove("exec.json")
    #coin='OMG'
    #target_sub_portfolios = target_sub_portfolios[target_sub_portfolios['name'].isin([coin+'/USD',coin+'-PERP'])]
    try:
        await executer_ws(exchange, request)
    except KeyboardInterrupt:# TODO: this is not caught :(
        pass
    except Exception as e:
        logging.exception(e,exc_info=True)
    finally:
        stats = await fetch_latencyStats(exchange, days=1, subaccount_nickname='SysPerp')
        print(f'latencystats:{stats}')

        with open("exec.json", "a") as file: file.write(']')
        result = exchange._localLog
        await exchange.close()

    return result

def ftx_ws_spread_main(*argv):
    argv=list(argv)
    if len(argv) == 0:
        argv.extend(['execute'])
    if len(argv) < 4:
        argv.extend(['ftx', 'debug'])
    print(f'running {argv}')
    loop = asyncio.new_event_loop()
    if argv[0] == 'execute':
        return loop.run_until_complete(ftx_ws_spread_main_wrapper(*argv,loop=loop))
    else:
        print(f'commands: execute')

if __name__ == "__main__":
    ftx_ws_spread_main(*sys.argv[1:])