import time
import ccxtpro
import asyncio
import pandas as pd

from ftx_rest_spread import *
from ftx_portfolio import live_risk
import logging
logging.basicConfig(level=logging.INFO)
from ftx_rest_spread import ExecutionLog

entry_tolerance = .6
time_budget = 5*60

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
                print('reconnect..')
                continue
            except Exception as e:
                logging.exception(e,exc_info=True)

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
            
    class Risk:
        def __init__(self):
            self.delta={}
            self.pv=0
            self.timestamp=(0,0)

    def __init__(self, config={}):
        super().__init__(config=config)
        self._localLog = ExecutionLog('dummy',[])
        self.limit = myFtx.LimitBreached()
        self.risk = myFtx.Risk()

    # updates risk. Should be handle_my_trade but needs a coroutine :(
    async def fetch_risk(self, params={}):
        # {coin:{'netDelta':netDelta,symbol1:{'volume','spot_price','diff','target'}]}]
        # [coin:{'netDelta':netDelta,'legs':{symbol:delta}]]
        positions = await self.fetch_positions()
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

        balances = await self.fetch_balance()
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

        absolute_risk=sum(abs(data['netDelta']) for data in self.risk.delta.values())
        if absolute_risk>self.limit.limit:
            logging.warn(self.limit)

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
            warnings.warn('!!!!!!!!!! duplicates orders removed !!!!!!!!!!')
    
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

        if order != None: self._localLog.children+=[log]

    @loop_and_callback
    async def monitor_fills(self):
        orders = await self.watch_my_trades()
        previous_risk = self.risk
        await self.fetch_risk()
        #risk_change=self.risk - previous_risk
        #logging.info(risk_change)

    ## redundant minutely risk check
    @loop_and_callback
    async def monitor_risk(self):
        await self.fetch_risk()
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
                               'edit_trigger_depth': 10,
                               'edit_price_depth': 1,
                               'stop_depth': 50},
                           'series': df['vwap'].filter(like='/trades/vwap')}
                      for df in trades_history_list if df['coin']==coin}
                 for coin in coin_list}

    # exclude coins with slow symbols
    volume_list = {coin:
                       coin_data for coin, coin_data in data_dict.items()
                   if all(data['volume'] * time_budget > max(1, np.abs(data['diff']))
                           for name, data in coin_data.items())}

    def basket_vwap_quantile(series_list,diff_list):
        series = pd.concat(series_list,axis=1).dropna(axis=0)-diff_list
        return series.sum(axis=1).diff().quantile(entry_tolerance)

    # get times series of target baskets, compute quantile of increments and add to last price
    # remove series
    light_list = {coin:
                      {'entry_level':
                           sum([float(exchange.market(name)['info']['price']) * data['diff']
                                for name, data in coin_data.items()])
                           + basket_vwap_quantile([data['series'] for data in coin_data.values()],[data['diff'] for data in coin_data.values()])}
                      | {name:
                             {field: field_data for field, field_data in data.items() if field != 'series'}
                         for name, data in coin_data.items()}
                  for coin, coin_data in volume_list.items()}

    print('Executing:')
    print(weights)
    return light_list

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
        print('reconnect..')
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
    target_sub_portfolios = (await diff_portoflio(exchange, argv[3]))
    global current_delta,previous_delta
    balances = (await exchange.fetch_balance(params={}))['info']['result']
    positions = pd.DataFrame([r['info'] for r in await exchange.fetch_positions(params={})],
                             dtype=float)# 'showAvgPrice':True})
    #prices = (await exchange.fetch_tickers(params={}))[symbol]['info']['result']
    #prices = (await exchange.fetch_tickers(params={}))[symbol]['info']['result']
    current_delta = await live_risk(exchange,exchange.markets_by_id)
    zero_delta = pd.DataFrame(index=list(target_sub_portfolios.loc[target_sub_portfolios['underlying'].isin(current_delta.index)==False,'underlying'].unique()))
    current_delta = current_delta.append(zero_delta).fillna(0)
    previous_delta = current_delta

    request = await execution_request(exchange, target_sub_portfolios)

    #coin='OMG'
    #target_sub_portfolios = target_sub_portfolios[target_sub_portfolios['name'].isin([coin+'/USD',coin+'-PERP'])]

    start_time = datetime.now().timestamp()
    try:
        if len(target_sub_portfolios)==0: warnings.warn('nothing to execute')
        await executer_ws(exchange, request)
        end_time = datetime.now().timestamp()
        with pd.ExcelWriter('Runtime/execution_diagnosis.xlsx', engine='xlsxwriter') as writer:
            pd.DataFrame(await exchange.fetch_orders(params={'start_time': start_time, 'end_time': end_time})).to_excel(
                writer, sheet_name='fills')
            pd.DataFrame(audit).to_excel(writer, sheet_name='audit')

    except Exception as e:
        logging.exception(e,exc_info=True)
    finally:
        await exchange.cancel_all_orders()
        # asyncio.run(clean_dust(exchange))
        stats = await fetch_latencyStats(exchange, days=1, subaccount_nickname='SysPerp')
        print(f'latencystats:{stats}')
        await exchange._localLog.populateFill(exchange)
#        print(exchange._localLog.bpCost())
        await exchange.close()

    return exchange._localLog()

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