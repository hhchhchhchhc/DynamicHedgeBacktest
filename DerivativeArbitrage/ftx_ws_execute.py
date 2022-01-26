import time
import ccxtpro
import asyncio

import pandas as pd

from ftx_rest_spread import *
from ftx_portfolio import live_risk
import logging
logging.basicConfig(level=logging.INFO)

def loop_and_callback(func):
    @functools.wraps(func)
    async def wrapper_loop(*args, **kwargs):
        while True:
            try:
                value = await func(*args, **kwargs)
            except myFtx.DoneDeal as e:
                logging.info(e)
                raise e
                break
            except myFtx.LimitBreached as e:
                logging.info(e)
                raise e
                break
            except ccxt.base.errors.RequestTimeout as e:
                print('reconnect..')
                continue
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
            
    class Risk(dict):
        def __init__(self):
            super().__init__()
    
    def __init__(self, config={}):
        super().__init__(config=config)
        self._localLog = myFtx.myLogging()
        self.limit = myFtx.LimitBreached()
        self.risk = myFtx.Risk()
        self.pv = None

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
        self.pv = sum(balance['total'] * (float(self.market(coin + '/USD')['info']['price']) if coin !='USD' else 1) for coin,balance in balances.items() if coin in self.currencies)

        balances = {key:
                        {key + '/USD':
                             {'delta':
                                  float(balance['total']) * float(self.market(key + '/USD')['info']['price'])
                              }
                         }
                    for key, balance in balances.items() if key in self.currencies.keys() and key != 'USD'}

        self.risk = {key:
                         (positions[key] if key in positions.keys() else {})
                         | (balances[key] if key in balances.keys() else {})
                         | {'netDelta': sum(positions[key][symbol]['delta'] for symbol in
                                            (positions[key].keys() if key in positions.keys() else {}))
                                        + (balances[key][key + '/USD']['delta'] if key in balances.keys() else 0)}
                     for key in positions.keys() | balances.keys()}



        absolute_risk=sum(abs(data['netDelta']) for data in self.risk.values())
        if absolute_risk>self.limit.limit:
            raise self.limit

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

    #size in coin
    #scalers are in number of bid asks
    async def update_orders(self,symbol,size,depth,edit_trigger_scaler,edit_price_scaler,stop_scaler):
        orders = await self.fetch_open_orders(symbol=symbol)
        if len(orders)>3:
            for item in orders[:-1]:
                await self.cancel_order(item['id'])
            warnings.warn('!!!!!!!!!! duplicates orders removed !!!!!!!!!!')
    
        price,opposite_side = float(self.markets[symbol]['info']['ask' if size<0 else 'bid']),float(self.market(symbol)['info']['ask' if size>0 else 'bid'])
        mid = 0.5*(price+opposite_side)
        priceIncrement = float(self.markets[symbol]['info']['priceIncrement'])
        sizeIncrement = float(self.markets[symbol]['info']['sizeIncrement'])
        if np.abs(size) < sizeIncrement:
            order_size =0
            return None
        else:
            order_size = int(np.abs(size)/sizeIncrement)*sizeIncrement

        # triggers are in units of bid/ask at size
        stop_trigger = max(1,int((stop_scaler * np.abs(opposite_side-price))/priceIncrement))*priceIncrement
        edit_trigger = max(1,int((edit_trigger_scaler * np.abs(opposite_side-price))/priceIncrement))*priceIncrement
        edit_price = opposite_side - max(1,int(edit_price_scaler * np.abs(opposite_side - price) * (1 if size>0 else -1))/priceIncrement)*priceIncrement

        if not orders:
            await self.create_limit_order(symbol, 'buy' if size>0 else 'sell', order_size, price=edit_price, params={'postOnly': True})

        for item in orders:
            # panic stop. we could rather place a trailing stop: more robust to latency, less generic.
            if (1 if item['side']=='buy' else -1)*(opposite_side-item['price'])>stop_trigger:
                order = await self.edit_order(item['id'], symbol, 'market', 'buy' if size>0 else 'sell', order_size)
            # chase
            if (1 if item['side']=='buy' else -1)*(price-item['price'])>edit_trigger:
                order = await self.edit_order(item['id'], symbol, 'limit', 'buy' if size>0 else 'sell', order_size,price=edit_price,params={'postOnly': True})
            #print(str(price/item['price']-1) + ' ' + str(opposite_side / item['price'] - 1))
    
        return None

    # a mix of watch_ticker and handle_ticker. Can't inherit since handle needs coroutine.
    @loop_and_callback
    async def execute_on_update(self,symbol,coin_request):
        top_of_book = await self.watch_ticker(symbol)
        coin = self.markets[symbol]['base']
        symbol_request = coin_request[symbol]

        # size to do
        size = symbol_request['target']
        netDelta = 0
        if coin in self.risk:
            netDelta = self.risk[coin]['netDelta']
            if symbol in self.risk[coin]:
                size -= self.risk[coin][symbol]['delta']

        size = np.sign(size)*min([np.abs(size),symbol_request['exec_parameters']['slice_size']])

        # if increases risk, go passive
        if np.abs(netDelta+size)>np.abs(netDelta):
            # wait for good level
            current_basket_price = sum(float(self.markets[symbol]['info']['price'])*symbol_data['target'] for symbol,symbol_data in coin_request.items() if symbol in self.markets)
            if current_basket_price<coin_request['entry_level']:
                depth = 0
                edit_trigger_scaler=symbol_request['exec_parameters']['edit_trigger_scaler']
                edit_price_scaler=symbol_request['exec_parameters']['edit_price_scaler']
                stop_scaler=999999
                order = await self.update_orders(symbol,size,0,edit_trigger_scaler,edit_price_scaler,stop_scaler)
        else:
            depth = 0
            edit_trigger_scaler=symbol_request['exec_parameters']['edit_trigger_scaler']
            edit_price_scaler=symbol_request['exec_parameters']['edit_price_scaler']
            stop_scaler=symbol_request['exec_parameters']['edit_price_scaler']
            order = await self.update_orders(symbol,size,0,edit_trigger_scaler,edit_price_scaler,stop_scaler)

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
                          {'volume': df['vwap'].filter(like='/trades/volume').mean().values[0] / frequency.total_seconds(),
                           'spot_price': weights.loc[df['symbol'], 'spot_price'],
                           'diff': weights.loc[df['symbol'], 'diff'],
                           'target': weights.loc[df['symbol'], 'target'],
                           'exec_parameters': {  # depend on risk tolerance other execution parameters...
                               'slice_size': max([float(exchange.market(df['symbol'])['info']['minProvideSize']),
                                                  0.1 * weights.loc[df['symbol'], 'diff']]),  # in usd
                               'edit_trigger_scaler': 10,
                               'edit_price_scaler': 1,
                               'stop_scaler': 50},
                           'series': df['vwap'].filter(like='/trades/vwap')}
                      for df in trades_history_list if df['coin']==coin}
                 for coin in coin_list}

    # exclude coins with slow symbols
    volume_list = {coin:
                       coin_data for coin, coin_data in data_dict.items()
                   if min([data['volume'] * time_budget / data['spot_price'] / max(1, np.abs(data['diff']))
                           for name, data in coin_data.items()]) > 1}

    # get times series of target baskets, compute quantile of increments and add to last price
    # remove series
    light_list = {coin:
                      {'entry_level':
                           sum([float(exchange.market(name)['info']['price']) * data['target'] for name, data in
                                coin_data.items()])
                           + sum([data['series'] * data['target'] for name, data in
                                  coin_data.items()]).dropna().diff().quantile(0.6).values[0]}
                      | {name:
                             {field: field_data for field, field_data in data.items() if field != 'series'}
                         for name, data in coin_data.items()}
                  for coin, coin_data in volume_list.items()}

    print('Executing:')
    print(weights)
    return light_list

async def executer_ws(exchange, targets):
    try:
        await asyncio.gather(*([exchange.monitor_fills()]+#,exchange.monitor_risk()]+
                               [exchange.execute_on_update(symbol, target)
                                for coin,target in targets.items() for symbol,symbol_data in target.items() if symbol in exchange.markets.keys()]
                               ))
    except myFtx.LimitBreached as e:
        logging.exception(e)
        #break
    except ccxt.base.errors.RequestTimeout as e:
        print('reconnect..')
        #continue
    except Exception as e:
        logging.exception(e)

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

    request = await execution_request(exchange, target_sub_portfolios)

    #coin='OMG'
    #target_sub_portfolios = target_sub_portfolios[target_sub_portfolios['name'].isin([coin+'/USD',coin+'-PERP'])]

    start_time = datetime.now().timestamp()
    log = ExecutionLog('dummy', [])
    try:
        if len(target_sub_portfolios)==0: warnings.warn('nothing to execute')
        log = await executer_ws(exchange, request)
        end_time = datetime.now().timestamp()
        with pd.ExcelWriter('Runtime/execution_diagnosis.xlsx', engine='xlsxwriter') as writer:
            pd.DataFrame(await exchange.fetch_orders(params={'start_time': start_time, 'end_time': end_time})).to_excel(
                writer, sheet_name='fills')
            pd.DataFrame(audit).to_excel(writer, sheet_name='audit')
        print(log.bpCost())

    except Exception as e:
        logging.exception(e)
    finally:
        await exchange.cancel_all_orders()
        # asyncio.run(clean_dust(exchange))
        stats = await fetch_latencyStats(exchange, days=1, subaccount_nickname='SysPerp')
        print(f'latencystats:{stats}')
        await log.populateFill(exchange)
        await exchange.close()

    return log.to_df()

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