import json
import time
import ccxtpro
import asyncio
import functools

import numpy as np
import pandas as pd
from ftx_utilities import *
from ftx_portfolio import live_risk,diff_portoflio
from ftx_history import fetch_trades_history
from ftx_ftx import mkt_at_size,fetch_latencyStats
import logging
logging.basicConfig(level=logging.INFO)

entry_tolerance = 0.5 # green light if basket better than median
edit_trigger_tolerance = np.sqrt(30) # chase on 30s stdev
stop_tolerance = np.sqrt(5*60) # stop on 5min stdev
time_budget = 5*60 # used in transaction speed screener
delta_limit = 5. # delta limit / pv
slice_factor = 0.1 # cut in 10

class myFtx(ccxtpro.ftx):
    class DoneDeal(Exception):
        def __init__(self,message):
            super().__init__(message)

    class LimitBreached(Exception):
        def __init__(self,limit=None,check_frequency=60):
            super().__init__()
            self.limit = limit
            self.check_frequency = check_frequency

    def __init__(self, config={}):
        super().__init__(config=config)
        self._localLog = [] #just a list of flattened dict, overwriting the same json over and over
        self.limit = myFtx.LimitBreached()
        self.pv = None
        self.risk_state = {}
        self.exec_parameters = {}
        # self.myTrades = {} # native
        # self.orders = {} # native
        # self.tickers= {} native
        # self.orderbooks = {} # native
        # self.trades = {} # native

    def __del__(self):
        asyncio.run(self.cancel_all_orders())

    def to_json(self,dict2add):
        self._localLog += dict2add
        with open('events.json', 'w') as file:
            json.dump(self._localLog, file)

    # initialize all state and does some filtering (weeds out slow underlyings; should be in strategy)
    # target_sub_portfolios = {coin:{entry_level_increment,
    #                     symbol1:{'spot_price','diff','target'}]}]
    async def build_state(self, weights,
                          entry_tolerance = entry_tolerance,
                          edit_trigger_tolerance = edit_trigger_tolerance, # chase on 30s stdev
                          stop_tolerance = stop_tolerance, # stop on 5min stdev
                          time_budget = time_budget,
                          delta_limit = delta_limit, # delta limit / pv
                          slice_factor = slice_factor): # cut in 10):

        self.limit.delta_limit = delta_limit

        frequency = timedelta(minutes=1)
        end = datetime.now()
        start = end - timedelta(hours=1)

        trades_history_list = await asyncio.gather(*[fetch_trades_history(
            self.market(symbol)['id'], self, start, end, frequency=frequency)
            for symbol in weights['name']])

        weights['name'] = weights['name'].apply(lambda s: self.market(s)['symbol'])
        weights.set_index('name', inplace=True)
        coin_list = weights['coin'].unique()

        # {coin:{symbol1:{data1,data2...},sumbol2:...}}
        data_dict = {coin:
                         {data['symbol']:
                              {'diff': weights.loc[data['symbol'], 'diffCoin'],
                               'target': weights.loc[data['symbol'], 'optimalCoin'],
                               'volume': data['vwap'].filter(like='/trades/volume').mean().values[0] / frequency.total_seconds(),# in coin per second
                               'series': data['vwap'].filter(like='/trades/vwap')}
                          for data in trades_history_list if data['coin']==coin}
                     for coin in coin_list}

        # exclude coins too slow or symbol diff too small
        data_dict = {coin: {symbol:data
                            for symbol, data in coin_data.items() if np.abs(data['diff']) > float(self.markets[symbol]['info']['minProvideSize'])}
                     for coin, coin_data in data_dict.items() if
                     all(data['volume'] * time_budget > np.abs(data['diff']) for data in coin_data.values())
                     and any(np.abs(data['diff']) > float(self.markets[symbol]['info']['minProvideSize']) for symbol, data in coin_data.items())}

        def basket_vwap_quantile(series_list,diff_list,quantile):
            series = pd.concat(series_list,axis=1).dropna(axis=0)-diff_list
            return series.sum(axis=1).diff().quantile(quantile)
        def z_score(series,z_score):# stdev of 1s prices * quantile
            series = series.dropna(axis=0)
            return series.std().values[0]*z_score

        # get times series of target baskets, compute quantile of increments and add to last price
        # remove series
        self.exec_parameters = {'timestamp':self.seconds()} \
                               |{coin:
                                     {'entry_level':
                                          sum([float(self.markets[name]['info']['price']) * data['diff']
                                               for name, data in coin_data.items()])
                                          + basket_vwap_quantile([data['series'] for data in coin_data.values()],
                                                                 [data['diff'] for data in coin_data.values()],
                                                                 entry_tolerance)}
                                     | {symbol:
                                         {
                                             'diff': data['diff'],
                                             'target': data['target'],
                                             'slice_size': max([float(self.markets[symbol]['info']['minProvideSize']),
                                                                slice_factor * np.abs(data['diff'])]),  # in coin
                                             'edit_trigger_depth': z_score(data['series'],edit_trigger_tolerance),
                                             'edit_price_depth': 0, # floored at priceIncrement. Used for aggressive, overriden for passive.
                                             'stop_depth': z_score(data['series'],stop_tolerance),
                                             'priceIncrement': float(self.markets[symbol]['info']['priceIncrement']),
                                             'sizeIncrement': float(self.markets[symbol]['info']['minProvideSize']),
                                         }
                                         for symbol, data in coin_data.items()}
                                 for coin, coin_data in data_dict.items()}

        self.risk_state = {coin:
                               {'netDelta':0,
                                'netDelta_timestamp':end.timestamp()}
                               | {symbol:
                                   {
                                       'delta': 0,
                                       'delta_timestamp':end.timestamp(),
                                   }
                                   for symbol, data in coin_data.items()}
                           for coin, coin_data in data_dict.items()}

        await self.fetch_risk(params={'useMarkets'})

        with open('request.json', 'w') as file:
            json.dump(flatten(self.exec_parameters),file)

    # updates risk in USD. Should be handle_my_trade but needs a coroutine :(
    # all symbols not present when state is built are ignored !
    # if some tickers are not initialized, just use markets
    async def fetch_risk(self, params=[]):
        positions = await self.fetch_positions()
        position_timestamp = self.seconds()-self.exec_parameters['timestamp']
        balances = await self.fetch_balance()
        balances_timestamp = self.seconds()-self.exec_parameters['timestamp']

        # mark to tickers (live) by default, markets (stale) otherwise
        marks = {symbol: self.mark(symbol)
                 for coin_data in self.exec_parameters.values() if type(coin_data)==dict
                 for symbol in coin_data.keys()
                 if symbol in self.tickers} \
                |{symbol:
                      np.median([float(self.markets[symbol]['info']['bid']), float(self.markets[symbol]['info']['ask']), float(self.markets[symbol]['info']['last'])])
                  for coin_data in self.exec_parameters.values() if type(coin_data)==dict
                  for symbol in coin_data.keys()
                  if symbol in self.markets and symbol not in self.tickers }

        time_spent = balances_timestamp - position_timestamp
        if time_spent*1000 > 200:  # 15ms is approx throttle, so 200 is a lot
            logging.info(f'balances vs positions: {time_spent*1000} ms')

        # delta is noisy for perps, so override to delta 1.
        for position in positions:
            if float(position['info']['size'])!=0:
                coin = self.markets[position['symbol']]['base']
                if coin in self.risk_state and position['symbol'] in self.risk_state[coin]:
                    self.risk_state[coin][position['symbol']]['delta'] = float(position['notional']*(1 if position['side'] == 'long' else -1)) \
                        if self.markets[position['symbol']]['type'] == 'future' \
                        else float(position['info']['size']) * marks[position['symbol']]
                    self.risk_state[coin][position['symbol']]['delta_timestamp']=position_timestamp

        for coin, balance in balances.items():
            if coin in self.currencies.keys() and coin != 'USD' and balance['total']!=0 and coin in self.risk_state and coin+'/USD' in self.risk_state[coin]:
                symbol = coin+'/USD'
                self.risk_state[coin][symbol]['delta'] = balance['total'] * marks[symbol]
                self.risk_state[coin][symbol]['delta_timestamp'] = balances_timestamp

        for coin,coin_data in self.risk_state.items():
            coin_data['netDelta']= sum([data['delta'] for symbol,data in coin_data.items() if symbol in self.markets and 'delta' in data.keys()])
            coin_data['netDelta_timestamp'] = max([position_timestamp,balances_timestamp])

        self.pv = sum(coin_data[coin+'/USD']['delta'] for coin, coin_data in self.risk_state.items() if coin+'/USD' in coin_data.keys()) + balances['USD']['total']

        self.to_json([{'eventType':'risk',
                       'timestamp':position_timestamp if ':' in symbol else balances_timestamp,
                       'symbol':symbol,
                       'delta':data['delta'],
                       'coin':coin,
                       'netDelta':coin_data['netDelta'],
                       'pv(wrong timestamp)':self.pv}
                      for coin,coin_data in self.risk_state.items() for symbol,data in coin_data.items() if symbol in self.markets])

    def mark(self,symbol):
        data = self.tickers[symbol] if symbol in self.tickers else self.markets[symbol]['info']
        return np.median([float(data['bid']), float(data['ask']), float(data['last'])])

    # size in usd
    def mkt_at_depth_orderbook(self,symbol,size):
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
    async def update_orders(self,symbol,size,orderbook_depth,edit_trigger_depth,edit_price_depth,stop_depth=None):
        orders = self.orders if self.orders else await self.fetch_open_orders(symbol=symbol)
        if len(orders)>2:
            for item in orders[:-1]:
                await self.cancel_order(item['id'])
            logging.warning('!!!!!!!!!! duplicates orders removed !!!!!!!!!!')
        #TODO: https://help.ftx.com/hc/en-us/articles/360052595091-Ratelimits-on-FTX

        price = self.tickers[symbol]['ask' if size<0 else 'bid']
        opposite_side = self.tickers[symbol]['ask' if size>0 else 'bid']
        mark = np.median([price,opposite_side,self.tickers[symbol]['last']])

        priceIncrement = self.exec_parameters[self.markets[symbol]['base']][symbol]['priceIncrement']
        sizeIncrement = self.exec_parameters[self.markets[symbol]['base']][symbol]['sizeIncrement']

        if stop_depth:
            stop_trigger = max(1,int(stop_depth / priceIncrement))*priceIncrement
        edit_trigger = max(1,int(edit_trigger_depth / priceIncrement))*priceIncrement
        edit_price = opposite_side - (1 if size>0 else -1)*max(1,int(edit_price_depth/priceIncrement))*priceIncrement

        order = None
        if not orders:
            order = await self.create_limit_order(symbol, 'buy' if size>0 else 'sell', np.abs(size), price=edit_price, params={'postOnly': True})
        else:
            # panic stop. we could rather place a trailing stop: more robust to latency, but less generic.
            if stop_depth \
                    and (1 if orders[0]['side']=='buy' else -1)*(opposite_side-orders[0]['price'])>stop_trigger \
                    and orders[0]['remaining']>sizeIncrement:
                order = await self.edit_order(orders[0]['id'], symbol, 'market', 'buy' if size>0 else 'sell',None)
            # chase
            if (1 if orders[0]['side']=='buy' else -1)*(price-orders[0]['price'])>edit_trigger \
                    and np.abs(edit_price-orders[0]['price'])>=priceIncrement \
                    and orders[0]['remaining']>sizeIncrement:
                order = await self.edit_order(orders[0]['id'], symbol, 'limit', 'buy' if size>0 else 'sell', None ,price=edit_price,params={'postOnly': True})

        return order

    def loop_and_callback(func):
        @functools.wraps(func)
        async def wrapper_loop(*args, **kwargs):
            while True:
                try:
                    value = await func(*args, **kwargs)
                except myFtx.DoneDeal as e:
                    logging.info(f'{e} done')
                    return
                except ccxt.base.errors.InsufficientFunds as e:
                    logging.info(str(e))# + '-> cancel all and finish with minimum increment')
                    #await args[0].cancel_all_orders()
                    #for coin,coin_data in args[0].exec_parameters.items() if coin self.currencies:
                    #    for symbol,data in coin_data.items():
                    #        if symbol in args[0].markets:
                    #            data['slice_size']=data['sizeIncrement']
                    continue
                except myFtx.LimitBreached as e:
                    logging.info(e)
                    # raise e # not a hard limit for now
                    continue
                except ccxt.base.errors.RequestTimeout as e:
                    logging.info('reconnect..')
                    continue
                except Exception as e:
                    logging.exception(e, exc_info=True)

        return wrapper_loop

    # on each top of book update, update market_state and send orders
    # tunes aggressiveness according to risk
    @loop_and_callback
    async def execute_on_update(self,symbol):
        coin = self.markets[symbol]['base']
        top_of_book = await self.watch_ticker(symbol)
        mark = self.mark(symbol)
        #risk
        risk = self.risk_state[coin][symbol]
        netDelta = self.risk_state[coin]['netDelta']

        # size to do: aim at target, slice, round to sizeIncrement
        params = self.exec_parameters[coin][symbol]
        size = params['target'] - risk['delta']/mark
        size = np.sign(size)*min([np.abs(size),params['slice_size']])
        sizeIncrement = params['sizeIncrement']
        if np.abs(size) < sizeIncrement:
            size =0
            raise myFtx.DoneDeal(symbol)
        else:
            size = np.sign(size)*int(np.abs(size)/sizeIncrement)*sizeIncrement
        assert(np.abs(size)>=sizeIncrement)

        # if increases risk, go passive
        if np.abs(netDelta+size)>np.abs(netDelta):
            if self.exec_parameters[coin]['entry_level'] is None: # for a purely risk reducing execise
                raise myFtx.DoneDeal(symbol)
                return
            # set limit at target quantile
            current_basket_price = sum(self.mark(symbol)*self.exec_parameters[coin][symbol]['diff']
                                       for symbol in self.exec_parameters[coin].keys() if symbol in self.markets)
            edit_price_depth = max([0,(current_basket_price-self.exec_parameters[coin]['entry_level'])/params['diff']])#TODO: sloppy logic assuming no correlation
            edit_price_depth = (np.abs(netDelta+size)-np.abs(netDelta))*params['stop_depth'] # pnl vol ~ profit if done
            edit_trigger_depth=params['edit_trigger_depth']
            order = await self.update_orders(symbol,size,orderbook_depth=0,edit_trigger_depth=edit_trigger_depth,edit_price_depth=edit_price_depth,stop_depth=None)
        else:
            edit_trigger_depth=params['edit_trigger_depth']
            edit_price_depth=params['edit_price_depth']
            stop_depth=params['stop_depth']
            order = await self.update_orders(symbol,size,orderbook_depth=0,edit_trigger_depth=edit_trigger_depth,edit_price_depth=edit_price_depth,stop_depth=stop_depth)

        if order != None:
            self.to_json([{'eventType':'top_of_book'}
                          |{key:self.tickers[symbol][key] for key in ['timestamp','symbol','bid','bidVolume','ask','askVolume','last']}
                          |{'timestamp': self.tickers[symbol]['timestamp'] - self.exec_parameters['timestamp']}
                          |{'coin':coin}])
            self.to_json([{'eventType':'order'}
                          |{key:order[key] for key in ['timestamp','symbol','side','price','amount','type','id','filled']}
                          |{'timestamp': order['timestamp'] - self.exec_parameters['timestamp']}
                          |{'coin':coin}])

    @loop_and_callback
    async def monitor_fills(self):
        fills = await self.watch_my_trades()
        def translate(key):
            if key=='takerOrMaker': return 'type'
            elif key=='order': return 'id'
            else: return key

        self.to_json([{'eventType':'fill'}
                      |{translate(key):fill[key] for key in ['symbol','side','price','amount','takerOrMaker','order']}
                      |{'timestamp':fill['timestamp']-self.exec_parameters['timestamp']}
                      |{'feeUSD':fill['fee']['cost']*(1 if fill['fee']['currency']=='USD' else self.tickers[fill['fee']['currency']+'/USD']['ask'])}
                      for fill in fills])
        await self.fetch_risk()

    ## redundant minutely risk check
    @loop_and_callback
    async def monitor_risk(self):
        await self.fetch_risk()

        self.limit.limit = self.pv * self.limit.delta_limit
        absolute_risk = sum(abs(data['netDelta']) for data in self.risk_state.values())
        if absolute_risk > self.limit.limit:
            logging.warning(f'absolute_risk {absolute_risk} > {self.limit.limit}')

        await asyncio.sleep(self.limit.check_frequency)

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

    if argv[0]=='sysperp':
        future_weights = pd.read_excel('Runtime/ApprovedRuns/current_weights.xlsx')
        diff = await diff_portoflio(exchange, future_weights)
        selected_coins = ['REN']#target_portfolios.sort_values(by='USDdiff', key=np.abs, ascending=False).iloc[2]['underlying']
        target_portfolio=diff[diff['coin'].isin(selected_coins)]
        await exchange.build_state(target_portfolio,
                                   entry_tolerance=entry_tolerance,
                                   edit_trigger_tolerance=edit_trigger_tolerance,
                                   stop_tolerance=stop_tolerance,
                                   time_budget=time_budget,
                                   delta_limit=delta_limit,
                                   slice_factor=slice_factor)
    elif argv[0]=='flatten': # only works for basket with 2 symbols
        future_weights = pd.DataFrame(columns=['name','optimalWeight'])
        diff = await diff_portoflio(exchange, future_weights)
        smallest_risk = diff.groupby(by='coin')['currentCoin'].agg(lambda series: series.apply(np.abs).min())
        target_portfolio=diff
        target_portfolio['optimalCoin'] = diff.apply(lambda f: smallest_risk[f['coin']]*np.sign(f['currentCoin']),axis=1)
        target_portfolio['diffCoin'] = target_portfolio['optimalCoin'] - target_portfolio['currentCoin']
        await exchange.build_state(target_portfolio,
                                   entry_tolerance=0.99,
                                   edit_trigger_tolerance=np.sqrt(5),
                                   stop_tolerance=np.sqrt(30),
                                   time_budget=999,
                                   delta_limit=999,
                                   slice_factor=0.5)
    elif argv[0]=='unwind':
        future_weights = pd.DataFrame(columns=['name','optimalWeight'])
        target_portfolio = await diff_portoflio(exchange, future_weights)
        target_portfolio['optimalCoin'] = 0
        target_portfolio['diffCoin'] = target_portfolio['optimalCoin'] - target_portfolio['currentCoin']
        await exchange.build_state(target_portfolio,
                                   entry_tolerance=0.99,
                                   edit_trigger_tolerance=np.sqrt(5),
                                   stop_tolerance=np.sqrt(30),
                                   time_budget=999,
                                   delta_limit=delta_limit,
                                   slice_factor=0.5)
    else:
        print('what?')
        return

    try:
        await asyncio.gather(*([exchange.monitor_fills()]+#,exchange.monitor_risk()]+#,exchange.watch_orders()]+
                               [exchange.execute_on_update(symbol)
                                for coin_data in exchange.risk_state.values()
                                for symbol in coin_data.keys() if symbol in exchange.markets]
                               ))
    except myFtx.LimitBreached as e:
        logging.exception(e,exc_info=True)
        #break
    except KeyboardInterrupt:# TODO: this is not caught :(
        pass
    except Exception as e:
        logging.exception(e,exc_info=True)
    finally:
        stats = await fetch_latencyStats(exchange, days=1, subaccount_nickname='SysPerp')
        print(f'latencystats:{stats}')
        await exchange.close()

    return

def ftx_ws_spread_main(*argv):
    argv=list(argv)
    if len(argv) == 0:
        argv.extend(['execute'])
    if len(argv) < 2:
        argv.extend(['sysperp'])
    if len(argv) < 4:
        argv.extend(['ftx', 'debug'])
    print(f'running {argv}')
    loop = asyncio.new_event_loop()
    if argv[0] == 'execute':
        return loop.run_until_complete(ftx_ws_spread_main_wrapper(*argv[1:],loop=loop))
    else:
        print(f'commands: sysperp, flatten,unwind')

if __name__ == "__main__":
    ftx_ws_spread_main(*sys.argv[1:])