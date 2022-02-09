import ccxtpro
import functools

import numpy as np
import pandas as pd

from ftx_utilities import *
from ftx_portfolio import live_risk,diff_portoflio,liveIM
from ftx_history import fetch_trades_history
from ftx_ftx import mkt_at_size,fetch_latencyStats,fetch_futures
import logging
logging.basicConfig(level=logging.INFO)

max_nb_coins = 7  # TODO: sharding needed
entry_tolerance = 0.5 # green light if basket better than median
edit_trigger_tolerance = np.sqrt(60/3600) # chase on 1m stdev
stop_tolerance = np.sqrt(30*60/3600) # stop on 30min stdev
time_budget = 500*60 # used in transaction speed screener
delta_limit = 0.2 # delta limit / pv
slice_factor = 0.5 # % of request
edit_price_tolerance=np.sqrt(30/3600)#price on 1s std

class myFtx(ccxtpro.ftx):
    class DoneDeal(Exception):
        def __init__(self,symbol):
            super().__init__('{} done'.format(symbol))

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
        self.margin_headroom = None
        self.risk_state = {}
        self.order_state= {} # redundant with native self.orders
        self.fill_state = {}  # redundant with native self.myTrade
        self.exec_parameters = {}
        self.margin_calculator = None
        self.done = []
        # self.myTrades = {} # native
        # self.orders = {} # native
        # self.tickers= {} native
        # self.orderbooks = {} # native
        # self.trades = {} # native

#    def __del__(self):
#        asyncio.run(self.cancel_all_orders())

    def to_json(self,dict2add):
        self._localLog += dict2add
        with open('Runtime/logs/'+datetime.utcnow().strftime("%Y-%m-%d-%Hh")+'_events.json', 'w') as file:
            json.dump(self._localLog, file,cls=NpEncoder)

    # initialize all state and does some filtering (weeds out slow underlyings; should be in strategy)
    # target_sub_portfolios = {coin:{entry_level_increment,
    #                     symbol1:{'spot_price','diff','target'}]}]
    async def build_state(self, weights,
                          entry_tolerance = entry_tolerance,
                          edit_trigger_tolerance = edit_trigger_tolerance, # chase on 30s stdev
                          edit_price_tolerance = edit_price_tolerance,
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
        full_dict = {coin:
                         {data['symbol']:
                              {'diff': weights.loc[data['symbol'], 'diffCoin'],
                               'target': weights.loc[data['symbol'], 'optimalCoin'],
                               'spot_price': weights.loc[data['symbol'], 'spot_price'],
                               'volume': data['vwap'].filter(like='/trades/volume').mean().values[0] / frequency.total_seconds(),# in coin per second
                               'series': data['vwap'].filter(like='/trades/vwap')}
                          for data in trades_history_list if data['coin']==coin}
                     for coin in coin_list}

        # exclude coins too slow or symbol diff too small, limit to max_nb_coins
        diff_threshold = sorted(
            [max([np.abs(weights.loc[data['symbol'], 'diffUSD'])
                  for data in trades_history_list if data['coin']==coin])
             for coin in coin_list])[-min(max_nb_coins,len(coin_list))]

        data_dict = {coin: {symbol:data
                            for symbol, data in coin_data.items() if np.abs(data['diff']) > max(diff_threshold/data['spot_price'], float(self.markets[symbol]['info']['minProvideSize']))}
                     for coin, coin_data in full_dict.items() if
                     all(data['volume'] * time_budget > np.abs(data['diff']) for data in coin_data.values())
                     and any(np.abs(data['diff']) > max(diff_threshold/data['spot_price'], float(self.markets[symbol]['info']['minProvideSize'])) for symbol, data in coin_data.items())}
        if data_dict =={}:
            raise myFtx.DoneDeal('empty query: too small or too slow')

        def basket_vwap_quantile(series_list,diff_list,quantile):
            series = pd.concat(series_list,axis=1).dropna(axis=0)-diff_list
            return series.sum(axis=1).diff().quantile(quantile)
        def z_score(series,z_score):# stdev of 1s prices * quantile
            series = series.dropna(axis=0)
            return series.std().values[0]*z_score

        # get times series of target baskets, compute quantile of increments and add to last price
        # remove series
        self.exec_parameters = {'timestamp':self.seconds()} \
                               |{sys.intern(coin):
                                     {sys.intern('entry_level'):
                                          sum([float(self.markets[symbol]['info']['price']) * data['diff']
                                               for symbol, data in coin_data.items()])
                                          + basket_vwap_quantile([data['series'] for data in coin_data.values()],
                                                                 [data['diff'] for data in coin_data.values()],
                                                                 entry_tolerance)}
                                     | {sys.intern(symbol):
                                         {
                                             sys.intern('diff'): data['diff'],
                                             sys.intern('target'): data['target'],
                                             sys.intern('slice_size'): max([float(self.markets[symbol]['info']['minProvideSize']),
                                                                slice_factor * np.abs(data['diff'])]),  # in coin
                                             sys.intern('edit_trigger_depth'): z_score(data['series'],edit_trigger_tolerance),
                                             sys.intern('edit_price_depth'): z_score(data['series'],edit_price_tolerance), # floored at priceIncrement. Used for aggressive, overriden for passive.
                                             sys.intern('stop_depth'): z_score(data['series'],stop_tolerance),
                                             sys.intern('priceIncrement'): float(self.markets[symbol]['info']['priceIncrement']),
                                             sys.intern('sizeIncrement'): float(self.markets[symbol]['info']['minProvideSize']),
                                         }
                                         for symbol, data in coin_data.items()}
                                 for coin, coin_data in data_dict.items()}

        self.risk_state = {sys.intern(coin):
                               {sys.intern('netDelta'):0}
                               | {sys.intern(symbol):
                                   {
                                       sys.intern('delta'): 0,
                                       sys.intern('delta_timestamp'):end.timestamp(),
                                       sys.intern('delta_id'): None
                                   }
                                   for symbol, data in coin_data.items()}
                           for coin, coin_data in data_dict.items()}

        coin_list = [coin for coin,coin_data in data_dict.items() for symbol in coin_data.keys()]
        futures = pd.DataFrame(await fetch_futures(self))
        account_leverage = float(futures.iloc[0]['account_leverage'])
        collateralWeight = futures.set_index('underlying')['collateralWeight'].to_dict()
        imfFactor = futures.set_index('new_symbol')['imfFactor'].to_dict()
        self.margin_calculator = liveIM(account_leverage,collateralWeight,imfFactor)

        # populates risk, pv and IM
        await self.fetch_risk()

        # orders and fills
        self.order_state = {sys.intern(coin):
                               {sys.intern(symbol):[]
                                   for symbol, data in coin_data.items()}
                           for coin, coin_data in data_dict.items()}

        self.fill_state = {sys.intern(coin):
                               {sys.intern(symbol):[]
                                   for symbol, data in coin_data.items()}
                           for coin, coin_data in data_dict.items()}

        # logs
        with open('Runtime/logs/'+datetime.utcnow().strftime("%Y-%m-%d-%Hh")+'_request.json', 'w') as file:
            json.dump(flatten(self.exec_parameters),file,cls=NpEncoder)
        with pd.ExcelWriter('Runtime/logs/latest.xlsx', engine='xlsxwriter') as writer:
            vwap_dataframe = pd.concat([data['vwap'].filter(like='vwap').fillna(method='ffill') for data in trades_history_list],axis=1,join='outer')
            vwap_dataframe.to_excel(writer, sheet_name='vwap')
            size_dataframe = pd.concat([data['vwap'].filter(like='volume').fillna(method='ffill') for data in trades_history_list], axis=1, join='outer')
            size_dataframe.to_excel(writer, sheet_name='volume')

    # updates risk in USD. Should be handle_my_trade but needs a coroutine :(
    # all symbols not present when state is built are ignored !
    # if some tickers are not initialized, just use markets
    async def fetch_risk(self, params=[]):
        risks= await asyncio.gather(*[self.fetch_positions(),self.fetch_balance()])
        positions = risks[0]
        balances = risks[1]
        risk_timestamp = self.seconds()

        # mark to tickers (live) by default, markets (stale) otherwise
        marks = {symbol: self.mark(symbol)
                 for coin_data in self.exec_parameters.values() if type(coin_data)==dict
                 for symbol in coin_data.keys()
                 if symbol in self.tickers} \
                |{symbol:
                      np.median([float(self.markets[symbol]['info']['bid']), float(self.markets[symbol]['info']['ask']), float(self.markets[symbol]['info']['last'])])
                  for coin_data in self.exec_parameters.values() if type(coin_data)==dict
                  for symbol in coin_data.keys()
                  if symbol in self.markets and symbol not in self.tickers}

        # delta is noisy for perps, so override to delta 1.
        for position in positions:
            if float(position['info']['size'])!=0:
                coin = self.markets[position['symbol']]['base']
                if coin in self.risk_state and position['symbol'] in self.risk_state[coin]:
                    self.risk_state[coin][position['symbol']]['delta'] = position['notional']*(1 if position['side'] == 'long' else -1) \
                        if self.markets[position['symbol']]['type'] == 'future' \
                        else float(position['info']['size']) * marks[position['symbol']]*(1 if position['side'] == 'long' else -1)
                    self.risk_state[coin][position['symbol']]['delta_timestamp']=risk_timestamp

        for coin, balance in balances.items():
            if coin in self.currencies.keys() and coin != 'USD' and balance['total']!=0 and coin in self.risk_state and coin+'/USD' in self.risk_state[coin]:
                symbol = coin+'/USD'
                self.risk_state[coin][symbol]['delta'] = balance['total'] * marks[symbol]
                self.risk_state[coin][symbol]['delta_timestamp'] = risk_timestamp

        for coin,coin_data in self.risk_state.items():
            coin_data['netDelta']= sum([data['delta'] for symbol,data in coin_data.items() if symbol in self.markets and 'delta' in data.keys()])

        # update pv
        self.pv = sum(coin_data[coin+'/USD']['delta'] for coin, coin_data in self.risk_state.items() if coin+'/USD' in coin_data.keys()) + balances['USD']['total']

        #compute IM
        spot_weight={}
        future_weight={}
        for position in positions:
            if float(position['info']['netSize'])!=0:
                future_weight |= {position['symbol']: {'weight': float(position['info']['netSize']), 'mark':self.mark(position['symbol'])}}
        for coin,balance in balances.items():
            if coin!='USD' and coin in self.currencies and balance['total']!=0:
                spot_weight |= {(coin):{'weight':balance['total'],'mark':self.mark(coin+'/USD')}}
        IM = self.margin_calculator.margins(balances['USD']['total'],spot_weight,future_weight)['IM']

        # fetch IM, in fact...
        account_info=(await self.privateGetAccount())['result']
        self.margin_headroom = float(account_info['totalPositionSize'])*(float(account_info['openMarginFraction'])-float(account_info['initialMarginRequirement']))

        self.to_json([{'eventType':'risk',
                       'coin': coin,
                       'symbol': symbol,
                       'timestamp':risk_timestamp-self.exec_parameters['timestamp'],
                       'delta':data['delta'],
                       'netDelta': coin_data['netDelta'],
                       'pv(wrong timestamp)':self.pv,
                       'margin_headroom':self.margin_headroom,
                       'IM_discrepancy':IM - float(account_info['totalPositionSize'])*(float(account_info['marginFraction']))}
                      for coin,coin_data in self.risk_state.items() for symbol,data in coin_data.items() if symbol in self.markets])

    def process_fill(self,fill, coin, symbol):
        found_order = next((order for order in self.order_state[coin][symbol] if order['id'] == fill['order']), None)
        if fill['id'] in [_fill['id'] for _fill in self.fill_state[coin][symbol]]: 
            return # already processed 
        if not found_order is None:
            # update order_state
            found_order['filled'] += fill['amount']
            found_order['remaining'] -= fill['amount']
            found_order['status']='closed' if found_order['remaining']==0 else 'open'

            # update fill_state
            self.fill_state[coin][symbol] += [fill]
            
            # update risk_state
            fill_size = fill['amount'] * (1 if fill['side'] == 'buy' else -1)*fill['price']
            self.risk_state[coin][symbol]['delta'] += fill_size
            self.risk_state[coin][symbol]['delta_timestamp'] = fill['timestamp']/1000
            latest_delta = self.risk_state[coin][symbol]['delta_id']
            self.risk_state[coin][symbol]['delta_id'] = max(0 if latest_delta is None else latest_delta,int(fill['id']))
            self.risk_state[coin]['netDelta'] += fill_size
        else:
            logging.exception("order {} not found".format(
                fill['order']))  # next(order if order['id']==fill['order'] for order in self.order_state)

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
    async def peg_or_stopout(self,symbol,size,orderbook_depth,edit_trigger_depth,edit_price_depth,stop_depth=None):
        coin = self.markets[symbol]['base']
        #orders = self.orders if self.orders else await self.fetch_open_orders(symbol=symbol)
        #orders = await self.fetch_open_orders(symbol=symbol)
        orders = [order for order in self.order_state[coin][symbol] if order['status']=='open']
        if len(orders)>2:
            for item in orders[:-1]:
                await self.cancel_order(item['id'])
            logging.warning('!!!!!!!!!! duplicates orders removed !!!!!!!!!!')
        #TODO: https://help.ftx.com/hc/en-us/articles/360052595091-Ratelimits-on-FTX

        price = self.tickers[symbol]['ask' if size<0 else 'bid']
        opposite_side = self.tickers[symbol]['ask' if size>0 else 'bid']
        mark = np.median([price,opposite_side,self.tickers[symbol]['last']])

        priceIncrement = self.exec_parameters[coin][symbol]['priceIncrement']
        sizeIncrement = self.exec_parameters[coin][symbol]['sizeIncrement']

        if stop_depth:
            stop_trigger = max(1,int(stop_depth / priceIncrement))*priceIncrement
        edit_trigger = max(1,int(edit_trigger_depth / priceIncrement))*priceIncrement
        edit_price = opposite_side - (1 if size>0 else -1)*max(1,int(edit_price_depth/priceIncrement))*priceIncrement

        try:
            order = None
            if len(orders)==0:
                order = await self.create_limit_order(symbol, 'buy' if size>0 else 'sell', np.abs(size), price=edit_price, params={'postOnly': True})
            else:
                order_distance = (1 if orders[0]['side']=='buy' else -1)*(opposite_side-orders[0]['price'])
                # panic stop. we could rather place a trailing stop: more robust to latency, but less generic.
                if stop_depth \
                        and order_distance>stop_trigger \
                        and orders[0]['remaining']>sizeIncrement:
                    order = await self.edit_order(orders[0]['id'], symbol, 'market', 'buy' if size>0 else 'sell',None)
                # chase
                if order_distance>edit_trigger \
                        and np.abs(edit_price-orders[0]['price'])>=priceIncrement \
                        and orders[0]['remaining']>sizeIncrement:
                    order = await self.edit_order(orders[0]['id'], symbol, 'limit', 'buy' if size>0 else 'sell', None ,price=edit_price,params={'postOnly': True})
        except Exception as e:
            logging.exception('{} {} {} at {} raised {}'.format('buy' if size>0 else 'sell', np.abs(size), symbol, price, e))

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
                except KeyboardInterrupt:
                    raise(KeyboardInterrupt)
                except Exception as e:
                    logging.exception(e, exc_info=True)

        return wrapper_loop

    # on each top of book update, update market_state and send orders
    # tunes aggressiveness according to risk
    @loop_and_callback
    async def order_master(self,symbol):
        coin = self.markets[symbol]['base']
        top_of_book = await self.watch_ticker(symbol)
        mark = self.mark(symbol)
        params = self.exec_parameters[coin][symbol]

        # in case fills arrived before risk updates
        coin_fills = await asyncio.gather(*[self.fetch_my_trades(symbol=symbol_,
                                                                 params={'minId':self.risk_state[coin][symbol_]['delta_id']+1})
                                            for symbol_ in self.risk_state[coin].keys() if symbol_ in self.markets and not self.risk_state[coin][symbol_]['delta_id'] is None])
        for symbol_fills in coin_fills:
            for fill in symbol_fills:
                self.process_fill(fill, coin, symbol)
                logging.warning('{}:caught fill {} {} in {} loop'.format(
                    fill['timestamp']/1000-self.exec_parameters['timestamp'],
                    fill['amount']*(1 if fill['side']=='buy' else -1),
                    fill['symbol'],
                    symbol))

        #risk
        delta = self.risk_state[coin][symbol]['delta']/mark
        delta_timestamp = self.risk_state[coin][symbol]['delta_timestamp']
        netDelta = self.risk_state[coin]['netDelta']/mark

        # size to do: aim at target, slice, round to sizeIncrement
        size = params['target'] - delta
        sizeIncrement = params['sizeIncrement']
        if np.abs(size) < sizeIncrement:
            size =0
            self.done+=[symbol]
            if all(symbol_ in self.done
                   for coin,coin_data in self.exec_parameters.items() if coin in self.currencies
                   for symbol_ in coin_data.keys() if symbol_ in self.markets):
                log_reader()
                raise myFtx.DoneDeal('all done')
            else:
                raise myFtx.DoneDeal(symbol)
        else:
            size = np.sign(size)*int(min([np.abs(size),params['slice_size'],np.abs(self.margin_headroom)/mark])/sizeIncrement)*sizeIncrement
        assert(np.abs(size)>=sizeIncrement)

        order = None
        # if increases risk, go passive
        if np.abs(netDelta+size)>np.abs(netDelta):
            if self.exec_parameters[coin]['entry_level'] is None: # for a purely risk reducing execise
                self.done += [symbol]
                raise myFtx.DoneDeal(symbol)
            # set limit at target quantile
            #current_basket_price = sum(self.mark(symbol)*self.exec_parameters[coin][symbol]['diff']
            #                           for symbol in self.exec_parameters[coin].keys() if symbol in self.markets)
            #edit_price_depth = max([0,(current_basket_price-self.exec_parameters[coin]['entry_level'])/params['diff']])#TODO: sloppy logic assuming no correlation
            edit_price_depth = (np.abs(netDelta+size)-np.abs(netDelta))/np.abs(size)*params['edit_price_depth'] # equate: profit if done ~ marginal risk * stdev
            edit_trigger_depth=params['edit_trigger_depth']
            order = await self.peg_or_stopout(symbol,size,orderbook_depth=0,edit_trigger_depth=edit_trigger_depth,edit_price_depth=edit_price_depth,stop_depth=None)
        else:
            edit_trigger_depth=params['edit_trigger_depth']
            edit_price_depth=params['edit_price_depth']
            stop_depth=params['stop_depth']
            order = await self.peg_or_stopout(symbol,size,orderbook_depth=0,edit_trigger_depth=edit_trigger_depth,edit_price_depth=edit_price_depth,stop_depth=stop_depth)

        if not order is None:
            self.order_state[coin][symbol]+=[order]

            self.to_json([{'eventType':'order_master','coin':coin}
                          |{key:self.tickers[symbol][key] for key in ['symbol','bid','bidVolume','ask','askVolume','last']}
                          |{'timestamp': self.tickers[symbol]['timestamp']/1000 - self.exec_parameters['timestamp']},
                          {'eventType':'local_risk','coin':coin,'symbol':symbol,
                           'delta':delta,'netDelta':netDelta,'delta_timestamp':delta_timestamp-self.exec_parameters['timestamp']},
                          {'eventType':'order','coin':coin}
                          |{key:order[key] for key in ['symbol','side','price','amount','type','id','filled']}
                          |{'timestamp': order['timestamp']/1000 - self.exec_parameters['timestamp']}])

    @loop_and_callback
    async def monitor_fills(self):
        fills = await self.watch_my_trades()
        def translate(key):
            if key=='takerOrMaker': return 'type'
            elif key=='order': return 'id'
            else: return key

        # await self.fetch_risk() is safer but slower. we have monitor_risk to reconcile
        for fill in fills:
            coin = self.markets[fill['symbol']]['base']
            symbol = fill['symbol']
            self.process_fill(fill,coin,symbol)

            self.to_json([{'eventType': 'fill'}
                          | {translate(key): fill[key] for key in
                             ['symbol', 'side', 'price', 'amount', 'takerOrMaker', 'order']}
                          | {'timestamp': fill['timestamp'] / 1000 - self.exec_parameters['timestamp'],
                             'coin': coin,
                             'delta_timestamp': self.risk_state[coin][symbol]['delta_timestamp'] - self.exec_parameters['timestamp'],
                             'delta_id':fill['id'],
                             'delta': self.risk_state[coin][symbol]['delta'],
                             'netDelta': self.risk_state[coin]['netDelta'],
                             'ask': self.tickers[symbol]['ask'], 'bid': self.tickers[symbol]['bid'],
                             'feeUSD': fill['fee']['cost'] * (1 if fill['fee']['currency'] == 'USD' else
                                                              self.tickers[fill['fee']['currency'] + '/USD']['ask'])}])
            logging.info('{} fill at {}: {} {} {} at {}'.format(symbol,
                                fill['timestamp'] / 1000 - self.exec_parameters['timestamp'],
                                fill['side'],fill['amount'],symbol,fill['price']))

            current = self.risk_state[coin][symbol]['delta']
            initial =self.exec_parameters[coin][symbol]['target'] * fill['price'] -self.exec_parameters[coin][symbol]['diff'] * fill['price']
            target=self.exec_parameters[coin][symbol]['target'] * fill['price']
            logging.info('{} risk at {}sec: {}% done [current {}, initial {}, target {}]'.format(
                symbol,
                self.risk_state[coin][symbol]['delta_timestamp']- self.exec_parameters['timestamp'],
                (current-initial)/(target-initial)*100,
                current,
                initial,
                target))

        #futures_array = np.array([self.risk_state[coin][symbol]['delta'] / self.mark(symbol) if ':' in symbol else 0.0
        #                          for coin_data in self.risk_state.values() for symbol in coin_data.keys() if
        #                          symbol in self.markets])
        #spot_array = np.array([self.risk_state[coin][symbol]['delta'] if not ':' in symbol else 0.0
        #                          for coin_data in self.risk_state.values() for symbol in coin_data.keys() if
        #                          symbol in self.markets])
        #self.margin = self.margin_calculator()#TODO:later....

    ## redundant minutely risk check
    @loop_and_callback
    async def monitor_risk(self):
        await self.fetch_risk()

        self.limit.limit = self.pv * self.limit.delta_limit
        absolute_risk = sum(abs(data['netDelta']) for data in self.risk_state.values())
        if absolute_risk > self.limit.limit:
            logging.warning(f'absolute_risk {absolute_risk} > {self.limit.limit}')
        if self.margin_headroom < self.pv/100:
            logging.warning(f'IM {self.margin_headroom}  < 1%')

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
    await exchange.cancel_all_orders()

    try:
        if argv[0]=='sysperp':
            future_weights = pd.read_excel('Runtime/ApprovedRuns/current_weights.xlsx')
            futures=await fetch_futures(exchange)
            target_portfolio = await diff_portoflio(exchange, future_weights)
            #selected_coins = ['REN']#target_portfolios.sort_values(by='USDdiff', key=np.abs, ascending=False).iloc[2]['underlying']
            #target_portfolio=diff[diff['coin'].isin(selected_coins)]
            target_portfolio['optimalCoin']*=0.8
            await exchange.build_state(target_portfolio,
                                       entry_tolerance=entry_tolerance,
                                       edit_trigger_tolerance=edit_trigger_tolerance,
                                       edit_price_tolerance=edit_price_tolerance,
                                       stop_tolerance=stop_tolerance,
                                       time_budget=time_budget,
                                       delta_limit=delta_limit,
                                       slice_factor=slice_factor)
        elif argv[0]=='flatten': # only works for basket with 2 symbols
            future_weights = pd.DataFrame(columns=['name','optimalWeight'])
            diff = await diff_portoflio(exchange, future_weights)
            smallest_risk = diff.groupby(by='coin')['currentCoin'].agg(lambda series: series.apply(np.abs).min() if series.shape[0]>1 else 0)
            target_portfolio=diff
            target_portfolio['optimalCoin'] = diff.apply(lambda f: smallest_risk[f['coin']]*np.sign(f['currentCoin']),axis=1)
            target_portfolio['diffCoin'] = target_portfolio['optimalCoin'] - target_portfolio['currentCoin']
            await exchange.build_state(target_portfolio,
                                       entry_tolerance=0.99,
                                       edit_trigger_tolerance=np.sqrt(5),
                                       edit_price_tolerance=0,
                                       stop_tolerance=np.sqrt(30),
                                       time_budget=999,
                                       delta_limit=delta_limit,
                                       slice_factor=0.5)
        elif argv[0]=='unwind':
            future_weights = pd.DataFrame(columns=['name','optimalWeight'])
            target_portfolio = await diff_portoflio(exchange, future_weights)
            target_portfolio['optimalCoin'] = 0
            target_portfolio['diffCoin'] = target_portfolio['optimalCoin'] - target_portfolio['currentCoin']
            await exchange.build_state(target_portfolio,
                                       entry_tolerance=0.99,
                                       edit_trigger_tolerance=edit_trigger_tolerance,
                                       edit_price_tolerance=edit_price_tolerance,
                                       stop_tolerance=stop_tolerance,
                                       time_budget=999,
                                       delta_limit=delta_limit,
                                       slice_factor=slice_factor)
        else:
            logging.exception('what ?',exc_info=True)

        await asyncio.gather(*([exchange.monitor_fills(),exchange.monitor_risk()]+#,exchange.watch_orders()
                               [exchange.order_master(symbol)
                                for coin_data in exchange.risk_state.values()
                                for symbol in coin_data.keys() if symbol in exchange.markets]
                               ))
    except myFtx.LimitBreached as e:
        logging.warning(e,exc_info=True)
        #break
    except KeyboardInterrupt:# TODO: this is not caught :(
        pass
    except myFtx.DoneDeal as e:
        logging.info(e)
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
        argv.extend(['sysperp'])
    if len(argv) < 3:
        argv.extend(['ftx', 'debug'])
    print(f'running {argv}')
    loop = asyncio.new_event_loop()
    if argv[0] in ['sysperp', 'flatten','unwind']:
        return loop.run_until_complete(ftx_ws_spread_main_wrapper(*argv,loop=loop))
    else:
        print(f'commands: sysperp, flatten,unwind')

if __name__ == "__main__":
    ftx_ws_spread_main(*sys.argv[1:])