import asyncio

import ccxt
import ccxtpro
import numpy as np
from ccxtpro.base.cache import ArrayCacheBySymbolById
import functools
import threading

from ftx_utilities import *
from ftx_portfolio import diff_portoflio,MarginCalculator
from ftx_history import fetch_trades_history
from ftx_ftx import fetch_latencyStats,fetch_futures

max_nb_coins = 50  # TODO: sharding needed
entry_tolerance = 0.5 # green light if basket better than median
edit_trigger_tolerance = np.sqrt(60/60) # chase on 1m stdev
stop_tolerance = np.sqrt(10) # stop on 10min stdev
time_budget = 30*60 # 30m. used in transaction speed screener
delta_limit = 0.2 # delta limit / pv
slice_factor = 0.25 # % of request
edit_price_tolerance=np.sqrt(10/60)#price on 10s std

class myFtx(ccxtpro.ftx):
    def __init__(self, config={}):
        super().__init__(config=config)
        #self.lock = threading.Lock()
        self.rest_semaphor = asyncio.Semaphore(safe_gather_limit)

        self.orders_lifecycle = dict()
        self.risk_reconciliations = []
        self.limit = myFtx.LimitBreached()

        self.exec_parameters = {}
        self.running_symbols = []

        self.risk_state = {}
        self.pv = None
        self.usd_balance = None # it's handy for marginal calcs
        self.margin_headroom = None
        self.margin_calculator = None
        self.calculated_IM = None

        #limit = self.safe_integer(self.options, 'ordersLimit', 1000)
        #self.orders = ArrayCacheBySymbolById(limit)
        #limit = self.safe_integer(self.options, 'tradesLimit', 1000)
        #self.myTrades = ArrayCacheBySymbolById(limit)
        self.latest_order_reconcile_timestamp = 0
        self.latest_fill_reconcile_timestamp = 0

    # --------------------------------------------------------------------------------------------
    # ---------------------------------- various helpers -----------------------------------------
    # --------------------------------------------------------------------------------------------

    class DoneDeal(Exception):
        def __init__(self,symbol):
            super().__init__('{} done'.format(symbol))
            self.symbol = symbol

    class LimitBreached(Exception):
        def __init__(self,limit=None,check_frequency=600):
            super().__init__()
            self.limit = limit
            self.check_frequency = check_frequency

    def loop_and_callback(func):
        @functools.wraps(func)
        async def wrapper_loop(*args, **kwargs):
            self=args[0]
            while True:
                try:
                    value = await func(*args, **kwargs)
                except myFtx.DoneDeal as e:
                    args[0].running_symbols.remove(e.symbol)
                    if len(args[0].running_symbols)>0: return
                    else: raise e
                except ccxt.NetworkError as e:
                    args[0].logger.debug(str(e))
                    args[0].logger.info(
                        '----------------------------- reconnecting ---------------------------------------')
                    continue
                except Exception as e:
                    self.logger.exception(e, exc_info=True)
                    raise e
        return wrapper_loop

    def synchronized(wrapped):
        lock = threading.RLock()

        @functools.wraps(wrapped)
        def _wrapper(*args, **kwargs):
            with lock:
                return wrapped(*args, **kwargs)

        return _wrapper

    def find_clientID_from_fill(self,fill):
        '''find order by id, even if still in flight
        all layers events must carry id if known !! '''
        try:
            found = next(clientID for clientID, events in self.orders_lifecycle.items() if
                         any('id' in event and event['id'] == fill['order'] for event in events))
        except StopIteration as e:# could still be in flight --> lookup
            try:
                found = next(clientID for clientID,events in self.orders_lifecycle.items() if
                     any(event['price'] == fill['price']
                     and event['amount'] == fill['amount']
                     and event['symbol'] == fill['symbol']
                     #and x['type'] == fill['type'] # sometimes None in fill
                     and event['side'] == fill['side'] for event in events))
            except StopIteration as e:
                raise Exception("fill {} not found".format(fill['symbol']))
        return found

    def mid(self,symbol):
        data = self.tickers[symbol] if symbol in self.tickers else self.markets[symbol]['info']
        return 0.5*(float(data['bid'])+float(data['ask']))

    def amount_to_precision(self, symbol, amount):
        market = self.market(symbol)
        return self.decimal_to_precision(amount, ccxt.ROUND, market['precision']['amount'], self.precisionMode, self.paddingMode)

    def build_logging(self):
        '''3 handlers: >=debug, ==info and >=warning'''
        class MyFilter(object):
            '''this is to restrict info logger to info only'''
            def __init__(self, level):
                self.__level = level

            def filter(self, logRecord):
                return logRecord.levelno <= self.__level

        # logs
        handler_warning = logging.FileHandler('Runtime/logs/warning.log', mode='w')
        handler_warning.setLevel(logging.WARNING)
        handler_warning.setFormatter(logging.Formatter(f"%(levelname)s: %(message)s"))
        self.logger.addHandler(handler_warning)

        handler_info = logging.FileHandler('Runtime/logs/info.log', mode='w')
        handler_info.setLevel(logging.INFO)
        handler_info.setFormatter(logging.Formatter(f"%(levelname)s: %(message)s"))
        handler_info.addFilter(MyFilter(logging.INFO))
        self.logger.addHandler(handler_info)

        handler_debug = logging.FileHandler('Runtime/logs/debug.log', mode='w')
        handler_debug.setLevel(logging.DEBUG)
        handler_debug.setFormatter(logging.Formatter(f"%(levelname)s: %(message)s"))
        self.logger.addHandler(handler_debug)

        handler_alert = logging.handlers.SMTPHandler(mailhost='smtp.google.com',
                                                     fromaddr='david@pronoia.link',
                                                     toaddrs=['david@pronoia.link'],
                                                     subject='auto alert',
                                                     credentials=('david@pronoia.link', ''),
                                                     secure=None)
        handler_alert.setLevel(logging.CRITICAL)
        handler_alert.setFormatter(logging.Formatter(f"%(levelname)s: %(message)s"))
        # self.logger.addHandler(handler_alert)

        self.logger.setLevel(logging.DEBUG)

    # --------------------------------------------------------------------------------------------
    # ---------------------------------- OMS             -----------------------------------------
    # --------------------------------------------------------------------------------------------
    # orders_lifecycle is a dictionary of blockchains, one per order intended.
    # each block has a lifecyle_state in ['pending_new','sent','pending_cancel','acknowledged','partial_fill','canceled','rejected','filled']
    # lifecycle_xxx do:
    # 1) resolve clientID
    # 2) validate block, notably maintaining orders_pending_new (which is orders_lifecycle[x][-1]['lifecycle_state']=='pending_new')
    # 3) build a new block from messages received
    # 4) mines it to the blockchain
    #
    all_states = set(['pending_new','sent','pending_cancel','pending_replace','acknowledged','partial_fill'])
    open_states = set(['pending_new','sent','pending_cancel','pending_replace','acknowledged','partial_fill'])
    editable_states = set(['acknowledged','partial_fill'])
    cancelable_states = set(['sent','acknowledged','partial_fill'])
    # --------------------------------------------------------------------------------------------

    def pending_new_histories(self, coin):
        symbols = [symbol
                   for symbol in self.exec_parameters[coin].keys()
                   if symbol in self.markets]
        return [{clientID:data}
                for clientID,data in self.orders_lifecycle.items()
                if data[0]['symbol'] in symbols
                and data[-1]['lifecycle_state'] == 'pending_new']

    def open_order_histories(self,symbol):
        '''returns blockchain which current state open_order_histories not filled or cancelled'''
        try:
            return [data
                for data in self.orders_lifecycle.values()
                if data[0]['symbol'] == symbol
                and data[-1]['lifecycle_state'] in self.open_states]
        except:
            pass

    def lifecycle_pending_new(self, order_event):
        '''self.orders_lifecycle = {clientId:[{key:data}]}'''
        #1) resolve clientID
        clientOrderId = order_event['clientOrderId']
        
        #2) validate block
        pass
        
        #3) make new block
        
        ## order details
        symbol = order_event['symbol']
        coin = self.markets[symbol]['base']

        nowtime = order_event['timestamp'] - self.exec_parameters['timestamp']
        eventID = clientOrderId + '_' + str(nowtime)
        current = {'eventID': eventID,
                   'lifecycle_state': 'pending_new',
                   #'timestamp': nowtime, already in order_event 
                   'id': None} | order_event
        
        ## risk details
        risk_data = self.risk_state[coin]
        current |= {'risk_timestamp':risk_data[symbol]['delta_timestamp']-self.exec_parameters['timestamp'],
                       'delta':risk_data[symbol]['delta'],
                       'netDelta': risk_data['netDelta'],
                       'pv(wrong timestamp)':self.pv,
                       'margin_headroom':self.margin_headroom,
                       'IM_discrepancy':self.calculated_IM - self.margin_headroom}

        ## mkt details
        if symbol in self.tickers:
            mkt_data = self.tickers[symbol]
            timestamp = mkt_data['timestamp']
        else:
            mkt_data = self.markets[symbol]['info']|{'bidVolume':0,'askVolume':0}#TODO: should have all risk group
            timestamp = self.exec_parameters['timestamp']
        current |= {'mkt_timestamp': timestamp - self.exec_parameters['timestamp']}\
                        | {key: mkt_data[key] for key in ['bid', 'bidVolume', 'ask', 'askVolume']}

        #4) mine genesis block
        self.orders_lifecycle[clientOrderId] = [current]

    def lifecycle_sent(self, order_event):
        # 1) resolve clientID
        clientOrderId = order_event['clientOrderId']

        # 2) validate block
        past = self.orders_lifecycle[clientOrderId][-1]
        if past['lifecycle_state'] not in ['pending_new','acknowledged','partial_fill']:
            self.logger.warning('order {} was {}'.format(past['clientOrderId'],past['lifecycle_state']))
            return

        # 3) new block
        nowtime = int(self.milliseconds() - self.exec_parameters['timestamp'])
        eventID = clientOrderId + '_' + str(nowtime)
        current = {'eventID':eventID,
                   'lifecycle_state':'sent',
                   'timestamp': nowtime,
                   'id': order_event['id'],
                   'remaining': order_event['remaining'] if order_event['side'] =='buy' else -order_event['remaining']}

        if order_event['status'] == 'closed':
            current['lifecycle_state'] = 'rejected'

        #4) mine
        self.orders_lifecycle[clientOrderId] += [current]

    def lifecycle_pending_cancel(self, clientOrderId, result):
        # 1) resolve clientID
        clientOrderId = clientOrderId

        # 2) validate block
        past = self.orders_lifecycle[clientOrderId][-1]
        if past['lifecycle_state'] not in self.open_states:
            self.logger.warning('order {} re-canceled'.format(past['clientOrderId']))
            return

        # 3) new block
        nowtime = int(self.milliseconds() - self.exec_parameters['timestamp'])
        eventID = clientOrderId + '_' + str(nowtime)
        current = result | {'eventID': eventID,
                   'lifecycle_state': 'pending_cancel',
                   'timestamp': nowtime}

        # 4) mine
        self.orders_lifecycle[clientOrderId] += [current]

    def lifecycle_ackowledgment(self,order_event,trigger):
        # 1) resolve clientID
        clientOrderId = order_event['clientOrderId']

        # 2) validate block...is this needed?
        past = self.orders_lifecycle[clientOrderId][-1]
        if past['lifecycle_state'] in (self.open_states - set(['acknowledged','partial_fill'])):
            self.logger.warning('order {} re-ackowledged'.format(clientOrderId))
            return

        # 3) new block
        nowtime = int(self.milliseconds() - self.exec_parameters['timestamp'])
        eventID = clientOrderId + '_' + str(nowtime)
        current = {'eventID': eventID,
                   'lifecycle_state': 'pending_cancel',
                   'sent_timestamp': nowtime,
                   'id': order_event['id'],
                   'trigger': trigger}

        if order_event['status'] in ['new', 'open', 'triggered']:
            current['lifecycle_state'] = 'acknowledged'
        elif order_event['status'] in ['closed']:
            #TODO: not obviously rejected
            current['lifecycle_state'] = 'filled' if order_event['remaining'] == 0 else 'rejected'
        elif order_event['status'] in ['canceled']:
            current['lifecycle_state'] = 'canceled'
        else:
            raise Exception('unknown status{}'.format(order_event['status']))

        # 4) mine
        self.orders_lifecycle[clientOrderId] += [current]

    def lifecycle_fill(self,fill,trigger):
        # 1) resolve clientID
        clientOrderId = self.find_clientID_from_fill(fill)

        # 2) validate block
        past = self.orders_lifecycle[clientOrderId][-1]
        if past['lifecycle_state'] not in self.open_states:
            self.logger.warning('order {} re-filled'.format(past['clientOrderId']))
            return

        # 3) new block
        nowtime = fill['timestamp'] - self.exec_parameters['timestamp']
        eventID = clientOrderId + '_' + str(nowtime)
        fill.pop('id') # would conflict with fill['order']
        current = past
        for key,data in {'eventID': eventID,
                   'timestamp': nowtime,
                   'trigger': trigger,
                   'remaining': past['remaining']-fill['amount']}.items():
            current[key] = data
        assert (past['remaining']-current['remaining'] == fill['amount'])

        symbol = fill['symbol']
        coin = self.markets[symbol]['base']
        if current['remaining'] < self.exec_parameters[coin][symbol]['sizeIncrement']/2:
            current['lifecycle_state'] = 'filled'
            prices = np.array([event['price'] for event in self.orders_lifecycle[clientOrderId] if 'fill_timestamp' in event])
            amounts = np.array([event['amount'] for event in self.orders_lifecycle[clientOrderId] if 'fill_timestamp' in event])
            current |= {'avgPrice': np.dot(prices,amounts)/sum(amounts)}
        else:
            current['lifecycle_state'] = 'partial_fill'

        # 4) mine
        self.orders_lifecycle[clientOrderId] += [current]

    #@synchronized
    async def lifecycle_to_json(self,filename = 'Runtime/logs/latest_events.json'):
        async with aiofiles.open(filename,mode='w') as file:
            await file.write(json.dumps(self.orders_lifecycle, cls=NpEncoder))
        shutil.copy2(filename, 'Runtime/logs/'+datetime.fromtimestamp(self.exec_parameters['timestamp']/1000).strftime("%Y-%m-%d-%H-%M")+'_events.json')

    #@synchronized
    async def risk_reconciliation_to_json(self,filename = 'Runtime/logs/latest_risk_reconciliations.json'):
        async with aiofiles.open(filename,mode='w') as file:
            await file.write(json.dumps(self.risk_reconciliations, cls=NpEncoder))
        shutil.copy2(filename, 'Runtime/logs/'+datetime.fromtimestamp(self.exec_parameters['timestamp']/1000).strftime("%Y-%m-%d-%H-%M")+'_risk_reconciliations.json')

    #@synchronized
    async def reconcile_fills(self):
        '''fetch fills, to recover missed messages'''
        self.latest_fill_reconcile_timestamp = self.milliseconds()
        fetched_fills = await self.fetch_my_trades(since=max(self.exec_parameters['timestamp'],self.latest_fill_reconcile_timestamp-1000), limit=1000)
        for fill in fetched_fills:
            self.myTrades.append(fill)
            self.lifecycle_fill(fill,trigger='reconciled')
        # hopefully using another lock instance
        await self.lifecycle_to_json()

    # @synchronized
    async def reconcile_orders(self):
        '''fetch orders, recover missed messages and discard other pending_new'''
        fetched_orders = await self.fetch_orders(since=max(self.exec_parameters['timestamp'],self.latest_order_reconcile_timestamp)-1000, limit=1000)
        self.latest_order_reconcile_timestamp = self.milliseconds()

        # forcefully acknowledge
        for order in fetched_orders:
            #self.orders.append(order)
            past = self.orders_lifecycle[order['clientOrderId']][-1]
            current = {'trigger':'reconciled'}
            if order['status'] == 'open' and (past['lifecycle_state'] not in ['acknowledged']):
                self.lifecycle_ackowledgment(order,trigger='reconciled')
            elif order['status'] == 'closed' and order['remaining'] == 0 and past['lifecycle_state'] not in ['filled']:
                current['lifecycle_state'] = 'filled'
                current['trigger'] = 'reconciled'
            elif ((order['status'] == 'canceled') or (order['status'] == 'closed' and order['remaining'] != 0)) and past['lifecycle_state'] not in ['canceled','rejected']:
                self.lifecycle_cancel(order['clientOrderId'],trigger='reconciled')
            else:
                pass

        # hopefully using another lock instance
        await self.lifecycle_to_json()

    # ---------------------------------------------------------------------------------------------
    # ---------------------------------- PMS -----------------------------------------
    # ---------------------------------------------------------------------------------------------

    async def build_state(self, weights,
                          entry_tolerance = entry_tolerance,
                          edit_trigger_tolerance = edit_trigger_tolerance, # chase on 30s stdev
                          edit_price_tolerance = edit_price_tolerance,
                          stop_tolerance = stop_tolerance, # stop on 5min stdev
                          time_budget = time_budget,
                          delta_limit = delta_limit, # delta limit / pv
                          slice_factor = slice_factor): # cut in 10):
        '''initialize all state and does some filtering (weeds out slow underlyings; should be in strategy)
            target_sub_portfolios = {coin:{entry_level_increment,
            symbol1:{'spot_price','diff','target'}]}]'''
        self.build_logging()

        self.limit.delta_limit = delta_limit

        frequency = timedelta(minutes=1)
        end = datetime.now()
        start = end - timedelta(hours=1)

        trades_history_list = await safe_gather([fetch_trades_history(
            self.market(symbol)['id'], self, start, end, frequency=frequency)
            for symbol in weights['name']],semaphore=self.rest_semaphor)

        weights['diffCoin'] = weights['optimalCoin'] - weights['currentCoin']
        weights['diffUSD'] = weights['diffCoin'] * weights['spot_price']
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
                            for symbol, data in coin_data.items() if np.abs(data['diff']) >= max(diff_threshold/data['spot_price'], float(self.markets[symbol]['info']['minProvideSize']))}
                     for coin, coin_data in full_dict.items() if
                     all(data['volume'] * time_budget > np.abs(data['diff']) for data in coin_data.values())
                     and any(np.abs(data['diff']) >= max(diff_threshold/data['spot_price'], float(self.markets[symbol]['info']['minProvideSize'])) for symbol, data in coin_data.items())}
        if data_dict =={}:
            self.logger.info('nothing to do')
            raise myFtx.DoneDeal('all')

        def basket_vwap_quantile(series_list,diff_list,quantile):
            series = pd.concat(series_list,axis=1).dropna(axis=0)-diff_list
            return series.sum(axis=1).diff().quantile(quantile)
        def z_score(series,z_score):# stdev of 1min prices * quantile
            series = series.dropna(axis=0)
            return series.std().values[0]*z_score

        # get times series of target baskets, compute quantile of increments and add to last price
        # remove series
        self.exec_parameters = {'timestamp':end.timestamp()*1000} \
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
                                             sys.intern('spot'): self.mid(symbol)
                                         }
                                         for symbol, data in coin_data.items()}
                                 for coin, coin_data in data_dict.items()}

        self.risk_state = {sys.intern(coin):
                               {sys.intern('netDelta'):0}
                               | {sys.intern(symbol):
                                   {
                                       sys.intern('delta'): 0,
                                       sys.intern('delta_timestamp'):end.timestamp()*1000,
                                       sys.intern('delta_id'): None
                                   }
                                   for symbol, data in coin_data.items()}
                           for coin, coin_data in data_dict.items()}

        futures = pd.DataFrame(await fetch_futures(self))
        account_leverage = float(futures.iloc[0]['account_leverage'])
        collateralWeight = futures.set_index('underlying')['collateralWeight'].to_dict()
        imfFactor = futures.set_index('new_symbol')['imfFactor'].to_dict()
        self.margin_calculator = MarginCalculator(account_leverage, collateralWeight, imfFactor)

        # populates risk, pv and IM
        self.latest_order_reconcile_timestamp = self.exec_parameters['timestamp']
        self.latest_fill_reconcile_timestamp = self.exec_parameters['timestamp']
        await self.reconcile_orders()
        await self.reconcile_risk()

        vwap_dataframe = pd.concat([data['vwap'].filter(like='vwap').fillna(method='ffill') for data in trades_history_list],axis=1,join='outer').fillna(method='bfill')
        vwap_dataframe=vwap_dataframe.apply(np.log).diff()
        size_dataframe = pd.concat([data['vwap'].filter(like='volume').fillna(method='ffill') for data in trades_history_list], axis=1, join='outer').fillna(method='bfill')
        to_parquet(pd.concat([vwap_dataframe,size_dataframe], axis=1, join='outer'),'Runtime/logs/latest_minutely.parquet')
        shutil.copy2('Runtime/logs/latest_minutely.parquet',
                     'Runtime/logs/' + datetime.fromtimestamp(self.exec_parameters['timestamp'] / 1000).strftime(
                         "%Y-%m-%d-%Hh") + '_minutely.parquet')

        with open('Runtime/logs/latest_request.json', 'w') as file:
            json.dump({symbol:data
                        for coin,coin_data in self.exec_parameters.items() if coin in self.currencies
                        for symbol,data in coin_data.items() if symbol in self.markets}, file, cls=NpEncoder)
        shutil.copy2('Runtime/logs/latest_request.json','Runtime/logs/' + datetime.fromtimestamp(self.exec_parameters['timestamp'] / 1000).strftime(
                         "%Y-%m-%d-%Hh") + '_request.json')

    async def reconcile_risk(self):
        '''update risk using rest
        all symbols not present when state is built are ignored !
        if some tickers are not initialized, it just uses markets'''
        risks = await safe_gather([self.fetch_positions(),self.fetch_balance(),self.reconcile_fills()],semaphore=self.rest_semaphor)
        positions = risks[0]
        balances = risks[1]
        risk_timestamp = self.milliseconds()

        # delta is noisy for perps, so override to delta 1.
        for position in positions:
            if float(position['info']['size'])!=0:
                coin = self.markets[position['symbol']]['base']
                if coin in self.risk_state and position['symbol'] in self.risk_state[coin]:
                    self.risk_state[coin][position['symbol']]['delta'] = position['notional']*(1 if position['side'] == 'long' else -1) \
                        if self.markets[position['symbol']]['type'] == 'future' \
                        else float(position['info']['size']) * self.mid(position['symbol'])*(1 if position['side'] == 'long' else -1)
                    self.risk_state[coin][position['symbol']]['delta_timestamp']=risk_timestamp

        for coin, balance in balances.items():
            if coin in self.currencies.keys() and coin != 'USD' and balance['total']!=0 and coin in self.risk_state and coin+'/USD' in self.risk_state[coin]:
                symbol = coin+'/USD'
                self.risk_state[coin][symbol]['delta'] = balance['total'] * self.mid(symbol)
                self.risk_state[coin][symbol]['delta_timestamp'] = risk_timestamp

        for coin,coin_data in self.risk_state.items():
            coin_data['netDelta']= sum([data['delta'] for symbol,data in coin_data.items() if symbol in self.markets and 'delta' in data.keys()])

        # update pv and usd balance
        self.usd_balance = balances['USD']['total']
        self.pv = sum(coin_data[coin+'/USD']['delta'] for coin, coin_data in self.risk_state.items() if coin+'/USD' in coin_data.keys()) + self.usd_balance

        #compute IM
        spot_weight={}
        future_weight={}
        for position in positions:
            if float(position['info']['netSize'])!=0:
                mid = self.mid(position['symbol'])
                future_weight |= {position['symbol']: {'weight': float(position['info']['netSize'])*mid, 'mark':mid}}
        for coin,balance in balances.items():
            if coin!='USD' and coin in self.currencies and balance['total']!=0:
                mid = self.mid(coin+'/USD')
                spot_weight |= {(coin):{'weight':balance['total']*mid,'mark':mid}}
        (spot_weight,future_weight) = MarginCalculator.add_pending_orders(self, spot_weight, future_weight)

        self.margin_calculator.estimate(self.usd_balance, spot_weight, future_weight)
        self.calculated_IM = sum(value for value in self.margin_calculator.estimated_IM.values())
        # fetch IM
        account_info = (await self.privateGetAccount())['result']
        self.margin_calculator.actual(account_info) if float(account_info['totalPositionSize']) else self.pv
        self.margin_headroom = self.margin_calculator.actual_IM

        # log risk
        if True:
            self.risk_reconciliations += [{'lifecycle_state': 'remote_risk', 'symbol':symbol_, 'timestamp':self.milliseconds()}
                                      for coin_,coin_data in self.risk_state.items()
                                      for symbol_ in coin_data.keys() if symbol_ in self.markets]

    # --------------------------------------------------------------------------------------------
    # ---------------------------------- WS loops             -----------------------------------------
    # --------------------------------------------------------------------------------------------

    @loop_and_callback
    async def monitor_fills(self):
        fills = await self.watch_my_trades()#limit=1)
        return

    def handle_my_trade(self, client, message):
        '''maintains risk_state, event_records, logger.info
        await self.reconcile_state() is safer but slower. we have monitor_risk to reconcile'''
        super().handle_my_trade(client, message)
        data = self.safe_value(message, 'data')
        fill = self.parse_trade(data)
        symbol = fill['symbol']
        coin = self.markets[symbol]['base']

        # update risk_state
        data = self.risk_state[coin][symbol]
        fill_size = fill['amount'] * (1 if fill['side'] == 'buy' else -1) * fill['price']
        data['delta'] += fill_size
        data['delta_timestamp'] = fill['timestamp']
        latest_delta = data['delta_id']
        data['delta_id'] = max(latest_delta or 0, int(fill['order']))
        self.risk_state[coin]['netDelta'] += fill_size

        # log event
        self.lifecycle_fill(fill)

        # logger.info
        self.logger.info('{} fill at {}: {} {} {} at {}'.format(symbol,
                                                                fill['timestamp'] - self.exec_parameters['timestamp'],
                                                                fill['side'], fill['amount'], symbol, fill['price']))

        current = self.risk_state[coin][symbol]['delta']
        initial = self.exec_parameters[coin][symbol]['target'] * fill['price'] - self.exec_parameters[coin][symbol][
            'diff'] * fill['price']
        target = self.exec_parameters[coin][symbol]['target'] * fill['price']
        self.logger.info('{} risk at {} ms: {}% done [current {}, initial {}, target {}]'.format(
            symbol,
            self.risk_state[coin][symbol]['delta_timestamp'] - self.exec_parameters['timestamp'],
            (current - initial) / (target - initial) * 100,
            current,
            initial,
            target))

    @loop_and_callback
    async def monitor_risk(self):
        '''redundant minutely risk check'''#TODO: would be cool if this could interupt other threads and restart it when margin is ok.
        await self.reconcile_risk()

        self.limit.limit = self.pv * self.limit.delta_limit
        absolute_risk = sum(abs(data['netDelta']) for data in self.risk_state.values())
        if absolute_risk > self.limit.limit:
            self.logger.warning(f'absolute_risk {absolute_risk} > {self.limit.limit}')
        if self.margin_headroom < self.pv/100:
            self.logger.warning(f'IM {self.margin_headroom}  < 1%')

        await asyncio.sleep(self.limit.check_frequency)

    @loop_and_callback
    async def monitor_orders(self):
        orders = await self.watch_orders()

    def handle_order(self, client, message):
        '''maintains orders, pending_new, event_records'''
        super().handle_order(client, message)
        data = self.safe_value(message, 'data')
        order = self.parse_order(data)
        assert order['clientOrderId']

        self.lifecycle_ackowledgment(order) # status new, triggered, open or canceled

    # --------------------------------------------------------------------------------------------
    # ---------------------------------- order placement -----------------------------------------
    # --------------------------------------------------------------------------------------------

    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        '''if ackowledged, place order. otherwise just reconcile
        orders_pending_new is blocking'''
        coin = self.markets[symbol]['base']
        if self.pending_new_histories(coin):#TODO: rather incorporate orders_pending_new in risk, rather than block
            self.logger.info('orders {} still in flight. holding off {}'.format(
                [order.keys() for order in self.pending_new_histories(coin)],
                params['clientOrderId']))
            await self.reconcile_orders()
        else:
            # set pending_new -> send rest -> if success, leave pending_new and give id. Pls note it may have been caught by handle_order by then.
            self.lifecycle_pending_new({'clientOrderId': params['clientOrderId'],
                                      'symbol': symbol,
                                      'type': type,
                                      'side': side,
                                      'amount': amount,
                                      'remaining': amount if side =='buy' else -amount,
                                      'price': price,
                                      'timestamp': int(float(params['clientOrderId'].split('_')[2]))})
            order = await super().create_order(symbol, type, side, amount, price, params)
            self.lifecycle_sent(order)

    async def cancel_order(self, symbol, params):
        '''set in flight, send cancel, set as pending cancel, set as cancelled or insist'''
        coin = self.markets[symbol]['base']
        self.pending_new_histories[coin] = (self.pending_new_histories[coin] if coin in self.pending_new_histories else []) + [params['clientOrderId']]

        result = await super().cancel_order(None, symbol,params)

        if result == "Order queued for cancellation":
            self.lifecycle_pending_cancel(params['clientOrderId'], result)
        else: # could be in flight, or unknown
            self.logger.warning('cancel for {} fails with {}'.format(params['clientOrderId'],result))
            await asyncio.sleep(.1)
            await self.cancel_order(symbol, params)

    async def peg_or_stopout(self,symbol,size,orderbook_depth,edit_trigger_depth,edit_price_depth,stop_depth=None):
        '''creates of edit orders, pegging to orderbook
        size in coin, already filtered
        skips if any pending_new, cancels duplicates, add any cancellations to event_records
        extensive exception handling
        '''
        coin = self.markets[symbol]['base']

        #TODO: https://help.ftx.com/hc/en-us/articles/360052595091-Ratelimits-on-FTX

        price = self.tickers[symbol]['ask' if size<0 else 'bid']
        opposite_side = self.tickers[symbol]['ask' if size>0 else 'bid']
        mid = self.tickers[symbol]['mid']

        priceIncrement = self.exec_parameters[coin][symbol]['priceIncrement']
        sizeIncrement = self.exec_parameters[coin][symbol]['sizeIncrement']

        if stop_depth:
            stop_trigger = float(self.price_to_precision(symbol,stop_depth))
        edit_trigger = float(self.price_to_precision(symbol,edit_trigger_depth))
        edit_price = float(self.price_to_precision(symbol,opposite_side - (1 if size>0 else -1)*max(priceIncrement*1.1,edit_price_depth)))

        try:
            # look for open orders on the symbol. if dupes, remove latest.
            event_histories = self.open_order_histories(symbol)
            if len(event_histories) > 1:
                first_pending_new = np.argmin(np.array([data[0]['timestamp'] for data in event_histories]))
                for i,order in event_histories.enumerate():
                    if i == first_pending_new: continue
                    await self.cancel_order(symbol,params={'clientOrderId':order['clientOrderId'],'trigger':'duplicates'})
                    self.logger.warning('canceled duplicate {} order {}'.format(symbol,order['clientOrderId']))
                order = event_histories[first_pending_new][-1]

            # if none, create genesis
            if len(event_histories)==0:
                clientOrderId = 'new_{}_{}'.format(symbol,str(self.milliseconds()-self.exec_parameters['timestamp']))
                order = await self.create_order(symbol, 'limit', 'buy' if size>0 else 'sell', np.abs(size), price=edit_price,
                                                params={'postOnly': True, 'clientOrderId':clientOrderId})
            # if only one, stopout or peg or wait
            else:
                order = event_histories[-1]
                if set('side','price','remaining','clientOrderId').issubset(set(order.keys())):
                    order_distance = (1 if order['side']=='buy' else -1)*(opposite_side-order['price'])
                # panic stop. we could rather place a trailing stop: more robust to latency, but less generic.

                if stop_depth \
                        and order['lifecycle_state'] in self.cancelable_states\
                        and order_distance>stop_trigger \
                        and order['remaining']>sizeIncrement:

                    await self.cancel_order(symbol=symbol, params={'clientOrderId': order['clientOrderId'], 'trigger': False})
                    clientMktOrderId = 'stop_{}_{}'.format(symbol,str(self.milliseconds() - self.exec_parameters['timestamp']))
                    order = await self.create_order(symbol, 'market', 'buy' if size>0 else 'sell',order['remaining'],
                                            params={'clientOrderId':clientMktOrderId})
                # peg limit order
                elif (order_distance>edit_trigger) \
                        and order['lifecycle_state'] in self.editable_states \
                        and (np.abs(edit_price-order['price'])>=priceIncrement) \
                        and (order['remaining']>sizeIncrement):
                    await self.cancel_order(symbol=symbol, params={'clientOrderId': order['clientOrderId'],'trigger':False})
                    clientOrderId = 'chase_{}_{}'.format(symbol,str(self.milliseconds()-self.exec_parameters['timestamp']))
                    order = await self.create_order(symbol, 'limit', 'buy' if size > 0 else 'sell', np.abs(size),price=edit_price,
                                            params={'postOnly': True, 'clientOrderId': clientOrderId})
                #wait
                else:
                    pass

        ### see error_hierarchy in DerivativeArbitrage/venv/Lib/site-packages/ccxt/base/errors.py
        except ccxt.InsufficientFunds as e: # is ExchangeError
            cost = self.margin_calculator.margin_cost(coin, mid, size, self.usd_balance)
            self.logger.warning(f'marginal cost {cost}, vs margin_headroom {self.margin_headroom} and calculated_IM {self.calculated_IM}')
        except ccxt.InvalidOrder as e: # is ExchangeError
            if "Order already queued for cancellation" in str(e):
                self.logger.warning(str(e) + str(order))
            elif ("Order already closed" in str(e)) or ("Order not found" in str(e)):
                self.logger.warning(str(e) + str(order))
            elif ("Size too small for provide" or "Size too small") in str(e):
                # usually because filled btw remaining checked and order modify
                self.logger.warning('{}: {} too small {}...or {} < {}'.format(order,np.abs(size),sizeIncrement,clientOrderId.split('_')[2],'order[timestamp]'))
            else:
                self.logger.warning(str(e) + str(order))
        except ccxt.ExchangeError as e:  # is base error
            if "Must modify either price or size" in str(e):
                self.logger.warning(str(e) + str(order))
            else:
                self.logger.warning(str(e), exc_info=True)
        except ccxt.NetworkError as e:
            self.logger.warning(str(e))
        except Exception as e:
            self.logger.exception('{} {} {} at {} raised {}'.format('buy' if size > 0 else 'sell', np.abs(size), symbol, price, e))
            raise e

    async def watch_ticker(self, symbol, params={}):
        '''watch_order_book is faster than watch_tickers so we DON'T LISTEN TO TICKERS. Dirty...'''
        raise Exception("watch_order_book is faster than watch_tickers so we DON'T LISTEN TO TICKERS. Dirty...")

    @loop_and_callback
    async def order_master(self,symbol):
        '''on each top of book update, update market_state and send orders
        tunes aggressiveness according to risk.
        populates event_records, maintains pending_new'''
        coin = self.markets[symbol]['base']

        # watch_order_book is faster than watch_tickers so we DON'T LISTEN TO TICKERS. Dirty...
        order_book = await self.watch_order_book(symbol)
        self.tickers[symbol]={'symbol':order_book['symbol'],
                              'timestamp':order_book['timestamp'],
                              'bid':order_book['bids'][0][0],
                              'ask':order_book['asks'][0][0],
                              'mid':0.5*(order_book['bids'][0][0]+order_book['asks'][0][0]),
                              'bidVolume':order_book['bids'][0][1],
                              'askVolume':order_book['asks'][0][1]}
        mid = self.tickers[symbol]['mid']

        params = self.exec_parameters[coin][symbol]

        # in case fills arrived after risk updates
        #     latest_delta_refresh = {x:data['delta_id'] for x,data in self.risk_state[coin].items() if x in self.markets and data['delta_id']}
        #     coin_fills = await safe_gather([self.fetch_my_trades(symbol=x,params={'minId':delta_id + 1})
        #                                         for x,delta_id in latest_delta_refresh.items()])
        #     for symbol_fills in coin_fills:
        #         for fill in symbol_fills:
        #             if int(fill['id'])>max([delta_id for delta_id in latest_delta_refresh.values()]):
        #                 self.process_fill(fill, coin, symbol)
        #                 self.logger.warning('{}:caught fill {} {} in {} loop'.format(
        #                     fill['timestamp']/1000-self.exec_parameters['timestamp'],
        #                     fill['amount']*(1 if fill['side']=='buy' else -1),
        #                     fill['symbol'],
        #                     symbol))

        #risk
        delta = self.risk_state[coin][symbol]['delta']/mid
        delta_timestamp = self.risk_state[coin][symbol]['delta_timestamp']
        netDelta = self.risk_state[coin]['netDelta']/mid

        # size to do: aim at target, slice, round to sizeIncrement
        size = params['target'] - delta
        # size < slice_size and margin_headroom
        size = np.sign(size)*float(self.amount_to_precision(symbol, min([np.abs(size), params['slice_size']])))
        if (np.abs(size) < self.exec_parameters[coin][symbol]['sizeIncrement']/2):
            self.logger.info(f'{symbol} done')
            raise myFtx.DoneDeal(symbol)
        # if not enough margin, hold it# TODO: this may be slow, and depends on orders anyway
        #if self.margin_calculator.margin_cost(coin, mid, size, self.usd_balance) > self.margin_headroom:
        #    await asyncio.sleep(60) # TODO: I don't know how to stop/restart a thread..
        #    self.logger.info('margin {} too small for order size {}'.format(size*mid, self.margin_headroom))
        #    return None

        # if increases risk, go passive
        if np.abs(netDelta+size)-np.abs(netDelta)>0:
            if self.exec_parameters[coin]['entry_level'] is None: # for a purely risk reducing exercise
                self.logger.info(f'dont do {symbol}, would increase risk')
                raise myFtx.DoneDeal(symbol)
            # set limit at target quantile
            #current_basket_price = sum(self.mid(symbol)*self.exec_parameters[coin][symbol]['diff']
            #                           for symbol in self.exec_parameters[coin].keys() if symbol in self.markets)
            #edit_price_depth = max([0,(current_basket_price-self.exec_parameters[coin]['entry_level'])/params['diff']])#TODO: sloppy logic assuming no correlation
            edit_price_depth = (np.abs(netDelta+size)-np.abs(netDelta))/np.abs(size)*params['edit_price_depth'] # equate: profit if done ~ marginal risk * stdev
            edit_trigger_depth=params['edit_trigger_depth']
            await self.peg_or_stopout(symbol,size,orderbook_depth=0,edit_trigger_depth=edit_trigger_depth,edit_price_depth=edit_price_depth,stop_depth=None)
        # if decrease risk, go aggressive
        else:
            edit_trigger_depth=params['edit_trigger_depth']
            edit_price_depth=params['edit_price_depth']
            stop_depth=params['stop_depth']
            await self.peg_or_stopout(symbol,size,orderbook_depth=0,edit_trigger_depth=edit_trigger_depth,edit_price_depth=edit_price_depth,stop_depth=stop_depth)

async def ftx_ws_spread_main_wrapper(*argv,**kwargs):
    allDone = False
    while not allDone:
        try:
            exchange = myFtx({
                'asyncioLoop': kwargs['loop'] if 'loop' in kwargs else None,
                'newUpdates': True,
                # 'watchOrderBookLimit': 1000,
                'enableRateLimit': True,
                'apiKey': apiKey,
                'secret': secret}) if argv[1] == 'ftx' else None
            exchange.verbose = False
            exchange.headers = {'FTX-SUBACCOUNT': argv[2]}
            exchange.authenticate()
            await exchange.load_markets()

            if argv[0]=='sysperp':
                future_weights = pd.read_excel('Runtime/ApprovedRuns/current_weights.xlsx')
                target_portfolio = await diff_portoflio(exchange, future_weights)
                if target_portfolio.empty: return
                #selected_coins = ['REN']#target_portfolios.sort_values(by='USDdiff', key=np.abs, ascending=False).iloc[2]['underlying']
                #target_portfolio=diff[diff['coin'].isin(selected_coins)]
                await exchange.build_state(target_portfolio,
                                           entry_tolerance=entry_tolerance,
                                           edit_trigger_tolerance = edit_trigger_tolerance,
                                           edit_price_tolerance = edit_price_tolerance,
                                           stop_tolerance = stop_tolerance,
                                           time_budget = time_budget,
                                           delta_limit = delta_limit,
                                           slice_factor = slice_factor)

            elif argv[0]=='spread':
                coin=argv[3]
                cash_name = coin+'/USD'
                future_name = coin + '-PERP'
                cash_price = float(exchange.market(cash_name)['info']['price'])
                future_price = float(exchange.market(future_name)['info']['price'])
                target_portfolio = pd.DataFrame(columns=['coin','name','optimalCoin','currentCoin','spot_price'],data=[
                    [coin,cash_name,float(argv[4])/cash_price,0,cash_price],
                    [coin,future_name,-float(argv[4])/future_price,0,future_price]])
                if target_portfolio.empty: return

                await exchange.build_state(target_portfolio,
                                           entry_tolerance=0.99,
                                           edit_trigger_tolerance = np.sqrt(5),
                                           edit_price_tolerance = 0,
                                           stop_tolerance = np.sqrt(30),
                                           time_budget = 999,
                                           delta_limit = delta_limit,
                                           slice_factor = 0.5)

            elif argv[0]=='flatten': # only works for basket with 2 symbols
                future_weights = pd.DataFrame(columns=['name','optimalWeight'])
                diff = await diff_portoflio(exchange, future_weights)
                smallest_risk = diff.groupby(by='coin')['currentCoin'].agg(lambda series: series.apply(np.abs).min() if series.shape[0]>1 else 0)
                target_portfolio=diff
                target_portfolio['optimalCoin'] = diff.apply(lambda f: smallest_risk[f['coin']]*np.sign(f['currentCoin']),axis=1)
                if target_portfolio.empty: return

                await exchange.build_state(target_portfolio,
                                           entry_tolerance=entry_tolerance,
                                           edit_trigger_tolerance = edit_trigger_tolerance,
                                           edit_price_tolerance = edit_price_tolerance,
                                           stop_tolerance = stop_tolerance,
                                           time_budget = time_budget,
                                           delta_limit = delta_limit,
                                           slice_factor = slice_factor)

            elif argv[0]=='unwind':
                future_weights = pd.DataFrame(columns=['name','optimalWeight'])
                target_portfolio = await diff_portoflio(exchange, future_weights)
                if target_portfolio.empty: return

                await exchange.build_state(target_portfolio,
                                           entry_tolerance=.99,
                                           edit_trigger_tolerance=edit_trigger_tolerance,
                                           edit_price_tolerance=edit_price_tolerance,
                                           stop_tolerance=stop_tolerance,
                                           time_budget=999999,
                                           delta_limit=delta_limit,
                                           slice_factor=slice_factor)

            else:
                exchange.logger.exception(f'unknown command {argv[0]}',exc_info=True)
                raise Exception(f'unknown command {argv[0]}',exc_info=True)

            exchange.running_symbols = [symbol
                                        for coin_data in exchange.risk_state.values()
                                        for symbol in coin_data.keys() if symbol in exchange.markets]
            await safe_gather([exchange.monitor_fills(),exchange.monitor_risk(),exchange.monitor_orders()]+
                                   [exchange.order_master(symbol)
                                    for symbol in exchange.running_symbols],semaphore=exchange.rest_semaphor)

        except myFtx.DoneDeal as e:
            allDone = True
        except Exception as e:
            exchange.logger.exception(e,exc_info=True)
            raise e
        else:
            stats = await fetch_latencyStats(exchange, days=1, subaccount_nickname='SysPerp')
            exchange.logger.info(f'latencystats:{stats}')
        finally:
            await exchange.cancel_all_orders()
            await exchange.close()
            exchange.logger.info('exchange closed')

    return

def ftx_ws_spread_main(*argv):
    argv=list(argv)
    if len(argv) == 0:
        argv.extend(['sysperp'])
    if len(argv) < 3:
        argv.extend(['ftx', 'SysPerp'])
    logging.info(f'running {argv}')
    loop = asyncio.new_event_loop()
    if argv[0] in ['sysperp', 'flatten','unwind','spread']:
        return loop.run_until_complete(ftx_ws_spread_main_wrapper(*argv,loop=loop))
    else:
        logging.info(f'commands: sysperp [ftx][debug], flatten [ftx][debug],unwind [ftx][debug], spread [ftx][debug][coin][cash in usd]')

if __name__ == "__main__":
    ftx_ws_spread_main(*sys.argv[1:])