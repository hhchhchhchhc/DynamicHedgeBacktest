import logging
import sys

import ccxtpro
import pandas as pd
from ccxtpro.base.cache import ArrayCacheBySymbolById
import functools
from copy import deepcopy

from ftx_utilities import *
from ftx_portfolio import collateralWeightInitial,diff_portoflio
from ftx_history import fetch_trades_history
from ftx_ftx import fetch_latencyStats,fetch_futures

max_nb_coins = 7  # TODO: sharding needed
entry_tolerance = 0.5 # green light if basket better than median
edit_trigger_tolerance = np.sqrt(60/3600) # chase on 1m stdev
stop_tolerance = np.sqrt(30*60/3600) # stop on 30min stdev
time_budget = 99999999999 # used in transaction speed screener
delta_limit = 0.2 # delta limit / pv
slice_factor = 0.5 # % of request
edit_price_tolerance=np.sqrt(30/3600)#price on 1s std

### low level class to compute margins
class liveIM:
    def __init__(self, account_leverage, collateralWeight, imfFactor):  # imfFactor by symbol not coin
        self._account_leverage = account_leverage
        self._collateralWeight = collateralWeight
        self._imfFactor = imfFactor
        self._collateralWeightInitial = {coin: collateralWeightInitial({'underlying': coin, 'collateralWeight': data})
                                         for coin, data in collateralWeight.items()}

    def futureMargins(self, weights):  # weights = {symbol: 'weight, 'mid
        im_fut = sum([
            abs(data['weight']) * max(1.0 / self._account_leverage,
                                      self._imfFactor[symbol] * np.sqrt(abs(data['weight']) / data['mid']))
            for symbol, data in weights.items()])
        mm_fut = sum([
            max([0.03 * data['weight'], 0.6 * im_fut])
            for symbol, data in weights.items()])

        return (im_fut, mm_fut)

    def spotMargins(self, weights):
        collateral = sum([
            data['weight'] if data['weight'] < 0
            else data['weight'] * min(self._collateralWeight[coin],
                                      1.1 / (1 + self._imfFactor[coin + '/USD:USD'] * np.sqrt(
                                          abs(data['weight']) / data['mid'])))
            for coin, data in weights.items()])
        # https://help.ftx.com/hc/en-us/articles/360053007671-Spot-Margin-Trading-Explainer
        im_short = sum([
            0 if data['weight'] > 0
            else -data['weight'] * max(1.1 / self._collateralWeightInitial[coin] - 1,
                                       self._imfFactor[coin + '/USD:USD'] * np.sqrt(abs(data['weight']) / data['mid']))
            for coin, data in weights.items()])
        mm_short = sum([
            0 if data['weight'] > 0
            else -data['weight'] * max(1.03 / self._collateralWeightInitial[coin] - 1,
                                       0.6 * self._imfFactor[coin + '/USD:USD'] * np.sqrt(
                                           abs(data['weight']) / data['mid']))
            for coin, data in weights.items()])

        return (collateral, im_short, mm_short)

    def margins(self, usd_balance, spot_weights, future_weights):
        (collateral, im_short, mm_short) = self.spotMargins(spot_weights)
        (im_fut, mm_fut) = self.futureMargins(future_weights)
        IM = collateral + usd_balance + 0.1 * min([0, usd_balance]) - im_fut - im_short
        MM = collateral + usd_balance + 0.03 * min([0, usd_balance]) - mm_fut - mm_short
        return {'IM': IM, 'MM': MM}

    # not used, as would need all risks not only self.risk_state
    def margins_from_exchange(self,exchange):
        future_weight = {symbol: {'weight': data['delta'], 'mid': exchange.mid(symbol)}
                         for coin,coin_data in exchange.risk_state.items() if coin in exchange.currencies
                         for symbol,data in coin_data.items() if symbol in exchange.markets and exchange.markets[symbol]['contract']}
        spot_weight = {coin: {'weight': data['delta'], 'mid': exchange.mid(symbol)}
                         for coin,coin_data in exchange.risk_state.items() if coin in exchange.currencies
                         for symbol,data in coin_data.items() if symbol in exchange.markets and exchange.markets[symbol]['spot']}

        ## add orders as if done
        for order in exchange.orders:
            if order['status'] == 'open':
                symbol = order['symbol']
                if exchange.markets[symbol]['spot']:
                    coin = exchange.markets[symbol]['base']
                    if coin not in spot_weight: spot_weight[coin] = {'weight': 0, 'mid': exchange.mid(symbol)}
                    spot_weight[coin]['weight'] += order['amount'] * (1 if order['side'] == 'buy' else -1)*exchange.mid(symbol)
                else:
                    if symbol not in future_weight: future_weight[symbol] = {'weight': 0, 'mid': exchange.mid(symbol)}
                    future_weight[symbol]['weight'] += order['amount'] * (1 if order['side'] == 'buy' else -1)*exchange.mid(symbol)

        (collateral, im_short, mm_short) = self.spotMargins(spot_weight)
        (im_fut, mm_fut) = self.futureMargins(future_weight)
        IM = collateral + exchange.usd_balance + 0.1 * min([0, exchange.usd_balance]) - im_fut - im_short
        MM = collateral + exchange.usd_balance + 0.03 * min([0, exchange.usd_balance]) - mm_fut - mm_short

        return {'IM': IM, 'MM': MM}

    # margin impact of an order
    def margin_cost(self, symbol, mid, size, usd_balance):
        if symbol in self._imfFactor:  # --> derivative
            (im_fut, mm_fut) = self.futureMargins({symbol: {'weight': size * mid, 'mid': mid}})
            return -im_fut
        elif symbol in self._collateralWeight:  # --> coin
            (collateral, im_short, mm_short) = self.spotMargins({symbol: {'weight': size * mid, 'mid': mid}})
            usd_balance_chg = -size * mid
            return collateral + usd_balance_chg + 0.1 * (
                        min([0, usd_balance + usd_balance_chg]) - min([0, usd_balance])) - im_short
        logging.exception(f'IM impact: {symbol} neither derivative or cash')

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
        self.event_records = []
        self.limit = myFtx.LimitBreached()

        self.exec_parameters = {}
        self.done = []

        self.risk_state = {}
        self.pv = None
        self.usd_balance = None # it's handy for marginal calcs
        self.margin_headroom = None
        self.margin_calculator = None
        self.calculated_IM = None

        limit = self.safe_integer(self.options, 'ordersLimit', 1000)
        self.orders = ArrayCacheBySymbolById(limit)
        self.unacknowledged_orders = dict()

    def build_logging(self):
        # logs
        handler_warning = logging.FileHandler('Runtime/logs/warning.log', mode='w')
        handler_warning.setLevel(logging.WARNING)
        handler_warning.setFormatter(logging.Formatter(f"%(levelname)s: %(message)s"))
        self.logger.addHandler(handler_warning)

        handler_info = logging.FileHandler('Runtime/logs/info.log', mode='w')
        handler_info.setLevel(logging.INFO)
        handler_info.setFormatter(logging.Formatter(f"%(levelname)s: %(message)s"))
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

    def log_order_event(self,event_type,order_event):
        ## generic info
        symbol = order_event['symbol']
        coin = self.markets[symbol]['base']
        event_record = {'event_type':event_type,
                       'event_timestamp':self.milliseconds()-self.exec_parameters['timestamp'],
                       'coin': coin,
                       'symbol': symbol}

        ## order/fill/ack details. except for global risk.
        if len(order_event.keys())>1:
            if event_type=='order':
                event_record |= {key: order_event[key] for key in
                             ['side', 'price', 'amount', 'type', 'id', 'filled', 'clientOrderId','acknowledged']}
            elif event_type=='fill':
                def translate(key):
                    if key == 'takerOrMaker': return 'type'
                    elif key == 'order': return 'id'
                    elif key == 'id': return 'clientOrderId'
                    else: return key
                event_record |= {translate(key): order_event[key] for key in
                                 ['side', 'price', 'amount', 'takerOrMaker', 'order','id','acknowledged']}
            elif event_type=='acknowledge':
                event_record |= {key: order_event[key] for key in
                                 ['side', 'price', 'amount', 'type', 'id', 'filled', 'clientOrderId','acknowledged']}
            elif event_type=='cancel':
                event_record |= {key: order_event[key] for key in
                                 ['side', 'price', 'amount', 'type', 'id', 'filled', 'clientOrderId', 'acknowledged']}

        ## risk details
        risk_data = self.risk_state[coin]
        event_record |= {'risk_timestamp':risk_data[symbol]['delta_timestamp']-self.exec_parameters['timestamp'],
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
        event_record |= {'mkt_timestamp': timestamp - self.exec_parameters['timestamp']}\
                        | {key: mkt_data[key] for key in ['bid', 'bidVolume', 'ask', 'askVolume']}

        self.event_records += [event_record]
        filename = 'Runtime/logs/latest_events.json'
        with open(filename,mode='w') as file:
            json.dump(self.event_records, file,cls=NpEncoder)
        shutil.copy2(filename, 'Runtime/logs/'+datetime.fromtimestamp(self.exec_parameters['timestamp']/1000).strftime("%Y-%m-%d-%Hh")+'_events.json')

    def filter_orders(self,symbol,key,value):
        if self.orders and (symbol in self.orders.hashmap):
            return [order for id,order in self.orders.hashmap[symbol].items() if order[key]==value]
        else:
            return []

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

        trades_history_list = await asyncio.gather(*[fetch_trades_history(
            self.market(symbol)['id'], self, start, end, frequency=frequency)
            for symbol in weights['name']])

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
            self.logger.info(myFtx.DoneDeal('all done: too small or too slow'))
            raise myFtx.DoneDeal('all done: too small or too slow')

        def basket_vwap_quantile(series_list,diff_list,quantile):
            series = pd.concat(series_list,axis=1).dropna(axis=0)-diff_list
            return series.sum(axis=1).diff().quantile(quantile)
        def z_score(series,z_score):# stdev of 1s prices * quantile
            series = series.dropna(axis=0)
            return series.std().values[0]*z_score

        # get times series of target baskets, compute quantile of increments and add to last price
        # remove series
        self.exec_parameters = {'timestamp':self.milliseconds()} \
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
                                       sys.intern('delta_timestamp'):end.timestamp()*1000,
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

        with pd.ExcelWriter('Runtime/logs/latest_request.xlsx', engine='xlsxwriter', mode='w') as writer:
            request = pd.DataFrame({symbol:data
                                 for coin,coin_data in self.exec_parameters.items() if coin in self.currencies
                                 for symbol,data in coin_data.items() if symbol in self.markets})
            request.to_excel(writer, sheet_name='request')
            vwap_dataframe = pd.concat([data['vwap'].filter(like='vwap').fillna(method='ffill') for data in trades_history_list],axis=1,join='outer')
            vwap_dataframe.to_excel(writer, sheet_name='vwap')
            size_dataframe = pd.concat([data['vwap'].filter(like='volume').fillna(method='ffill') for data in trades_history_list], axis=1, join='outer')
            size_dataframe.to_excel(writer, sheet_name='volume')
        with open('Runtime/logs/' + datetime.fromtimestamp(self.exec_parameters['timestamp']/1000).strftime(
                "%Y-%m-%d-%Hh") + '_request.json', 'w') as file:
            json.dump([{symbol:data
                        for coin,coin_data in self.exec_parameters.items() if coin in self.currencies
                        for symbol,data in coin_data.items() if symbol in self.markets}], file, cls=NpEncoder)
        with open('Runtime/logs/latest_request.json', 'w') as file:
            json.dump(flatten(self.exec_parameters), file, cls=NpEncoder)

    async def fetch_risk(self, params=[]):
        ''' updates risk in USD. Should be handle_my_trade but needs a coroutine :(
            all symbols not present when state is built are ignored !
            if some tickers are not initialized, just use markets'''
        risks= await asyncio.gather(*[self.fetch_positions(),self.fetch_balance()])
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
                future_weight |= {position['symbol']: {'weight': float(position['info']['netSize'])*mid, 'mid':mid}}
        for coin,balance in balances.items():
            if coin!='USD' and coin in self.currencies and balance['total']!=0:
                mid = self.mid(coin+'/USD')
                spot_weight |= {(coin):{'weight':balance['total']*mid,'mid':mid}}

        # calculate and also fetch IM
        account_info = (await self.privateGetAccount())['result']
        self.calculated_IM = self.margin_calculator.margins(self.usd_balance,spot_weight,future_weight)['IM']
        self.margin_headroom = float(account_info['totalPositionSize']) * (float(account_info['openMarginFraction']) - float(account_info['initialMarginRequirement'])) if float(account_info['totalPositionSize']) else self.pv

        [self.log_order_event('remote_risk',{'symbol':symbol})
         for coin,coin_data in self.risk_state.items()
         for symbol in coin_data.keys() if symbol in self.markets]

    def mid(self,symbol):
        data = self.tickers[symbol] if symbol in self.tickers else self.markets[symbol]['info']
        return 0.5*(float(data['bid'])+float(data['ask']))

    def sweep_price(self, symbol, size):
        '''slippage of a mkt order: https://www.sciencedirect.com/science/article/pii/S0378426620303022'''
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

    async def peg_or_stopout(self,symbol,size,orderbook_depth,edit_trigger_depth,edit_price_depth,stop_depth=None):
        '''creates of edit orders, pegging to orderbook
        size in coin, already filtered'''
        coin = self.markets[symbol]['base']
        #orders = self.orders if self.orders else await self.fetch_open_orders(symbol=symbol)
        #orders = await self.fetch_open_orders(symbol=symbol)
        #orders = [order for order in self.order_state[coin][symbol] if order['status']=='open']


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
            orders = self.filter_orders(symbol, 'status', 'open')+[order for order in self.unacknowledged_orders.values() if order['symbol']==symbol]
            if len(orders) > 1:
                for order in orders[:-1]:
                    self.log_order_event('cancel', order | {'acknowledged': 'duplicate'})
                    await self.cancel_order(order['id'])

            order = None
            if len(orders)==0:
                order = await self.create_limit_order(symbol, 'buy' if size>0 else 'sell', np.abs(size), price=edit_price, params={'postOnly': True,'clientOrderId':f'initial_{symbol}_{str(self.milliseconds())}'})
            else:
                order_distance = (1 if orders[0]['side']=='buy' else -1)*(opposite_side-orders[0]['price'])
                # panic stop. we could rather place a trailing stop: more robust to latency, but less generic.
                if stop_depth \
                        and order_distance>stop_trigger \
                        and orders[0]['remaining']>sizeIncrement:
                    result = await asyncio.gather(*[self.create_market_order(symbol, 'buy' if size>0 else 'sell',orders[0]['remaining'],params={'clientOrderId':f'initial_{symbol}_{str(self.milliseconds())}'}),self.cancel_order(orders[0]['id'])])
                    order = result[0]

                # chase
                if order_distance>edit_trigger \
                        and np.abs(edit_price-orders[0]['price'])>=priceIncrement \
                        and orders[0]['remaining']>sizeIncrement:
                    order = await self.edit_order(orders[0]['id'], symbol, 'limit', 'buy' if size>0 else 'sell', None ,price=edit_price,params={'postOnly': True,'clientOrderId':f'initial_{symbol}_{str(self.milliseconds())}'})
        except ccxt.InsufficientFunds as e:
            cost = self.margin_calculator.margin_cost(symbol, mid, size, self.usd_balance)
            self.logger.warning(f'marginal cost {cost}, vs margin_headroom {self.margin_headroom} and calculated_IM {self.calculated_IM}')
        except ccxt.InvalidOrder as e:
            if "Order already queued for cancellation" in str(e):
                self.logger.warning(str(e) + str(orders[0]))
            elif ("Order already closed" in str(e)) or ("Order not found" in str(e)):
                fetched_fills = await self.fetch_my_trades(symbol,since=self.exec_parameters['timestamp'],limit=1000)
                fetched_orders = await self.fetch_orders(symbol,since=self.exec_parameters['timestamp'],limit=1000)
                self.logger.warning(str(e) + str(orders[0]))
            elif ("Size too small for provide" or "Size too small") in str(e):
                self.logger.warning('{}: {} too small {}'.format(orders[0],np.abs(size),sizeIncrement))
            else:
                self.logger.warning(str(e) + str(orders[0]))
        except ccxt.ExchangeError as e:
            if "Must modify either price or size" in str(e):
                self.logger.warning(str(e) + str(orders[0]))
            else:
                self.logger.warning(str(e))
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(str(e))
            await asyncio.sleep(.1)
        except ccxt.NetworkError as e:
            self.logger.warning(str(e))
        except Exception as e:
            self.logger.exception('{} {} {} at {} raised {}'.format('buy' if size > 0 else 'sell', np.abs(size), symbol, price, e))
            raise e

        return order

    def loop_and_callback(func):
        @functools.wraps(func)
        async def wrapper_loop(*args, **kwargs):
            self=args[0]
            while True:
                try:
                    value = await func(*args, **kwargs)
                except myFtx.DoneDeal as e:
                    if 'all done' in str(e):
                        raise e
                    else:
                        return
                except ccxt.DDoSProtection as e:
                    self.logger.critical(e)
                    raise e
                except ccxt.RequestTimeout as e:
                    self.logger.warning(str(e))
                    continue
                except ccxt.NetworkError as e:
                    self.logger.warning(str(e))
                    continue
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as e:
                    self.logger.exception(e, exc_info=True)
                    raise e

        return wrapper_loop

    async def watch_ticker(self, symbol, params={}):
        '''watch_order_book is faster than watch_tickers so we DON'T LISTEN TO TICKERS. Dirty...'''
        raise Exception("watch_order_book is faster than watch_tickers so we DON'T LISTEN TO TICKERS. Dirty...")

    @loop_and_callback
    async def order_master(self,symbol):
        '''on each top of book update, update market_state and send orders
        tunes aggressiveness according to risk'''
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
        #     coin_fills = await asyncio.gather(*[self.fetch_my_trades(symbol=x,params={'minId':delta_id + 1})
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
        sizeIncrement = params['sizeIncrement']
        if np.abs(size) < sizeIncrement:
            self.done+=[symbol]
            if all(symbol_ in self.done
                   for coin,coin_data in self.exec_parameters.items() if coin in self.currencies
                   for symbol_ in coin_data.keys() if symbol_ in self.markets):
                self.logger.info('all done')
                raise myFtx.DoneDeal('all done')
            else:
                self.logger.info(f'{symbol} done')
                raise myFtx.DoneDeal(symbol)
        else:
            size = np.sign(size)*float(self.amount_to_precision(symbol, min([np.abs(size), params['slice_size'], self.margin_headroom / mid])))
            if np.abs(size)<sizeIncrement:
                self.logger.warning(f'size {size} too small?')
                #raise myFtx.LimitBreached(f'size {size} too small: margin_headroom {self.margin_headroom}?')

        order = None
        # if increases risk, go passive
        if np.abs(netDelta+size)>np.abs(netDelta):
            if self.exec_parameters[coin]['entry_level'] is None: # for a purely risk reducing execise
                self.done += [symbol]
                self.logger.info(myFtx.DoneDeal(symbol))
                raise myFtx.DoneDeal(symbol)
            # set limit at target quantile
            #current_basket_price = sum(self.mid(symbol)*self.exec_parameters[coin][symbol]['diff']
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

        if order is not None:
            #self.order_state[coin][symbol]+=[order]
            acknowledged = (symbol in self.orders.hashmap) and (order['id'] in self.orders.hashmap[symbol])
            if not acknowledged:
                self.unacknowledged_orders[order['id']] = order

            self.log_order_event('order',order|{'acknowledged':acknowledged})

    @loop_and_callback
    async def monitor_fills(self):
        '''logs fills'''
        fills = await self.watch_my_trades(limit=1)

        # await self.fetch_risk() is safer but slower. we have monitor_risk to reconcile
        fill = fills[-1]
        try:
            coin = self.markets[fill['symbol']]['base']
        except:
            pass
        symbol = fill['symbol']

        # update risk_state
        data = self.risk_state[coin][symbol]
        fill_size = fill['amount'] * (1 if fill['side'] == 'buy' else -1)*fill['price']
        data['delta'] += fill_size
        data['delta_timestamp'] = fill['timestamp']*1000
        latest_delta = data['delta_id']
        data['delta_id'] = max(latest_delta or 0,int(fill['id']))
        self.risk_state[coin]['netDelta'] += fill_size

        self.log_order_event('fill', fill | {'acknowledged': '?'})

        self.logger.info('{} fill at {}: {} {} {} at {}'.format(symbol,
                            fill['timestamp'] - self.exec_parameters['timestamp'],
                            fill['side'],fill['amount'],symbol,fill['price']))

        current = self.risk_state[coin][symbol]['delta']
        initial =self.exec_parameters[coin][symbol]['target'] * fill['price'] -self.exec_parameters[coin][symbol]['diff'] * fill['price']
        target=self.exec_parameters[coin][symbol]['target'] * fill['price']
        self.logger.info('{} risk at {}sec: {}% done [current {}, initial {}, target {}]'.format(
            symbol,
            self.risk_state[coin][symbol]['delta_timestamp']- self.exec_parameters['timestamp'],
            (current-initial)/(target-initial)*100,
            current,
            initial,
            target))

    @loop_and_callback
    async def monitor_risk(self):
        '''redundant minutely risk check'''
        await self.fetch_risk()

        self.limit.limit = self.pv * self.limit.delta_limit
        absolute_risk = sum(abs(data['netDelta']) for data in self.risk_state.values())
        if absolute_risk > self.limit.limit:
            self.logger.warning(f'absolute_risk {absolute_risk} > {self.limit.limit}')
        if self.margin_headroom < self.pv/100:
            self.logger.warning(f'IM {self.margin_headroom}  < 1%')

        await asyncio.sleep(self.limit.check_frequency)

    def handle_order(self, client, message):
        '''logs newly ackownledged orders. If present in self.orders then amend'''
        super().handle_order(client, message)
        data = self.safe_value(message, 'data')
        order = self.parse_order(data)

        unacknowledged = order['id'] in self.unacknowledged_orders
        if unacknowledged:
            self.unacknowledged_orders.__delitem__(order['id'])

        order['timestamp']=self.milliseconds() - self.exec_parameters['timestamp']
        self.log_order_event('acknowledge', order | {'acknowledged': not unacknowledged})

    @loop_and_callback
    async def monitor_orders(self):
        orders = await self.watch_orders()

async def ftx_ws_spread_main_wrapper(*argv,**kwargs):
    exchange = myFtx({
        'asyncioLoop': kwargs['loop'] if 'loop' in kwargs else None,
        'newUpdates': True,
        #'watchOrderBookLimit': 1000,
        'enableRateLimit': True,
        'apiKey': apiKey,
        'secret': secret}) if argv[1]=='ftx' else None
    exchange.verbose = False
    exchange.headers =  {'FTX-SUBACCOUNT': argv[2]}
    exchange.authenticate()
    await exchange.cancel_all_orders()

    try:
        if argv[0]=='sysperp':
            future_weights = pd.read_excel('Runtime/ApprovedRuns/current_weights.xlsx')
            futures=await fetch_futures(exchange)
            target_portfolio = await diff_portoflio(exchange, future_weights)
            if target_portfolio.empty: return
            #selected_coins = ['REN']#target_portfolios.sort_values(by='USDdiff', key=np.abs, ascending=False).iloc[2]['underlying']
            #target_portfolio=diff[diff['coin'].isin(selected_coins)]
            target_portfolio['optimalCoin']*=1
            await exchange.build_state(target_portfolio,
                                       entry_tolerance=entry_tolerance,
                                       edit_trigger_tolerance=edit_trigger_tolerance,
                                       edit_price_tolerance=edit_price_tolerance,
                                       stop_tolerance=stop_tolerance,
                                       time_budget=time_budget,
                                       delta_limit=delta_limit,
                                       slice_factor=slice_factor)


        elif argv[0]=='spread':
            coin=argv[3]
            cash_name = coin+'/USD'
            future_name = coin + '-PERP'
            future_name = coin + '-PERP'
            cash_price = float(exchange.market(cash_name)['info']['price'])
            future_price = float(exchange.market(future_name)['info']['price'])
            target_portfolio = pd.DataFrame(columns=['coin','name','optimalCoin','currentCoin','spot_price'],data=[
                [coin,cash_name,float(argv[4])/cash_price,0,cash_price],
                [coin,future_name,-float(argv[4])/future_price,0,future_price]])
            if target_portfolio.empty: return

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
            if target_portfolio.empty: return

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
            if target_portfolio.empty: return

            await exchange.build_state(target_portfolio,
                                       entry_tolerance=0.99,
                                       edit_trigger_tolerance=edit_trigger_tolerance,
                                       edit_price_tolerance=edit_price_tolerance,
                                       stop_tolerance=stop_tolerance,
                                       time_budget=999,
                                       delta_limit=delta_limit,
                                       slice_factor=slice_factor)
        else:
            exchange.logger.exception(f'unknown command {argv[0]}',exc_info=True)
            raise Exception(f'unknown command {argv[0]}',exc_info=True)

        await asyncio.gather(*([exchange.monitor_fills(),exchange.monitor_risk(),exchange.monitor_orders()]+#
                               [exchange.order_master(symbol)
                                for coin_data in exchange.risk_state.values()
                                for symbol in coin_data.keys() if symbol in exchange.markets]
                               ))
    except myFtx.DoneDeal as e:
        exchange.logger.info(e)
    except Exception as e:
        exchange.logger.exception(e,exc_info=True)
        raise e
    finally:
        stats = await fetch_latencyStats(exchange, days=1, subaccount_nickname='SysPerp')
        exchange.logger.info(f'latencystats:{stats}')
        await exchange.close()

    return

def ftx_ws_spread_main(*argv):
    argv=list(argv)
    if len(argv) == 0:
        argv.extend(['sysperp'])
    if len(argv) < 3:
        argv.extend(['ftx', 'debug'])
    logging.info(f'running {argv}')
    loop = asyncio.new_event_loop()
    if argv[0] in ['sysperp', 'flatten','unwind','spread']:
        return loop.run_until_complete(ftx_ws_spread_main_wrapper(*argv,loop=loop))
    else:
        logging.info(f'commands: sysperp [ftx][debug], flatten [ftx][debug],unwind [ftx][debug], spread [ftx][debug][coin][cash in usd]')

if __name__ == "__main__":
    ftx_ws_spread_main(*sys.argv[1:])