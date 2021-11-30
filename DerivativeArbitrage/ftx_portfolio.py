# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from ftx_utilities import *
from ftx_ftx import *

## calc various margins for a cash and carry.
# weights is position size, not %
# note: weight.shape = futures.shape, but returns a shape = futures.shape+1 !
# TODO: speed up with nd.array
class ExcessMargin:
    def __init__(self,futures,equity,
                    long_blowup=LONG_BLOWUP,short_blowup=SHORT_BLOWUP,nb_blowups=NB_BLOWUPS,params={'positive_carry_on_balances': False}):
        ## inputs
        self._account_leverage=futures['account_leverage'].values[0]
        self._collateralWeight=futures['collateralWeight'].values
        self._imfFactor = futures['imfFactor'].values
        self._mark = futures['mark'].values
        self._collateralWeightInitial = futures.apply(collateralWeightInitial,axis=1).values
        self._equity=float(equity)
        self._long_blowup= float(long_blowup)
        self._short_blowup = float(short_blowup)
        self._nb_blowups = int(nb_blowups)
        self._params=params
    # unused for now
    def update(self,mark):
        self._mark = mark

    def baseMargins(self,x,mark):
        n = len(x)
        # TODO: staked counts towards MM not IM
        # https://help.ftx.com/hc/en-us/articles/360031149632
        collateral = np.array([
            x[i] if x[i] < 0
            else x[i] * min(self._collateralWeight[i],
                            1.1 / (1 + self._imfFactor[i] * np.sqrt(abs(x[i]) / mark[i])))
            for i in range(n)])
        # https://help.ftx.com/hc/en-us/articles/360053007671-Spot-Margin-Trading-Explainer
        im_short = np.array([
            0 if x[i] > 0
            else -x[i] * max(1.1 / self._collateralWeightInitial[i] - 1,
                             self._imfFactor[i] * np.sqrt(abs(x[i]) / mark[i]))
            for i in range(n)])
        mm_short = np.array([
            0 if x[i] > 0
            else -x[i] * max(1.03 / self._collateralWeightInitial[i] - 1,
                             0.6 * self._imfFactor[i] * np.sqrt(abs(x[i]) / mark[i]))
            for i in range(n)])
        im_fut = np.array([
            abs(x[i]) * max(1.0 / self._account_leverage, self._imfFactor[i] * np.sqrt(abs(x[i]) / mark[i]))
            for i in range(n)])
        mm_fut = np.array([
            max([0.03 * x[i], 0.6 * im_fut[i]])
            for i in range(n)])

        return (collateral,im_short,mm_short,im_fut,mm_fut)

    def call(self,x):
        n=len(x)

        # blowups deal with a spike situation on the nb_blowups biggest deltas
        # long: new freeColl = (1+ds)w-(ds+blow)-fut_mm(1+ds+blow)
        # freeColl move = w ds-(ds+blow)-mm(ds+blow) = blow(1-mm) -ds(1-w+mm) ---> ~blow
        # short: new freeColl = -(1+ds)+(ds+blow)-fut_mm(1+ds+blow)-spot_mm(1+ds)
        # freeColl move = blow - fut_mm(ds+blow)-spot_mm ds = blow(1-fut_mm)-ds(fut_mm+spot_mm) ---> ~blow
        blowup_idx=np.argpartition(np.apply_along_axis(abs,0,x), -self._nb_blowups)[-self._nb_blowups:]
        blowups=np.zeros(n)
        for i in range(n): 
            for j in range(len(blowup_idx)):
                i=blowup_idx[j]
                blowups[i] = x[i]*self._long_blowup if x[i]>0 else -x[i]*self._short_blowup

        # assume all coins go either LONG_BLOWUP or SHORT_BLOWUP..what is the margin impact incl future pnl ?
        (collateral_up,im_short_up,mm_short_up,im_fut_up,mm_fut_up) = self.baseMargins(x*(1+LONG_BLOWUP),self._mark*(1+LONG_BLOWUP))
        MM_up = collateral_up - mm_fut_up - mm_short_up - x * LONG_BLOWUP # the futures pnl
        (collateral_down,im_short_down,mm_short_down,im_fut_down,mm_fut_down) = self.baseMargins(x*(1-SHORT_BLOWUP),self._mark*(1-SHORT_BLOWUP))
        MM_down = collateral_down - mm_fut_down - mm_short_down + x * SHORT_BLOWUP # the futures pnl
        (collateral,im_short,mm_short,im_fut,mm_fut) = self.baseMargins(x,self._mark)
        MM = collateral - mm_fut - mm_short - blowups

        IM = collateral - im_fut - im_short
        totalIM = self._equity -sum(x) - 0.1* max([0,sum(x)-self._equity]) + sum(IM)
        totalMM = self._equity -sum(x) - 0.03 * max([0, sum(x) - self._equity]) + min([sum(MM),sum(MM_up),sum(MM_down)])

        return {'totalIM':totalIM,
                'totalMM':totalMM,
                'IM':IM,
                'MM':MM}

### list of dicts positions (resp. balances) assume unique 'future' (resp. 'coin')
### positions need netSize, future, initialMarginRequirement, maintenanceMarginRequirement, realizedPnl, unrealizedPnl
### balances need coin, total
### careful: carry on balances cannot be overal positive.
def carry_portfolio_greeks(exchange,futures,params={'positive_carry_on_balances':False}):
    markets = exchange.fetch_markets()
    coin_details = fetch_coin_details(exchange)  ### * (1+500*taker fee)
    futures = fetch_futures(exchange)

    greeks = pd.DataFrame(columns=pd.MultiIndex.from_tuples([], names=['underlyingType',"underlying", "margining", "expiry","name","contractType"]))
    updated=str(datetime.now())
    rho=0.4

    for x in positions:
        if float(x['optimalWeight']) !=0.0:

            future_item=next(item for item in futures if item['symbol'] == x['future'])
            coin = future_item['underlying']
            underlyingType=getUnderlyingType(coin_details.loc[coin]) if coin in coin_details.index else 'index'
            funding_stats =exchange.publicGetFuturesFutureNameStats({'future_name': future_item['name']})['result']

            size = float(x['netSize'])
            chg = float(future_item['change24h'])
            f=float(future_item['mark'])
            s = float(future_item['index'])
            if future_item['type']=='perpetual':
                t=0.0
                carry= - size*s*float(funding_stats['nextFundingRate'])*24*365.25
            else:
                days_diff = (dateutil.parser.isoparse(future_item['expiry']) - datetime.now(tz=timezone.utc))
                t=days_diff.days/365.25
                carry = - size*f * numpy.log(f / s) / t

            margin_coin = 'USD'  ## always USD on FTX
            if margin_coin == 'USD':
                greeks[(underlyingType,
                    str(coin),
                    margin_coin,
                    future_item['expiry'],
                    future_item['name'],
                    future_item['type'])]= pd.Series({
                        (updated,'PV'):0,
                        (updated, 'ref'): f,
                        (updated,'Delta'):size*f,
                        (updated,'ShadowDelta'):size*f*(1+rho*t),
                        (updated,'Gamma'):size*f*rho*t*(1+rho*t),
                        (updated,'IR01'):size*t*f/10000,
                        (updated,'Carry'):carry,
                        (updated,'collateralValue'):0,
                        (updated,'IM'): float(x['initialMarginRequirement'])*numpy.abs(size)*f,
                        (updated,'MM'): float(x['maintenanceMarginRequirement'])*numpy.abs(size)*f,
                            })
            else:
                greeks[(underlyingType,
                    str(coin),
                    margin_coin,
                    future_item['expiry'],
                    future_item['name'],
                    future_item['type'])] = pd.Series({
                    (updated, 'PV'): 0,
                    (updated, 'ref'): f,
                    (updated, 'Delta'): size / f*s,
                    (updated, 'ShadowDelta'): size / f * s * (1 + rho * t),
                    (updated, 'Gamma'): size / f *s* rho * t * (1 + rho * t),
                    (updated, 'IR01'): size*t*s/f/10000,
                    (updated, 'Carry'): carry,
                    (updated, 'collateralValue'): 0,
                    (updated, 'IM'): float(x['collateralUsed']),
                    (updated, 'MM'): float(x['maintenanceMarginRequirement']) * size ,
                })

            margin_cash=float(x['realizedPnl'])+float(x['unrealizedPnl'])
            try:
                for item in balances:
                    if item['coin'] == margin_coin: item['total']=float(item['total'])+margin_cash
            except:
                balances.append({'total':margin_cash,'coin':margin_coin})

#        margin_greeks=pd.Series(index=list(zip([updated]*10,['PV','ref','Delta','ShadowDelta','Gamma','IR01','Carry','collateralValue','IM','MM'])),
#                   data=[margin_cash,1.0,0.0,0.0,0,0,0,margin_cash,0,0])# + float(x['realizedPnl'])
#        if (margin_coin, 'USD', None, margin_coin, 'spot') in greeks.columns:
#            greeks[('usdFungible',margin_coin, 'USD', None, margin_coin, 'spot')]=margin_greeks.add(greeks[(margin_coin, 'USD', None, margin_coin, 'spot')],fill_value=0.0)
#        else:
#            greeks[('usdFungible',margin_coin, 'USD', None, margin_coin, 'spot')]=margin_greeks ### 'usdFungible' for now...

    stakes = pd.DataFrame(exchange.privateGetStakingBalances()['result']).set_index('coin')
    for x in balances:
        try:
            market_item = next(item for item in markets if item['id'] == x['coin']+'/USD')
            s = float(market_item['info']['price'])
            chg = float(market_item['info']['change24h'])
        except: ## fails for USD
            s = 1.0
            chg = 0.0

        coin = x['coin']
        underlyingType=getUnderlyingType(coin_details.loc[coin])

        size=float(x['total'])
        if size!=0:
            staked = float(stakes.loc[coin,'staked']) if coin in stakes.index else 0
            collateralValue=size*s*(coin_details.loc[coin,'collateralWeight'] if size>0 else 1)-staked*s
            ### weight(initial)=weight(total)-5% for all but stablecoins/ftt(0) and BTC (2.5)
            im=(1.1 / (coin_details.loc[coin,'collateralWeight']-0.05) - 1) * s * -size if (size<0) else 0.0
            mm=(1.03 / (coin_details.loc[coin,'collateralWeight']-0.05) - 1) * s * -size if (size<0) else 0.0
            ## prevent positive carry on balances (no implicit lending/staking)
            carry=size*s* (float(coin_details.loc[coin,('borrow')]) if (size<0) else 0)
            delta = size*s if coin!='USD' else 0

            newgreeks=pd.Series({
                    (updated,'PV'):size*s,
                    (updated, 'ref'): s,
                    (updated,'Delta'):delta,
                    (updated,'ShadowDelta'):delta,
                    (updated,'Gamma'):0,
                    (updated,'IR01'):0,
                    (updated,'Carry'):carry,
                    (updated,'collateralValue'): collateralValue,
                    (updated,'IM'): im,
                    (updated,'MM'): mm})
            if (underlyingType,coin,'USD',None,coin,'spot') in greeks.columns:
                greeks[(underlyingType,
                        coin,
                        'USD',
                        None,
                        coin,
                        'spot')] = greeks[(underlyingType,
                        coin,
                        'USD',
                        None,
                        coin,
                        'spot')] + newgreeks
            else:
                greeks[(underlyingType,
                coin,
                'USD',
                None,
                coin,
                'spot')]=newgreeks

    ## add a sum column
    greeks.sort_index(axis=1, level=[0, 1, 3, 5], ascending=[True, True, True, True],inplace=True)
    greeks[('sum',
            None,
            None,
            None,
            None,
            None)] = greeks.sum(axis=1)
    return greeks

### optimizer using https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp
#eq_cons = {'type': 'eq',
#           'fun' : lambda x: array([2*x[0] + x[1] - 1]),
#           'jac' : lambda x: array([2.0, 1.0])}
#res = minimize(rosen, x0, method='SLSQP', jac=rosen_der,
#               constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True},
#               bounds=bounds)


def live_risk():
    exchange = open_exchange('ftx_benoit','CashAndCarry')
    futures = pd.DataFrame(fetch_futures(exchange, includeExpired=False, includeIndex=True)).set_index('name')
    markets = pd.DataFrame([r['info'] for r in exchange.fetch_markets()]).set_index('name')

    positions = pd.DataFrame([r['info'] for r in exchange.fetch_positions(params={})],dtype=float)#'showAvgPrice':True})
    positions['coin'] = positions['future'].apply(lambda f: f.split('-')[0])
    positions = positions[positions['netSize'] != 0.0].set_index('coin').fillna(0.0)

    balances=pd.DataFrame(exchange.fetch_balance(params={})['info']['result'],dtype=float)#'showAvgPrice':True})
    balances=balances[balances['total']!=0.0].set_index('coin').fillna(0.0)

    greeks=balances.join(positions,how='outer')
    greeks['futureDelta'] = positions.apply(lambda f: f['netSize'] * futures.loc[f['future'], 'mark'], axis=1)
    greeks['spotDelta'] = balances.apply(lambda f: f['total'] * (1.0 if f.name=='USD' else float(markets.loc[f.name+'/USD', 'price'])), axis=1)
    result=greeks[['futureDelta','spotDelta']].fillna(0.0)
    result['netDelta'] = result['futureDelta'] + result['spotDelta']
    result['futureMark'] = positions.apply(lambda f: futures.loc[f['future'], 'mark'], axis=1)
    result['futureIndex'] = positions.apply(lambda f: futures.loc[f['future'], 'index'], axis=1)
    result['spotMark'] = balances.apply(lambda f: (1.0 if f.name=='USD' else float(markets.loc[f.name+'/USD', 'price'])), axis=1)
    result.loc['total', ['futureDelta', 'spotDelta', 'netDelta']] = result[['futureDelta', 'spotDelta', 'netDelta']].sum()

    account_info = pd.DataFrame(exchange.privateGetAccount()['result']).iloc[0, 1:8]
    result.loc['total', account_info.index] = account_info.values

    return result

#print(live_risk())

def process_fills(exchange,spot_fills,future_fills):
    if (spot_fills.empty)|(future_fills.empty): return pd.DataFrame()
    spot_fills['time']=spot_fills['time'].apply(dateutil.parser.isoparse)
    future_fills['time']=future_fills['time'].apply(dateutil.parser.isoparse)
    # TODO: its not that simple if partials, but...
    fill1 = spot_fills if spot_fills['time'].min()<future_fills['time'].min() else future_fills
    fill2 = spot_fills if spot_fills['time'].min() >= future_fills['time'].min() else future_fills
    symbol1 = fill1['market'].iloc[0]
    symbol2 = fill2['market'].iloc[0]

    ## TODO: more rigorous to see in USD than bps
    result=pd.Series()
    result['size'] = min([(spot_fills['price'] * spot_fills['size']).sum(),
                            (future_fills['price'] * future_fills['size']).sum()])
    result['avg_spot_level']=(spot_fills['size']*spot_fills['price']).sum() / spot_fills['size'].sum()
    result['avg_future_level']=(future_fills['size']*future_fills['price']).sum() / future_fills['size'].sum()
    result['realized_premium_bps']=(result['avg_future_level']*future_fills['size'].sum()-\
                                    result['avg_spot_level']*future_fills['size'].sum())\
                                    /result['size']*10000
    result['final_delta']=( spot_fills.loc[spot_fills['side']=='buy','size'].sum() \
                        -   spot_fills.loc[spot_fills['side'] == 'sell', 'size'].sum() \
                        +   future_fills.loc[future_fills['side'] == 'buy', 'size'].sum() \
                        -   future_fills.loc[future_fills['side'] == 'sell', 'size'].sum())\
                                  *fill2.loc[fill2.index.max(),'price']
    # premium_mid: we use nearest trades instead of candles
    s2 = fill1['time'].apply(lambda t: fetch_nearest_trade(exchange,symbol2,t,target_depth=1)[0])
    premium = (s2-fill1['price'])*fill1['size']* (1 if spot_fills['time'].min()<future_fills['time'].min() else -1)
    result['premium_nearest_transaction']= premium.sum()/result['size']*10000
    # delta_pnl: we use nearest trade at the end
    (result['start_time'],result['end_time'])=\
        (spot_fills['time'].min().tz_convert(None),future_fills['time'].min().tz_convert(None)) if spot_fills['time'].min()<future_fills['time'].min() \
            else (future_fills['time'].min().tz_convert(None),spot_fills['time'].min().tz_convert(None))
   # result['end_time'] = max(spot_fills['time'].max(), future_fills['time'].max())
    final_spot=fetch_nearest_trade(exchange,spot_fills['market'].iloc[0],result['end_time'])[0]
    final_future = fetch_nearest_trade(exchange,future_fills['market'].iloc[0],result['end_time'])[0]
    result['delta_pnl'] = \
        (spot_fills['size']*(final_spot-spot_fills['price']).sum()\
        +(future_fills['size']*(final_future-future_fills['price']).sum())\
        )/result['size']*10000
    result['fee']=(spot_fills['fee']+future_fills['fee']).sum()/result['size']

    return result

def run_fills_analysis(exchange_name='ftx', end = datetime.now(),start = datetime.now()- timedelta(days=30)):
    exchange=open_exchange(exchange_name)
    futures = pd.DataFrame(fetch_futures(exchange,includeExpired=False))

    fill_analysis = pd.DataFrame()
    all_fills=pd.DataFrame(exchange.privateGetFills(
                        {'start_time':int(start.timestamp()),'end_time':int(end.timestamp())}
                        )['result'],dtype=float)
    funding_paid = pd.DataFrame(exchange.privateGetFundingPayments(
                        {'start_time':int(start.timestamp()),'end_time':int(end.timestamp())}
                        )['result'],dtype=float)
    if all_fills.empty: return (fill_analysis,all_fills)
    for future in set(all_fills['market']).intersection(set(futures['name'])):
        underlying=future.split('-')[0]
        spot_fills=all_fills[all_fills['market']==underlying+'/USD']
        future_fills=all_fills[all_fills['market']==future]
        if (spot_fills.empty)|(future_fills.empty): continue

        result=process_fills(exchange,spot_fills,future_fills)

        funding_received= -funding_paid.loc[
            funding_paid['future'].apply(lambda f: f.split('-')[0])==underlying,
            'payment']
        result['funding'] = funding_received.sum()
        fill_analysis[underlying]=result

    return (fill_analysis,all_fills)

if False:
    (fill_analysis, all_fills)=run_fills_analysis('ftx',
            end = datetime.now(),
            start = datetime.now()- timedelta(days=7))
    with pd.ExcelWriter('fills.xlsx', engine='xlsxwriter') as writer:
        fill_analysis.to_excel(writer,sheet_name='fill_analysis')
        all_fills.to_excel(writer,sheet_name='all_fills')