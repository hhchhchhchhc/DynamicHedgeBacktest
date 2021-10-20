# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import sys
from ftx_utilities import *
from ftx_history import *
from ftx_ftx import *

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')

import ccxt  # noqa: E402
from datetime import datetime,timezone,timedelta,date
import time
import pickle

print('CCXT Version:', ccxt.__version__)

def enricher(exchange,futures):
    coin_details=pd.DataFrame(exchange.publicGetWalletCoins()['result'],dtype=float).set_index('id')
    futures = pd.merge(futures, coin_details[['spotMargin','tokenizedEquity','collateralWeight','usdFungible','fiat']], how='left', left_on='underlying', right_index=True)

    markets = exchange.fetch_markets()
    futures['spot_ticker'] = futures.apply(lambda x: str(find_spot_ticker(markets, x, 'name')), axis=1)
    futures['expiryTime']=futures.apply(lambda x:
        dateutil.parser.isoparse(x['expiry']).replace(tzinfo=None) if x['type']=='future' else np.NaN,axis=1)#.replace(tzinfo=timezone.utc)

    ### only if active and  spot trades
    futures=futures[(futures['enabled']==True)&(futures['type']!="move")]
    futures=futures[futures.apply(lambda f: float(find_spot_ticker(markets,f,'ask')),axis=1)>0.0] #### only if spot trades

    ########### add borrows
    borrows = fetch_coin_details(exchange)
    futures = pd.merge(futures, borrows[['borrow','lend','funding_volume']], how='left', left_on='underlying', right_index=True)
    futures['quote_borrow'] = float(borrows.loc['USD', 'borrow'])
    futures['quote_lend'] = float(borrows.loc['USD', 'lend'])
    #### need borrow to be present
    futures = futures[futures['borrow']>=-999]

    ########### naive basis for all futures
    futures.loc[futures['type']=='future','basis_mid'] = futures[futures['type']=='future'].apply(lambda f: calc_basis(f['mark'],f['index'],f['expiryTime'],datetime.now()),axis=1)
    futures.loc[futures['type'] == 'perpetual','basis_mid'] = futures[futures['type'] == 'perpetual'].apply(lambda f:
                                                                         float(fetch_funding_rates(exchange,f['name'])['result']['nextFundingRate']) * 24 * 365.25,axis=1)
    return futures

def basis_scanner(exchange,futures,hy_history,point_in_time=datetime.now(), depths=[0],
                  holding_period_slippage_assumption=timedelta(days=3),
                  signal_horizon=timedelta(days=3),
                  slippage_scaler=0.25,
                  params={'naive':True, 'history':True}):###perps assumed held 1 w for slippage calc
    borrows = fetch_coin_details(exchange)
    markets = exchange.fetch_markets()

    futures['spot_ticker'] = futures.apply(lambda x: str(find_spot_ticker(markets, x, 'name')), axis=1)
    futures['expiryTime']=futures.apply(lambda x:
        dateutil.parser.isoparse(x['expiry']).replace(tzinfo=None) if x['type']=='future' else point_in_time+holding_period_slippage_assumption,axis=1)#.replace(tzinfo=timezone.utc)

    ### only if active and  spot trades
    futures=futures[(futures['enabled']==True)&(futures['type']!="move")]
    futures=futures[futures.apply(lambda f: float(find_spot_ticker(markets,f,'ask')),axis=1)>0.0] #### only if spot trades

    #### need borrow to be present
    futures = futures[futures['borrow']>=-999]

    ########### naive basis for all futures
    futures.loc[futures['type']=='future','basis_mid'] = futures[futures['type']=='future'].apply(lambda f: calc_basis(f['mark'],f['index'],f['expiryTime'],point_in_time),axis=1)
    futures.loc[futures['type'] == 'perpetual','basis_mid'] = futures[futures['type'] == 'perpetual'].apply(lambda f:
                                                                         float(fetch_funding_rates(exchange,f['name'])['result']['nextFundingRate']) * 24 * 365.25,axis=1)

    ############ add slippage, fees and speed
    fees=(exchange.fetch_trading_fees()['taker']+exchange.fetch_trading_fees()['maker'])/2
    for size in depths:
        ### relative semi-spreads incl fees, and speed
        if size==0:
            futures['spot_ask_in_0'] = fees+futures.apply(lambda f: 0.5*(float(find_spot_ticker(markets, f, 'ask'))/float(find_spot_ticker(markets, f, 'bid'))-1), axis=1)*slippage_scaler
            futures['spot_bid_in_0'] = -futures['spot_ask_in_0']
            futures['future_ask_in_0'] = fees+0.5*(futures['ask'].astype(float)/futures['bid'].astype(float)-1)*slippage_scaler
            futures['future_bid_in_0'] = -futures['future_ask_in_0']
            futures['speed_in_0']=0##*futures['future_ask_in_0'] ### just 0
        else:
            futures['spot_ask_in_' + str(size)] = futures['spot_ticker'].apply(lambda x: mkt_depth(exchange,x, 'asks', size))*slippage_scaler+fees
            futures['spot_bid_in_' + str(size)] = futures['spot_ticker'].apply(lambda x: mkt_depth(exchange,x, 'bids', size))*slippage_scaler - fees
            futures['future_ask_in_'+str(size)] = futures['name'].apply(lambda x: mkt_depth(exchange,x,'asks',size))*slippage_scaler+fees
            futures['future_bid_in_' + str(size)] = futures['name'].apply(lambda x: mkt_depth(exchange,x, 'bids', size))*slippage_scaler - fees
            futures['speed_in_'+str(size)]=futures['name'].apply(lambda x:mkt_speed(exchange,x,size).seconds)

        #### rate slippage assuming perps are rolled every perp_holding_period
        #### use centred bid ask for robustness
        futures['bid_rate_slippage_in_' + str(size)] = futures.apply(lambda f: \
            (f['future_bid_in_' + str(size)]-f['spot_ask_in_' + str(size)]) \
            / np.max([1, (f['expiryTime'] - point_in_time).seconds* 365.25*24*3600]) ,axis=1)
        futures['ask_rate_slippage_in_' + str(size)] = futures.apply(lambda f: \
            (f['future_ask_in_' + str(size)] - f['spot_bid_in_' + str(size)]) \
            / np.max([1, (f['expiryTime'] - point_in_time).seconds * 365.25*24*3600]),axis=1)

    perps=futures[futures['type'] == 'perpetual']
    if (perps.index.empty == False)&(params['history']==True):
        ### EMA for basis

        gamma = 1.0 / signal_horizon.total_seconds()
        exp_time = pd.Series(index=hy_history.index,
                             data=[np.exp(gamma * (t.timestamp() - datetime.now(tz=timezone.utc).timestamp())) for t in
                                   hy_history.index])
        perps['avgBasis'] = perps.apply(
            lambda h: (hy_history[h['name']+'/rate/funding']
                       * exp_time).sum() / exp_time.sum(), axis=1)
        perps['stdevBasis'] = perps.apply(
            lambda h: (np.power(hy_history[h['name']+'/rate/funding']-h['avgBasis'],2)
                       * exp_time).sum() / exp_time.sum(), axis=1)
    #### easier for futures: skip history
    IMM = futures[futures['type'] == 'future']
    if IMM.index.empty == False:
        IMM['avgBasis']=IMM['basis_mid']
        IMM['stdevBasis']=0
    futures = pd.concat([perps, IMM], join='outer', axis=0)

    ### EMA for borrow
    futures['avgBorrow'] = futures.apply(
        lambda h: (hy_history[h['underlying']+'/rate/borrow']
                   * exp_time).sum() / exp_time.sum(), axis=1)
    futures['stdevBorrow'] = futures.apply(
        lambda h: (np.power(hy_history[h['underlying']+'/rate/borrow']-h['avgBorrow'],2)
                   * exp_time).sum() / exp_time.sum(), axis=1)
    avgUSDborrow = (hy_history['USD/rate/borrow']
                   * exp_time).sum() / exp_time.sum()
    stdUSDborrow = (np.power(hy_history['USD/rate/borrow']-avgUSDborrow,2)
               * exp_time).sum() / exp_time.sum()

    #### finally, adjust for fees, slippage, funding and collateral cost
    account_leverage=exchange.privateGetAccount()['result']

    if float(account_leverage['leverage']) >= 50: print("margin rules not implemented for leverage >=50")

    dummy_size=10000 ## IM is in ^3/2 not linear, but rule typically kicks in at a few M for optimal leverage of 20 so we linearize
    futures['futIM']=(futures['imfFactor']*np.sqrt(dummy_size/futures['mark'])).clip(lower=1/float(account_leverage['leverage']))

    ### note we don't assume staking, which stops earning over the next hour, and is free for IM after another hour (but always free for MM)
    ### we do assume that we are net short USD, so shorts generate usd borrow offset
    futures['adjLongCarry'] = (futures['avgBasis']+futures['bid_rate_slippage_in_' + str(size)] - avgUSDborrow)
    futures['adjShortCarry'] = (futures['avgBasis']+futures['ask_rate_slippage_in_' + str(size)] + futures['avgBorrow'] - avgUSDborrow)
    futures['xccyBasis']=0
    futures.loc[futures['adjLongCarry']>0,'xccyBasis'] = futures.loc[futures['adjLongCarry']>0,'adjLongCarry']
    futures.loc[futures['adjShortCarry']< 0, 'xccyBasis'] = -futures.loc[futures['adjShortCarry'] < 0, 'adjShortCarry']
    futures['maxPos']=0
    futures.loc[futures['adjLongCarry']>0,'maxPos'] = 1/(1+
                                                         (futures.loc[futures['adjLongCarry']>0,'futIM']
                                                         -futures.loc[futures['adjLongCarry']>0,'collateralWeight'])/1.1)
    futures.loc[futures['adjShortCarry']< 0,'maxPos'] = -1/(futures.loc[futures['adjShortCarry']<0,'futIM']
                                                          +1.1/(0.05+futures.loc[futures['adjShortCarry']<0,'collateralWeight'])-1)
    futures['maxCarry'] = np.abs(futures['maxPos'])*futures['xccyBasis']

    return futures

def futures_to_dataframe(futures,size=0,### change wanted greeks if size!=0
                         wanted_greeks=['avgBasis','stdevBasis','avgBasis_with_slippage','spot_ticker','borrow','quote_borrow','lend','quote_lend']):
### only one size for now
    data = pd.DataFrame(columns=pd.MultiIndex.from_tuples(list(zip(*map(futures.get, ['underlyingType', 'underlying', 'margining','expiry','name','type']))),
                                                              # (None,None,None,None,None)
                                                              names=['underlyingType', "underlying", "margining",
                                                                     "expiry", "name", "contractType"]),
                        index=wanted_greeks)
                        #index = list(zip([nowtime] * len(wanted_greeks), wanted_greeks)))

    for i,f in futures.iterrows():
        data[(f['underlyingType'], f['underlying'], f['margining'],f['expiry'],f['name'],f['type'])]=f[wanted_greeks]

    #data[('usdFungible','USD','USD',None,'USD','spot')] = [float(borrows.loc['USD','estimate'])]*len(wanted_greeks)

    data['updated']=datetime.now()
    data.set_index('updated',append=True,inplace=True)

    return data ### not using multiindex for now...