import pandas as pd
import scipy.optimize
from scipy.stats import norm
from pandas.tseries.frequencies import to_offset

from ftx_utilities import *
from ftx_history import *
from ftx_ftx import *

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
                                                                         float(fetch_funding_rates(exchange,f['name'])['result']['nextFundingRate'])* 24 * 365.25,axis=1)
    return futures

def excessUSD(x,futures):
    futures.apply(lambda f:
                  f['longWeight'] if E_long_t.loc[point_in_time, f['name']] > E_short_t.loc[
                      point_in_time, f['name']]
                  else f['shortWeight'],axis=1)

def basis_scanner(exchange, futures, hy_history, point_in_time='live', depths=0,
                  holding_period=timedelta(days=3),  # to convert slippag into rate
                  signal_horizon=timedelta(days=3),  # historical window for expectations
                  loss_tolerance=0.1,  # for markovitz
                  slippage_scaler=0.25,  # if scaler from order book
                  slippage_override=2e-4,  # external slippage override
                  concentration_limit=0.25,
                  params={'override_slippage':True}):             # use external rather than order book
    borrows = fetch_coin_details(exchange)
    markets = exchange.fetch_markets()

    ### remove blanks for this
    hy_history = hy_history.fillna(method='ffill',limit=2,inplace=False)

    #-----------------  screening --------------
    ### only if active and  spot trades
    futures=futures[(futures['enabled']==True)&(futures['type']!="move")]
    futures['spot_ticker'] = futures.apply(lambda x: str(find_spot_ticker(markets, x, 'name')), axis=1)
    futures=futures[futures.apply(lambda f: float(find_spot_ticker(markets,f,'ask')),axis=1)>0.0] #### only if spot trades
    #### need borrow to be present
    futures = futures[futures['borrow']>=-999]

    #-------------------- transaction costs---------------
    # add slippage, fees and speed. Pls note we will trade both legs, at entry and exit.
    # Unless slippage override, calculate from orderbook (override Only live is supported).

    if params['override_slippage']==True:
        futures['bid_rate_slippage_in_' + str(depths)]=slippage_override
        futures['ask_rate_slippage_in_' + str(depths)]=slippage_override
    else:
        fees=2*(exchange.fetch_trading_fees()['taker']+exchange.fetch_trading_fees()['maker'])
        ### relative semi-spreads incl fees, and speed
        if size==0:
            futures['spot_ask_in_0'] = fees+futures.apply(lambda f: 0.5*(float(find_spot_ticker(markets, f, 'ask'))/float(find_spot_ticker(markets, f, 'bid'))-1), axis=1)*slippage_scaler
            futures['spot_bid_in_0'] = -futures['spot_ask_in_0']
            futures['future_ask_in_0'] = fees+0.5*(futures['ask'].astype(float)/futures['bid'].astype(float)-1)*slippage_scaler
            futures['future_bid_in_0'] = -futures['future_ask_in_0']
            futures['speed_in_0']=0##*futures['future_ask_in_0'] ### just 0
        else:
            futures['spot_ask_in_' + str(depths)] = futures['spot_ticker'].apply(lambda x: mkt_depth(exchange,x, 'asks', depths))*slippage_scaler+fees
            futures['spot_bid_in_' + str(depths)] = futures['spot_ticker'].apply(lambda x: mkt_depth(exchange,x, 'bids', depths))*slippage_scaler - fees
            futures['future_ask_in_'+str(depths)] = futures['name'].apply(lambda x: mkt_depth(exchange,x,'asks',depths))*slippage_scaler+fees
            futures['future_bid_in_' + str(depths)] = futures['name'].apply(lambda x: mkt_depth(exchange,x, 'bids', depths))*slippage_scaler - fees
            futures['speed_in_'+str(depths)]=futures['name'].apply(lambda x:mkt_speed(exchange,x,depths).seconds)

        #### rate slippage assuming perps are rolled every perp_holding_period
        #### use centred bid ask for robustness
        futures['expiryTime'] = futures.apply(lambda x:
                                              dateutil.parser.isoparse(x['expiry']).replace(tzinfo=None) if x['type'] == 'future'
                                              else point_in_time + holding_period,
                                              axis=1)  # .replace(tzinfo=timezone.utc)

        futures['bid_rate_slippage_in_' + str(size)] = futures.apply(lambda f: \
            (f['future_bid_in_' + str(size)]-f['spot_ask_in_' + str(size)]) \
            / np.max([1, (f['expiryTime'] - point_in_time).seconds* 365.25*24*3600]) ,axis=1)
        futures['ask_rate_slippage_in_' + str(size)] = futures.apply(lambda f: \
            (f['future_ask_in_' + str(size)] - f['spot_bid_in_' + str(size)]) \
            / np.max([1, (f['expiryTime'] - point_in_time).seconds * 365.25*24*3600]),axis=1)

    #-------------- max weight under margin constraint--------------
    #### IM calc
    account_leverage=exchange.privateGetAccount()['result']
    if float(account_leverage['leverage']) >= 50: print("margin rules not implemented for leverage >=50")
    dummy_size=100000 ## IM is in ^3/2 not linear, but rule typically kicks in at a few M for optimal leverage of 20 so we linearize
    futIM = (futures['imfFactor']*np.sqrt(dummy_size/futures['mark'])).clip(lower=1/float(account_leverage['leverage']))
    ##### max weights ---> CHECK formulas, short < long no ??
    futures['longWeight'] = 1 / (1 + (futIM - futures['collateralWeight']) / 1.1)
    futures['shortWeight'] = -1 / (futIM + 1.1 / (0.05 + futures['collateralWeight']) - 1)

    #---------- compute max leveraged \int{carry moments}, long and short
    # for perps, compute carry history to estimate moments.
    # for future, funding is deterministic because rate change is compensated by carry change (well, modulo funding...)
    LongCarry = futures.apply(lambda f:
        f['longWeight']*(f['bid_rate_slippage_in_' + str(depths)]- hy_history['USD/rate/borrow']+
                        hy_history[f['name'] + '/rate/funding'] if f['type']=='perpetual'
                                else hy_history.loc[point_in_time, f['name'] + '/rate/funding']),
                        axis=1).T
    LongCarry.columns=futures['name'].tolist()

    ShortCarry = futures.apply(lambda f:
        f['shortWeight']*(f['ask_rate_slippage_in_' + str(depths)]- hy_history['USD/rate/borrow']+
                        hy_history[f['name'] + '/rate/funding'] if f['type']=='perpetual'
                                else hy_history.loc[point_in_time, f['name'] + '/rate/funding']
                        + hy_history[f['underlying'] + '/rate/borrow']),
                        axis=1).T
    ShortCarry.columns = futures['name'].tolist()

    ##### E_long/short_t is expectation of carry --> MUST REMOVE SPIKES
    ##### assume direction only depends on sign(E[long]-E[short]), no integral.
    # Freeze direction into Carry_t and assign max weights.
    E_long_t = LongCarry.ewm(times=hy_history.index,halflife=signal_horizon,axis=0).mean()
    E_short_t = ShortCarry.ewm(times=hy_history.index, halflife=signal_horizon, axis=0).mean()

    Carry_t = futures.apply(lambda f:
            LongCarry[f['name']] if E_long_t.loc[point_in_time,f['name']] > E_short_t.loc[point_in_time,f['name']]
            else ShortCarry[f['name']],
            axis=1).T
    Carry_t.columns = futures['name'].tolist()
    ### not sure how to define window as a timedelta :(
    intCarry_t=Carry_t.rolling(int(holding_period.total_seconds()/3600)).mean()

    futures['direction'] = futures.apply(lambda f:
                                         1 if E_long_t.loc[point_in_time, f['name']] > E_short_t.loc[
                                             point_in_time, f['name']]
                                         else -1,
                                         axis=1)

    futures['maxWeight'] = futures['longWeight']*(futures['direction']+1)/2\
                           -futures['shortWeight']*(futures['direction']-1)/2

    ### E_int and C_int are moments of the integral of annualized carry.
    E_int = intCarry_t.ewm(times=hy_history.index, halflife=signal_horizon, axis=0).mean().loc[point_in_time]
    C_int = intCarry_t.ewm(times=hy_history.index, halflife=signal_horizon, axis=0).cov().loc[point_in_time]

    E_USDborrow_int = hy_history['USD/rate/borrow'].rolling(int(holding_period.total_seconds()/3600)).mean()\
        .ewm(times=hy_history.index, halflife=signal_horizon, axis=0).mean().loc[point_in_time]
    ###### then use in convex optimiziation with lagrange multipliers w>0 and sum w=1
    # https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp

    # banal mean variance...risk_aversion hard to interpret
    #objective = lambda x: -(np.dot(x, E_int) - risk_aversion * np.dot(x, np.dot(covariance, x)))
    #objective_jac = lambda x: -(expectation - risk_aversion * np.dot(covariance, x))
    futures.set_index('name', inplace=True)
    maxWeight_list = futures['maxWeight']
    ### objective is E_int + USDborrow * min(0,sum(w)), subject to weight bounds, sum and loss probability ceiling
    objective = lambda x: -(np.dot(x,E_int)) - E_USDborrow_int * min([0,np.dot(x,maxWeight_list)])# - risk_aversion* np.dot(x,np.dot(covariance,x)))
    objective_jac= lambda x: -(E_int) - (maxWeight_list*E_USDborrow_int if np.dot(x,maxWeight_list)<0 else np.zeros(len(x)))

    n = len(futures.index)
    loss_tolerance_constraint = {'type': 'ineq', ##### maybe Bounds is simpler ?
                 'fun': lambda x: loss_tolerance - norm(loc=np.dot(x,E_int), scale=np.dot(x,np.dot(C_int,x))).cdf(0)}
                 ## dunno ,'jac': lambda x: 0}
    margin_constraint = {'type': 'ineq',
               'fun': lambda x: 1-sum(x),
               'jac': lambda x: -np.ones(n)}
    bounds = scipy.optimize.Bounds(lb=np.zeros(n), ub=concentration_limit*np.ones(n))

    # guess: normalized point in time expectation
    x0=np.array(E_int)/sum(E_int)

    res = scipy.optimize.minimize(objective, x0, method='SLSQP', jac=objective_jac,
            constraints = [loss_tolerance_constraint,margin_constraint],bounds = bounds,
            options = {'ftol': 1e-9, 'disp': True})

    futures['optimalWeight'] = res['x'] * maxWeight_list
    futures.loc['USD', 'optimalWeight'] = -np.dot(res['x'],maxWeight_list)
    futures['optimalCarry'] = res['x'] * E_int
    futures.loc['USD','optimalCarry'] = E_USDborrow_int * min([0,np.dot(res['x'],maxWeight_list)])

    for col in futures.sort_values(by='optimalCarry',ascending=False).head(5).index:
        all=pd.concat([LongCarry[col], ShortCarry[col], E_long_t[col], E_short_t[col], Carry_t[col], intCarry_t[col]],
                       axis=1)
        all.columns=['LongCarry', 'ShortCarry', 'E_long_t', 'E_short_t', 'Carry_t', 'intCarry_t']
        all.to_excel(col+'.xlsx',sheet_name=col)

    futures['E_int']=futures['symbol'].apply(lambda f: E_int.loc(point_in_time,f))
    futures[['symbol', 'borrow', 'quote_borrow', 'basis_mid', 'E_int', 'longWeight', 'shortWeight', 'direction', 'optimalWeight', 'optimalCarry']].\
        to_excel('optimal.xlsx',sheet_name='summary')

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