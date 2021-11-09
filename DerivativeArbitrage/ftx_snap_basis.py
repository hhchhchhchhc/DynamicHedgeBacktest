import pandas as pd
import scipy.optimize
from scipy.stats import norm
from pandas.tseries.frequencies import to_offset

from ftx_utilities import *
from ftx_portfolio import *
from ftx_history import *
from ftx_ftx import *

def enricher(exchange,futures):
    coin_details=pd.DataFrame(exchange.publicGetWalletCoins()['result'],dtype=float).set_index('id')
    markets = exchange.fetch_markets()
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

    futures['carryLong']=futures['basis_mid']-futures['quote_borrow']
    futures['carryShort']=-(futures['basis_mid']-futures['quote_borrow']+futures['borrow'])
    futures['direction_mid']=futures.apply(lambda f: 1 if (f['basis_mid']-f['quote_borrow']+0.5*f['borrow']>0) else -1, axis=1)
    futures['carry_mid'] = futures.apply(lambda f: f['carryLong'] if (f['direction_mid']>0) else f['carryShort'],axis=1)
    return futures.drop(columns=['carryLong','carryShort'])

def basis_scanner(exchange, futures, hy_history,
                  point_in_time='live',
                  slippage_override=2e-4,  # external slippage override
                  slippage_scaler=0.25,  # if scaler from order book
                  slippage_orderbook_depth=0,
                  holding_period=timedelta(days=3),  # to convert slippag into rate
                  signal_horizon=timedelta(days=3),  # historical window for expectations
                  concentration_limit=0.25,
                  loss_tolerance=0.1,  # for markovitz
                  marginal_coin_penalty=0.05,
                  params={'override_slippage':True}):             # use external rather than order book
    borrows = fetch_coin_details(exchange)
    markets = exchange.fetch_markets()
    ### remove blanks for this
    hy_history = hy_history.fillna(method='ffill',limit=2,inplace=False)
    # TODO: check hy_history is hourly

    #-----------------  screening --------------
    ### only if active and  spot trades
    futures=futures[(futures['enabled']==True)&(futures['type']!="move")]
    futures=futures[futures.apply(lambda f: float(find_spot_ticker(markets,f,'ask')),axis=1)>0.0] #### only if spot trades
    #### need borrow to be present
    futures = futures[futures['borrow']>=-999]

    #-------------------- transaction costs---------------
    # add slippage, fees and speed. Pls note we will trade both legs, at entry and exit.
    # Unless slippage override, calculate from orderbook (override Only live is supported).

    if params['override_slippage']==True:
        futures['bid_rate_slippage_in_' + str(slippage_orderbook_depth)]=slippage_override
        futures['ask_rate_slippage_in_' + str(slippage_orderbook_depth)]=slippage_override
    else:
        fees=2*(exchange.fetch_trading_fees()['taker']+exchange.fetch_trading_fees()['maker'])
        ### relative semi-spreads incl fees, and speed
        if slippage_orderbook_depth==0:
            futures['spot_ask_in_0'] = fees+futures.apply(lambda f: 0.5*(float(find_spot_ticker(markets, f, 'ask'))/float(find_spot_ticker(markets, f, 'bid'))-1), axis=1)*slippage_scaler
            futures['spot_bid_in_0'] = -futures['spot_ask_in_0']
            futures['future_ask_in_0'] = fees+0.5*(futures['ask'].astype(float)/futures['bid'].astype(float)-1)*slippage_scaler
            futures['future_bid_in_0'] = -futures['future_ask_in_0']
            futures['speed_in_0']=0##*futures['future_ask_in_0'] ### just 0
        else:
            futures['spot_ask_in_' + str(slippage_orderbook_depth)] = futures['spot_ticker'].apply(lambda x: mkt_depth(exchange,x, 'asks', slippage_orderbook_depth))*slippage_scaler+fees
            futures['spot_bid_in_' + str(slippage_orderbook_depth)] = futures['spot_ticker'].apply(lambda x: mkt_depth(exchange,x, 'bids', slippage_orderbook_depth))*slippage_scaler - fees
            futures['future_ask_in_'+str(slippage_orderbook_depth)] = futures['name'].apply(lambda x: mkt_depth(exchange,x,'asks',slippage_orderbook_depth))*slippage_scaler+fees
            futures['future_bid_in_' + str(slippage_orderbook_depth)] = futures['name'].apply(lambda x: mkt_depth(exchange,x, 'bids', slippage_orderbook_depth))*slippage_scaler - fees
            futures['speed_in_'+str(slippage_orderbook_depth)]=futures['name'].apply(lambda x:mkt_speed(exchange,x,slippage_orderbook_depth).seconds)

        #### rate slippage assuming perps are rolled every perp_holding_period
        #### use centred bid ask for robustness
        futures['expiryTime'] = futures.apply(lambda x:
                                              futures['expiryTime'] if x['type'] == 'future'
                                              else point_in_time + holding_period,
                                              axis=1)  # .replace(tzinfo=timezone.utc)

        futures['bid_rate_slippage_in_' + str(slippage_orderbook_depth)] = futures.apply(lambda f: \
            (f['future_bid_in_' + str(slippage_orderbook_depth)]-f['spot_ask_in_' + str(slippage_orderbook_depth)]) \
            / np.max([1, (f['expiryTime'] - point_in_time).seconds/(365.25*24*3600)]),axis=1)
        futures['ask_rate_slippage_in_' + str(slippage_orderbook_depth)] = futures.apply(lambda f: \
            (f['future_ask_in_' + str(slippage_orderbook_depth)] - f['spot_bid_in_' + str(slippage_orderbook_depth)]) \
            / np.max([1, (f['expiryTime'] - point_in_time).seconds/(365.25*24*3600)]),axis=1)

    #-------------- max weight under margin constraint--------------

    ##### max weights ---> CHECK formulas, short < long no ??
    future_im=futures.apply(lambda f: IM(f),axis=1) ## default size 10k
    futures['MaxLongWeight'] = 1 / (1 + (future_im - futures['collateralWeight']) / 1.1)
    futures['MaxShortWeight'] = -1 / (future_im + 1.1 / (0.05 + futures['collateralWeight']) - 1)

    #---------- compute max leveraged \int{carry moments}, long and short. To find direction, not weights.
    # for perps, compute carry history to estimate moments.
    # for future, funding is deterministic because rate change is compensated by carry change (well, modulo borrow...)
    MaxLongCarry = futures.apply(lambda f:
        f['MaxLongWeight']*(f['bid_rate_slippage_in_' + str(slippage_orderbook_depth)]- hy_history['USD/rate/borrow']+
                        hy_history[f['name'] + '/rate/funding'] if f['type']=='perpetual'
                                else hy_history.loc[point_in_time, f['name'] + '/rate/funding']),
                        axis=1).T
    MaxLongCarry.columns=futures['name'].tolist()

    MaxShortCarry = futures.apply(lambda f:
        f['MaxShortWeight']*(f['ask_rate_slippage_in_' + str(slippage_orderbook_depth)]- hy_history['USD/rate/borrow']+
                        hy_history[f['name'] + '/rate/funding'] if f['type']=='perpetual'
                                else hy_history.loc[point_in_time, f['name'] + '/rate/funding']
                        + hy_history[f['underlying'] + '/rate/borrow']),
                        axis=1).T
    MaxShortCarry.columns = futures['name'].tolist()

    ##### assume direction only depends on sign(E[long]-E[short]), no integral.
    ##### Freeze direction into Carry_t and assign max weights.

    # E_long/short_t is median of carry (to remove spikes)
    E_long_t = MaxLongCarry.rolling(int(signal_horizon.total_seconds()/3600)).median()
    E_short_t = MaxShortCarry.rolling(int(signal_horizon.total_seconds()/3600)).median()

    # freeze direction
    futures['direction'] = futures.apply(lambda f:
                                         1 if E_long_t.loc[point_in_time, f['name']] > E_short_t.loc[
                                             point_in_time, f['name']]
                                         else -1,
                                         axis=1)

    # compute \int(carry). We're done with direction so remove the max leverage.
    Carry_t = futures.apply(lambda f:
                            MaxLongCarry[f['name']]/f['MaxLongWeight'] if f['direction']>0
                            else -MaxShortCarry[f['name']]/f['MaxShortWeight'],
                            axis=1).T
    Carry_t.columns = futures['name'].tolist()
    integralCarry_t=Carry_t.rolling(int(holding_period.total_seconds()/3600)).mean()
    USDborrow_t=hy_history['USD/rate/borrow']
    USDborrow_int=USDborrow_t.rolling(int(holding_period.total_seconds() / 3600)).mean()

    ### E_int and C_int are median, and cov of the integral of annualized carry. E_USDborrow_int is mean.
    E_int = integralCarry_t.rolling(int(signal_horizon.total_seconds()/3600)).median().loc[point_in_time]
    C_int = integralCarry_t.ewm(times=hy_history.index, halflife=signal_horizon, axis=0).cov().loc[point_in_time]
    E_USDborrow_int = USDborrow_int.rolling(int(signal_horizon.total_seconds()/3600)).median().loc[point_in_time]

    ###### then use in convex optimiziation with lagrange multipliers w>0 and sum w=1
    # https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp
    futures.set_index('name', inplace=True)
    ### objective is E_int but:
    # - do not earn USDborrow if usd balance (1-sum)>0
    # - need 5% carry to add a coin
    objective = lambda x: -(np.dot(x,E_int) - E_USDborrow_int*max([0,1-sum(x)])) # don't receive borrow if usd positive
                          #+ marginal_coin_penalty*sum(np.array([np.min[1,np.abs(i)/0.001] for i in x]))

    objective_jac= lambda x: -(E_int + (E_USDborrow_int if 1-sum(x)>0 else np.zeros(len(x))) )
                           #  + marginal_coin_penalty*np.array([np.sign(i)/0.001 if np.abs(i)<0.001 else 0 for i in x])

    #subject to weight bounds, margin and loss probability ceiling
    n = len(futures.index)
    loss_tolerance_constraint = {'type': 'ineq',
                'fun': lambda x: loss_tolerance - norm(loc=np.dot(x,E_int), scale=np.dot(x,np.dot(C_int,x))).cdf(0)}
    margin_constraint = {'type': 'ineq',
                'fun': lambda x: excessIM(futures,x).sum()}
    ## TODO: implement with shock
    stopout_constraint = {'type': 'ineq',
                'fun': lambda x: excessMM(futures,x).sum()}

    bounds = scipy.optimize.Bounds(lb=np.asarray([0 if w>0 else -concentration_limit for w in futures['direction']]),
                                   ub=np.asarray([0 if w<0 else  concentration_limit for w in futures['direction']]))

    # guess: normalized point in time expectation
    x0=np.array(E_int)/sum(E_int)

    res = scipy.optimize.minimize(objective, x0, method='SLSQP', jac=objective_jac,
                                  constraints = [margin_constraint,stopout_constraint], # ,loss_tolerance_constraint
                                  bounds = bounds,
                                  options = {'ftol': 1e-6, 'disp': True})

    futures['spotCarry']=Carry_t.loc[point_in_time]
    futures['medianCarryInt']=E_int
    futures['optimalWeight'] = res['x']
    futures['ExpectedCarry'] = res['x'] * E_int
    futures['RealizedCarry'] = res['x'] * integralCarry_t.loc[point_in_time]
    futures['excessIM'] = excessIM(futures, futures['optimalWeight'])
    futures['excessMM'] = excessMM(futures, futures['optimalWeight'])

    futures.loc['USD', 'spotCarry'] = USDborrow_t.loc[point_in_time]
    futures.loc['USD', 'medianCarryInt'] = E_USDborrow_int
    futures.loc['USD', 'optimalWeight'] = 1-sum(res['x'])
    futures.loc['USD', 'ExpectedCarry'] = E_USDborrow_int * min([0,1-sum(res['x'])])
    futures.loc['USD', 'RealizedCarry'] = USDborrow_int.loc[point_in_time]* min([0,1-sum(res['x'])])
    futures.loc['USD', 'excessIM'] = excessIM(futures, futures['optimalWeight'])['USD']
    futures.loc['USD', 'excessMM'] = excessMM(futures, futures['optimalWeight'])['USD']

    futures.loc['total', 'ExpectedCarry'] = futures['ExpectedCarry'].sum()
    futures.loc['total', 'RealizedCarry'] = futures['RealizedCarry'].sum()
    futures.loc['total', 'excessIM'] = excessIM(futures, futures['optimalWeight']).sum()
    futures.loc['total', 'excessMM'] = excessMM(futures, futures['optimalWeight']).sum()
    futures.loc['total', 'lossProbability'] = loss_tolerance - loss_tolerance_constraint['fun'](res['x'])

    temporary=futures
    temporary['absWeight']=futures['optimalWeight'].apply(np.abs,axis=1)
    for col in temporary.sort_values(by='absWeight',ascending=False).drop(index=['USD','total']).head(5).index:
        all=pd.concat([MaxLongCarry[col],
                       MaxShortCarry[col],
                       E_long_t[col],
                       E_short_t[col],
                       Carry_t[col],
                       integralCarry_t[col]],
                       axis=1)
        all.columns=['MaxLongCarry', 'MaxShortCarry', 'E_long_t', 'E_short_t', 'Carry_t', 'integralCarry_t']
        all.to_excel(col+'.xlsx',sheet_name=col)
    futures.to_excel('futuresinfo.xlsx')
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