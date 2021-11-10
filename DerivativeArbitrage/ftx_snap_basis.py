import pandas as pd
import scipy.optimize
from scipy.stats import norm
from pandas.tseries.frequencies import to_offset

from ftx_utilities import *
from ftx_portfolio import *
from ftx_history import *
from ftx_ftx import *

# adds info, transcation costs, and basic screening
def enricher(exchange,input_futures,holding_period,
             slippage_override= -999, slippage_orderbook_depth= 0,
             slippage_scaler= 1.0, params={'override_slippage': True,'type_allowed':'perpetual','fee_mode':'retail'}):
    futures=pd.DataFrame(input_futures)
    markets=exchange.fetch_markets()

    # basic screening
    futures = futures[
        (futures['expired'] == False) & (futures['enabled'] == True) & (futures['type'] != "move")
        & (futures.apply(lambda f: float(find_spot_ticker(markets, f, 'ask')), axis=1) > 0.0)
        & (futures['tokenizedEquity'] != True)
        & (futures['type'] == params['type_allowed'])]
    
    ########### add borrows
    coin_details = pd.DataFrame(exchange.publicGetWalletCoins()['result'], dtype=float).set_index('id')
    borrows = fetch_coin_details(exchange)
    futures = pd.merge(futures, borrows[['borrow', 'lend', 'funding_volume']], how='left', left_on='underlying',
                       right_index=True)
    futures['quote_borrow'] = float(borrows.loc['USD', 'borrow'])
    futures['quote_lend'] = float(borrows.loc['USD', 'lend'])
    ########### naive basis for all futures
    futures.loc[futures['type'] == 'perpetual', 'basis_mid'] = futures.loc[futures['type'] == 'perpetual','name'].apply(
        lambda f:float(exchange.publicGetFuturesFutureNameStats({'future_name': f})['result']['nextFundingRate']) * 24 * 365.25)
    futures.loc[futures['type'] == 'future', 'basis_mid'] = futures[futures['type'] == 'future'].apply(
        lambda f: calc_basis(f['mark'], f['index'], f['expiryTime'], datetime.now()), axis=1)

    #### need borrow to be present
    futures = futures[futures['borrow']>=-999]

    # spot carries
    futures['carryLong']=futures['basis_mid']-futures['quote_borrow']
    futures['carryShort']=-(futures['basis_mid']-futures['quote_borrow']+futures['borrow'])
    futures['direction_mid']=futures.apply(lambda f: 1 if (f['basis_mid']-f['quote_borrow']+0.5*f['borrow']>0) else -1, axis=1)
    futures['carry_mid'] = futures.apply(lambda f: f['carryLong'] if (f['direction_mid']>0) else f['carryShort'],axis=1)

    # transaction costs
    costs=fetch_rate_slippage(futures, exchange, holding_period,
        slippage_override, slippage_orderbook_depth, slippage_scaler,
        params)
    futures = futures.join(costs, how = 'outer')

    ##### max weights ---> TODO CHECK formulas, short < long no ??
    future_im=futures.apply(lambda f: IM(f),axis=1) ## default size 10k
    futures['MaxLongWeight'] = 1 / (1.1 + (future_im - futures['collateralWeight']))
    futures['MaxShortWeight'] = -1 / (future_im + 1.1 / futures.apply(lambda f:collateralWeightInitial(f),axis=1) - 1)

    return futures.drop(columns=['carryLong','carryShort'])

def update(input_futures,point_in_time=datetime.now()+timedelta(weeks=1),history=pd.DataFrame()):
    futures=pd.DataFrame(input_futures)
    ########### add borrows
    futures['borrow']=futures['underlying'].apply(lambda f:history.loc[point_in_time,f+'/rate/borrow'])
    futures['borrow'] = futures['underlying'].apply(lambda f: history.loc[point_in_time, f + '/rate/borrow'])
    futures['lend'] = futures['underlying'].apply(lambda f:history.loc[point_in_time, f + '/rate/borrow'])*.9 # TODO:lending rate
    futures['quote_borrow'] = history.loc[point_in_time, 'USD/rate/borrow']
    futures['quote_lend'] = history.loc[point_in_time, 'USD/rate/borrow'] * .9  # TODO:lending rate
    ########### naive basis for all futures
    futures.loc[futures['type'] == 'perpetual', 'basis_mid'] = futures.loc[futures['type'] == 'perpetual','name'].apply(
        lambda f: history.loc[point_in_time, f + '/rate/funding'],axis=1)
    futures['mark']=futures['underlying'].apply(lambda f:history.loc[point_in_time,f+'/mark/o'])
    futures['index'] = futures['underlying'].apply(lambda f: history.loc[point_in_time, f + '/index/o'])
    futures.loc[futures['type'] == 'future','expiryTime'] = futures.loc[futures['type'] == 'future','underlying'].apply(
        lambda f: history.loc[point_in_time, f + '/rate/T'],axis=1)
    futures.loc[futures['type'] == 'future', 'basis_mid'] = futures[futures['type'] == 'future'].apply(
        lambda f: calc_basis(f['mark'], f['index'], f['expiryTime'], datetime.now()), axis=1)

    # spot carries
    futures['carryLong']=futures['basis_mid']-futures['quote_borrow']
    futures['carryShort']=-(futures['basis_mid']-futures['quote_borrow']+futures['borrow'])
    futures['direction_mid']=futures.apply(lambda f: 1 if (f['basis_mid']-f['quote_borrow']+0.5*f['borrow']>0) else -1, axis=1)
    futures['carry_mid'] = futures.apply(lambda f: f['carryLong'] if (f['direction_mid']>0) else f['carryShort'],axis=1)

    return futures.drop(columns=['carryLong','carryShort'])

#-------------------- transaction costs---------------
# add slippage, fees and speed. Pls note we will trade both legs, at entry and exit.
# Unless slippage override, calculate from orderbook (override Only live is supported).
def fetch_rate_slippage(input_futures, exchange: Exchange,holding_period,
                            slippage_override: int = -999, slippage_orderbook_depth: float = 0,
                            slippage_scaler: float = 1.0,params={'override_slippage':True,'fee_mode':'retail'}) -> None:
    futures=pd.DataFrame(input_futures)
    point_in_time=datetime.now()
    markets=exchange.fetch_markets()
    if params['override_slippage']==True:
        futures['spot_ask'] = slippage_override
        futures['spot_bid'] = -slippage_override
        futures['future_ask'] = slippage_override
        futures['future_bid'] = -slippage_override
    else: ## rubble calc:
        fees=(0.6*0.00015-0.0001+2*0.00006) if (params['fee_mode']=='hr') \
            else (exchange.fetch_trading_fees()['taker']+exchange.fetch_trading_fees()['maker']*0) #maker fees 0 with 26 FTT staked
        ### relative semi-spreads incl fees, and speed
        if slippage_orderbook_depth==0:
            futures['spot_ask'] = fees+futures.apply(lambda f: 0.5*(float(find_spot_ticker(markets, f, 'ask'))/float(find_spot_ticker(markets, f, 'bid'))-1), axis=1)*slippage_scaler
            futures['spot_bid'] = -futures['spot_ask']
            futures['future_ask'] = fees+0.5*(futures['ask'].astype(float)/futures['bid'].astype(float)-1)*slippage_scaler
            futures['future_bid'] = -futures['future_ask']
            #futures['speed']=0##*futures['future_ask'] ### just 0
        else:
            futures['spot_ask'] = futures['spot_ticker'].apply(lambda x: mkt_depth(exchange,x, 'asks', slippage_orderbook_depth))*slippage_scaler+fees
            futures['spot_bid'] = futures['spot_ticker'].apply(lambda x: mkt_depth(exchange,x, 'bids', slippage_orderbook_depth))*slippage_scaler - fees
            futures['future_ask'] = futures['name'].apply(lambda x: mkt_depth(exchange,x,'asks',slippage_orderbook_depth))*slippage_scaler+fees
            futures['future_bid'] = futures['name'].apply(lambda x: mkt_depth(exchange,x, 'bids', slippage_orderbook_depth))*slippage_scaler - fees
            #futures['speed_in_'+str(slippage_orderbook_depth)]=futures['name'].apply(lambda x:mkt_speed(exchange,x,slippage_orderbook_depth).seconds)

    #### rate slippage assuming perps are rolled every perp_holding_period
    #### use both bid and ask for robustness, but don't x2 for entry+exit
    futures['expiryTime'] = futures.apply(lambda x:
                                          x['expiryTime'] if x['type'] == 'future'
                                          else point_in_time + holding_period,
                                          axis=1)  # .replace(tzinfo=timezone.utc)

    # buy is negative, sell is positive
    buy_slippage=futures['future_bid'] - futures['spot_ask']
    bid_rate_slippage = futures.apply(lambda f: \
        (f['future_bid']- f['spot_ask']) \
        / np.max([1, (f['expiryTime'] - point_in_time).total_seconds()/3600])*365.25*24,axis=1) # no less than 1h
    sell_slippage=futures['future_ask'] - futures['spot_bid']
    ask_rate_slippage = futures.apply(lambda f: \
        (f['future_ask'] - f['spot_bid']) \
        / np.max([1, (f['expiryTime'] - point_in_time).total_seconds()/3600])*365.25*24,axis=1) # no less than 1h

    return pd.DataFrame({
        'buy_slippage': buy_slippage,
        'sell_slippage': sell_slippage,
        'bid_rate_slippage':bid_rate_slippage,
        'ask_rate_slippage':ask_rate_slippage,
    })

def basis_scanner(exchange, input_futures, hy_history,
                  point_in_time,
                  previous_weights,
                  holding_period,  # to convert slippag into rate
                  signal_horizon,  # historical window for expectations
                  concentration_limit=99,
                  loss_tolerance=1,  # for markovitz
                  marginal_coin_penalty=0):             # use external rather than order book
    futures=pd.DataFrame(input_futures)
    ### remove blanks for this
    hy_history = hy_history.fillna(method='ffill',limit=2,inplace=False)
    # TODO: check hy_history is hourly

    #---------- compute max leveraged \int{carry moments}, long and short. To find direction, not weights.
    # for perps, compute carry history to estimate moments.
    # for future, funding is deterministic because rate change is compensated by carry change (well, modulo borrow...)
    # TODO: we no longer have tx costs in carries # 0*f['bid_rate_slippage']
    MaxLongCarry = futures.apply(lambda f:
        f['MaxLongWeight']*(- hy_history['USD/rate/borrow']+
                        hy_history[f['name'] + '/rate/funding'] if f['type']=='perpetual'
                                else hy_history.loc[point_in_time, f['name'] + '/rate/funding']),
                        axis=1).T
    MaxLongCarry.columns=futures['name'].tolist()

    MaxShortCarry = futures.apply(lambda f:
        f['MaxShortWeight']*(- hy_history['USD/rate/borrow']+
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
    holding_hours=int(holding_period.total_seconds()/3600)
    integralCarry_t=Carry_t.rolling(holding_hours).mean()
    USDborrow_t=hy_history['USD/rate/borrow']
    USDborrow_int=USDborrow_t.rolling(holding_hours).mean()

    ### E_int and C_int are median, and cov of the integral of annualized carry. E_USDborrow_int is mean.
    E_int = integralCarry_t.rolling(int(signal_horizon.total_seconds()/3600)).median().loc[point_in_time]
    C_int = integralCarry_t.ewm(times=hy_history.index, halflife=signal_horizon, axis=0).cov().loc[point_in_time]
    E_USDborrow_int = USDborrow_int.rolling(int(signal_horizon.total_seconds()/3600)).median().loc[point_in_time]

    ###### then use in convex optimiziation with lagrange multipliers w>0 and sum w=1
    # https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp
    # TODO: need to decide when to do that
    futures.set_index('name', inplace=True)
    ### objective is E_int but:
    # - do not earn USDborrow if usd balance (1-sum)>0
    # - need 5% carry to add a coin
    buy_slippage=futures['buy_slippage'].values*365.25*24/holding_hours
    sell_slippage = futures['sell_slippage'].values*365.25*24/holding_hours
    # don't receive borrow if usd positive
    objective = lambda x: -(
            np.dot(x,E_int) \
            - E_USDborrow_int * max([0,1-sum(x)]) \
            + sum([(x - previous_weights)[i]*buy_slippage[i] if (x - previous_weights)[i]>0
                   else (x - previous_weights)[i]*sell_slippage[i] for i in range(len(x - previous_weights))])
    )
                          #+ marginal_coin_penalty*sum(np.array([np.min[1,np.abs(i)/0.001] for i in x]))

    objective_jac= lambda x: -(
            E_int
            - ( -E_USDborrow_int if 1-sum(x)>0 else np.zeros(len(x)))
            + np.array( [buy_slippage[i] if (x - previous_weights)[i] > 0
                         else sell_slippage[i] for i in range(len(x - previous_weights))])
    )
    #  + marginal_coin_penalty*np.array([np.sign(i)/0.001 if np.abs(i)<0.001 else 0 for i in x])
    progress_display=[]
    i=0
    def callbackF(x,progress_display):
        progress_display+=[pd.Series({
            'E_int':np.dot(x,E_int),
            'usdBorrowCost':- E_USDborrow_int * max([0,1-sum(x)]),
            'tx_cost':+ sum([(x - previous_weights)[i]*buy_slippage[i] if (x - previous_weights)[i]>0
                   else (x - previous_weights)[i]*sell_slippage[i] for i in range(len(x - previous_weights))]),
            'loss_tolerance_constraint':loss_tolerance_constraint['fun'](x),
            'margin_constraint': margin_constraint['fun'](x),
            'stopout_constraint': stopout_constraint['fun'](x)
        }).append(pd.Series(index=futures.index, data=x))]

    #subject to weight bounds, margin and loss probability ceiling
    n = len(futures.index)
    loss_tolerance_constraint = {'type': 'ineq',
                'fun': lambda x: loss_tolerance - norm(loc=np.dot(x,E_int), scale=np.dot(x,np.dot(C_int,x))).cdf(0)}
    margin_constraint = {'type': 'ineq',
                'fun': lambda x: excessIM(futures,x).sum()}
    stopout_constraint = {'type': 'ineq',
                'fun': lambda x: excessMM(futures,x).sum()}
    bounds = scipy.optimize.Bounds(lb=np.asarray([0 if w>0 else -concentration_limit for w in futures['direction']]),
                                   ub=np.asarray([0 if w<0 else  concentration_limit for w in futures['direction']]))

    # guess: normalized point in time expectation
    x0=np.array(E_int)/sum(E_int)
    callbackF(x0, progress_display)

    res = scipy.optimize.minimize(objective, x0, method='SLSQP', jac=objective_jac,
                                  constraints = [margin_constraint,stopout_constraint], # ,loss_tolerance_constraint
                                  bounds = bounds,
                                  callback=lambda x:callbackF(x,progress_display),
                                  options = {'ftol': 1e-4, 'disp': True})
    callbackF(res['x'], progress_display)

    futures['spotCarry']=Carry_t.loc[point_in_time]
    futures['medianCarryInt']=E_int
    futures['optimalWeight'] = res['x']
    futures['ExpectedCarry'] = E_int
    futures['RealizedCarry'] = integralCarry_t.loc[point_in_time]
    futures['excessIM'] = excessIM(futures, futures['optimalWeight'])
    futures['excessMM'] = excessMM(futures, futures['optimalWeight'])

    futures.loc['USD', 'spotCarry'] = USDborrow_t.loc[point_in_time]
    futures.loc['USD', 'medianCarryInt'] = E_USDborrow_int
    futures.loc['USD', 'optimalWeight'] = 1-sum(res['x'])
    futures.loc['USD', 'ExpectedCarry'] = E_USDborrow_int
    futures.loc['USD', 'RealizedCarry'] = USDborrow_int.loc[point_in_time]
    futures.loc['USD', 'excessIM'] = excessIM(futures, futures['optimalWeight'])['USD']
    futures.loc['USD', 'excessMM'] = excessMM(futures, futures['optimalWeight'])['USD']

    futures.loc['total', 'ExpectedCarry'] = futures['ExpectedCarry'].mean()
    futures.loc['total', 'RealizedCarry'] = futures['RealizedCarry'].mean()
    futures.loc['total', 'excessIM'] = excessIM(futures, futures['optimalWeight']).sum()
    futures.loc['total', 'excessMM'] = excessMM(futures, futures['optimalWeight']).sum()
    futures.loc['total', 'lossProbability'] = loss_tolerance - loss_tolerance_constraint['fun'](res['x'])

    futures['absWeight']=futures['optimalWeight'].apply(np.abs,axis=1)
    with pd.ExcelWriter('paths.xlsx', engine='xlsxwriter') as writer:
        futures.to_excel(writer, sheet_name='futureinfo')
        pd.concat(progress_display,axis=1).to_excel(writer, sheet_name='optimPath')
        for col in futures.sort_values(by='absWeight',ascending=False).drop(index=['USD','total']).head(5).index:
            all=pd.concat([MaxLongCarry[col],
                           MaxShortCarry[col],
                           E_long_t[col],
                           E_short_t[col],
                           Carry_t[col],
                           integralCarry_t[col]],
                           axis=1)
            all.columns=['MaxLongCarry', 'MaxShortCarry', 'E_long_t', 'E_short_t', 'Carry_t', 'integralCarry_t']
            all.to_excel(writer,sheet_name=col)

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