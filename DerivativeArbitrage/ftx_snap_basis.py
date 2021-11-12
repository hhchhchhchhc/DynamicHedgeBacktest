import pandas as pd
import scipy.optimize
from scipy.stats import norm
from pandas.tseries.frequencies import to_offset

from ftx_utilities import *
from ftx_portfolio import *
from ftx_history import *
from ftx_ftx import *

# adds info, transcation costs, and basic screening
def enricher(exchange,input_futures,holding_period,equity=1e5,
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
    futures.loc[futures['type'] == 'perpetual', 'basis_mid'] = futures[futures['type'] == 'perpetual'].apply(
        lambda f:float(exchange.publicGetFuturesFutureNameStats({'future_name': f.name})['result']['nextFundingRate']) * 24 * 365.25,axis=1)
    futures.loc[futures['type'] == 'future', 'basis_mid'] = futures[futures['type'] == 'future'].apply(
        lambda f: calc_basis(f['mark'], f['index'], f['expiryTime'], datetime.now()), axis=1)

    #### need borrow to be present
    futures = futures[futures['borrow']>=-999]

    # spot carries
    futures['carryLong']=futures['basis_mid']-futures['quote_borrow']
    futures['carryShort']=futures['basis_mid']-futures['quote_borrow']+futures['borrow']
    futures['direction_mid']=1
    futures.loc[futures['carryShort']+futures['carryLong']<0,'direction_mid']=-1
    futures['carry_mid'] = futures['carryLong']
    futures.loc[futures['direction_mid']<0,'carry_mid'] = futures['carryShort']

    # transaction costs
    costs=fetch_rate_slippage(futures, exchange, holding_period,
        slippage_override, slippage_orderbook_depth, slippage_scaler,
        params)
    futures = futures.join(costs, how = 'outer')

    ##### max weights ---> TODO CHECK formulas, short < long no ??
    future_im=futures.apply(lambda f:
            (f['imfFactor'] * np.sqrt(equity / f['mark'])).clip(min=1 / f['account_leverage']),
                            axis=1)
    futures['MaxLongWeight'] = 1 / (1.1 + (future_im - futures['collateralWeight']))
    futures['MaxShortWeight'] = -1 / (future_im + 1.1 / futures.apply(lambda f:collateralWeightInitial(f),axis=1) - 1)
    ##### using 1d optimization...failed !
#    IM_constraint=futures.apply(lambda f:
#                {'type': 'ineq','fun':(lambda w: excessMargin(pd.DataFrame(f).T,w)['IM'])},
#                                axis=1)
#    MM_constraint=futures.apply(lambda f:
#                {'type': 'ineq', 'fun': (lambda w: excessMargin(pd.DataFrame(f).T, w)['MM'])},
#                                axis=1)
#    futures[futures['direction_mid']>0,'MaxLongWeight'] = futures[futures['direction_mid']>0].apply(lambda f:
#                    scipy.optimize.minimize(fun=(lambda w: (w*f['carryLong'])[0]), x0=np.array([5]), method='SLSQP',
#                    constraints=[IM_constraint[f.name], MM_constraint[f.name]],  # ,loss_tolerance_constraint
#                    bounds=scipy.optimize.Bounds(lb=np.array([0]),ub=np.array([10])),
#                    options={'ftol': 1e-2, 'disp': False})['x'][0],
#                                                                                            axis=1)
#    futures[futures['direction_mid']<0,'MaxShortWeight'] = futures[futures['direction_mid']<0].apply(lambda f:
#                    scipy.optimize.minimize(fun=(lambda w: (w*f['carryShort'])[0]), x0=np.array([-5]), method='SLSQP',
#                    constraints=[IM_constraint[f.name], MM_constraint[f.name]],  # ,loss_tolerance_constraint
#                    bounds=scipy.optimize.Bounds(lb=np.array([-10]),ub=np.array([0])),
#                    options={'ftol': 1e-2, 'disp': False})['x'][0],
#                                                                                            axis=1)

    return futures.drop(columns=['carryLong','carryShort'])

def update(input_futures,point_in_time,history,equity,
           intLongCarry, intShortCarry, intUSDborrow,E_long,E_short,E_intUSDborrow):
    futures=pd.DataFrame(input_futures)

    ########### add borrows
    futures['borrow']=futures['underlying'].apply(lambda f:history.loc[point_in_time,f + '/rate/borrow'])
    futures['lend'] = futures['underlying'].apply(lambda f:history.loc[point_in_time, f + '/rate/borrow'])*.9 # TODO:lending rate
    futures['quote_borrow'] = history.loc[point_in_time, 'USD/rate/borrow']
    futures['quote_lend'] = history.loc[point_in_time, 'USD/rate/borrow'] * .9  # TODO:lending rate

    ########### naive basis for all futures
    futures.loc[futures['type'] == 'perpetual', 'basis_mid'] = futures[futures['type'] == 'perpetual'].apply(
        lambda f: history.loc[point_in_time, f.name + '/rate/funding'],axis=1)
    futures['mark']=futures.apply(
        lambda f:history.loc[point_in_time,f.name+'/mark/o'],axis=1)
    futures['index'] = futures.apply(
        lambda f: history.loc[point_in_time, f.name + '/indexes/open'],axis=1)
    futures.loc[futures['type'] == 'future','expiryTime'] = futures.loc[futures['type'] == 'future','underlying'].apply(
        lambda f: history.loc[point_in_time, f + '/rate/T'])
    futures.loc[futures['type'] == 'future', 'basis_mid'] = futures[futures['type'] == 'future'].apply(
        lambda f: calc_basis(f['mark'], f['index'], f['expiryTime'], datetime.now()), axis=1)

    # spot carries
    futures['carryLong']=futures['basis_mid']-futures['quote_borrow']
    futures['carryShort']=-(futures['basis_mid']-futures['quote_borrow']+futures['borrow'])
    futures['direction_mid']=1
    futures.loc[futures['carryShort']>futures['carryLong'],'direction_mid']=-1
    futures['carry_mid'] = futures['carryLong']
    futures.loc[futures['direction_mid']>0,'carry_mid'] = futures['carryShort']
    futures=futures.drop(columns=['carryLong','carryShort'])

    # carry expectation at point_in_time
    futures['intLongCarry']  = intLongCarry.loc[point_in_time]
    futures['intShortCarry'] = intShortCarry.loc[point_in_time]
    futures['intUSDborrow']  = intUSDborrow.loc[point_in_time]
    futures['E_long']        = E_long.loc[point_in_time]
    futures['E_short']      = E_short.loc[point_in_time]
    futures['E_intUSDborrow']  = E_intUSDborrow.loc[point_in_time]

    # initialize optimizer functions
    excess_margin = ExcessMargin(futures,equity=equity,
                 long_blowup=0.1, short_blowup=0.2, nb_blowups=3)
    return futures,excess_margin

# return rolling expectations of integrals
def build_derived_history(exchange, input_futures, hy_history,
                  holding_period,  # to convert slippage into rate
                  signal_horizon,  # historical window for expectations
                  verbose=False):             # use external rather than order book
    futures=pd.DataFrame(input_futures)
    ### remove blanks for this
    hy_history = hy_history.fillna(method='ffill',limit=2,inplace=False)
    # TODO: check hy_history is hourly
    holding_hours = int(holding_period.total_seconds() / 3600)

    #---------- compute max leveraged \int{carry moments}, long and short. To find direction, not weights.
    # for perps, compute carry history to estimate moments.
    # for future, funding is deterministic because rate change is compensated by carry change (well, modulo borrow...)
    # TODO: we no longer have tx costs in carries # 0*f['bid_rate_slippage']
    # TODO: doesn't work with futures (needs point in time argument)
    LongCarry = futures.apply(lambda f:
                    (- hy_history['USD/rate/borrow']+
                        hy_history[f.name + '/rate/funding']),
                        axis=1).T
    LongCarry.columns=futures.index.tolist()
    intLongCarry = LongCarry.rolling(holding_hours).mean()
    E_long= intLongCarry.rolling(int(signal_horizon.total_seconds()/3600)).median()

    ShortCarry = futures.apply(lambda f:
                    (- hy_history['USD/rate/borrow']+
                        hy_history[f.name + '/rate/funding']
                     + hy_history[f['underlying'] + '/rate/borrow']),
                        axis=1).T
    ShortCarry.columns = futures.index.tolist()
    intShortCarry = ShortCarry.rolling(holding_hours).mean()
    E_short = intShortCarry.rolling(int(signal_horizon.total_seconds()/3600)).median()

    USDborrow = hy_history['USD/rate/borrow']
    intUSDborrow = USDborrow.rolling(holding_hours).mean()
    E_intUSDborrow = intUSDborrow.rolling(int(signal_horizon.total_seconds()/3600)).median()
    
    if verbose:
        with pd.ExcelWriter('paths.xlsx', engine='xlsxwriter') as writer:
            futures.to_excel(writer, sheet_name='futureinfo')
            pd.concat(progress_display, axis=1).to_excel(writer, sheet_name='optimPath')
            for col in futures.sort_values(by='absWeight', ascending=False).drop(index=['USD', 'total']).head(5).index:
                all = pd.concat([intLongCarry[col],
                                intShortCarry[col],
                                E_long[col],
                                E_short[col]],axis = 1)
            all.columns = ['intLongCarry', 'intShortCarry','intUSDborrow', 'E_long', 'E_short','E_intUSDborrow']
            all.to_excel(writer, sheet_name=col)

    return (intLongCarry,intShortCarry,intUSDborrow,E_long,E_short,E_intUSDborrow)

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
            futures['future_ask'] = futures.apply(lambda x: mkt_depth(exchange,x.name,'asks',slippage_orderbook_depth))*slippage_scaler+fees
            futures['future_bid'] = futures.apply(lambda x: mkt_depth(exchange,x.name, 'bids', slippage_orderbook_depth))*slippage_scaler - fees
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

    holding_hours = int(holding_period.total_seconds() / 3600)
    return pd.DataFrame({
        'buy_slippage': buy_slippage*365.25*24/holding_hours,
        'sell_slippage': sell_slippage*365.25*24/holding_hours,
        'bid_rate_slippage':bid_rate_slippage,
        'ask_rate_slippage':ask_rate_slippage,
    })
def transaction_cost_calculator(dx,buy_slippage,sell_slippage):
    return sum([
        dx[i] * buy_slippage[i] if dx[i] > 0
        else dx[i] * sell_slippage[i]
        for i in range(len(dx))
    ])

def cash_carry_optimizer(exchange, input_futures,excess_margin,
                  previous_weights,
                  holding_period,  # to convert slippag into rate
                  signal_horizon,  # historical window for expectations
                  concentration_limit=99,
                  loss_tolerance=1,
                  equity=1e5,# for markovitz
                  verbose=False):             # use external rather than order book
    futures=pd.DataFrame(input_futures)

    ##### assume direction only depends on sign(E[long]-E[short]), no integral.
    
    # Freeze direction into Carry_t and assign max weights.
    futures['direction'] = 1
    futures.loc[
        futures['E_long']*futures['MaxLongWeight']-futures['E_short']*futures['MaxShortWeight']<0,
        'direction'] =-1
    
    # compute realized=\int(carry) and E[\int(carry)]. We're done with direction so remove the max leverage.
    intCarry=futures['intLongCarry']
    intCarry[futures['direction'] < 0] = futures.loc[futures['direction'] < 0,'intShortCarry']
    intUSDborrow=futures['intUSDborrow'].values[0]
    E_intCarry = futures['E_long']
    E_intCarry[futures['direction'] < 0] = futures.loc[futures['direction'] < 0, 'E_short']
    E_intUSDborrow=futures['E_intUSDborrow'].values[0]
    # TODO: covar pre-update
    # C_int = integralCarry_t.ewm(times=hy_history.index, halflife=signal_horizon, axis=0).cov().loc[point_in_time]

    ###### then use in convex optimiziation with lagrange multipliers w>0 and sum w=1
    # https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp

    ### objective is E_int but:
    # - do not earn USDborrow if usd balance (1-sum)>0
    buy_slippage=futures['buy_slippage'].values
    sell_slippage = futures['sell_slippage'].values
    xt=previous_weights.values
    # don't receive borrow if usd positive
    objective = lambda x: -(
            np.dot(x,E_intCarry) \
            + E_intUSDborrow * min([equity,sum(x)]) \
            + transaction_cost_calculator(x - xt,buy_slippage,sell_slippage)
    )
                          #+ marginal_coin_penalty*sum(np.array([np.min[1,np.abs(i)/0.001] for i in x]))
    objective_jac= lambda x: -(
            E_intCarry
            + (E_intUSDborrow if sum(x)<equity else np.zeros(len(x)))
            + np.array( [buy_slippage[i] if (x - xt)[i] > 0
                         else sell_slippage[i] for i in range(len(x - xt))])
    )

    # --------- breaks down pnl during optimization
    progress_display=[]
    def callbackF(x, progress_display, verbose=False):
        if verbose:
            progress_display += [pd.Series({
                'E_int': np.dot(x, E_intCarry),
                'usdBorrowRefund': E_intUSDborrow * min([equity, sum(x)]),
                'tx_cost': + sum([(x - xt)[i] * buy_slippage[i] if (x - xt)[i] > 0
                                  else (x - xt)[i] * sell_slippage[i] for i in
                                  range(len(x - xt))]),
                #TODO: covar pre-update
                # 'loss_tolerance_constraint': loss_tolerance_constraint['fun'](x),
                'margin_constraint': margin_constraint['fun'](x),
                'stopout_constraint': stopout_constraint['fun'](x)
            }).append(pd.Series(index=futures.index, data=x))]
        return []

    #subject to weight bounds, margin and loss probability ceiling
    n = len(futures.index)
    # TODO: covar pre-update
    #loss_tolerance_constraint = {'type': 'ineq',
    #            'fun': lambda x: loss_tolerance - norm(loc=np.dot(x,E_int), scale=np.dot(x,np.dot(C_int,x))).cdf(0)}
    margin_constraint = {'type': 'ineq',
                'fun': lambda x: excess_margin.call(x)['totalIM']}
    stopout_constraint = {'type': 'ineq',
                'fun': lambda x: excess_margin.call(x)['totalMM']}
    bounds = scipy.optimize.Bounds(lb=np.asarray([0 if w>0 else -concentration_limit*equity for w in futures['direction']]),
                                   ub=np.asarray([0 if w<0 else  concentration_limit*equity for w in futures['direction']]))

    # guess: normalized carry expectation, rescaled to max margins
    x0=equity*np.array(E_intCarry)/sum(E_intCarry)
    x1 = x0/np.max([1-margin_constraint['fun'](x0)/equity,1-stopout_constraint['fun'](x0)/equity])
    callbackF(x1, progress_display,verbose)

    res = scipy.optimize.minimize(objective, x1, method='SLSQP', jac=objective_jac,
                                  constraints = [margin_constraint,stopout_constraint], # ,loss_tolerance_constraint
                                  bounds = bounds,
                                  callback=lambda x:callbackF(x,progress_display,verbose),
                                  options = {'ftol': 1e-4*equity, 'disp': False})
    callbackF(res['x'], progress_display,verbose)

    summary=pd.DataFrame()
    summary['spotCarry']=futures['carry_mid']
    summary['CarryEstimator']=E_intCarry+E_intUSDborrow
    summary['optimalWeight'] = res['x']
    summary['ExpectedCarry'] = res['x'] * (E_intCarry+E_intUSDborrow)
    summary['RealizedCarry'] = previous_weights * (intCarry+intUSDborrow)
    summary['excessIM'] = excess_margin.call(res['x'])['IM']
    summary['excessMM'] = excess_margin.call(res['x'])['MM']

    weight_move=summary['optimalWeight']-previous_weights
    summary['transactionCost']=weight_move*futures['buy_slippage']
    summary.loc[weight_move<0, 'transactionCost'] = weight_move[weight_move < 0] * sell_slippage[weight_move < 0]

    summary.loc['USD', 'spotCarry'] = futures['quote_borrow'].values[0]
    summary.loc['USD', 'CarryEstimator'] = E_intUSDborrow
    summary.loc['USD', 'optimalWeight'] = equity-sum(res['x'])
    summary.loc['USD', 'ExpectedCarry'] = np.min([0,equity-sum(res['x'])])* E_intUSDborrow
    summary.loc['USD', 'RealizedCarry'] = np.min([0,equity-previous_weights.sum()])* intUSDborrow
    summary.loc['USD', 'excessIM'] = excess_margin.call(res['x'])['totalIM']-sum(excess_margin.call(res['x'])['IM'])
    summary.loc['USD', 'excessMM'] = excess_margin.call(res['x'])['totalMM']-sum(excess_margin.call(res['x'])['MM'])
    summary.loc['USD', 'transactionCost'] = 0

    summary.loc['total', 'spotCarry'] = summary['spotCarry'].mean()
    summary.loc['total', 'CarryEstimator'] = summary['CarryEstimator'].mean()
    summary.loc['total', 'optimalWeight'] = summary['optimalWeight'].sum() ## 000....
    summary.loc['total', 'ExpectedCarry'] = summary['ExpectedCarry'].sum()
    summary.loc['total', 'RealizedCarry'] = summary['RealizedCarry'].sum()
    summary.loc['total', 'excessIM'] = summary['excessIM'].sum()
    summary.loc['total', 'excessMM'] = summary['excessMM'].sum()
    # TODO: covar pre-update
    #futures.loc['total', 'lossProbability'] = loss_tolerance - loss_tolerance_constraint['fun'](res['x'])
    summary.loc['total', 'transactionCost'] = summary['transactionCost'].sum()

    summary['absWeight']=summary['optimalWeight'].apply(np.abs,axis=1)
    if verbose:
        with pd.ExcelWriter('paths.xlsx', engine='xlsxwriter') as writer:
            summary.to_excel(writer, sheet_name='futureinfo')
            pd.concat(progress_display, axis=1).to_excel(writer, sheet_name='optimPath')

    return summary
