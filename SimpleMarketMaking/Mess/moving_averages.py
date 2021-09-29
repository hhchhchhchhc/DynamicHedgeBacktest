import datetime
import numpy as np
import pandas as pd

from SimpleMarketMaking.Mess import backtest

directory = 'C:/Users/Tibor/Sandbox/'
start_time = datetime.datetime.now()

# my_backtest = backtest.Backtest(None)
# my_backtest.load_market_data(directory, 'dogeusdt', datetime.date(2021, 9, 8), 9)
# market_data = my_backtest.data_frame
#
# indices = [0]
# n = len(market_data.index)
# for i in range(1, n):
#     if market_data['milliseconds_since_epoch'].iloc[i] - market_data['milliseconds_since_epoch'].iloc[indices[-1]] >= 1000:
#         indices.append(i)
# market_data = market_data.iloc[indices]
# market_data.to_csv(directory + 'dogeusdt_secondly.csv')

market_data = pd.read_csv(directory + 'xrpusdt_secondly.csv')

for slope_look_back_sec in [30]:
    print('running ' + str(slope_look_back_sec) + ' seconds.')
    market_data['mid_price'] = market_data['bid_price'].add(market_data['ask_price']).div(2)
    alpha_short = 2 * 60
    slope_look_back = slope_look_back_sec
    alpha_long = 20 * 60
    smooth_mid_short = [market_data['mid_price'].iloc[0]]
    smooth_mid_long = [market_data['mid_price'].iloc[0]]
    slopes = [0]
    slope_start_index = 0
    n = len(market_data.index)
    pnls = [0]
    cumulative_pnls = [0]
    positions = [0]
    pnl = 0
    cumulative_pnl = 0
    progress = 0
    number_of_trades = 0
    for i in range(1, n):
        if np.floor(100 * i / n) > progress:
            progress = progress + 5
            print(str(progress) + '% ', end='')
        now = market_data['milliseconds_since_epoch'].iloc[i]
        time_span = (now - market_data['milliseconds_since_epoch'].iloc[i - 1])/1000
        time_decay_short = 1 - np.exp(-2 * time_span / alpha_short)
        time_decay_long = 1 - np.exp(-2 * time_span / alpha_long)
        smooth_mid_short.append(((1 - time_decay_short) * smooth_mid_short[i - 1]) + (time_decay_short * market_data['mid_price'].iloc[i]))
        smooth_mid_long.append(((1 - time_decay_long) * smooth_mid_long[i - 1]) + (time_decay_long * market_data['mid_price'].iloc[i]))

        while now - market_data['milliseconds_since_epoch'].iloc[slope_start_index] > 1000 * slope_look_back:
            slope_start_index = slope_start_index + 1
        slope = 0
        if slope_start_index < i:
            slope = 1000 * (smooth_mid_short[i] - smooth_mid_short[slope_start_index]) / (now - market_data['milliseconds_since_epoch'].iloc[slope_start_index]) / smooth_mid_short[i]
        slopes.append(slope)

        position = 0
        if positions[i - 1] == 0:
            if np.abs(slopes[i]) > 2e-5:
                position = np.sign(slopes[i])
        elif positions[i - 1] == -1:
            if slopes[i] <= 0:
                position = -1
            elif slopes[i] > 0:
                position = 0
        elif positions[i - 1] == 1:
            if slopes[i] >= 0:
                position = 1
            elif slopes[i] < 0:
                position = 0

        positions.append(position)
        if positions[i - 1] != positions[i]:
            number_of_trades = number_of_trades + 1
        if i < n - 1:
            pnl = position * (market_data['mid_price'].iloc[i + 1] - market_data['mid_price'].iloc[i])
        pnls.append(pnl)
        cumulative_pnl = cumulative_pnl + pnl
        cumulative_pnls.append(cumulative_pnl)

    market_data['smooth_mid_short'] = smooth_mid_short
    market_data['smooth_mid_long'] = smooth_mid_long
    market_data['slope'] = slopes
    market_data['position'] = positions
    market_data['cumulative pnl'] = cumulative_pnls
    market_data.to_csv('C:/Users/Tibor/Sandbox/results_xrpusdt_' + str(slope_look_back_sec) + '.csv')

    average_return = np.average(pnls)
    std_dev = np.std(pnls)
    sharp = average_return / std_dev

    print()
    print('number of trades = ' + str(number_of_trades))
    print('friction = ' + str(number_of_trades*0.00022))
    print('pnl = ' + str(1e-5*cumulative_pnl))

    maximum_drawdown = 0
    maximum_pnl = 0
    minimum_pnl = 0
    for pnl in cumulative_pnls:
        if pnl > maximum_pnl:
            maximum_pnl = pnl
            minimum_pnl = pnl
        if pnl < minimum_pnl:
            minimum_pnl = pnl
            if (maximum_pnl - minimum_pnl) > maximum_drawdown:
                maximum_drawdown = maximum_pnl - minimum_pnl
    print('max drawdown = ' + str(1e-5*maximum_drawdown))

end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
