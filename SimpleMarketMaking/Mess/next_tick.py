import datetime

import numpy as np
import pandas as pd

from SimpleMarketMaking.Mess import backtest

directory = 'C:/Users/Tibor/Sandbox/'
start_time = datetime.datetime.now()

my_backtest = backtest.Backtest(None)
my_backtest.load_market_data(directory, 'btcusdt', datetime.date(2021, 9, 1), 1)
imbalances = []
bid_moves = []
ask_moves = []
progress_pct = 0
for i in range(1, len(my_backtest.data_frame.index)):
    if np.floor(100 * i / len(my_backtest.data_frame.index)) > progress_pct:
        progress_pct = progress_pct + 10
        print(str(progress_pct) + '% ', end='')
    previous_tick = my_backtest.data_frame.iloc[i - 1]
    this_tick = my_backtest.data_frame.iloc[i]
    ask_size = previous_tick['ask_size']
    bid_size = previous_tick['bid_size']
    imbalance = (ask_size - bid_size) / (ask_size + bid_size)
    imbalances.append(imbalance)
    bid_move = this_tick['bid_price'] - previous_tick['bid_price']
    ask_move = this_tick['ask_price'] - previous_tick['ask_price']
    bid_moves.append(bid_move)
    ask_moves.append(ask_move)

results = pd.DataFrame()
results['imbalance'] = imbalances
results['bid_move'] = bid_moves
results['ask_move'] = ask_moves

stats = pd.DataFrame()
stats.index = np.union1d(results['bid_move'].unique(), results['ask_move'].unique())
stat = results['bid_move'].value_counts()
stats = pd.merge(stats, stat, how='left', left_index=True, right_index=True)
stat = results['ask_move'].value_counts()
stats = pd.merge(stats, stat, how='left', left_index=True, right_index=True)
stats = stats.rename(columns={'bid_move': 'bid_move_all'})
stats = stats.rename(columns={'ask_move': 'ask_move_all'})
for threshold in [.9, .99, .999, .9999]:
    stat = results['bid_move'][results['imbalance'] > threshold].value_counts()
    stats = pd.merge(stats, stat, how='left', left_index=True, right_index=True)
    stat = results['ask_move'][results['imbalance'] < -threshold].value_counts()
    stats = pd.merge(stats, stat, how='left', left_index=True, right_index=True)
    stats = stats.rename(columns={'bid_move': 'bid_move_' + str(threshold)})
    stats = stats.rename(columns={'ask_move': 'ask_move_' + str(threshold)})
stats.to_csv(directory + 'stats.csv')

end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
