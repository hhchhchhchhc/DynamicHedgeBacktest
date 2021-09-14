import datetime
import pyarrow.parquet as pq
import pandas as pd

directory = 'C:/Users/Tibor/Sandbox/'
start_time = datetime.datetime.now()

#### write data to file ###

# file = '20210907_62984_tob.parquet'
# path = directory + file
# xrpusdt = pq.read_table(path).to_pandas()
# xrpusdt = xrpusdt[['timestamp_ms', 'bid_price', 'bid_qty', 'ask_price', 'ask_qty']]
# xrpusdt['bid_price'] = 1e4*xrpusdt['bid_price']
# xrpusdt['bid_price'] = xrpusdt['bid_price'].astype(int)
# xrpusdt['ask_price'] = 1e4*xrpusdt['ask_price']
# xrpusdt['ask_price'] = xrpusdt['ask_price'].astype(int)
# xrpusdt['bid_qty'] = 10*xrpusdt['bid_qty']
# xrpusdt['bid_qty'] = xrpusdt['bid_qty'].astype(int)
# xrpusdt['ask_qty'] = 10*xrpusdt['ask_qty']
# xrpusdt['ask_qty'] = xrpusdt['ask_qty'].astype(int)
#
# xrpusdt = xrpusdt.rename(columns={'timestamp_ms': 'milisecondsSinceEpoch', 'bid_price': 'bidPrice', 'bid_qty': 'bidSize', 'ask_price': 'askPrice', 'ask_qty': 'askSize' })
# xrpusdt['datetime'] = xrpusdt['milisecondsSinceEpoch'].apply(lambda milisecondsSinceEpoch: datetime.datetime.utcfromtimestamp(milisecondsSinceEpoch/1000))

# start = datetime.datetime(2021, 9, 7, 12, 0)
# end = datetime.datetime(2021, 9, 7, 12, 1)
# xrpusdt = xrpusdt[(xrpusdt['datetime'] >= start) & (xrpusdt['datetime'] < end)]
# xrpusdt.to_csv(directory + 'sample_xrpusdt.csv', index=False)

### backtest ###

# xrpusdt = pd.read_csv(directory + 'sample_xrpusdt.csv')
#
# target_spread = 40
# skew_at_20_pct = 12
# skew_at_40_pct = 16
# skew_at_max_pos = 22
# max_pos = 100
# target_pos = 20
# mid_change_threshold = 10
#
# position = 0
# cash = 0
# bid = 0
# ask = 0
# last_mid = 0
# pnl = 0
#
# market_mids = []
# skews = []
# bids = []
# asks = []
# positions = []
# cashs = []
# pnls = []
# n = len(xrpusdt)
# for i in range(n):
#     if i < n - 1:
#         bidPrice = xrpusdt['bidPrice'].iloc[i]
#         askPrice = xrpusdt['askPrice'].iloc[i]
#         mid = 0.5*(bidPrice + askPrice)
#         market_mids.append(mid)
#         if bid > 0 and askPrice <= bid:
#             position = position + target_pos
#             cash = cash - target_pos*bid
#             bid = 0
#         if 0 < ask <= bidPrice:
#             position = position - target_pos
#             cash = cash + target_pos*ask
#             ask = 0
#         positions.append(position)
#         cashs.append(cash)
#         mid_changed = abs(mid - last_mid) >= mid_change_threshold
#         last_mid = mid
#         skew = 0
#         if 0.2 * max_pos <= position < 0.4 * max_pos:
#             skew = -skew_at_20_pct
#         elif 0.4 * max_pos <= position < max_pos:
#             skew = -skew_at_40_pct
#         elif position >= max_pos:
#             skew = -skew_at_max_pos
#         if -0.4*max_pos < position <= -0.2*max_pos:
#             skew = skew_at_20_pct
#         elif -max_pos < position <= -0/4*max_pos:
#             skew = skew_at_40_pct
#         elif position <= -max_pos:
#             skew = skew_at_max_pos
#         skews.append(skew)
#         if mid_changed or bid == 0 or ask == 0:
#             bid = mid + skew - 0.5*target_spread
#             ask = mid + skew + 0.5*target_spread
#     else:
#         market_mids.append(last_mid)
#         skews.append(0)
#         positions.append(0)
#         if position > 0:
#             cash = cash + position*xrpusdt['bidPrice'].iloc[-1]
#             cashs.append(cash)
#         if position < 0:
#             pnl = pnl - position*xrpusdt['askPrice'].iloc[-1]
#             cashs.append(cash)
#     bids.append(bid)
#     asks.append(ask)
#     pnl = cash + position * mid
#     pnls.append(pnl)
#
# xrpusdt['bid'] = bids
# xrpusdt['ask'] = asks
# xrpusdt['market_mid'] = market_mids
# xrpusdt['skew'] = skews
# xrpusdt['position'] = positions
# xrpusdt['pnl'] = pnls
#
# xrpusdt.to_csv(directory + 'backtest_xrpusdt.csv', index=False)

### output ###

backtest = pd.read_csv(directory + 'backtest_xrpusdt.csv')
backtest = backtest[::1000]
backtest.to_csv(directory + 'backtest_sample.csv')

end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time-start_time))



