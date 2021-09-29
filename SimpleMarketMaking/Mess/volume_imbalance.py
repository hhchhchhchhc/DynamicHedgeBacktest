import datetime
from SimpleMarketMaking.Mess import backtest

directory = 'C:/Users/Tibor/Sandbox/'
start_time = datetime.datetime.now()

my_backtest = backtest.Backtest(None)
my_backtest.load_market_data(directory, 'dogeusdt', datetime.date(2021, 9, 1), 1)
test = my_backtest.data_frame.head(100)
test['volume_imbalance'] = (test['ask_size'] - test['bid_size']) / (test['ask_size'] + test['bid_size'])
test['mid_price'] = 0.5*(test['bid_price'] + test['ask_price'])




end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
