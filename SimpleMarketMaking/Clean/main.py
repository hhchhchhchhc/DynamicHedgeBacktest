import datetime
import SimpleMarketMaking.Clean.market_data

start_time = datetime.datetime.now()

my_market_data = SimpleMarketMaking.Clean.market_data.MarketData('BTCUSDT')


end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
