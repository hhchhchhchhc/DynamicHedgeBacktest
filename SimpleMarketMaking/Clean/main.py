import datetime
import SimpleMarketMaking.Clean.market_data

start_time = datetime.datetime.now()

my_market_data = SimpleMarketMaking.Clean.market_data.MarketData('BTCUSDT')

trade_side_imbalance_bars = my_market_data.get_trade_side_imbalance_bars()
print(trade_side_imbalance_bars)

minutely = my_market_data.get_time_bars(60000)
print(minutely)
volume_bars = my_market_data.get_volume_bars(500000)
print(volume_bars)
dollar_bars = my_market_data.get_dollar_bars(500000*5000000)
print(dollar_bars)
tick_imbalance_bars = my_market_data.get_tick_imbalance_bars()
print(tick_imbalance_bars)


end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
