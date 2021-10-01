import datetime
import SimpleMarketMaking.Clean.market_data
import SimpleMarketMaking.Clean.config

start_time = datetime.datetime.now()

my_market_data = SimpleMarketMaking.Clean.market_data.MarketData('BTCUSDT')

# print('secondly bars')
# secondly_bars = my_market_data.get_time_bars(1000)
# print(secondly_bars)
# secondly_bars.to_csv('C:/Users/Tibor/Sandbox/secondly_bars.csv')
#
# print('tick bars')
# tick_bars = my_market_data.get_tick_bars(18)
# print(tick_bars)
# tick_bars.to_csv('C:/Users/Tibor/Sandbox/tick_bars.csv')
#
# print('volume bars')
# volume_bars = my_market_data.get_volume_bars(4050)
# print(volume_bars)
# volume_bars.to_csv('C:/Users/Tibor/Sandbox/volume_bars.csv')

print('dollar bars')
dollar_bars = my_market_data.get_dollar_bars(2000000)
print(dollar_bars)
dollar_bars.to_csv('C:/Users/Tibor/Sandbox/dollar_bars.csv')

print('tick imbalance bars')
tick_imbalance_bars = my_market_data.get_tick_imbalance_bars()
print(tick_imbalance_bars)
tick_imbalance_bars.to_csv('C:/Users/Tibor/Sandbox/tick_imbalance_bars.csv')

print('trade side imbalance bars')
trade_side_imbalance_bars = my_market_data.get_trade_side_imbalance_bars()
print(trade_side_imbalance_bars)
trade_side_imbalance_bars.to_csv('C:/Users/Tibor/Sandbox/trade_side_imbalance_bars.csv')

print('volume tick imbalance bars')
volume_tick_imbalance_bars = my_market_data.get_volume_tick_imbalance_bars()
print(volume_tick_imbalance_bars)
volume_tick_imbalance_bars.to_csv('C:/Users/Tibor/Sandbox/volume_tick_imbalance_bars.csv')

print('trade side imbalance bars')
volume_trade_side_imbalance_bars = my_market_data.get_volume_trade_side_imbalance_bars()
print(volume_trade_side_imbalance_bars)
volume_trade_side_imbalance_bars.to_csv('C:/Users/Tibor/Sandbox/volume_trade_side_imbalance_bars.csv')

print('dollar tick imbalance bars')
dollar_tick_imbalance_bars = my_market_data.get_dollar_tick_imbalance_bars()
print(dollar_tick_imbalance_bars)
dollar_tick_imbalance_bars.to_csv('C:/Users/Tibor/Sandbox/dollar_tick_imbalance_bars.csv')

print('dollar trade side imbalance bars')
dollar_trade_side_imbalance_bars = my_market_data.get_dollar_trade_side_imbalance_bars()
print(dollar_trade_side_imbalance_bars)
dollar_trade_side_imbalance_bars.to_csv('C:/Users/Tibor/Sandbox/dollar_trade_side_imbalance_bars.csv')

print('tick runs bars')
tick_runs_bars = my_market_data.get_tick_runs_bars()
print(tick_runs_bars)
tick_runs_bars.to_csv('C:/Users/Tibor/Sandbox/tick_runs_bars.csv')

print('trade side runs bars')
trade_side_runs_bars = my_market_data.get_trade_side_runs_bars()
print(trade_side_runs_bars)
trade_side_runs_bars.to_csv('C:/Users/Tibor/Sandbox/trade_side_runs_bars.csv')

print('volume tick runs bars')
volume_tick_runs_bars = my_market_data.get_volume_tick_runs_bars()
print(volume_tick_runs_bars)
volume_tick_runs_bars.to_csv('C:/Users/Tibor/Sandbox/volume_tick_runs_bars.csv')

print('volume trade side runs bars')
volume_trade_side_runs_bars = my_market_data.get_volume_trade_side_runs_bars()
print(volume_trade_side_runs_bars)
volume_trade_side_runs_bars.to_csv('C:/Users/Tibor/Sandbox/volume_trade_side_runs_bars.csv')

print('dollar tick runs bars')
dollar_tick_runs_bars = my_market_data.get_dollar_tick_runs_bars()
print(dollar_tick_runs_bars)
dollar_tick_runs_bars.to_csv('C:/Users/Tibor/Sandbox/dollar_tick_runs_bars.csv')

print('dollar trade side runs bars')
dollar_trade_side_runs_bars = my_market_data.get_dollar_trade_side_runs_bars()
print(dollar_trade_side_runs_bars)
dollar_trade_side_runs_bars.to_csv('C:/Users/Tibor/Sandbox/dollar_trade_side_runs_bars.csv')

end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
