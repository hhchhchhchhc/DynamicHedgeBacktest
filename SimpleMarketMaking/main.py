import datetime
import math

from SimpleMarketMaking import tools
from SimpleMarketMaking import backtest


def write_data_to_file(my_date: datetime.date) -> None:
    date_string = my_date.strftime('%Y%m%d')
    print('Writing XRPUSDT date ' + date_string + '.')
    file = date_string + '_62984_tob.parquet'
    path = directory + 'parquet/' + file
    data_frame = tools.create_pandas_data_frame_from_parquet_file(path)
    data_frame = tools.format_top_of_book_pandas_data_frame(data_frame)
    data_frame.to_csv(directory + 'market_data_xrpusdt_' + date_string + '.csv', index=False)
    return None


directory = 'C:/Users/Tibor/Sandbox/'
start_time = datetime.datetime.now()

# tools.s3_download_directory('C:/Users/Tibor/Sandbox', 'binance-historical', 'USDTvsBUSD-20210908-to-20210916')

# for d in range(9):
#     date = datetime.date(2021, 9, 8)
#     date = date + datetime.timedelta(days=d)
#     write_data_to_file(date)

for target_spread in [100]:
    my_backtest = backtest.Backtest(None)
    my_backtest.load_market_data(directory, datetime.date(2021, 9, 8), 9)
    my_backtest.target_spread = target_spread
    my_backtest.skew_at_max_pos = 10*target_spread
    my_backtest.run()
    pnl = my_backtest.pnl / 10000
    volume = my_backtest.get_total_volume_traded()
    number_of_trades = my_backtest.get_number_of_trades()
    y = my_backtest.get_yield()
    annualised_return = my_backtest.get_annualised_average_hourly_return()
    annualised_sharp_ratio = my_backtest.get_annualised_sharp_ratio()
    maximum_drawdown = my_backtest.get_maximum_drawdown()
    print('')
    print('target_spread = ' + str(target_spread) +
          ' pips\t: pnl = ' + str(pnl) +
          ' USD\t, number of trades = ' + str(number_of_trades) +
          '\t, volume = ' + str(volume) +
          '\t, annualised return = ' + str("{:.2f}".format(100*annualised_return)) +
          '%\t, sharp = ' + str("{:.2f}".format(annualised_sharp_ratio)) +
          '\t, MDD = ' + str("{:.2f}".format(100*maximum_drawdown)) +
          '%\t, yield = ' + str("{:.2f}".format(y)) +
          ' $/M$.')
    my_backtest_sampled = my_backtest.data_frame.iloc[::math.ceil(len(my_backtest.data_frame.index)/100000), :]
    my_backtest_sampled.to_csv(directory + 'results_backtest_pnls_' + str(target_spread) + '.csv')
    trades = my_backtest.generate_trades()
    trades.to_csv(directory + 'results_backtest_trades_' + str(target_spread) + '.csv')

end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
