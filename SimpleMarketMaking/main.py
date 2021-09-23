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

for target_spread in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
    my_backtest = backtest.Backtest(None)
    my_backtest.load_market_data(directory, datetime.date(2021, 9, 8), 9)
    my_backtest.target_spread = target_spread
    my_backtest.run()
    number_of_trades = my_backtest.cumulative_number_of_trade
    if number_of_trades >= 100:
        net_pnl = my_backtest.cumulative_net_pnl
        fees = my_backtest.cumulative_fee
        gross_pnl = my_backtest.cumulative_gross_pnl
        volume = my_backtest.cumulative_usd_volume
        total_gross_yield = my_backtest.get_total_gross_yield()
        total_net_yield = my_backtest.get_total_net_yield()
        maximum_drawdown = my_backtest.get_maximum_drawdown()
        number_of_quotes = sum(my_backtest.number_of_quotes)
        print('')
        print('target_spread = ' + str(target_spread) +
              ' pips: gross pnl = $' + str("{:,.2f}".format(gross_pnl/10000)) +
              ' ($' + str("{:,.2f}".format(my_backtest.annualise(gross_pnl)/10000)) + ' p.a.)'
              ', fees = $' + str("{:,.2f}".format(fees/10000)) +
              ' ($' + str("{:,.2f}".format(my_backtest.annualise(fees)/10000)) + ' p.a.)'
              ', net pnl = ' + str("{:,.2f}".format(net_pnl/10000)) +
              ' ($' + str("{:,.2f}".format(my_backtest.annualise(net_pnl)/10000)) + ' p.a.)'
              ', volume = $' + str("{:,.0f}".format(volume/10000)) +
              ' ($' + str("{:,.0f}".format(my_backtest.annualise(volume)/10000)) + ' p.a.)'
              ', number of quotes = ' + str("{:,.0f}".format(number_of_quotes)) +
              ' (' + str("{:,.0f}".format(my_backtest.annualise(number_of_quotes))) + ' p.a.)'
              ', number of trades = ' + str("{:,.0f}".format(number_of_trades)) +
              ' (' + str("{:,.0f}".format(my_backtest.annualise(number_of_trades))) + ' p.a.)'
              ', MDD = $' + str("{:,.2f}".format(maximum_drawdown/10000)) +
              ', total gross yield = ' + str("{:,.2f}".format(total_gross_yield)) +
              ' $/M$, total net yield = ' + str("{:,.2f}".format(total_net_yield)) +
              ' $/M$.')
        my_backtest_sampled = my_backtest.data_frame.iloc[::math.ceil(len(my_backtest.data_frame.index)/10000), :]
        my_backtest_sampled.to_csv(directory + 'results_backtest_sampled_' + str(target_spread) + '.csv')
        trades = my_backtest.generate_trades()
        trades.to_csv(directory + 'results_backtest_trades_' + str(target_spread) + '.csv')
        hourly_data = my_backtest.generate_hourly_data()
        hourly_data.to_csv(directory + 'results_hourly_data_' + str(target_spread) + '.csv')
    else:
        print()
        print('target_spread = ' + str(target_spread) + ': less than 100 trades')
        break

end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
