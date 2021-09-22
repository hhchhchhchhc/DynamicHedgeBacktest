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

for target_spread in [30]:
    my_backtest = backtest.Backtest(None)
    my_backtest.load_market_data(directory, datetime.date(2021, 9, 8), 1)
    my_backtest.target_spread = target_spread
    my_backtest.run()
    number_of_trades = my_backtest.get_number_of_trades()
    if number_of_trades >= 100:
        pnl = my_backtest.get_final_pnl()
        volume = my_backtest.get_total_volume_traded()
        total_yield = my_backtest.get_yield()
        annualised_return = my_backtest.get_annualised_return()
        annualised_standard_deviation = my_backtest.get_annualised_standard_deviation()
        annualised_sharp_ratio = my_backtest.get_annualised_sharp_ratio()
        maximum_drawdown = my_backtest.get_maximum_drawdown()
        number_of_quotes = sum(my_backtest.number_of_quotes)
        fees = sum(my_backtest.fees)
        print('')
        print('target_spread = ' + str(target_spread) +
              ' pips\t: pnl = ' + str("{:,.2f}".format(pnl/10000)) +
              ' USD,\t fees = ' + str("{:,.2f}".format(fees/10000)) +
              ' USD,\t net pnl = ' + str("{:,.2f}".format((pnl - fees)/10000)) +
              ' USD,\t number of quotes = ' + str("{:,.0f}".format(number_of_quotes)) +
              ',\t number of trades = ' + str("{:,.0f}".format(number_of_trades)) +
              ',\t volume = ' + str("{:,.0f}".format(volume)) +
              ' USD,\t annualised return = ' + str("{:,.2f}".format(100*annualised_return)) +
              '%,\t annualised standard deviation of returns = ' +
              str("{:,.2f}".format(100*annualised_standard_deviation)) +
              '%,\t annualised sharp = ' + str("{:,.2f}".format(annualised_sharp_ratio)) +
              ',\t MDD = ' + str("{:,.0f}".format(maximum_drawdown)) +
              ' USD,\t total yield = ' + str("{:,.2f}".format(total_yield)) +
              ' $/M$.')
        # my_backtest_sampled = my_backtest.data_frame.iloc[::math.ceil(len(my_backtest.data_frame.index)/100000), :]
        my_backtest_sampled = my_backtest.data_frame
        my_backtest_sampled.to_csv(directory + 'results_backtest_pnls_' + str(target_spread) + '.csv')
        trades = my_backtest.generate_trades()
        trades.to_csv(directory + 'results_backtest_trades_' + str(target_spread) + '.csv')
    else:
        print()
        print('target_spread = ' + str(target_spread) + ': less than 100 trades')

end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
