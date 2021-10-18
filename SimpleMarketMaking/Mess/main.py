import datetime

directory = 'C:/Users/Tibor/Sandbox/'
start_time = datetime.datetime.now()

# tools.s3_download_directory('C:/Users/Tibor/Sandbox', 'binance-historical', 'USDTvsBUSD-20210908-to-20210916')


# new comment

# for parameter in [0]:
#     parameter_name = 'no_param'
#     my_backtest = backtest.Backtest(None)
#     my_backtest.load_market_data(directory, 'btcusdt', datetime.date(2021, 9, 1), 2)
#     my_backtest.skew_multiplier = 2
#     my_backtest.target_spread = 2000
#     my_backtest.price_change_threshold = 200
#     my_backtest.run()
#     number_of_trades = my_backtest.cumulative_number_of_trade
#     if number_of_trades >= 1:
#         net_pnl = my_backtest.cumulative_net_pnl
#         fees = my_backtest.cumulative_fee
#         gross_pnl = my_backtest.cumulative_gross_pnl
#         volume = my_backtest.cumulative_usd_volume
#         total_gross_yield = my_backtest.get_total_gross_yield()
#         total_net_yield = my_backtest.get_total_net_yield()
#         maximum_drawdown = my_backtest.get_maximum_drawdown()
#         number_of_quotes = my_backtest.cumulative_number_of_quote
#         divider = 1e2
#         print('')
#         print(parameter_name + ' = ' + str(parameter) +
#               ' : gross pnl = $' + str("{:,.2f}".format(gross_pnl / divider)) +
#               ' ($' + str("{:,.2f}".format(my_backtest.annualise(gross_pnl) / divider)) + ' p.a.), fees = $'
#               + str("{:,.2f}".format(fees / divider)) +
#               ' ($' + str("{:,.2f}".format(my_backtest.annualise(fees) / divider)) + ' p.a.), net pnl = $'
#               + str("{:,.2f}".format(net_pnl / divider)) +
#               ' ($' + str("{:,.2f}".format(my_backtest.annualise(net_pnl) / divider)) + ' p.a.), volume = $'
#               + str("{:,.0f}".format(volume / divider)) +
#               ' ($' + str("{:,.0f}".format(my_backtest.annualise(volume) / divider)) + ' p.a.), number of quotes = '
#               + str("{:,.0f}".format(number_of_quotes)) +
#               ' (' + str("{:,.0f}".format(my_backtest.annualise(number_of_quotes))) + ' p.a.), number of trades = '
#               + str("{:,.0f}".format(number_of_trades)) +
#               ' (' + str("{:,.0f}".format(my_backtest.annualise(number_of_trades))) + ' p.a.), MDD = $'
#               + str("{:,.2f}".format(maximum_drawdown / divider)) +
#               ', total gross yield = ' + str("{:,.2f}".format(total_gross_yield)) +
#               ' $/M$, total net yield = ' + str("{:,.2f}".format(total_net_yield)) +
#               ' $/M$.')
#         my_backtest_sampled = my_backtest.data_frame.iloc[::math.ceil(len(my_backtest.data_frame.index) / 100000), :]
#         my_backtest_sampled.to_csv(directory + 'results_backtest_sampled_' + parameter_name + '_' +
#                                    str(parameter) + '.csv')
#         trades = my_backtest.generate_trades()
#         trades.to_csv(directory + 'results_backtest_trades_' + parameter_name + '_' +
#                       str(parameter) + '.csv')
#         hourly_data = my_backtest.generate_hourly_data()
#         hourly_data.to_csv(directory + 'results_hourly_data_' + parameter_name + '_' +
#                            str(parameter) + '.csv')
#         stats = my_backtest.generate_stats()
#         stats.to_csv(directory + 'results_stats_' + parameter_name + '_' +
#                      str(parameter) + '.csv')
#     else:
#         print()
#         print(parameter_name + ' = ' + str(parameter) + ': less than 100 trades')
#         break

end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
