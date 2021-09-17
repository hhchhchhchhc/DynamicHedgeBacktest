import datetime

import pandas as pd

from SimpleMarketMaking import tools
from SimpleMarketMaking import backtest


def write_data_to_file() -> None:
    file = '20210914_62984_tob.parquet'
    path = directory + file
    data_frame = tools.create_pandas_data_frame_from_parquet_file(path)
    data_frame = tools.extract_data_frame_interval(data_frame, datetime.datetime(2021, 9, 14, 9, 9, 46, 930000),
                                                   datetime.datetime(2021, 9, 14, 17, 12, 21))
    data_frame = tools.format_top_of_book_pandas_data_frame(data_frame)
    data_frame.to_csv(directory + 'sample_data_frame.csv', index=False)
    return None


def write_backtest() -> None:
    data_frame = pd.read_csv(directory + 'sample_data_frame.csv')
    for target_spread in range(85, 115, 5):
        my_backtest = backtest.Backtest(data_frame)
        my_backtest.target_spread = target_spread
        my_backtest.run()
        pnl = my_backtest.pnl/10000
        volume = my_backtest.get_total_volume_traded()
        number_of_trades = my_backtest.get_number_of_trades()
        y = my_backtest.get_yield()
        print('target_spread = ' + str(target_spread) +
              ' pips\t: pnl = ' + str(pnl) +
              ' USD\t, number of trades = ' + str(number_of_trades) +
              '\t, volume = ' + str(volume) +
              '\t, yield = ' + str("{:.2f}".format(y)) +
              ' $/M$.')
    return None


directory = 'C:/Users/Tibor/Sandbox/'
start_time = datetime.datetime.now()
write_backtest()
end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
