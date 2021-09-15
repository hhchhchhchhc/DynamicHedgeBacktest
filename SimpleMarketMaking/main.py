import datetime

import pandas as pd
import tools
from backtest import Backtest

directory = 'C:/Users/Tibor/Sandbox/'
start_time = datetime.datetime.now()

# write data to file ###

file = '20210914_62984_tob.parquet'
path = directory + file
data_frame = tools.create_pandas_data_frame_from_parquet_file(path)
data_frame = tools.format_top_of_book_pandas_data_frame(data_frame)
# data_frame = tools.extract_data_frame_interval(data_frame, datetime.datetime(2021, 9, 14, 8, 9),
#                                                datetime.datetime(2021, 9, 14, 16, 13))
data_frame.to_csv(directory + 'sample_data_frame.csv', index=False)

# backtest ###

# data_frame = pd.read_csv(directory + 'sample_data_frame.csv')
# backtest = Backtest(data_frame)
# backtest.run()
# result = backtest.data_frame
# result.to_csv(directory + 'backtest.csv', index=False)

# output ###

# backtest = pd.read_csv(directory + 'backtest.csv')
# backtest = backtest[::1000]
# backtest.to_csv(directory + 'sampled_backtest.csv')

end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
