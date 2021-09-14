import datetime
import pyarrow.parquet as pq
import pandas as pd

directory = 'C:/Users/Tibor/Sandbox/'

# file = '20210907_62984_tob.parquet'
# path = directory + file
# xrpusdt = pq.read_table(path).to_pandas()
# xrpusdt = xrpusdt[['timestamp_ms', 'bid_price', 'bid_qty', 'ask_price', 'ask_qty']]
# xrpusdt = xrpusdt.rename(columns={'timestamp_ms': 'milisecondsSinceEpoch', 'bid_price': 'bidPrice', 'bid_qty': 'bidSize', 'ask_price': 'askPrice', 'ask_qty': 'askSize' })
# xrpusdt['datetime'] = xrpusdt['milisecondsSinceEpoch'].apply(lambda milisecondsSinceEpoch: datetime.datetime.utcfromtimestamp(milisecondsSinceEpoch/1000))

# start = datetime.datetime(2021, 9, 7, 12, 0)
# end = datetime.datetime(2021, 9, 7, 12, 1)
# xrpusdt = xrpusdt[(xrpusdt['datetime'] >= start) & (xrpusdt['datetime'] < end)]
# xrpusdt.to_csv(directory + 'sample_xrpusdt.csv')

xrpusdt = pd.read_csv(directory + 'sample_xrpusdt.csv')
print(xrpusdt)

