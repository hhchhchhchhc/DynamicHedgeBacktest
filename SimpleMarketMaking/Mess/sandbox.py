from SimpleMarketMaking.Mess import tools
import pyarrow.parquet as pq
import pandas as pd

# tools.s3_download_directory('C:/Users/Tibor/Sandbox/latency', 'binance-historical', 'USDTvsBUSD')

data = pq.read_table('C:/Users/Tibor/Data/parquet/20210922_4610_tob.parquet').to_pandas()
latency = data['receive_timestamp_nanos'] - data['exchange_timestamp_nanos']
latency = 1e-6*latency
print(latency.describe())
