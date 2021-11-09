import pandas as pd
import numpy as np
import puffin.tardis_importer as ti
import math
import puffin.market_data as md
import puffin.tools as tools
import datetime
import matplotlib.pyplot as plt

ftx = pd.read_parquet('C:/Users/Tibor/Data/ftx_refdb1.parquet')
ftx['ExchangePairCode']

symbol = 'BTC'
date = '2021-10-14'
end_date = date[:-2] + f"{int(date[-2:]) + 1:02}" #set the end date to 1 day after

ti.batch_download(exchange='ftx', symbol_list=[f'{symbol}/USD', f'{symbol}-PERP'], start_date=date, end_date = end_date, data_types =['quotes', 'trades'])

instrument_id = tools.get_id_from_symbol(f'{symbol}/USD')
stripped_date = date.replace('-','')
trades = ti.generate_trade_data_from_gzip(f'{symbol}/USD', f'./datasets/ftx_trades_{date}_{symbol}-USD.csv.gz')
tob = ti.generate_tob_data_from_gzip(f'{symbol}/USD', f'./datasets/ftx_quotes_{date}_{symbol}-USD.csv.gz')
trades.to_parquet(f'./datasets/{stripped_date}_{instrument_id}_trade.parquet', index=False)
tob.to_parquet(f'./datasets/{stripped_date}_{instrument_id}_tob.parquet', index=False)

instrument_id2 = tools.get_id_from_symbol(f'{symbol}-PERP')
trades_perp = ti.generate_trade_data_from_gzip(f'{symbol}-PERP', f'./datasets/ftx_trades_{date}_{symbol}-PERP.csv.gz')
tob_perp = ti.generate_tob_data_from_gzip(f'{symbol}-PERP', f'./datasets/ftx_quotes_{date}_{symbol}-PERP.csv.gz')
trades_perp.to_parquet(f'./datasets/{stripped_date}_{instrument_id2}_trade.parquet', index=False)
tob_perp.to_parquet(f'./datasets/{stripped_date}_{instrument_id2}_tob.parquet', index=False)
