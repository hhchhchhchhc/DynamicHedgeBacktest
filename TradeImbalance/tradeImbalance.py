# testing trade imbalance on binance

import json, datetime, os, re
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# def md_pq_to_pandas(datadir : str, trade_date : datetime.datetime, instrument_id : int):
#     pq_filename_prefix = os.path.join(datadir, trade_date.strftime("%Y%m%d") + "_" + str(instrument_id) + "_")
#     trade_filename = pq_filename_prefix + "trade.parquet"
#     tradedf = pq.read_table(trade_filename).to_pandas()
#     return tradedf
#
# trades = md_pq_to_pandas('C:/Users/Tibor/AUK/data/', datetime.datetime(2021,8,29), 3651)
# trades["datetime"] = pd.to_datetime(trades["timestamp_ms"], unit='ms')
# trades = trades[["timestamp_ms", "datetime", "price", "qty", "buyer_is_market_maker"]]
# trades = trades.rename(columns={"timestamp": "timestamp_epoch", "datetime": "datetime", "price": "price", "qty": "size", "buyer_is_market_maker": "given"})
#
# # trades = trades[1:1000]
# lookback_window_btc = 5
#
# n = len(trades.index)
# lookback_window_start_index = 0
# imbalance = []
# horizon_btc = 5
# horizon_end_index = 0
# move = []
# progress_pct = 0
# for i in range(n):
#     if int(100*i/n) > progress_pct:
#         progress_pct = progress_pct + 1
#         print(datetime.datetime.now())
#         print(str(progress_pct) +'%')
#     if sum(trades["size"][lookback_window_start_index:i]) < lookback_window_btc:
#         imbalance.append(math.nan)
#     else:
#         while sum(trades["size"][lookback_window_start_index + 1:i]) >= lookback_window_btc:
#             lookback_window_start_index = lookback_window_start_index + 1
#         window = trades[lookback_window_start_index:i + 1]
#         given = window["size"][window["given"]].sum()
#         paid = window["size"].sum() - given
#         imbalance.append((paid - given)/(paid + given))
#     while sum(trades["size"][i + 1:horizon_end_index + 1]) < horizon_btc:
#         horizon_end_index = horizon_end_index + 1
#         if horizon_end_index >= n:
#             break
#     if horizon_end_index < n:
#         move.append(1e6*(trades.iloc[horizon_end_index]["price"] - trades.iloc[i]["price"])/trades.iloc[i]["price"])
#     else:
#         move.append(math.nan)
#
# trades["imbalance"] = imbalance
# trades["move_dpm"] = move
# trades = trades.dropna()
# trades.to_csv('imbalance.csv')

trades = pd.read_csv('imbalance.csv')
trades["bucket"] = trades["imbalance"].apply(lambda i: int(10*i))


def q1(x):
    return x.quantile(0.01)
def q5(x):
    return x.quantile(0.05)
def q10(x):
    return x.quantile(0.1)
def q20(x):
    return x.quantile(0.2)
def q30(x):
    return x.quantile(0.3)
def q40(x):
    return x.quantile(0.4)
def q50(x):
    return x.quantile(0.5)
def q60(x):
    return x.quantile(0.6)
def q70(x):
    return x.quantile(0.7)
def q80(x):
    return x.quantile(0.8)
def q90(x):
    return x.quantile(0.9)
def q95(x):
    return x.quantile(0.95)
def q99(x):
    return x.quantile(0.99)

stats = trades[["bucket", "move_dpm"]].groupby("bucket").agg({"move_dpm": [np.alen, np.mean, np.std, q1,q5,q10,q20,q30,q40,q50,q60,q70,q80,q90,q95,q99]})

stats.to_csv('stats.csv')

# trades = pd.read_csv('imbalance.csv')
# trades = trades[1:10000]
#
# n = len(trades.index)
# position = 0
# buys = []
# sells = []
# for i in range(n - 1):
#     if position < 1 and trades.iloc[i]["imbalance"] < -.999:
#         price = trades.iloc[i]["price"]
#         if trades.iloc[i]["given"]:
#             price = price + 0.01
#         if position == -1:
#             buys.append(price)
#             position = 0
#         if position == 0:
#             buys.append(price)
#             position = 1
#     if position > -1 and trades.iloc[i]["imbalance"] > .999:
#         price = trades.iloc[i]["price"] - 0.01
#         if trades.iloc[i]["given"]:
#             price = price + 0.01
#         if position == 1:
#             sells.append(price)
#             position = 0
#         if position == 0:
#             sells.append(price)
#             position = -1
#
# if position == 1:
#     price = trades.iloc[i]["price"] - 0.01
#     if trades.iloc[i]["given"]:
#         price = price + 0.01
#     sells.append(price)
#
# if position == -1:
#     price = trades.iloc[i]["price"]
#     if trades.iloc[i]["given"]:
#         price = price + 0.01
#     buys.append(price)
#
# print('bla')
