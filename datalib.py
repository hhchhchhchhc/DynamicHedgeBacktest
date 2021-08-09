import distutils.dir_util
import os.path
from calendar import monthrange

import numpy as np
import pandas as pd
import requests
from pathos.multiprocessing import ProcessingPool as Pool

# from multiprocessing import Pool

n_cpus = 4


class BinanceTradeLoader:
    def __init__(self, local_cache_root="/media/corvin/BigData/Crypto/binance/"):
        self.local_cache_root = local_cache_root
        self.one_sec_resampled_cache = {}

    # local_cache_root = "/media/corvin/BigData/Crypto/binance/futures/"
    def load_monthly_trades(self, symbol, month, year, contract):
        num_days = monthrange(year, month)[1]  # num_days = 28
        return pd.concat(self.load_daily_trades(symbol, day, month, year, contract) for day in range(1, num_days + 1))

    def load_daily_trades(self, symbol, day, month, year, contract, aggregated=True):
        path = self.get_path_daily_trades_by_contract(symbol, day, month, year, contract, aggregated)
        if contract in ["coin", "usdt"]:
            path = "futures/" + path
        elif contract == "spot":
            path = "spot/" + path
        else:
            print("Contract needs to be in ", ["coin", "usdt", "spot"])
            return
        file_path = self.local_cache_root + path
        if not os.path.isfile(file_path):
            base_url = "https://data.binance.vision/data/"
            # if contract in ["coin", "usdt"] :
            #     base_url += "futures/"
            # elif contract == "spot" :
            #     base_url += "spot/"
            url = base_url + path
            print("Backfilling from " + url)
            r = requests.get(url)
            # Create folder if not already there
            distutils.dir_util.mkpath(os.path.dirname(file_path))
            with open(file_path, mode='wb') as localfile:
                localfile.write(r.content)
                localfile.close()
        if not aggregated:
            result = pd.read_csv(file_path, header=None, usecols=[1, 2, 4, 5],
                                 #                    names = ["trade Id", "price", "qty", "quoteQty", "time", "isBuyerMaker"]
                                 names=["price", "qty", "time", "isBuyerMaker"]
                                 )
        else:
            result = pd.read_csv(file_path, header=None, usecols=[1, 2, 5, 6],
                                 #                    names = ["trade Id", "price", "qty", "quoteQty", "time", "isBuyerMaker"]
                                 names=["price", "qty", "time", "isBuyerMaker"]
                                 )
        result["time"] = pd.to_datetime(result["time"], unit="ms")
        return result

    def get_path_daily_trades_by_contract(self, symbol: str, day, month, year, contract, aggregated: bool):
        if contract not in ["coin", "usdt", "spot"]:
            raise (Exception("Contract needs to be in ", ["coin", "usdt", "spot"]))
        contract_dict = {"coin": "USD_PERP", "usdt": "USDT", "spot": "USDT"}
        contract_prefix = contract_dict[contract]
        contract_dir_dict = {"coin": "cm/", "usdt": "um/", "spot": ""}
        contract_dir_prefix = contract_dir_dict[contract]

        day = int(day)
        if day <= 9:
            day = "0" + str(day)
        else:
            day = str(day)
        if month <= 9:
            month = "0" + str(month)
        else:
            month = str(month)
        trades_name = "aggTrades"
        if not aggregated:
            trades_name = "trades"
        root_path = contract_dir_prefix + "daily/" + trades_name + "/" + symbol + contract_prefix + "/"
        return root_path + symbol + contract_prefix + "-" + trades_name + "-" + str(year) + "-" + str(
            month) + "-" + str(
            day) + ".zip"

    def load_daily_1_sec_resampled_trades(self, symbol, day, month, year, contract, aggregated=True):
        trades = self.load_daily_trades(symbol=symbol, day=day, month=month, year=year, contract=contract,
                                        aggregated=aggregated)
        trades["buy_usdt_volume"] = trades["price"] * trades["qty"] * (1 - trades["isBuyerMaker"])
        trades["usdt_volume"] = trades["price"] * trades["qty"]
        resampled = trades.resample(f'1s', on="time").agg(
            {"buy_usdt_volume": np.sum, "usdt_volume": np.sum, "qty": np.sum}).reset_index()
        # return trades[["time", "price"]]
        resampled_ohlc = trades[["time", "price"]].resample(f'1s', on="time").agg(
            {"price": ["first", "last", "min", "max"]})
        resampled_ohlc.columns = [{"first": "open", "last": "close", "min": "low", "max": "high"}[col[1]] for col in
                                  resampled_ohlc.columns.values]
        resampled = pd.merge(resampled, resampled_ohlc.reset_index(), on="time")
        resampled["buy_percentage"] = resampled["buy_usdt_volume"] / resampled["usdt_volume"]
        resampled["vwap"] = resampled["usdt_volume"] / resampled["qty"]
        resampled["vwap"].fillna(method="ffill", inplace=True)
        resampled["buy_percentage"].fillna(0.5, inplace=True)
        return resampled.drop(columns=["usdt_volume", "buy_usdt_volume"])

    def load_daily_trades_coin_margined(self, symbol, day, month, year):
        return self.load_daily_trades(symbol, day, month, year, "coin")

    def load_daily_trades_usdt_margined(self, symbol, day, month, year):
        return self.load_daily_trades(symbol, day, month, year, "usdt")

    def load_daily_trades_spot(self, symbol, day, month, year):
        return self.load_daily_trades(symbol, day, month, year, "spot")

    def load_monthly_trades_usdt_margined(self, symbol, month, year):
        return self.load_monthly_trades(symbol, month, year, "usdt")

    def make_daily_lambda(self, symbol, day, month, year, resample_interval, f):
        trades = self.load_daily_trades_usdt_margined(symbol, day, month, year)
        resampled = f(trades, resample_interval)
        del trades
        return resampled

    def make_monthly_volume_vwap_and_imbalance(self, symbol, month, year, resample_interval, contract="usdt"):
        num_days = monthrange(year, month)[1]
        with Pool(n_cpus) as p:
            return pd.concat(p.map(lambda day: self.make_daily_volume_vwap_and_imbalance(symbol, day, month, year,
                                                                                         resample_interval, contract),
                                   range(1, num_days + 1))).reset_index(drop=True)

        # return pd.concat(
        #     self.make_daily_volume_vwap_and_imbalance(symbol, day, month, year, resample_interval, contract) for day in
        #     range(1, num_days + 1)).reset_index(drop=True)

    def make_daily_volume_vwap_and_imbalance(self, symbol, day, month, year, resample_interval, contract="usdt"):
        if resample_interval < 1000:
            raise (Exception("Currently can't resample sub 1 second"))
        if (symbol, day, month, year, contract) in self.one_sec_resampled_cache:
            resampled1sec = self.one_sec_resampled_cache[(symbol, day, month, year, contract)]
        else:
            # trades = self.load_daily_trades(symbol, day, month, year, contract)
            # resampled1sec = BinanceTradeLoader.make_volume_vwap_and_imbalance_from_trades(trades, 1000)
            resampled1sec = self.load_daily_1_sec_resampled_trades(symbol=symbol, day=day, month=month, year=year,
                                                                   contract=contract)
            self.one_sec_resampled_cache[(symbol, day, month, year, contract)] = resampled1sec
        resampled = BinanceTradeLoader.make_volume_vwap_and_imbalance_from_resampled(resampled1sec, resample_interval)
        return resampled

    def make_volume_vwap_and_imbalance_from_resampled(resampled, resample_interval):
        resampled["usdt_volume"] = resampled["vwap"] * resampled["qty"]
        resampled["buy_usdt_volume"] = resampled["buy_percentage"] * resampled["usdt_volume"]
        resampled = resampled.resample(f'{resample_interval}ms', on="time").agg(
            {"buy_usdt_volume": np.sum, "usdt_volume": np.sum, "qty": np.sum, "open": "first", "high": np.max,
             "close": "last", "low": np.min}).reset_index()

        resampled["buy_percentage"] = resampled["buy_usdt_volume"] / resampled["usdt_volume"]
        resampled["vwap"] = resampled["usdt_volume"] / resampled["qty"]
        resampled["vwap"].fillna(method="ffill", inplace=True)
        resampled["buy_percentage"].fillna(0.5, inplace=True)
        return resampled.drop(columns=["usdt_volume", "buy_usdt_volume"])

    def make_volume_vwap_and_imbalance_from_trades(trades, resample_interval=1000):
        trades["buy_usdt_volume"] = trades["price"] * trades["qty"] * (1 - trades["isBuyerMaker"])
        # trades["sell_usdt_volume"] = trades["price"] * trades["qty"] * trades["isBuyerMaker"]
        trades["usdt_volume"] = trades["price"] * trades["qty"]
        resampled = trades.resample(f'{resample_interval}ms', on="time").agg(
            {"buy_usdt_volume": np.sum, "usdt_volume": np.sum, "qty": np.sum}).reset_index()
        resampled["buy_percentage"] = resampled["buy_usdt_volume"] / resampled["usdt_volume"]
        resampled["vwap"] = resampled["usdt_volume"] / resampled["qty"]
        resampled["vwap"].fillna(method="ffill", inplace=True)
        resampled["buy_percentage"].fillna(0.5, inplace=True)
        return resampled.drop(columns=["usdt_volume", "buy_usdt_volume"])

    # def make_volume_vwap_and_buyvolume(trades, resample_interval=1000):
    #     trades["buy_usdt_volume"] = trades["price"] * trades["qty"] * (1 - trades["isBuyerMaker"])
    #     #     trades["sell_usdt_volume"] = trades["price"] * trades["qty"] *trades["isBuyerMaker"]
    #     trades["usdt_volume"] = trades["price"] * trades["qty"]
    #     resampled = trades.resample(f'{resample_interval}ms', on="time").agg(
    #         {"buy_usdt_volume": np.sum, "usdt_volume": np.sum, "qty": np.sum}).reset_index()
    #     resampled["buy_percentage"] = resampled["buy_usdt_volume"] / (resampled["usdt_volume"])
    #     resampled["vwap"] = resampled["usdt_volume"] / resampled["qty"]
    #     resampled["vwap"].fillna(method="ffill", inplace=True)
    #     resampled["buy_percentage"].fillna(0.5, inplace=True)
    #     #     return resampled.drop(columns = ["usdt_volume", "buy_usdt_volume", "sell_usdt_volume"])
    #     return resampled

    def clear_cache(self):
        self.one_sec_resampled_cache = {}
