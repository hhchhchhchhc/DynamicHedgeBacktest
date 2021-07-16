import pandas as pd
import requests
import os.path
import distutils.dir_util
from calendar import monthrange

class BinanceTradeLoader:
    def __init__(self, local_cache_root):
        self.local_cache_root = local_cache_root

    # local_cache_root = "/media/corvin/BigData/Crypto/binance/futures/"
    def load_monthly_trades(self, symbol, month, year, contract):
        num_days = monthrange(year, month)[1] # num_days = 28
        return pd.concat(self.load_daily_trades( symbol, day, month, year, contract) for day in range(1,num_days + 1))

    def load_daily_trades(self, symbol, day, month, year, contract):
        path = self.get_path_daily_trades_by_contract(symbol, day, month, year, contract)
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
        result = pd.read_csv(file_path, header=None, usecols=[1, 2, 4, 5],
                             #                    names = ["trade Id", "price", "qty", "quoteQty", "time", "isBuyerMaker"]
                             names=["price", "qty", "time", "isBuyerMaker"]
                             )
        result["time"] = pd.to_datetime(result["time"], unit="ms")
        return result

    def get_path_daily_trades_by_contract(self, symbol, day, month, year, contract):
        if contract not in ["coin", "usdt", "spot"]:
            print("Contract needs to be in ", ["coin", "usdt", "spot"])
            return
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
        root_path = contract_dir_prefix + "daily/trades/" + symbol + contract_prefix + "/"
        return root_path + symbol + contract_prefix + "-trades-" + str(year) + "-" + str(month) + "-" + str(
            day) + ".zip"

    def load_daily_trades_coin_margined(self, symbol, day, month, year):
        return self.load_daily_trades(symbol, day, month, year, "coin")

    def load_daily_trades_usdt_margined(self, symbol, day, month, year):
        return self.load_daily_trades(symbol, day, month, year, "usdt")
    def load_daily_trades_spot(self, symbol, day, month, year):
        return self.load_daily_trades(symbol, day, month, year, "spot")
    def load_monthly_trades_usdt_margined(self, symbol, month, year):
        return self.load_monthly_trades(symbol, month, year, "usdt")
