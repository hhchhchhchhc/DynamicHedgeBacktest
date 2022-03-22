import requests
import pandas as pd
import sys

import asyncio
import csv
from tardis_client import TardisClient, Channel
from tardis_dev import datasets, get_exchange_details
from datetime import datetime, timezone
import logging
import json

def get_data_feeds(exchange = 'deribit'):
    exchange_details = get_exchange_details(exchange)

    # iterate over and download all data for every symbol
    for symbol in exchange_details["datasets"]["symbols"]:
        # alternatively specify datatypes explicitly ['trades', 'incremental_book_L2', 'quotes'] etc
        # see available options https://docs.tardis.dev/downloadable-csv-files#data-types
        data_types = ['options_chain']#symbol["dataTypes"]
        symbol_id = symbol["id"]
        from_date = symbol["availableSince"]
        to_date = symbol["availableTo"]

        # skip groupped symbols
        if symbol_id in ['PERPETUALS', 'SPOT', 'FUTURES']:
            continue

        print(f"Downloading {exchange} {data_types} for {symbol_id} from {from_date} to {to_date}")

        # each CSV dataset format is documented at https://docs.tardis.dev/downloadable-csv-files#data-types
        # see https://docs.tardis.dev/downloadable-csv-files#download-via-client-libraries for full options docs
        datasets.download(
            exchange=exchange,
            data_types=data_types,
            from_date=str(datetime(year=2022,month=3,day=20,tzinfo=timezone.utc)),
            to_date=str(datetime(year=2022,month=3,day=21,tzinfo=timezone.utc)),
            symbols=[symbol_id],
            api_key="TD.5sHAB2mPdsHYylP6.Y9KLWdd3XIeOpgv.7KnfWYukxDDuyFu.WUi-a3IvCvWnavo.iekVlRGaldwInlo.QJSZ",
            # path where CSV data will be downloaded into
            download_dir="Runtime/Tardis"
        )

async def save_historical_deribit_index_data_to_csv():
    tardis_client = TardisClient()
    messages = tardis_client.replay(
        exchange="deribit",
        from_date="2019-06-01",
        to_date="2019-06-02",
        filters=[Channel(name="deribit_price_index", symbols=["btc_usd", "eth_usd"])],
    )
    with open("Runtime/Tardis/deribit_index_data.csv", mode="w") as csv_file:
        fieldnames = ["symbol", "price", "timestamp"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        async for local_timestamp, message in messages:
            data = message["params"]["data"]
            writer.writerow({"symbol": data["index_name"], "price": data["price"], "timestamp": data["timestamp"]})

    print("finished")


def tardis_main(*argv):
    argv = list(argv)
    if len(argv) < 1:
        argv.extend(['rest'])

    if argv[0] == 'rest':
        return get_data_feeds()
    elif argv[0] == 'ws':
        return asyncio.run(save_historical_deribit_index_data_to_csv())
    else:
        url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false'
        response = requests.get(url)
        mktcaps = pd.DataFrame(response.json())
        borrowable = mktcaps[:88]

        url = 'https://api.coingecko.com/api/v3/coins/search'
        response = requests.get(url)
        token_description = response.json()
        rebase_tokens = [item['name'] for item in token_description if item['category_id'] == 'rebase-tokens']
        print(rebase_tokens)

if __name__ == "__main__":
    tardis_main(*sys.argv[1:])
