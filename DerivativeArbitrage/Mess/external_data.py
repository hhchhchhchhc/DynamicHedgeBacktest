import csv,sys
import requests,asyncio
import pandas as pd
from datetime import *
from tardis_client import TardisClient, Channel
from tardis_dev import datasets, get_exchange_details

def get_data_feeds(start, end, exchange = 'deribit'):
    exchange_details = get_exchange_details(exchange)

    datasets.download(
        exchange=exchange,
        data_types=['derivative_ticker'],
        from_date=str(start),
        to_date=str(end),
        symbols=['PERPETUALS','FUTURES'],
        api_key="TD.5sHAB2mPdsHYylP6.Y9KLWdd3XIeOpgv.7KnfWYukxDDuyFu.WUi-a3IvCvWnavo.iekVlRGaldwInlo.QJSZ",
        # path where CSV data will be downloaded into
        download_dir="Runtime/Tardis"
    )

    datasets.download(
        exchange=exchange,
        data_types=['options_chain'],
        from_date=str(start),
        to_date=str(end),
        symbols=['OPTIONS'],
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
        days = [datetime(year=2022,month=3,day=1),datetime(year=2022,month=3,day=1)]
        #return asyncio.gather(*[get_data_feeds(start,end) for start,end in list(zip(days[:-1],days[1:]))])
        return get_data_feeds(days[0],days[1])
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
