import csv,sys
import requests,asyncio,functools
import pandas as pd
from datetime import *
from tardis_client import TardisClient, Channel
from tardis_dev import datasets, get_exchange_details

def async_wrap(f):
    @functools.wraps(f)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        p = functools.partial(f, *args, **kwargs)
        return await loop.run_in_executor(executor, p)
    return run

def get_data_feeds(start, end, symbols = 'OPTION_CHAIN', exchange = 'deribit'):
    #exchange_details = get_exchange_details(exchange)

    # datasets.download(
    #     exchange=exchange,
    #     data_types=['derivative_ticker'],
    #     from_date=str(start),
    #     to_date=str(end),
    #     symbols=['PERPETUALS','FUTURES'],
    #     api_key="TD.5sHAB2mPdsHYylP6.Y9KLWdd3XIeOpgv.7KnfWYukxDDuyFu.WUi-a3IvCvWnavo.iekVlRGaldwInlo.QJSZ",
    #     # path where CSV data will be downloaded into
    #     download_dir="Runtime/Tardis"
    # )

    return datasets.download(
        exchange=exchange,
        data_types=['trades','book_snapshot_5'],
        from_date=str(start),
        to_date=str(end),
        symbols=symbols,
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

ribbon_auctions = [
    {'trade_date': datetime(2022,4,15,10)-timedelta(days=7),
     'expiry_date': datetime(2022,4,15,10),
     'strike': 4000,
     'volume': 38000},
    {'trade_date': datetime(2022, 4, 1, 10),
     'expiry_date': '8APR22',
     'strike': 3700,  # 3700
     'volume': 38000},
    {'trade_date': datetime(2022, 3, 25, 10),
     'expiry_date': '1APR22',
     'strike': 3500,#3700
     'volume': 38000},
    {'trade_date': datetime(2022, 3, 18, 10),
     'expiry_date': '25MAR22',
     'strike': 3200,
     'volume': 38000},
    {'trade_date': datetime(2022, 3, 11, 10),
     'expiry_date': '18MAR22',
     'strike': 3000,
     'volume': 38000},
    {'trade_date': datetime(2022, 3, 4, 10),
     'expiry_date': '11MAR22',
     'strike': 3100,
     'volume': 38000},
    {'trade_date': datetime(2022, 2, 25, 10),
     'expiry_date': '4MAR22',
     'strike': 3000,
     'volume': 38000},
    {'trade_date': datetime(2022, 2, 18, 10),
     'expiry_date': '25FEB22',
     'strike': 3300,
     'volume': 38000},
]

def tardis_main(*argv):
    argv = list(argv)
    if len(argv) < 1:
        argv.extend(['rest'])

    if argv[0] == 'rest':
        symbols = ['ETH-'+auction['expiry_date']+'-'+str(auction['strike'])+'-C' for auction in ribbon_auctions[1:]]
        coro = [async_wrap(get_data_feeds(auction['trade_date']-timedelta(hours=6),auction['trade_date']+timedelta(hours=2),[symbol]))
                for symbol,auction in zip(symbols,ribbon_auctions[1:])]
        data = asyncio.gather(*coro)
        return data
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
