import datetime, argparse
from SpreadExecution.Mess.backtest import Backtest
from concurrent.futures import ProcessPoolExecutor
import puffin


def one_run(latency: int, data_dir: str, date: datetime.datetime, base_asset: str, quote_asset: str):
    my_backtest = Backtest(data_dir, date, base_asset, quote_asset)
    my_backtest.run(latency, True)
    my_backtest.run(latency, False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='date to run for')
    parser.add_argument('--threads', help='number of threads/processes to use')
    parser.add_argument('--count', help='number of points to run')
    parser.add_argument('--latency-ns', help='latency in nanoseconds')
    parser.add_argument('--data-dir', help='data directory')
    parser.add_argument('--base-asset', help='base asset (uppercase)')
    parser.add_argument('--quote-asset', help='quote asset (uppercase)')
    args = parser.parse_args()

    date = datetime.datetime.strptime(args.date, '%Y-%m-%d') if args.date is not None else datetime.datetime(2021, 10, 14)
    threads = int(args.threads) if args.threads is not None else 16
    count = int(args.count) if args.count is not None else 16
    latency_ns = int(args.latency_ns) if args.latency_ns is not None else 150000000
    data_dir = args.data_dir if args.data_dir is not None else puffin.config.source_directory + 'data/inputs/'
    base_asset = args.base_asset if args.base_asset is not None else 'BTC'
    quote_asset = args.quote_asset if args.quote_asset is not None else 'USD'

    with ProcessPoolExecutor(threads) as pool:
        for i in range(count):
            pool.submit(one_run, latency_ns, data_dir, date, base_asset, quote_asset)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print('--- ran in ' + str(end_time - start_time))
