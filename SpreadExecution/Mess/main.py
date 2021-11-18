import datetime
from SpreadExecution.Mess.backtest import Backtest
from concurrent.futures import ProcessPoolExecutor


def main():
    my_backtest = Backtest()
    my_backtest.run(150000000, True)
    my_backtest.run(150000000, False)
    # with ProcessPoolExecutor(8) as pool:
    #     for latency in [0, 10000000, 25000000, 50000000, 100000000, 150000000, 300000000, 450000000]:
    #         pool.submit(my_backtest.run, latency)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print('--- ran in ' + str(end_time - start_time))
