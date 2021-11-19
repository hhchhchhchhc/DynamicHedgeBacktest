import datetime
from SpreadExecution.Mess.backtest import Backtest
from concurrent.futures import ProcessPoolExecutor


def one_run(latency: int):
    my_backtest = Backtest()
    my_backtest.run(latency, True)
    my_backtest.run(latency, False)


def main():
    with ProcessPoolExecutor(16) as pool:
        for i in range(16):
            pool.submit(one_run, 150000000)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print('--- ran in ' + str(end_time - start_time))
