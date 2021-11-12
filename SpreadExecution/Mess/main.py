import datetime
from SpreadExecution.Mess.backtest import Backtest
from concurrent.futures import ProcessPoolExecutor


def one_run():
    my_backtest = Backtest()
    my_backtest.run()


def main():
    one_run()
    # with ProcessPoolExecutor(8) as pool:
    #     for j in range(8):
    #         pool.submit(one_run)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print('--- ran in ' + str(end_time - start_time))
