import datetime
from backtest import Backtest
import config as _config
import pandas as pd
from market_data import MarketData
from concurrent.futures import ProcessPoolExecutor


def main():
    run_nu(nu=1.0)


def create_tick_bars():
    date = datetime.date(2021, 10, 1)
    market_data = MarketData('XRPUSDT')
    market_data.load_formatted_trade_data_from_csv(date)
    tick_bars = market_data.get_tick_bars(10)
    tick_bars.to_csv(_config.source_directory + 'tick_bars.csv')


def run_phi(phi: float) -> None:
    print('running phi = ' + str("{:,.2f}".format(phi)))
    start_date = datetime.date(2021, 10, 1)
    number_of_days = 1
    bar = _config.Bar.ONE_SECOND
    strategy = _config.Strategy.ASMM_PHI
    parameters = {'phi': phi}
    my_backtest = Backtest('XRPUSDT', bar, strategy, parameters, start_date, number_of_days)
    results, summary = my_backtest.run()
    results.to_csv(_config.source_directory + 'new_results_' + str("{:,.2f}".format(phi)) + '.csv', index=False)
    summary.to_csv(_config.source_directory + 'new_summary_' + str("{:,.2f}".format(phi)) + '.csv', index=False)


def run_nu(nu: float) -> None:
    print('running nu = ' + str("{:,.2f}".format(nu)))
    start_date = datetime.date(2021, 10, 1)
    number_of_days = 1
    bar = _config.Bar.TEN_TICKS
    strategy = _config.Strategy.ASMM_HIGH_LOW
    parameters = {'nu': nu}
    my_backtest = Backtest('XRPUSDT', bar, strategy, parameters, start_date, number_of_days)
    results, summary = my_backtest.run()
    results.to_csv(_config.source_directory + 'new_results_high_low_' + str("{:,.2f}".format(nu)) + '.csv', index=False)
    summary.to_csv(_config.source_directory + 'new_summary_high_low_' + str("{:,.2f}".format(nu)) + '.csv', index=False)


def run_nus():
    nus = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    with ProcessPoolExecutor(5) as pool:
        for nu in nus:
            pool.submit(run_nu, nu)


def run_phis():
    phis = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    with ProcessPoolExecutor(5) as pool:
        for phi in phis:
            pool.submit(run_phi, phi)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print('--- ran in ' + str(end_time - start_time))


def create_secondly_bars():
    for d in range(22):
        date = datetime.date(2021, 9, 1) + datetime.timedelta(days=d)
        date_string = date.strftime('%Y%m%d')
        my_market_data = MarketData('XRPUSDT')
        try:
            my_market_data.load_formatted_trade_data_from_csv(date)
            secondly_bars = my_market_data.get_time_bars(1000)
            secondly_bars.to_csv(
                _config.source_directory + 'formatted/secondly/' + date_string + '_Binance_XRPUSDT_trades.csv')
        except FileNotFoundError as error:
            print(repr(error))


def format_files():
    for d in range(22):
        date = datetime.date(2021, 9, 1) + datetime.timedelta(days=d)
        try:
            my_market_data = MarketData('XRPUSDT')
            my_market_data.load_trade_data_from_parquet(date)
            my_market_data.load_top_of_book_data_from_parquet(date)
            my_market_data.generate_formatted_trades_data()
            my_market_data.generate_formatted_top_of_book_data()
            date_string = date.strftime('%Y%m%d')
            my_market_data.trades_formatted.to_csv(_config.source_directory + 'formatted/trades/' + date_string +
                                                   '_Binance_XRPUSDT_trades.csv')
            my_market_data.top_of_book_formatted.to_csv(_config.source_directory + 'formatted/tobs/' + date_string +
                                                        '_Binance_XRPUSDT_tobs.csv')
        except FileNotFoundError as error:
            print(repr(error))


def test_volume_imbalance(data: pd.DataFrame):
    look_backs = range(1, 6)
    horizons = range(1, 6)

    n = len(data.index)
    returns = pd.DataFrame(index=data.index, columns=['horizon_' + str(horizon) for horizon in horizons])
    for i in data.index:
        for horizon in horizons:
            if i + horizon < n:
                returns.iloc[i, horizon - 1] = (data.loc[i + horizon, 'vwap'] / data.loc[i, 'vwap']) - 1

    volume_imbalance = pd.DataFrame(index=data.index,
                                    columns=['look_back_' + str(look_back) for look_back in look_backs])
    cumulative_total_volume = 0
    cumulative_given_volume = 0
    cumulative_total_volumes = []
    cumulative_given_volumes = []
    for i in data.index:
        cumulative_total_volume = cumulative_total_volume + data.loc[i, 'volume']
        cumulative_given_volume = cumulative_given_volume + data.loc[i, 'volume_given']
        cumulative_total_volumes.append(cumulative_total_volume)
        cumulative_given_volumes.append(cumulative_given_volume)
        for look_back in look_backs:
            if i >= look_back:
                total_volume = cumulative_total_volumes[i] - cumulative_total_volumes[i - look_back]
                given_volume = cumulative_given_volumes[i] - cumulative_given_volumes[i - look_back]
                imbalance = (total_volume - (2 * given_volume)) / total_volume
                volume_imbalance.iloc[i, look_back - 1] = imbalance
            elif i == look_back - 1:
                total_volume = cumulative_total_volumes[i]
                given_volume = cumulative_given_volumes[i]
                imbalance = (total_volume - (2 * given_volume)) / total_volume
                volume_imbalance.iloc[i, look_back - 1] = imbalance


def create_all_sorts_of_bars(my_market_data: MarketData):
    print('secondly bars')
    secondly_bars = my_market_data.get_time_bars(1000)
    print(secondly_bars)
    secondly_bars.to_csv(_config.source_directory + 'formatted/20210901_Binance_BTCUSDT_secondly_bars.csv')

    print('tick bars')
    tick_bars = my_market_data.get_tick_bars(18)
    print(tick_bars)
    tick_bars.to_csv(_config.source_directory + 'tick_bars.csv')

    print('volume bars')
    volume_bars = my_market_data.get_volume_bars(4050)
    print(volume_bars)
    volume_bars.to_csv(_config.source_directory + 'volume_bars.csv')

    print('dollar bars')
    dollar_bars = my_market_data.get_dollar_bars(2000000)
    print(dollar_bars)
    dollar_bars.to_csv(_config.source_directory + 'dollar_bars.csv')

    print('tick imbalance bars')
    tick_imbalance_bars = my_market_data.get_tick_imbalance_bars()
    print(tick_imbalance_bars)
    tick_imbalance_bars.to_csv(_config.source_directory + 'tick_imbalance_bars.csv')

    print('trade side imbalance bars')
    trade_side_imbalance_bars = my_market_data.get_trade_side_imbalance_bars()
    print(trade_side_imbalance_bars)
    trade_side_imbalance_bars.to_csv(_config.source_directory + 'trade_side_imbalance_bars.csv')

    print('volume tick imbalance bars')
    volume_tick_imbalance_bars = my_market_data.get_volume_tick_imbalance_bars()
    print(volume_tick_imbalance_bars)
    volume_tick_imbalance_bars.to_csv(_config.source_directory + 'volume_tick_imbalance_bars.csv')

    print('trade side imbalance bars')
    volume_trade_side_imbalance_bars = my_market_data.get_volume_trade_side_imbalance_bars()
    print(volume_trade_side_imbalance_bars)
    volume_trade_side_imbalance_bars.to_csv(_config.source_directory + 'volume_trade_side_imbalance_bars.csv')

    print('dollar tick imbalance bars')
    dollar_tick_imbalance_bars = my_market_data.get_dollar_tick_imbalance_bars()
    print(dollar_tick_imbalance_bars)
    dollar_tick_imbalance_bars.to_csv(_config.source_directory + 'dollar_tick_imbalance_bars.csv')

    print('dollar trade side imbalance bars')
    dollar_trade_side_imbalance_bars = my_market_data.get_dollar_trade_side_imbalance_bars()
    print(dollar_trade_side_imbalance_bars)
    dollar_trade_side_imbalance_bars.to_csv(_config.source_directory + 'dollar_trade_side_imbalance_bars.csv')

    print('tick runs bars')
    tick_runs_bars = my_market_data.get_tick_runs_bars()
    print(tick_runs_bars)
    tick_runs_bars.to_csv(_config.source_directory + 'tick_runs_bars.csv')

    print('trade side runs bars')
    trade_side_runs_bars = my_market_data.get_trade_side_runs_bars()
    print(trade_side_runs_bars)
    trade_side_runs_bars.to_csv(_config.source_directory + 'trade_side_runs_bars.csv')

    print('volume tick runs bars')
    volume_tick_runs_bars = my_market_data.get_volume_tick_runs_bars()
    print(volume_tick_runs_bars)
    volume_tick_runs_bars.to_csv(_config.source_directory + 'volume_tick_runs_bars.csv')

    print('volume trade side runs bars')
    volume_trade_side_runs_bars = my_market_data.get_volume_trade_side_runs_bars()
    print(volume_trade_side_runs_bars)
    volume_trade_side_runs_bars.to_csv(_config.source_directory + 'volume_trade_side_runs_bars.csv')

    print('dollar tick runs bars')
    dollar_tick_runs_bars = my_market_data.get_dollar_tick_runs_bars()
    print(dollar_tick_runs_bars)
    dollar_tick_runs_bars.to_csv(_config.source_directory + 'dollar_tick_runs_bars.csv')

    print('dollar trade side runs bars')
    dollar_trade_side_runs_bars = my_market_data.get_dollar_trade_side_runs_bars()
    print(dollar_trade_side_runs_bars)
    dollar_trade_side_runs_bars.to_csv(_config.source_directory + 'dollar_trade_side_runs_bars.csv')
