from ftx_history import *
from s3 import *

def build_fine_history():
    exchange=open_exchange('ftx')
    futures = pd.DataFrame(fetch_futures(exchange,includeExpired=False))

    funding_threshold = 1e4
    volume_threshold = 1e6
    type_allowed='perpetual'
    backtest_window=timedelta(weeks=200)

    enriched=enricher(exchange, futures)
    pre_filtered = enriched[
        (enriched['expired'] == False)
        & (enriched['funding_volume'] * enriched['mark'] > funding_threshold)
        & (enriched['volumeUsd24h'] > volume_threshold)
        & (enriched['tokenizedEquity']!=True)
        & (enriched['type']==type_allowed)]

    #### get history ( this is sloooow)
    history = build_history(pre_filtered, exchange, timeframe='15s', end=datetime.today(),
                  start=datetime.today() - backtest_window).to_parquet("15shistory.parquet")
    s3_upload_file('15shistory.parquet',
                   'gof.crypto.shared', 'ftx_fine_history')
    return None
