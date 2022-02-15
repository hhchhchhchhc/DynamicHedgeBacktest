import logging

import strategies,ftx_portfolio,ftx_snap_basis,ftx_ws_execute,ftx_history,datetime
logging.basicConfig(logging.ERROR)

if __name__ == "__main__":
    ftx_history.ftx_history_main('build','max','ftx',7)
    logging.error('ftx_history_main ok')
    strategies.strategies_main('sysperp',datetime.timedelta(days=2),datetime.timedelta(hours=2))
    logging.error('sysperp ok')
    ftx_snap_basis.enricher_wrapper('ftx',['perpetual','future'])
    strategies.strategies_main('basis ok')
    ftx_portfolio.ftx_portoflio_main('risk','ftx', 'debug')
    strategies.strategies_main('risk ok')
    ftx_portfolio.ftx_portoflio_main('plex', 'ftx', 'debug')
    strategies.strategies_main('plex ok ')
    ftx_portfolio.ftx_portoflio_main('fromOptimal', 'ftx', 'debug')
    strategies.strategies_main('fromOptimal ok')
    strategies.strategies_main('backtest')
    logging.error('backtest ok')