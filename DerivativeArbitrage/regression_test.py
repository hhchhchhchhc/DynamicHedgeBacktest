import strategies,ftx_portfolio,ftx_snap_basis,ftx_ws_execute,ftx_history,datetime

if __name__ == "__main__":
    ftx_history.ftx_history_main('build','max','ftx',7)
    strategies.strategies_main('sysperp',datetime.timedelta(days=2),datetime.timedelta(hours=2))
    strategies.strategies_main('backtest')
    ftx_snap_basis.enricher_wrapper('ftx',['perpetual','future'])
    ftx_portfolio.ftx_portoflio_main('risk','ftx', 'debug')
    ftx_portfolio.ftx_portoflio_main('plex', 'ftx', 'debug')
    ftx_portfolio.ftx_portoflio_main('execprogress', 'ftx', 'debug')