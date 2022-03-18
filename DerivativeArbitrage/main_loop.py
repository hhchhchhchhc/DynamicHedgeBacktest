from ftx_utilities import *
from ftx_portfolio import ftx_portoflio_main
from strategies import strategies_main
from ftx_ws_execute import ftx_ws_spread_main
from time import sleep
from datetime import *

launchtime_minutes = [datetime.now().minute]
throttle_minutes = 1
skip=[]#'ftx_ws_spread_main']#'strategies_main','ftx_ws_spread_main']


def loop_main(*argv):
    argv=list(argv)
    if len(argv) == 0:
        argv.extend(['loop'])
    if len(argv) < 3:
        argv.extend(['ftx', 'debug'])

    if (argv[0]=='loop'):
        while True:
            minutes_to_sleep = min([t - datetime.now().minute for t in launchtime_minutes])
            if minutes_to_sleep < 0: minutes_to_sleep += 60

            print(f'awaiting refresh time for {minutes_to_sleep} mins')
            sleep(60 * max(0, minutes_to_sleep))
            while True:
                try:
                    if 'strategies_main' in skip: break
                    strategies_main('sysperp')
                except Exception as e:
                    print(f'restarting strategy after {throttle_minutes} mins. Failed with {str(e)}')
                    sleep(throttle_minutes * 60)
                    continue
                else:
                    break
            print(f'weights refreshed. {throttle_minutes} mins before exec ')
            sleep(throttle_minutes * 60)

            while True:
                try:
                    if 'ftx_ws_spread_main' not in skip:
                        ftx_ws_spread_main('sysperp',*argv[1:])
                except Exception as e:
                    print(f'restarting exec after {throttle_minutes} mins. Failed with {str(e)}')
                    sleep(throttle_minutes * 60)
                    continue
                else:
                    print(f'weights executed. Running plex')
                    ftx_portoflio_main('plex', *argv[1:])
                    print(f'plex ran')
                    break
            sleep(throttle_minutes * 60)
        else:
            raise Exception('unknown command')

if __name__ == "__main__":
    loop_main(*sys.argv[1:])