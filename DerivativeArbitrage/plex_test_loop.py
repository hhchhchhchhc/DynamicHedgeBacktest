from ftx_portfolio import *
from time import sleep

i=1
while i<1000:
    try:
        run_plex('ftx_auk', 'CashAndCarry',dirname='C:/Users/david/Dropbox/auk/test_loop/')
        run_plex('ftx_auk', 'SystematicPerp', dirname='C:/Users/david/Dropbox/auk/test_loop/')
        run_plex('ftx', 'margintest', dirname='C:/Users/david/Dropbox/auk/test_loop/')
        run_plex('ftx', 'plexspottest', dirname='C:/Users/david/Dropbox/auk/test_loop/')
        run_plex('ftx', 'plextest2', dirname='C:/Users/david/Dropbox/auk/test_loop/')
    except Exception as e:
        print(e)
    sleep(5 * 60)
    i=i+1