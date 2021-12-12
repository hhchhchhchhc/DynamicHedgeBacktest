from ftx_portfolio import *
from time import sleep

i=1
while i<1000:
#    run_plex('ftx_auk', 'CashAndCarry',dirname='C:/Users/david/Dropbox/auk/test_loop/')
#    run_plex('ftx_auk', 'SystematicPerp', dirname='C:/Users/david/Dropbox/auk/test_loop/')
#    run_plex('ftx', 'margintest', dirname='C:/Users/david/Dropbox/auk/test_loop/')
#    run_plex('ftx', 'plexspottest', dirname='C:/Users/david/Dropbox/auk/test_loop/')
    run_plex('ftx', 'plextest2', dirname='C:/Users/david/Dropbox/auk/test_loop/')
#    except Exception as e:
#        print(e)
    sleep(10)
    i=i+1