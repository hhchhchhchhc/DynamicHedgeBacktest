from ftx_portfolio import *
from time import sleep

i=1
while i<100:
#    run_plex('ftx_auk', 'CashAndCarry',dirname='C:/Users/david/Dropbox/auk/test_loop/')
#    run_plex('ftx_auk', 'SystematicPerp', dirname='C:/Users/david/Dropbox/auk/test_loop/')
    run_plex('ftx', 'margintest', dirname='C:/Users/david/Dropbox/auk/test_loop/')
#    except Exception as e:
#        print(e)
    sleep(5)
    i=i+1