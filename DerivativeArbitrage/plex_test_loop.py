from ftx_portfolio import *
from time import sleep

i=1
while i<100:
    try:
        #    run_plex('ftx_auk', 'SystematicPerp')#
        run_plex('ftx_auk', 'CashAndCarry',dirname='Runtime/test_loop/')
    except:
        sleep(5*60)
    i=i+1