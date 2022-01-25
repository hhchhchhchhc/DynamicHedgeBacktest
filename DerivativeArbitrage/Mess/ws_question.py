import time
import sys
import ccxtpro
import asyncio
from copy import deepcopy

import logging
logging.basicConfig(level=logging.INFO)

class myExchange(ccxtpro.some_exchange):
    def handle_xxx(self, client, message):
        # 'whatever needs to be done syncronously and immediately: eg updating state')
    async def watch_state(self):
        # that waits for data and returns it when it comes. Once.
    async def react_to_state(self,data):
        # can use data and also the state of some_exchange
    async def loop_watch_state(self):
        while True:
            data=self.watch_state()
            self.react_to_state(data)

def main(loop):
    exchange=myExchange(loop) # what does exchange do with loop? Seems to work without.
    states_to_watch=['orderbooks', 'trades', 'balances','sometimes', 'markets']
    while True:
        await asyncio.gather([exchange.loop_watch_state() for state in states_to_watch])

loop = asyncio.get_event_loop()
loop.run_until_complete(main(loop))