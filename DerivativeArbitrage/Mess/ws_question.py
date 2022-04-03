import ccxtpro
import asyncio

class myExchange(ccxtpro.some_exchange):
    def __init__(self,*args,**kwargs):
        super(myExchange, self).__init__(*args,**kwargs) # kwargs include 'loop'
        _data = logic1(*args,**kwargs)
    def handle_xxx(self, client, message):
        # called by ccxt in the background upon message receipt. syncronous.
    async def watch_state(self):
        # that waits for data and returns it when it comes. Once.
    async def async_react_to_state(self,i):
        print(i)
        await asyncio.sleep(i/10)
    def sync_react_to_state(self):
        pass
    async def loop_watch_state(self):
        i=0
        while True:
            try:
                i=i+1
                data = await self.async_react_to_state()
                # handle_xxx is called by ccxt in the background upon message receipt
                self.react_to_state(data)
            # except NonCriticalError as e:
            #     # for instance connection error
            #     continue
            except Exception as e:
                break
            finally:
                cleanup()

def main(*argv,**kwargs):
    exchange=myExchange(*argv,**kwargs)
    states_to_watch=['orderbooks', 'trades', 'balances','sometimes', 'markets']
    while True:
        await asyncio.gather([exchange.loop_watch_state() for state in states_to_watch])

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(symbols=['DAWN/USD','DAWN/USD:USD'],size=[10,-10],loop=loop))