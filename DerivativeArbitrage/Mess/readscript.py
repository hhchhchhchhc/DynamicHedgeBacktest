import json
import pandas as pd
import datetime
import ftx_utilities

with open('events.json','r') as file:
    d = json.load(file)
    events=pd.DataFrame(d)
with open('request.json', 'r') as file:
    d = json.load(file)
    request = pd.Series(d)
with pd.ExcelWriter('logs.xlsx', engine='xlsxwriter') as writer:
    request.to_excel(writer,sheet_name='request')
    events.to_excel(writer, sheet_name='events')

# i = 1
# while i < 1:
#     try:
#         data = pd.concat(await asyncio.gather(*(
#                 [funding_history(f, exchange, start, end, dirname)
#                  for (i, f) in futures[futures['type'] == 'perpetual'].iterrows()] +
#                 [rate_history(f, exchange, end, start, timeframe, dirname)
#                  for (i, f) in futures.iterrows()] +
#                 [spot_history(f + '/USD', exchange, end, start, timeframe, dirname)
#                  for f in futures['underlying'].unique()] +
#                 [borrow_history(f, exchange, end, start, dirname)
#                  for f in (list(futures.loc[futures['spotMargin'], 'underlying'].unique()) + ['USD'])]
#         )), join='outer', axis=1)
#         break
#     except Exception as e:
#         if dirname == '':
#             logging.exception(e)
#             break
#         else:  # RuntimeError('throttle queue is over maxCapacity')
#             logging.info(f"looping over history, tries {i}")
#             i = +1