import datetime
from SimpleMarketMaking.Mess import tools

directory = 'C:/Users/Tibor/Sandbox/'
start_time = datetime.datetime.now()

for d in range(9):
    date = datetime.date(2021, 9, 8)
    date = date + datetime.timedelta(days=d)
    tools.write_data_to_file(date, directory)

end_time = datetime.datetime.now()
print('--- ran in ' + str(end_time - start_time))
