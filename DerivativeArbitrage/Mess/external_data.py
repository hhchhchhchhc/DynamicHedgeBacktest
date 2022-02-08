import requests
import pandas as pd
url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false'
response = requests.get(url)
mktcaps = pd.DataFrame(response.json())
borrowable = mktcaps[:88]

url = 'https://api.coingecko.com/api/v3/coins/search'
response = requests.get(url)
token_description = response.json()
rebase_tokens = [item['name'] for item in token_description if item['category_id']=='rebase-tokens']
print(rebase_tokens)