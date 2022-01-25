import requests
url = 'https://api.coingecko.com/api/v3/coins/search'
response = requests.get(url)
token_description = response.json()
rebase_tokens = [item['name'] for item in token_description if item['category_id']=='rebase-tokens']
print(rebase_tokens)