import math,scipy
import requests

from deribit_history import *

class black_scholes:
    @staticmethod
    def d1(S, K, V, T):
        return (math.log(S / float(K)) + (V ** 2 / 2) * T) / (V * math.sqrt(T))

    @staticmethod
    def d2(S, K, V, T):
        return black_scholes.d1(S, K, V, T) - (V * math.sqrt(T))

    @staticmethod
    def pv(S, K, V, T, cp):
        if cp == 'C':
            return S * scipy.stats.norm.cdf(black_scholes.d1(S, K, V, T)) - K * scipy.stats.norm.cdf(
                black_scholes.d2(S, K, V, T))
        elif cp == 'P':
            return K * scipy.stats.norm.cdf(-black_scholes.d2(S, K, V, T)) - S * scipy.stats.norm.cdf(
                -black_scholes.d1(S, K, V, T))
        else:
            return black_scholes.pv(S, K, V, T, 'P') + black_scholes.pv(S, K, V, T, 'C')

    @staticmethod
    def delta(S, K, V, T, cp):
        '''for a 1% move'''
        delta = scipy.stats.norm.cdf(black_scholes.d1(S, K, V, T))
        if cp == 'C':
            delta = delta
        elif cp == 'P':
            delta = (delta - 1)
        elif cp =='S':
            delta = (2 * delta - 1)

        return delta * S * 0.01

    @staticmethod
    def gamma(S, K, V, T, cp):
        '''for a 1% move'''
        gamma = scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T)) / (S * V * math.sqrt(T))
        return gamma * S * 0.01 * S * 0.01 * (1 if cp != 'S' else 2)

    @staticmethod
    def vega(S, K, V, T, cp):
        '''for a 10% move'''
        vega = (S * math.sqrt(T) * scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T)))
        return vega * V * 0.1 * (1 if cp != 'S' else 2)

    @staticmethod
    def theta(S, K, V, T, cp):
        '''for 1h'''
        theta = -((S * V * scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T))) / (2 * math.sqrt(T)))
        return theta / 24/365.25 * (1 if cp != 'S' else 2)

def deribit_smile_tardis(currency,whatever):
    ## perp, volindex...
    rest_history = deribit_history_main('just use',[currency],'deribit','cache')[0]

    ## tardis
    history = pd.read_csv(f"Runtime/Deribit_Mktdata_database/deribit_options_chain_2019-07-01_{currency}.csv",nrows=1e5)
    history.dropna(subset=['underlying_price','strike_price','local_timestamp','mark_iv','expiration'],inplace=True)
    #history['expiry'] = history['expiration'].apply(lambda t: pd.to_datetime(int(t), unit='us'))
    history['timestamp'] = history['timestamp'].apply(lambda t: pd.to_datetime(int(t), unit='us'))
    #history['otm_delta'] = history.apply(lambda d: (d['delta'] if d['delta']<0.5 else d['delta']-1) if d['type'] == 'call' else (d['delta'] if d['delta']>-0.5 else d['delta']+1),axis=1)
    history['stddev_factor'] = history.apply(
        lambda d: np.log(d['underlying_price']/d['strike_price'])/np.sqrt((d['expiration']-d['local_timestamp'])/1e6/3600/24/365.25), axis=1)

    #return history

    per_hour = history.groupby(pd.Grouper(key="timestamp", freq="1h"))
    ATM = pd.Series()
    for (hour, hour_data) in per_hour.__iter__():
        print(hour_data[['underlying_price', 'local_timestamp']].mean())
        s_t = hour_data[['underlying_price', 'local_timestamp']].mean()
        s = s_t['underlying_price']
        t = s_t['local_timestamp']

        # simple_ATM, later replace 0.25 by some vega amount floor
        ATM_data = hour_data[np.abs(hour_data['stddev_factor']) < 0.25]
        interpolator = ATM_data.set_index('stddev_factor')['mark_iv']
        interpolator.loc[0] = np.NaN
        interpolator.interpolate(method='index', inplace=True).sort_index()
        ATM[hour] = interpolator.loc[0]

        simple_ATM = dict()
        for side in ['bid', 'ask']:
            simple_ATM[side] = (ATM_data['mark_iv'] * \
                                ATM_data[side + '_amount'] * ATM_data['vega'] \
                                / (ATM_data[side + '_amount'] * ATM_data['vega']).sum()
                                ).sum(min_count=1)
            # vega*amount weighted regress (eg heston) of side_iv on otm_delta,expiry (later bid and ask)
            # vega*amount weighted regress (eg heston) of side_iv on otm_delta,expiry (later bid and ask)

def deribit_smile_genesisvolatility(currency,start='2019-01-01',end='2019-01-02',timeframe='1h'):
    payload='{\"query\":\"query ConstantMaturityAtm1Min($symbol: BTCOrETHEnumType, $dateStart: String, $dateEnd: String, $interval: String)' \
            '{\\n  ConstantMaturityAtm1Min(symbol:$symbol, dateStart:$dateStart, dateEnd: $dateEnd, interval: $interval) ' \
            '{\\n    date\\n    atm7\\n    atm30\\n    atm60\\n    atm90\\n    atm180\\n  }\\n}\\n\",' \
            '\"variables\":{\"symbol\":\"'+currency+'\",\"dateStart\":\"'+start+'\",\"dateEnd\":\"'+end+'\",\"interval\":\"'+timeframe+'\"}}'

    payload = "{\"query\":\"query FixedMaturityAtm($exchange: ExchangeEnumType, $symbol:BTCOrETHEnumType)" \
              "{\\n  FixedMaturityAtm(exchange:$exchange, symbol: $symbol) " \
              "{\\n    date\\n    atm7\\n    atm30\\n    atm60\\n    atm90\\n    atm180\\n    currency\\n  }" \
              "\\n}\",\"variables\":{\"exchange\":\"deribit\",\"symbol\":\""+currency+"\"}}"
    url = "https://app.pinkswantrading.com/graphql"
    payload = "{\"query\":\"query FixedMaturityAtmLite($exchange: ExchangeEnumType, $symbol:BTCOrETHEnumType){\\n  FixedMaturityAtm(exchange:$exchange, symbol: $symbol) {\\n    date\\n    atm7\\n    atm30\\n    atm60\\n    atm90\\n    atm180\\n    currency\\n  }\\n}\",\"variables\":{\"exchange\":\"deribit\",\"symbol\":\"BTC\"}}"
    headers = {
        'x-oracle': api_params.loc['genesisvolatility','value'],
        'Content-Type': 'application/json',
        'accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9'
    }

    response = requests.request("GET", url, headers=headers, data=payload).json()

def deribit_smile_main(*argv):
    argv = list(argv)
    if len(argv) == 0:
        argv.extend(['genesisvolatility'])
    if len(argv) < 2:
        argv.extend(['ETH'])
    if len(argv) < 3:
        argv.extend([1])
    print(f'running {argv}')

    if argv[0] == 'genesisvolatility':
        deribit_smile_genesisvolatility(argv[1])
    elif argv[0] == 'tardis':
        deribit_smile_tardis(argv[1],argv[2])
    else:
        raise Exception('unknown request ' + argv[0])

if __name__ == "__main__":
    deribit_smile_main(*sys.argv[1:])