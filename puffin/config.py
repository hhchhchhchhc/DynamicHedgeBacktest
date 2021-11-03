from enum import Enum

import pandas as pd

config = pd.DataFrame(data={'id': [70916, 4610, 70510, 4623, 63048, 4627, 72202, 4656, 70374, 4670, 73714, 63021, 71426,
                                   62984,30096, 30166, 30187, 30590, 30653, 30702, 30735, 30746, 30774, 30788, 30800, 
                                   30813, 30846, 30847, 30908, 30920, 30937, 30951, 30984, 31001, 58847, 58867, 58870, 
                                   58871, 58874, 58876, 59025, 59027, 59028, 69584],
                            'symbol': ['ADABUSD', 'ADAUSDT', 'BNBBUSD', 'BNBUSDT', 'BTCBUSD', 'BTCUSDT', 'DOGEBUSD',
                                       'DOGEUSDT', 'ETHBUSD', 'ETHUSDT', 'SOLBUSD', 'SOLUSDT', 'XRPBUSD', 'XRPUSDT',
                                       'AAVE-PERP', 'BCH-PERP', 'BNB-PERP', 'BTC-PERP', 'DOGE-PERP', 'ETH-PERP', 'HT-PERP', 
                                       'KNC-PERP', 'LINK-PERP', 'LTC-PERP', 'MATIC-PERP', 'MKR-PERP', 'OKB-PERP', 'OMG-PERP', 
                                       'TOMO-PERP', 'TRX-PERP', 'UNI-PERP', 'USDT-PERP', 'XRP-PERP', 'YFI-PERP', 'GRT-PERP', 
                                       'RUNE-PERP', 'SNX-PERP', 'SOL-PERP', 'SUSHI-PERP', 'SXP-PERP', '1INCH-PERP', 'ALPHA-PERP', 
                                       'BAND-PERP', 'CEL-PERP'],

                            'tick_size': [0.0001, 0.0001, 0.01, 0.01, 0.01, 0.01, 0.00001, 0.00001, 0.01, 0.01, 0.001,
                                          0.001, 0.0001, 0.0001,0.01, 0.05, 0.0025, 1.0, 5e-07, 0.1, 0.0005, 0.0001, 
                                          0.0005, 0.01, 2.5e-06, 0.5, 0.0005, 0.0005, 5e-05, 2.5e-06, 0.001, 0.0001, 
                                          2.5e-05, 5.0, 5e-05, 0.0005, 0.0005, 0.0025, 0.0001, 0.0005, 0.0001, 5e-05, 
                                          0.001, 0.0005],

                            'step_size': [1, 1, 0.01, 0.01, 0.001, 0.001, 1, 1, 0.001, 0.001, 1, 1, 0.1, 0.1, 0.01, 0.001,
                             0.1, 0.0001, 1.0, 0.001, 0.01, 0.1, 0.1, 0.01, 1.0, 0.001, 0.01, 0.1, 0.1, 1.0, 0.1, 1.0, 1.0,
                              0.001, 1.0, 0.1, 0.1, 0.01, 0.5, 0.1, 1.0, 1.0, 0.1, 0.1],

                            'minimum_order_size_base': [1, 1, 0.01, 0.01, 0.001, 0.001, 1, 1, 0.001, 0.001, 1, 1, 0.1,
                                                        0.1, 0.01, 0.001, 0.1, 0.0001, 1, 0.001, 0.01,
                                                        0.1, 0.1, 0.01, 1, 0.001, 0.01, 0.1,
                                                        0.1, 1, 0.1, 1, 1, 0.001, 1,
                                                        0.1, 0.1, 0.01, 0.5, 0.1, 1, 1,
                                                        0.1, 0.1],
                            'minimum_order_size_quote': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                                                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                                                        
                            'exchange':['binance', 'binance', 'binance', 'binance', 'binance', 'binance', 'binance',
                                        'binance', 'binance', 'binance', 'binance', 'binance', 'binance', 'binance',
                                        'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 
                                        'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 
                                        'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx', 'ftx']})

source_directory: str = '/Users/rahmanw/Dev/Puffin/Notebooks/datasets/'


class Strategy(Enum):
    ASMM_PHI = 1
    ASMM_HIGH_LOW = 2
    ROLL_MODEL = 3


class Bar(Enum):
    ONE_SECOND = 1
    TEN_TICKS = 2


class AlphaModel(Enum):
    NONE = 0
    MOMENTUM = 1


class Direction(Enum):
    SHORT = -1
    NEUTRAL = 0
    LONG = 1
