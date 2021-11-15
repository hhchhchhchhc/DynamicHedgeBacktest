import puffin.tardis_importer as ti
symbol = 'XRPUSDT'
date = '2021-10/14'
quotes_xrpusdt = ti.generate_tob_data_from_gzip('XRPUSDT', 'C:/Users/Tibor/PycharmProjects/Puffin/BinanceUsdtBusdArbitrage/JupyterNotebooks/datasets/binance-futures_quotes_2021-10-14_XRPUSDT.csv.gz')
quotes_xrpbusd = ti.generate_tob_data_from_gzip('XRPBUSD', 'C:/Users/Tibor/PycharmProjects/Puffin/BinanceUsdtBusdArbitrage/JupyterNotebooks/datasets/binance-futures_quotes_2021-10-14_XRPBUSD.csv.gz')
quotes_busdusdt = ti.generate_tob_data_from_gzip('BUSDUSDT', 'C:/Users/Tibor/PycharmProjects/Puffin/BinanceUsdtBusdArbitrage/JupyterNotebooks/datasets/binance_quotes_2021-10-14_BUSDUSDT.csv.gz')
print(quotes_busdusdt)
