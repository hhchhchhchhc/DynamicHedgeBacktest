#!/usr/bin/env python3
import pandas as pd

from ftx_history import *

from scipy.interpolate import CubicSpline
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.model_selection import TimeSeriesSplit,cross_val_score,cross_val_predict,train_test_split
from sklearn.linear_model import ElasticNet,LinearRegression,LassoCV

class LaplaceTransformer(FunctionTransformer):
    '''horizon_window in hours'''
    def __init__(self,horizon_window: int):
        super().__init__(lambda x: x.ewm(times = x.index,halflife=timedelta(hours=horizon_window)).mean())
        self._horizon_window = horizon_window
    def get_feature_names_out(self):
        return f'ewma{self._horizon_window}'
class ShiftTransformer(FunctionTransformer):
    def __init__(self,horizon_window):
        super().__init__(func=
                         lambda x: x.shift(periods=horizon_window),
                         inverse_func=
                         lambda x: x.shift(periods=-horizon_window))
        self._horizon_window = horizon_window
    def get_feature_names_out(self):
        return f'shift{self._horizon_window}'
class FwdMeanTransformer(TransformedTargetRegressor):
    def __init__(self,horizon_window):
        super().__init__(lambda x: x.shift(periods=-horizon_window-1).rolling(i).mean())
        self._horizon_window = horizon_window
    def get_feature_names_out(self):
        return f'fwdmean{self._horizon_window}'

class ColumnNames:
    @staticmethod
    def funding(coin): return f'{coin}-PERP/rate/funding'
    @staticmethod
    def price(coin): return f'{coin}/price/c'
    @staticmethod
    def borrowOI(coin): return f'{coin}/rate/size'
    @staticmethod
    def volume(coin): return f'{coin}/price/volume'
    @staticmethod
    def borrow(coin): return f'{coin}/rate/borrow'

def ftx_forecaster_main(*args):
    coins = ['ETH','AAVE']
    features = ['funding','price']
    horizon_windows = [1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 60, 72, 84, 168]

    labels = lambda df: df['funding']-df['borrow'] # to apply to a single coin df
    holding_windows = [1,4,8,12,24,36,48]

    models = [LinearRegression]
    n_split = 7
    pca_n = None

    # grab data
    data = ftx_history_main('get', coins, 'ftx', 1000)
    data_list = []
    label_list = []
    for coin in coins:
        for feature in features:
            feature_data = data[getattr(ColumnNames,feature)(coin)]

            if feature == 'price':
                feature_data = feature_data.diff() / feature_data

            laplace_expansion = FeatureUnion([(f'ewma{horizon}',LaplaceTransformer(horizon)) for horizon in horizon_windows])
            dimensionality_reduction = PCA(pca_n) if pca_n else 'passthrough'
            data_list += [Pipeline([('laplace_expansion', laplace_expansion),
                     ('dimensionality_reduction', dimensionality_reduction)]).fit_transform(feature_data)]

        feature_data = pd.concat(data_list,axis=1,how='inner')
        label_list += [data[getattr(ColumnNames,'funding')(coin)]-data[getattr(ColumnNames,'borrow')(coin)]]


    model = LassoCV(cv=TimeSeriesSplit(n_split))
    pipe = Pipeline([('model',model)])

    res = pipe.fit(data).predict()


    # laplace transform
    laplace_transformer = LaplaceTransformer(horizon_windows)
    list_df =[]
    for feature in data.columns:
        laplace_expansion = laplace_transformer.fit_transform(data[feature]).dropna()
        laplace_expansion.columns = pd.MultiIndex.from_tuples([(*feature,c) for c in laplace_expansion.columns],names=data.columns.names+['mode'])
        list_df += [laplace_expansion]

    data = pd.concat(list_df,join='outer',axis=1)

if __name__ == "__main__":
    ftx_forecaster_main(*sys.argv[1:])