import numpy as np
import pandas as pd
from hurst import compute_Hc, random_walk


class Features(object):
    '''
    class to take in OHLC data and generate features
    '''

    @staticmethod
    def cal_close_avg(df, w):
        # Current close/avg
        d = {}
        for i in w:
            ds = (df.Close / (df.Close.rolling(i).mean()))
            d[f'FactorCurCloseDivAvg_{i}'] = ds
        print('*. Current close/avg')
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_volume_avg(df, w):
        # Current volume/avg
        d = {}
        for i in w:
            ds = (df.Volume / (df.Volume.rolling(i).mean()))
            d[f'FactorCurVolumeDivAvg_{i}'] = ds
        print('*. Current volume/avg')
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_bk_return(df, w):
        # Period return up to now
        d = {}
        for i in w:
            ds = df.Close.pct_change(i)
            d[f'FactorBackReturn_{i}'] = ds
        print('*. Period return up to now')
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_fwd_return(df, w):
        # Period forward return up to w
        d = {}
        for i in w:
            ds = df.Close.pct_change(i).shift(-i)
            d[f'FactorFwdReturn_{i}'] = ds
        print('*. Period forward return up to w')
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_high_low(df, w):
        # rolling high/rolling low
        d = {}
        for i in w:
            ds = df.High.rolling(i).max() / df.Low.rolling(i).min()
            d[f'FactorHighDivLow_{i}'] = ds
        print('*. Rolling high/rolling low')
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_max_mean(df, w):
        # rolling high/rolling mean
        d = {}
        for i in w:
            ds = df.High.rolling(i).max() / df.Close.rolling(i).mean()
            d[f'FactorMaxDivMean_{i}'] = ds
        print('*. Rolling high/rolling mean')
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_min_mean(df, w):
        # rolling min/rolling mean
        d = {}
        for i in w:
            ds = df.Low.rolling(i).min() / df.Close.rolling(i).mean()
            d[f'FactorMinDivMean_{i}'] = ds
        print('*. Rolling min/rolling mean')
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_max_mean_volume(df, w):
        # rolling high/rolling mean for volume
        d = {}
        for i in w:
            ds = df.Volume.rolling(i).max() / df.Volume.rolling(i).mean()
            d[f'FactorMaxDivMeanVolume_{i}'] = ds
        print('*. Rolling high/rolling mean for volume')
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_min_mean_volume(df, w):
        # rolling min/rolling mean for volume
        d = {}
        for i in w:
            ds = df.Volume.rolling(i).min() / df.Volume.rolling(i).mean()
            d[f'FactorMinDivMeanVolume_{i}'] = ds
        print('*. Rolling min/rolling mean for volume')
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_std(df, w):
        # rolling std
        d = {}
        for i in w:
            ds = df.Close.rolling(i).std()
            d[f'FactorStd_{i}'] = ds
        print('*. Rolling std ')
        return pd.concat(d, axis=1)

    @staticmethod
    def Closing(df, w):
        # rolling std
        d = {}
        for i in w:
            ds = df.Close.shift(i)
            d[f'FactorClose_{i}'] = ds

        # Include current by default
        d[f'FactorClose_{0}'] = df.Close
        print('*. Closing only')
        return pd.concat(d, axis=1)

    @staticmethod
    def Volume(df, w):
        # rolling std
        d = {}
        for i in w:
            ds = df.Volume.shift(i)
            d[f'FactorVolume_{i}'] = ds

        # Include current by default
        d[f'FactorVolume_{0}'] = df.Volume
        print('*. Volume only')
        return pd.concat(d, axis=1)

    @staticmethod
    def r_zscore(df, w):
        # z_score Tina
        d = {}
        for i in w:
            ds = (np.log(df.Volume) - np.log(df.Volume).rolling(i).mean()) / np.log(df.Volume).rolling(i).std()
            d[f'FactorZ_Volume_{i}'] = ds
        print('*. Z-score')
        return pd.concat(d, axis=1)

    @staticmethod
    def Parkinson(df, w):
        # z_score Tina
        d = {}

        for i in w:
            ds = np.log(df.High / df.Low) ** 2
            d[f'FactorParkinson_{i}'] = 0.5 * np.sqrt(ds.rolling(i).sum()) / np.sqrt(np.log(2) * i / (365 * 1440))
        print('*. Parkinson Vol')
        return pd.concat(d, axis=1)

    @staticmethod
    def Hurst(df, w):
        d = {}
        get_hurst = lambda x: compute_Hc(x, kind='price')[0]

        #w = [180, 360]

        for i in w:
            if i > 100:  # need at least 100 data points else hurst function fails
                d[f'FactorHurst{i}'] = df.Close.rolling(i).apply(get_hurst)
        print('*. Hurst')
        return pd.concat(d, axis=1)

    @staticmethod
    def generate_factors(df_panel, factor_functions, w):
        '''
        generate factors on panel data with rolling window w
        '''
        l = []
        for f in factor_functions:
            #factor = df_panel.groupby('Ticker').apply(f, w)
            factor = f(df_panel,w)
            l.append(factor)
        return pd.concat(l, axis=1)

    @staticmethod
    def generate_features(data,factor_functions,lag):
        data_features = Features.generate_factors(data, factor_functions, lag)
        combine = pd.concat([data, data_features], axis=1)
        data_ml = combine[~combine['trade_return'].isna()].dropna()
        return data_ml.copy()
