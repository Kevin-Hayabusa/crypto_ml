import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SingleVectorBT():
    '''
    Class for basic meanrevert strategy in vectorized backtesting

     Attributes
        ==========
        symbol: str
           financial instrument to work with
        start: str
            start date for data selection
        end: str
            end date for data selection
        tc: float
            proportional transaction costs (e.g. 0.5% = 0.005) per trade
        Methods
        =======
        get_data:
            retrieves and prepares the base data set
        run_strategy:
            runs the backtest for the momentum-based strategy
        plot_results:
            plots the performance of the strategy compared to the symbol
    '''

    def __init__(self,symbol,start,end,tc):
        self.symbol=symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
    def get_data(self,path,type):
        '''
        :param path: file path for import
        :param type: file type CSV or Feather
        :return:dataframe with cleaned data
        '''
        if type=='feather':
            df = pd.read_feather(os.path.join(path, f'{self.symbol}.{type}'))
            df = df.set_index('Open Time')
        elif type =='csv':
            df = pd.read_csv(os.path.join(path, f'{self.symbol}.{type}'))
            df = df.set_index('Open Time')
        else:
            print('type not supported')
        self.data = df.loc[self.start:self.end]
    def run_startegy(self):
        pass
    def plot_results(self):
        if self.results is None:
            print('No results to plot yet. Run a strategy')
        title='%s | TC = %.4f' % (self.symbol, self.tc)
        self.results[['cum_returns', 'cum_strategy']].plot(title=title,
                                                     figsize=(10, 6))
        plt.show()
    def calculate_perf(self):
        def calc_drawdown(cum_rets):
            # Calculate the running maximum
            running_max = np.maximum.accumulate(cum_rets.dropna())

            # Ensure the value never drops below 1
            running_max[running_max < 1] = 1

            # Calculate the percentage drawdown
            drawdown = (cum_rets) / running_max - 1
            return drawdown
        daily_return =self.results['strategy'].resample('D').agg(lambda x: (x + 1).prod() - 1)
        cum_return = (1+daily_return).cumprod()
        self.sharpe = np.mean(daily_return) / np.std(daily_return) * (252 ** 0.5)
        drawdown = calc_drawdown(cum_return)
        self.max_dd = drawdown.min()
class SingleVectorBT_MR(SingleVectorBT):
    def run_strategy(self,lookback,threshold):
        '''
        :param lookback:lookback period for the strategy
        :param threshold:threshold for entry and exit
        :return: strategy return
        '''
        self.lookback=lookback
        self.threshold=threshold

        data=self.data.copy().dropna()
        data['Mean'] = data.Close.rolling(self.lookback).mean()
        data['Std'] = data.Close.rolling(self.lookback).std(ddof=0)
        data['Zscore'] = (data.Close - data.Mean) / data.Std
        data = data.dropna()
        data['return'] = data.Close.pct_change()

        # sell signals
        data['position'] = np.where(data['Zscore'] > self.threshold, -1, np.nan)
        # buy signals
        data['position'] = np.where(data['Zscore'] < -self.threshold, 1, data.position)

        # exit when revert to mean
        data['position'] = np.where(data['Zscore'] * data['Zscore'].shift(1) < 0, 0, data.position)
        data['position'] = data.position.ffill().fillna(0)
        # calculate strategy daily return
        data['strategy'] = data['position'].shift(1) * data['return']

        # substract transaction cost
        trades = (data['position'].diff().fillna(0) != 0)
        data.loc[trades, 'strategy'] -= tc

        # calculate strategy returns
        data['cum_returns'] = (1 + data['return']).cumprod()
        data['cum_strategy'] = (1 + data['strategy']).cumprod()
        self.raw_return = data['cum_returns'].iloc[-1] - 1
        self.total_return = data['cum_strategy'].iloc[-1] - 1
        self.results=data
        self.calculate_perf()
        print(f'raw return:{self.raw_return},total return:{self.total_return},sharpe:{self.sharpe},max_dd:{self.max_dd}')

if __name__=='__main__':
    path = '../../data/'
    type = 'feather'
    symbol = 'BTCUSDT_1HOUR'
    lookback = 10
    threshold = 2.5
    tc = 0
    start='2022-02-01'
    end = '2022-07-01'

    mrbt = SingleVectorBT_MR(symbol,start,end,tc)
    mrbt.get_data(path,type)
    #mrbt.run_strategy(lookback)
    mrbt.run_strategy(lookback,threshold)
    mrbt.plot_results()

