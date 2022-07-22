import numpy as np

from Algos.BacktestBase import *

class MeanRevertEvent(BacktestBase):
    def run_simple(self,lookback,threshold):
        msg = f'\n\nRunning MR simple strategy | lookback {lookback} days;threshold: {threshold}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 100)

        self.data['Mean'] = self.data.Close.rolling(lookback).mean()
        self.data['Std'] = self.data.Close.rolling(lookback).std(ddof=0)
        self.data['Zscore'] = (self.data.Close - self.data.Mean) / self.data.Std
        self.data['Return'] = self.data.Close.pct_change()
        self.data['price']=self.data.Close
        self.data['Net_equity']=np.NaN
        self.data['positions'] = np.NaN

        for bar in range(lookback,len(self.data)):
            if self.position == 0: #no position
                if self.data['Zscore'].iloc[bar]> threshold:
                    self.go_short(bar,amount='all')
                    self.position=-1
                if self.data['Zscore'].iloc[bar]< -threshold:
                    self.go_long(bar,amount='all')
                    self.position= 1
            elif self.position in [1,-1]:
                if self.data['Zscore'].iloc[bar]*self.data['Zscore'].iloc[bar-1] <0: #zscore change sign
                    if self.position == 1:
                        self.place_sell_order(bar,units=self.units)
                    if self.position ==-1:
                        self.place_buy_order(bar,units=-self.units)
                    self.position=0

            self.data.loc[self.data.index[bar],'Net_equity']=self.units * self.data.price[bar] + self.amount
            self.data.loc[self.data.index[bar], 'positions'] = self.units

        self.close_out(bar)
        self.calculate_perf()
        self.trade_summary()
    def run_scale(self,lookback,thresholds):
        msg = f'\n\nRunning MR scale strategy | lookback {lookback} days;threshold: {thresholds}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 100)

        self.data['Mean'] = self.data.Close.rolling(lookback).mean()
        self.data['Std'] = self.data.Close.rolling(lookback).std(ddof=0)
        self.data['Zscore'] = (self.data.Close - self.data.Mean) / self.data.Std
        self.data['Return'] = self.data.Close.pct_change()
        self.data['price'] = self.data.Close
        self.data['Net_equity'] = np.NaN
        self.data['positions'] = np.NaN
        levels=len(thresholds)
        for bar in range(lookback,len(self.data)):

            if self.position == 0:
                self.size=int(self.amount/levels/self.data.price.iloc[bar])
                #no position, looking to enter long/short
                if self.data['Zscore'].iloc[bar]> thresholds[0]:
                    #enter into short with 1/n of total amounts
                    self.go_short(bar,units=self.size)
                    self.position=-1
                    print('short 1 unit')
                # elif self.data['Zscore'].iloc[bar]< -thresholds[0]:
                #     #enter into long with 1/n of total amount
                #     self.go_long(bar,units=self.size)
                #     self.position=1
                #     print('long 1 unit')
            if self.position == -1:
                #already shorted 1 unit, looking to increase or exit
                if self.data['Zscore'].iloc[bar]> thresholds[1]:
                    #increase short by remaining amount/levels
                    self.go_short(bar,units=self.size)
                    self.position=-2
                    print('short 1 unit, total 2')
                elif self.data['Zscore'].iloc[bar]<0:
                    #buy cover
                    print('current positions',self.units)
                    self.go_long(bar,units=-self.units)
                    self.position=0
                    print('buy all unit and close, total 1')
            if self.position ==-2:
                #already shorted 2 unit, looking to increae or exit
                if self.data['Zscore'].iloc[bar]> thresholds[2]:
                    self.go_short(bar,units=self.size)
                    self.position=-3
                    print('short 1 unit, total 3')
                elif 0 <self.data['Zscore'].iloc[bar]<thresholds[0]:
                    #partial exit
                    self.go_long(bar,units=self.size)
                    self.position=-1
                    print('buy 1 unit,total 1')
                elif self.data['Zscore'].iloc[bar] <0:
                    #buy cover
                    self.go_long(bar,units=-self.units)
                    self.position=0
                    print('buy all units and close,total 2')
            if self.position == -3:
            # already shorted 3 unit, looking to exit
                if thresholds[0] <self.data['Zscore'].iloc[bar]<thresholds[1]:
                    #partial exit
                    self.go_long(bar,units=self.size)
                    self.position=-2
                    print('buy 1 unit,total 2')
                elif 0 <self.data['Zscore'].iloc[bar]<thresholds[0]:
                    # partial exit
                    self.go_long(bar, units=self.size*2)
                    self.position = -1
                    print('buy 2 unit,total 1')
                elif self.data['Zscore'].iloc[bar] <0:
                    #buy cover
                    self.go_long(bar,units=-self.units)
                    self.position=0
                    print('buy all units and close,total 3')

            if self.position == 1:
                #already long 1 unit, looking to increase or exit
                if self.data['Zscore'].iloc[bar]< -thresholds[1]:
                    #increase short by remaining amount/levels
                    self.go_long(bar,units=self.size)
                    self.position= 2
                    print('long 1 unit, total 2')
                elif self.data['Zscore'].iloc[bar]>0:
                    #sell cover
                    print('current positions',self.units)
                    self.go_short(bar,units=self.units)
                    self.position=0
                    print('sell all unit and close, total 1')
            if self.position ==2:
                #already long 2 unit, looking to increae or exit
                if self.data['Zscore'].iloc[bar]< -thresholds[2]:
                    self.go_long(bar,units=self.size)
                    self.position=3
                    print('long 1 unit, total 3')
                elif 0 >self.data['Zscore'].iloc[bar]>-thresholds[0]:
                    #partial exit
                    self.go_short(bar,units=self.size)
                    self.position= 1
                    print('sell 1 unit,total 1')
                elif self.data['Zscore'].iloc[bar] >0:
                    #sell cover
                    self.go_short(bar,units=self.units)
                    self.position=0
                    print('sell all units and close,total 2')
            if self.position == 3:
            # already long 3 unit, looking to exit
                if thresholds[0] >self.data['Zscore'].iloc[bar]>-thresholds[1]:
                    #partial exit
                    self.go_short(bar,units=self.size)
                    self.position=2
                    print('sell 1 unit,total 2')
                elif 0 >self.data['Zscore'].iloc[bar]>-thresholds[0]:
                    # partial exit
                    self.go_short(bar, units=self.size*2)
                    self.position = 1
                    print('sell 2 unit,total 1')
                elif self.data['Zscore'].iloc[bar] >0:
                    #sell cover
                    self.go_short(bar,units=self.units)
                    self.position=0
                    print('sell all units and close,total 3')
            self.data.loc[self.data.index[bar],'Net_equity']=self.units * self.data.price[bar] + self.amount
            self.data.loc[self.data.index[bar], 'positions'] = self.units
        self.close_out(bar)
        self.calculate_perf()

    def calculate_perf(self):
        def calc_drawdown(cum_rets):
            # Calculate the running maximum
            running_max = np.maximum.accumulate(cum_rets.dropna())

            # Ensure the value never drops below 1
            running_max[running_max < 1] = 1

            # Calculate the percentage drawdown
            drawdown = (cum_rets) / running_max - 1
            return drawdown
        self.data['strategy_return']=self.data.Net_equity.pct_change()
        self.data['cum_strategy']=(1+self.data['strategy_return']).cumprod()
        self.data['cum_returns']=(1+self.data['Return']).cumprod()
        daily_return = self.data['strategy_return'].resample('D').agg(lambda x: (x + 1).prod() - 1)
        cum_return = (1 + daily_return).cumprod()
        self.sharpe = np.mean(daily_return) / np.std(daily_return) * (252 ** 0.5)
        drawdown = calc_drawdown(cum_return)
        self.max_dd = drawdown.min()
        print(f'total:{cum_return.iloc[-1]-1}|sharpe:{self.sharpe}|max_dd:{self.max_dd}')
    def trade_summary(self):
        self.data.loc[self.data.positions!=0,'sign']=1
        self.data['trades']=self.data.sign.shift(1)*self.data.strategy_return
        events = np.split(self.data.trades, np.where(np.isnan(self.data.trades.values))[0])
        events = [ev[~np.isnan(ev.values)] for ev in events if not isinstance(ev, np.ndarray)]
        # removing empty DataFrames
        events = [ev for ev in events if not ev.empty]
        d={}
        for e in events:
            if len(e)==1: #special case of 1 trade
                d[e.index[0]]=e[0]
            else:
                cum = (1+e).cumprod() #cum returns
                d[e.index[0]]=cum[-1]/cum[0]-1
        trades = pd.DataFrame.from_dict(d,orient='index')
        trades.columns=['pnl']
        self.data.loc[trades.index,'pnl']=trades.values

    def plot_result(self):
        self.results=self.data.dropna()
        if self.results is None:
            print('No results to plot yet. Run a strategy')
        title='%s | FTC = %.4f,PTC = %.4f'% (self.symbol, self.ftc,self.ptc)
        self.results[['cum_returns', 'cum_strategy']].plot(title=title,
                                                     figsize=(10, 6))

if __name__ == '__main__':
    path = '../../data/'
    type = 'feather'
    symbol = 'BTCUSDT_1HOUR'
    amount = 1000000
    start = '2022-03-01'
    end = '2022-07-01'
    lookback = 20
    threshold = 2.5

    bb = MeanRevertEvent(symbol, start, end, amount,verbose=True)
    bb.get_data(path,type)
    bb.run_simple(lookback,threshold)
    #bb.run_scale(lookback, [2,2.5,3])
    bb.plot_result()
    plt.show()