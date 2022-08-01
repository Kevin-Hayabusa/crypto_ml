import numpy as np
import pandas as pd
from hurst import compute_Hc, random_walk
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn import  metrics
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from datetime import  date,datetime

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
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_volume_avg(df, w):
        # Current volume/avg
        d = {}
        for i in w:
            ds = (df.Volume / (df.Volume.rolling(i).mean()))
            d[f'FactorCurVolumeDivAvg_{i}'] = ds
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_bk_return(df, w):
        # Period return up to now
        d = {}
        for i in w:
            ds = df.Close.pct_change(i)
            d[f'FactorBackReturn_{i}'] = ds
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_fwd_return(df, w):
        # Period forward return up to w
        d = {}
        for i in w:
            ds = df.Close.pct_change(i).shift(-i)
            d[f'FactorFwdReturn_{i}'] = ds
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_high_low(df, w):
        # rolling high/rolling low
        d = {}
        for i in w:
            ds = df.High.rolling(i).max() / df.Low.rolling(i).min()
            d[f'FactorHighDivLow_{i}'] = ds
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_max_mean(df, w):
        # rolling high/rolling mean
        d = {}
        for i in w:
            ds = df.High.rolling(i).max() / df.Close.rolling(i).mean()
            d[f'FactorMaxDivMean_{i}'] = ds
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_min_mean(df, w):
        # rolling min/rolling mean
        d = {}
        for i in w:
            ds = df.Low.rolling(i).min() / df.Close.rolling(i).mean()
            d[f'FactorMinDivMean_{i}'] = ds
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_max_mean_volume(df, w):
        # rolling high/rolling mean for volume
        d = {}
        for i in w:
            ds = df.Volume.rolling(i).max() / df.Volume.rolling(i).mean()
            d[f'FactorMaxDivMeanVolume_{i}'] = ds
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_min_mean_volume(df, w):
        # rolling min/rolling mean for volume
        d = {}
        for i in w:
            ds = df.Volume.rolling(i).min() / df.Volume.rolling(i).mean()
            d[f'FactorMinDivMeanVolume_{i}'] = ds
        return pd.concat(d, axis=1)

    @staticmethod
    def cal_std(df, w):
        # rolling std
        d = {}
        for i in w:
            ds = df.Close.rolling(i).std()
            d[f'FactorStd_{i}'] = ds
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
        return pd.concat(d, axis=1)

    @staticmethod
    def r_zscore(df, w):
        # z_score Tina
        d = {}
        for i in w:
            ds = (np.log(df.Volume) - np.log(df.Volume).rolling(i).mean()) / np.log(df.Volume).rolling(i).std()
            d[f'FactorZ_Volume_{i}'] = ds

        return pd.concat(d, axis=1)

    @staticmethod
    def Parkinson(df, w):
        # z_score Tina
        d = {}

        for i in w:
            ds = np.log(df.High / df.Low) ** 2
            d[f'FactorParkinson_{i}'] = 0.5 * np.sqrt(ds.rolling(i).sum()) / np.sqrt(np.log(2) * i / (365 * 1440))

        return pd.concat(d, axis=1)

    @staticmethod
    def Hurst(df, w):
        d = {}
        Hurst = lambda x: compute_Hc(x, kind='price')[0]

        w = [180, 360]

        for i in w:
            if i > 100:  # else hurst function fails
                d[f'FactorHurst{i}'] = df.Close.rolling(i).apply(Hurst)

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
class MachineLearning(object):
    def __init__(self,data):
        self.data=data
    def GenerateFeature(self,lag,factor_functions,return_threhold):
        self.data=Features.generate_features(self.data, factor_functions, lag)
        self.data['results'] = np.where(self.data.trade_return > return_threhold, 1, 0)
    def RunModel(self,max_depth,n_estimators,random_state,IndexField,Forecast_Field,Features_Field,start,mid,end):
        self.Train, self.TrainX, self.TrainY, self.Test, self.TestX, self.TestY = MachineLearning.Train_Test_Build(self.data, IndexField,
                                                                                     Features_Field,
                                                                                     Forecast_Field, start,
                                                                                     mid,
                                                                                     end)
        self.Model = RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators)
        self.Model.fit(self.TrainX, self.TrainY)
        print("Model trained")
        print("-------------")
        # print("Estimators :", Estimators)
        print("Features   :", self.Model.n_features_in_)
        # print("OOB        :", Model.oob_score_)

        # Generate some forecasts
        self.ForecastsTrainY = self.Model.predict(self.TrainX)
        self.ForecastsTestY = self.Model.predict(self.TestX)

        Mat = confusion_matrix(self.TestY, self.ForecastsTestY)
        print('')
        print('* Testing set')
        MachineLearning.Metrics(Mat)

        Mat = confusion_matrix(self.TrainY, self.ForecastsTrainY)
        print('')
        print('* Training set')
        MachineLearning.Metrics(Mat)
        metrics.plot_confusion_matrix(self.Model,self.TestX,self.TestY)
        plt.savefig('MeanRevert/Results/ConfusionMatrix.png')
        self.RecallTest = metrics.recall_score(self.TestY,self.ForecastsTestY)
        self.PrecisionTest = metrics.precision_score(self.TestY,self.ForecastsTestY)
        self.F1Test = metrics.f1_score(self.TestY,self.ForecastsTestY)
        self.AccuracyTest = metrics.accuracy_score(self.TestY,self.ForecastsTestY)
        self.AucTest = metrics.roc_auc_score(self.TestY,self.Model.predict_proba(self.TestX)[:,1])
        self.TrainSkew = np.sum(self.TrainY)/len(self.TrainY)
        self.TestSkew = np.sum(self.TestY)/len(self.TestY)
        self.TrainLen = len(self.TrainX)
        self.TestLen = len(self.TestX)
        #print(metrics.classification_report(self.TestY, self.ForecastsTestY))
    def RollingValidation(self,max_depth,n_estimators,random_state,IndexField,Forecast_Field,Features_Field,start_date,end_date,train_period,test_period):
        dates = MachineLearning.GenerateDates(start_date, end_date, train_period, test_period)
        results = []
        for i, d in dates.iterrows():
            start = d['start'].strftime('%Y-%m-%d')
            mid = d['mid'].strftime('%Y-%m-%d')
            end = d['end'].strftime('%Y-%m-%d')
            self.RunModel(max_depth,n_estimators,random_state,IndexField, Forecast_Field, Features_Field, start, mid, end)
            results.append([start, mid, end, ml.PrecisionTest, ml.RecallTest, ml.F1Test, ml.AccuracyTest, ml.TrainSkew,
                            ml.TrainLen, ml.TestLen])
        df = pd.DataFrame(results)
        df.columns = ['start', 'mid', 'end', 'precision', 'recall', 'f1', 'accuracy', 'trainskew', 'trainlen',
                      'testlen']
        return df
    def RollingTesting(self,max_depth,n_estimators,random_state,IndexField,Forecast_Field,Features_Field,start_date,end_date,train_period,test_period):
        dates = MachineLearning.GenerateDates(start_date, end_date, train_period, test_period)

        for i, d in dates.iterrows():
            start = d['start'].strftime('%Y-%m-%d')
            mid = d['mid'].strftime('%Y-%m-%d')
            end = d['end'].strftime('%Y-%m-%d')
            self.RunModel(max_depth, n_estimators, random_state, IndexField, Forecast_Field, Features_Field, start, mid,
                          end)
            y_pred = self.Model.predict_proba(self.TestX)[:,1]
            self.data.loc[self.TestX.index,'predict']=y_pred

    def StressTest(self,n_depth,n_estimators):
        Train = np.empty(0)
        Test = np.empty(0)
        Depth_plot = np.empty(0)
        Tree_plot = np.empty(0)
        for Depth in n_depth:
            Model = RandomForestClassifier(max_depth=Depth, random_state=4021, n_estimators=50)
            Model.fit(self.TrainX, self.TrainY)

            Train_Error = np.sum(Model.predict(self.TrainX) == self.TrainY) / len(self.TrainY)
            Train = np.append(Train, Train_Error)

            Test_Error = np.sum(Model.predict(self.TestX) == self.TestY) / len(self.TestY)
            Test = np.append(Test, Test_Error)

            Depth_plot = np.append(Depth_plot, Depth)

        fig, axs = plt.subplots(2, 1)
        fig.set_size_inches(18.5, 10)

        # Create the  plot
        axs[0].scatter(Depth_plot, Train, s=10)

        axs[0].set_xlabel('Depth')
        axs[0].set_ylabel('Acc')
        axs[0].set_title('Train')
        axs[1].scatter(Depth_plot, Test, color='r', s=10)
        # axs[1].scatter(Depth_plot, Test_2, color = 'r', s= 10)
        axs[1].set_xlabel('Depth')
        axs[1].set_ylabel('Acc')
        axs[1].set_title('Test')
        fig.savefig('MeanRevert/Results/depth.png')


        Train = np.empty(0)
        Test = np.empty(0)
        Depth_plot = np.empty(0)
        Tree_plot = np.empty(0)

        for tree in n_estimators:
            Model = RandomForestClassifier(max_depth=5, random_state=4021, n_estimators=tree)
            Model.fit(self.TrainX, self.TrainY)

            Train_Error = np.sum(Model.predict(self.TrainX) == self.TrainY) / len(self.TrainY)
            Train = np.append(Train, Train_Error)

            Test_Error = np.sum(Model.predict(self.TestX) == self.TestY) / len(self.TestY)
            Test = np.append(Test, Test_Error)

            Tree_plot = np.append(Tree_plot, tree)

        fig, axs = plt.subplots(2, 1)
        fig.set_size_inches(18.5, 10)

        # Create the  plot
        axs[0].scatter(Tree_plot, Train, s=10)

        axs[0].set_xlabel('Tree')
        axs[0].set_ylabel('Acc')
        axs[0].set_title('Train')
        axs[1].scatter(Tree_plot, Test, color='r', s=10)

        axs[1].set_xlabel('Tree')
        axs[1].set_ylabel('Acc')
        axs[1].set_title('Test')
        fig.savefig('MeanRevert/Results/tree.png')
    def FeatureImportance(self):
        result_test = permutation_importance(self.Model, self.TestX, self.TestY, n_repeats=10,
                                             random_state=0,scoring=['f1','precision','recall', 'neg_log_loss'])
        fig = MachineLearning.Plot_Importance(result_test, 'f1', self.TrainX)

        return fig
    @staticmethod
    def Sample_Data(Data, Field=-1, UpperDate=-1, LowerDate=-1):
        # if no field is defined
        if Field == -1:
            # If no upper date is defined
            if UpperDate == -1:
                return Data[LowerDate:].sort_index()

            # If no lower date is not define
            if LowerDate == -1:
                return Data[:UpperDate].sort_index()

            return Data[LowerDate:UpperDate].sort_index()
        else:
            # If no upper date is defined
            if UpperDate == -1:
                return Data[(Data[Field] >= LowerDate)].sort_values(by=[Field])

            # If no lower date is not define
            if LowerDate == -1:
                return Data[(Data[Field] < UpperDate)].sort_values(by=[Field])
            return Data[(Data[Field] < UpperDate) & (Data[Field] >= LowerDate)].sort_values(by=[Field])

    @staticmethod
    def Train_Test_Build(Data,Index_Field ,Features_Field,Forecast_Field, Start, Mid, End):
        print('* Train / Test split with ~')
        print('         - Forecast field:', Forecast_Field)
        print('         - Training      :', Start, "->", Mid)
        print('         - Testing       :', Mid, "->", End)
        print('')

        # Filter by date to create train and test sets
        Train = MachineLearning.Sample_Data(Data, Field=Index_Field,LowerDate=Start, UpperDate=Mid)
        Test = MachineLearning.Sample_Data(Data, Field=Index_Field, LowerDate=Mid, UpperDate=End)

        TrainX = Train[Features_Field]
        TestX = Test[Features_Field]
        TrainY = Train[[Forecast_Field]]
        TestY = Test[[Forecast_Field]]

        # summarise the % split
        print("* Test Length   :", len(TestY[Forecast_Field]))
        print("* Train Length  :", len(TrainY[Forecast_Field]))

        # # Convert to numpy arrays
        #TrainX = TrainX.values
        TrainY = TrainY.values.ravel()

        #TestX = TestX.values
        TestY = TestY.values.ravel()

        print('* X shape       :', np.shape(TrainX))
        print('* Y shape       :', np.shape(TrainY))

        print('Train Skew      : ', np.sum(TrainY) / len(TrainY))
        print('Test Skew       : ', np.sum(TestY) / len(TestY))

        return Train, TrainX, TrainY, Test, TestX, TestY
    @staticmethod
    def Metrics(Mat):

        P = Mat[1, 1] / (Mat[0, 1] + Mat[1, 1])
        R = Mat[1, 1] / (Mat[1, 0] + Mat[1, 1])

        print('  - Accuracy :', round((Mat[0, 0] + Mat[1, 1]) / (Mat[1, 1] + Mat[1, 0] + Mat[0, 1] + Mat[0, 0]), 3))
        print('  - Precision:', round(P, 3))
        print('  - Recall   :', round(R, 3))
        print('  - F1       :', round(2 * P * R / (P + R), 3))
    @staticmethod
    def Plot_Importance(Results_Test, Metric, Data):
        fig, axs = plt.subplots(1, 1, figsize=(15, 10))

        sorted_idx = Results_Test[Metric].importances_mean.argsort()
        axs.boxplot(Results_Test[Metric].importances[sorted_idx].T, vert=False, labels=Data.columns[sorted_idx])
        axs.set_title("Permutation Importances (Test set) " + Metric)
        fig.tight_layout()
        return fig
    @staticmethod
    def Compute_ROC(Model,DataX, DataY, step):
        # Compute the model probabilities for the data points.
        Prob = Model.predict_proba(DataX)
        DataY = DataY.astype(bool)

        # We need only keep one of the entries (since they sum to 1)
        # Keep the first one
        Prob = Prob[:, 1]

        TPR = np.empty(0)
        FPR = np.empty(0)
        Prec = np.empty(0)
        F1 = np.empty(0)
        XEnt = np.empty(0)

        # Then forecast over a range of thresholds and compare to actuals.
        for T in np.arange(0, 1, step):
            Prob_T = Prob > T

            # The compute metrics
            R = np.sum(Prob_T & DataY) / np.sum(DataY)
            TPR = np.append(TPR, R)

            FPR = np.append(FPR, np.sum(Prob_T & ~DataY) / (len(DataY) - np.sum(DataY)))
            P = np.sum(Prob_T & DataY) / (np.sum(Prob_T & ~DataY) + np.sum(Prob_T & DataY))
            Prec = np.append(Prec, P)
            F1 = np.append(F1, 2 * P * R / (R + P))
            # XEnt = np.append(XEnt, -np.sum(Prob_T * np.log(DataY + 0.00000000001)))

        return pd.DataFrame({'T': np.arange(0, 1, step), 'TPR': TPR, 'FPR': FPR, 'Prec': Prec, 'F1': F1})
    @staticmethod
    def plot_ROC(Curve_Test,Curve_Train):
        fig, axs = plt.subplots(2, 4)
        fig.set_size_inches(18.5, 10)

        min_range, max_range = 0.01, 50
        x = np.linspace(min_range, max_range, num=100)

        Best_T = Curve_Test['T'][np.argmax(Curve_Test['F1'])]
        F1 = Curve_Test['F1'][np.argmax(Curve_Test['F1'])]
        Precision = Curve_Test['Prec'][np.argmax(Curve_Test['F1'])]
        Recall = Curve_Test['TPR'][np.argmax(Curve_Test['F1'])]

        Best_T_Train = Curve_Train['T'][np.argmax(Curve_Train['F1'])]
        F1_Train = Curve_Train['F1'][np.argmax(Curve_Train['F1'])]
        Precision_Train = Curve_Train['Prec'][np.argmax(Curve_Train['F1'])]
        Recall_Train = Curve_Train['TPR'][np.argmax(Curve_Train['F1'])]

        # Create the  plot
        axs[0, 0].scatter(Curve_Test['FPR'], Curve_Test['TPR'], s=10)
        axs[0, 0].axline((0, 0), slope=1, color='r', lw=0.5)
        axs[0, 0].set_xlabel('FPR')
        axs[0, 0].set_ylabel('TPR')
        axs[0, 0].set_title('Test: ROC')
        axs[0, 0].set_xlim([0, 1])
        axs[0, 0].set_ylim([0, 1])

        axs[0, 1].scatter(Curve_Test['T'], Curve_Test['F1'], s=1.5)
        axs[0, 1].set_xlabel('Threshold')
        axs[0, 1].set_ylabel('F1')
        axs[0, 1].set_title('Test: F1 (' + str(round(F1, 3)) + ') T = ' + str(round(Best_T, 3)))
        axs[0, 1].axvline(x=Best_T, lw=0.5, alpha=0.5, color='b')
        axs[0, 1].set_xlim([0, 1])
        axs[0, 1].set_ylim([0, 1])

        axs[0, 2].scatter(Curve_Test['T'], Curve_Test['Prec'], s=1.5)
        axs[0, 2].axvline(x=Best_T, lw=0.5, alpha=0.5, color='b')
        axs[0, 2].set_xlabel('Threshold')
        axs[0, 2].set_ylabel('Precision')
        axs[0, 2].set_title('Test: Precision (' + str(round(Precision, 3)) + ')')
        axs[0, 2].set_xlim([0, 1])
        axs[0, 2].set_ylim([0, 1])

        axs[0, 3].scatter(Curve_Test['T'], Curve_Test['TPR'], s=1.5)
        axs[0, 3].axvline(x=Best_T, lw=0.5, alpha=0.5, color='b')
        axs[0, 3].set_xlabel('Threshold')
        axs[0, 3].set_ylabel('Recall')
        axs[0, 3].set_title('Test: Recall (' + str(round(Recall, 3)) + ')')
        axs[0, 3].set_xlim([0, 1])
        axs[0, 3].set_ylim([0, 1])

        axs[1, 0].scatter(Curve_Train['FPR'], Curve_Train['TPR'], s=10)
        axs[1, 0].axline((0, 0), slope=1, color='r', lw=0.5)
        axs[1, 0].set_xlabel('FPR')
        axs[1, 0].set_ylabel('TPR')
        axs[1, 0].set_title('Train: ROC')
        axs[1, 0].set_xlim([0, 1])
        axs[1, 0].set_ylim([0, 1])

        axs[1, 1].scatter(Curve_Train['T'], Curve_Train['F1'], s=1.5)
        axs[1, 1].axvline(x=Best_T_Train, lw=0.5, alpha=0.5, color='b')
        axs[1, 1].set_xlabel('Threshold')
        axs[1, 1].set_ylabel('F1')
        axs[1, 1].set_title('Train: F1 (' + str(round(F1_Train, 3)) + ') T = ' + str(round(Best_T_Train, 3)))
        axs[1, 1].set_xlim([0, 1])
        axs[1, 1].set_ylim([0, 1])

        axs[1, 2].scatter(Curve_Train['T'], Curve_Train['Prec'], s=1.5)
        axs[1, 2].axvline(x=Best_T_Train, lw=0.5, alpha=0.5, color='b')
        axs[1, 2].set_xlabel('Threshold')
        axs[1, 2].set_ylabel('Precision')
        axs[1, 2].set_title('Train: Precision (' + str(round(Precision_Train, 3)) + ')')
        axs[1, 2].set_xlim([0, 1])
        axs[1, 2].set_ylim([0, 1])

        axs[1, 3].scatter(Curve_Train['T'], Curve_Train['TPR'], s=1.5)
        axs[1, 3].axvline(x=Best_T_Train, lw=0.5, alpha=0.5, color='b')
        axs[1, 3].set_xlabel('Threshold')
        axs[1, 3].set_ylabel('Recall')
        axs[1, 3].set_title('Train: Recall (' + str(round(Recall_Train, 3)) + ')')
        axs[1, 3].set_xlim([0, 1])
        axs[1, 3].set_ylim([0, 1])
        return fig
    @staticmethod
    def GenerateDates(start,end,train_period,test_period):
        dates = pd.DataFrame()
        d = start
        interval=train_period+test_period
        while d + relativedelta(months=interval) <= end:
            # print(d,end)
            mid=d+relativedelta(months=train_period)
            ed = d + relativedelta(months=interval)
            dates = dates.append([[d, mid,ed]])
            d = d + relativedelta(months=test_period)
        dates.columns = ['start','mid', 'end']
        return dates
if __name__ == '__main__':
    start = '2021-05-01'
    mid ='2022-04-01'
    end = '2022-05-01'

    data = pd.read_feather('./MeanRevert/Results/backtesting_vt.feather')
    lag = [30, 60, 180]
    return_threhold = 0.0008
    max_depth = 6
    n_estimators = 60
    random_state = 101

    factor_functions = [Features.cal_close_avg, Features.cal_bk_return, Features.cal_high_low, Features.cal_max_mean,
                        Features.cal_min_mean,
                        Features.cal_volume_avg, Features.cal_max_mean_volume, Features.cal_min_mean_volume,
                        Features.cal_std, Features.r_zscore, Features.Parkinson]

    ml = MachineLearning(data)
    ml.GenerateFeature(lag,factor_functions,return_threhold)

    Forecast_Field = 'results'
    Features_Field = [idx for idx in ml.data.columns if idx[:6] == 'Factor']

    ml.RunModel(max_depth,n_estimators,random_state,'Open Time',Forecast_Field,Features_Field,start,mid,end)
    fig = ml.FeatureImportance()
    fig.savefig('MeanRevert/Results/feature_importance.png')
    Curve_Test = MachineLearning.Compute_ROC(ml.Model,ml.TestX,ml.TestY,0.001)
    Curve_Train = MachineLearning.Compute_ROC(ml.Model, ml.TrainX, ml.TrainY, 0.001)
    fig = MachineLearning.plot_ROC(Curve_Test,Curve_Train)
    fig.savefig('MeanRevert/Results/ROC.png')
    df = ml.RollingValidation(max_depth,n_estimators,random_state,'Open Time',Forecast_Field,Features_Field,datetime.strptime(start,'%Y-%m-%d').date(),datetime.strptime(end,'%Y-%m-%d').date(),2,1)
    plot = df[['precision','recall','f1','accuracy','trainskew']].plot()
    plot.get_figure().savefig('MeanRevert/Results/rolling_validation.png')
    ml.StressTest(range(2,30,1),range(30,100,5))
    # ml.RollingTesting(max_depth,n_estimators,random_state,'Open Time',Forecast_Field,Features_Field,datetime.strptime(start,'%Y-%m-%d').date(),datetime.strptime(end,'%Y-%m-%d').date(),2,1)
    # d = pd.merge(data,ml.data[['Open Time','predict']],left_on='Open Time',right_on='Open Time',how='left')








