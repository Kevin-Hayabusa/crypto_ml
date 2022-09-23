import os

import numpy as np
import pandas as pd
from Features import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn import  metrics
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from datetime import  date,datetime
from sklearn.model_selection import train_test_split

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
def Train_Test_Build(Data, Index_Field, Features_Field, Forecast_Field, Start, Mid, End):
    print('* Train / Test split with ~')
    print('         - Forecast field:', Forecast_Field)
    print('         - Training      :', Start, "->", Mid)
    print('         - Testing       :', Mid, "->", End)
    print('')

    # Filter by date to create train and test sets
    Train = Sample_Data(Data, Field=Index_Field, LowerDate=Start, UpperDate=Mid)
    Test = Sample_Data(Data, Field=Index_Field, LowerDate=Mid, UpperDate=End)

    TrainX = Train[Features_Field]
    TestX = Test[Features_Field]
    TrainY = Train[[Forecast_Field]]
    TestY = Test[[Forecast_Field]]

    # summarise the % split
    print("* Test Length   :", len(TestY[Forecast_Field]))
    print("* Train Length  :", len(TrainY[Forecast_Field]))

    # # Convert to numpy arrays
    # TrainX = TrainX.values
    #TrainY = TrainY.values.ravel()

    # TestX = TestX.values
    #TestY = TestY.values.ravel()

    print('* X shape       :', np.shape(TrainX))
    print('* Y shape       :', np.shape(TrainY))

    print('Train Skew      : ', np.sum(TrainY) / len(TrainY))
    print('Test Skew       : ', np.sum(TestY) / len(TestY))

    return Train, TrainX, TrainY, Test, TestX, TestY

if __name__ == '__main__':
    start = '2020-01-01'
    mid ='2022-04-01'
    end = '2022-05-01'
    data_path = './MeanRevert/Results/'
    data = pd.read_feather(os.path.join(data_path,'backtesting_vt.feather'))
    lag = [30, 60, 180]
    return_threhold = 0.0008
    max_depth = 6
    n_estimators = 60
    random_state = 101

    #register all the factor calculation functions
    factor_functions = [Features.cal_close_avg, Features.cal_bk_return, Features.cal_high_low, Features.cal_max_mean,
                        Features.cal_min_mean,
                        Features.cal_volume_avg, Features.cal_max_mean_volume, Features.cal_min_mean_volume,
                        Features.cal_std, Features.r_zscore, Features.Parkinson,Features.Hurst]

    factor_data = Features.generate_features(data,factor_functions,lag)
    factor_data['results'] = np.where(factor_data.trade_return > return_threhold, 1, 0)
    factor_data.reset_index().to_feather(os.path.join(data_path,'factor_data.feather'))
    IndexField = 'Open Time'
    Forecast_Field = 'results'
    Features_Field = [idx for idx in factor_data.columns if idx[:6] == 'Factor']

    Train, TrainX, TrainY, Test, TestX, TestY = Train_Test_Build(factor_data,IndexField,Features_Field,Forecast_Field,start,mid,end)
    Model = RandomForestClassifier(max_depth=max_depth,random_state=random_state,n_estimators=n_estimators,class_weight='balanced')
    Model.fit(TrainX,TrainY)
    print("Model trained using " + '\n'.join(Features_Field))
    print("Features   :", Model.n_features_in_)
    # Generate some forecasts
    ForecastsTrainY = Model.predict(TrainX)
    ForecastsTestY = Model.predict(TestX)

    Mat = confusion_matrix(TestY, ForecastsTestY)
    metrics.plot_confusion_matrix(Model,TestX,TestY)
    RecallTest = metrics.recall_score(TestY, ForecastsTestY)
    PrecisionTest = metrics.precision_score(TestY, ForecastsTestY)
    F1Test = metrics.f1_score(TestY, ForecastsTestY)
    AccuracyTest = metrics.accuracy_score(TestY, ForecastsTestY)
    AucTest = metrics.roc_auc_score(TestY, Model.predict_proba(TestX)[:, 1])
    TrainSkew = np.sum(TrainY) / len(TrainY)
    TestSkew = np.sum(TestY) / len(TestY)
    TrainLen = len(TrainX)
    TestLen = len(TestX)
    print('Model Performance:')
    print(f'PrecisionTest:{PrecisionTest}\nRecallTest:{RecallTest}\nF1Test:{F1Test}\nAccuracyTest:{AccuracyTest}\nTrainSkew:{TrainSkew}\nTestSkew:{TestSkew}')

    # Use the model output as the indicator to compute maxdd etc using model output
    Indicator = pd.DataFrame(data=ForecastsTestY,index=TestY.index,columns=['PredictY'])
    #factor_data_test = factor_data.loc[Indicator.index].copy()
    data_ml = pd.merge(data,Indicator,how='outer',left_index=True, right_index=True)
    data_ml = data_ml.iloc[Indicator.index[0] - 1:, ]
    data_ml['PredictY_mask']=data_ml.PredictY.shift(-1)
    data_ml['PredictY_mask'] = data_ml['PredictY_mask'].ffill()
    data_ml['position_ml']=data_ml.position * data_ml.PredictY_mask

    # calculate strategy daily return
    data_ml['strategy_ml'] = data_ml['position_ml'].shift(1) * data_ml['return']
    # substract transaction cost
    trades = (data_ml['position_ml'].diff().fillna(0) != 0)
    #data_ml.loc[trades, 'strategy_ml'] -= tc
    # calculate strategy returns
    data_ml['cum_strategy_ml'] = (1 + data_ml['strategy_ml']).cumprod()
    total_return = data_ml['cum_strategy_ml'].iloc[-1] - 1




