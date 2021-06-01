import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math

def to_seconds(x):
    h, m, s = x.split(':')
    result = int(h) * 60 * 60 + int(m) * 60 + int(s)
    return result

def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def count_MSE(y_true, pred):
    res = 0
    for i in range(len(y_true)):
        res += ((y_true[i] - pred[i])**2)
    return res

def count_MAE(y_true, pred):
    res = 0
    for i in range(len(y_true)):
        res += (abs(y_true[i] - pred[i]))
    return res

def create_frame(ddf):
    ts_lat = exponential_smoothing((ddf.Latitude-ddf.Latitude.mean())/ddf.Latitude.std(), 0.4)
    ts_lon = exponential_smoothing((ddf.Longitude-ddf.Longitude.mean())/ddf.Longitude.std(), 0.4)
    ts_hei = exponential_smoothing((ddf.Height-ddf.Height.mean())/ddf.Height.std(), 0.3)
    ts_time = exponential_smoothing((ddf.Time-ddf.Time.mean())/ddf.Time.std(), 0.3)
    ts_rad = exponential_smoothing((ddf.radius-ddf.radius.mean())/ddf.radius.std(), 0.3)
    ts_alpha = exponential_smoothing(ddf.alpha, 0.3)
        
    # тут есть случаи, когда высота одинаковая, поэтому идёт деление на 0
    try:
        mse_hei = count_MSE((ddf.Height-ddf.Height.mean())/ddf.Height.std(), ts_hei)
    except:
        mse_hei = 0
    mse_time = count_MSE((ddf.Time-ddf.Time.mean())/ddf.Time.std(), ts_time)
    mse_rad = count_MSE((ddf.radius-ddf.radius.mean())/ddf.radius.std(), ts_rad)
    
    
    # тут есть случаи, когда высота одинаковая, поэтому идёт деление на 0
    try:
        mae_hei = count_MAE((ddf.Height-ddf.Height.mean())/ddf.Height.std(), ts_hei)
    except:
        mae_hei = 0
    mae_time = count_MAE((ddf.Time-ddf.Time.mean())/ddf.Time.std(), ts_time)
    mae_rad = count_MAE((ddf.radius-ddf.radius.mean())/ddf.radius.std(), ts_rad)
    mae_alpha = count_MAE(ddf.alpha, ts_alpha)
        
    return [mse_hei, mse_time, mse_rad, 
            mae_hei, mae_time, mae_rad, mae_alpha, len(ddf),
            ddf.radius.mean(), ddf.radius.std(), 
            np.mean(ts_rad), np.std(ts_rad),
            ]


def PREDICT(filename):
    model_filename = 'forest_model.pkl'
    forest_model = pickle.load(open(model_filename, 'rb'))
    
    columns = ['Time', 'ID', 'Latitude', 'Longitude', 'Height', 'Code', 'Name']
    bad_23 = pd.read_csv(filename, sep=" ", header=None)
    bad_23.columns = columns

    bad_names = bad_23.ID.value_counts().index
    bad_choice_ind = dict(zip(range(len(bad_names)), bad_names))

    trace = []
    for i in bad_choice_ind.keys(): #цикл по всем ключам (позывным)

        ddf = bad_23[bad_23.ID == bad_choice_ind[i]].copy().reset_index(drop=True)
        ddf.Time = ddf.Time.apply(to_seconds)
    
        ddf['radius'] = ddf.Longitude * ddf.Longitude + ddf.Latitude * ddf.Latitude
        ddf['alpha'] = (ddf.Longitude / ddf.Latitude).apply(math.atan)
    
        res = create_frame(ddf)
        
        trace.append([bad_choice_ind[i]] + res)
    test_frame = pd.DataFrame(trace)
    
    columns = ['id', 'mse_hei', 'mse_time', 'mse_rad',
                 'mae_hei', 'mae_time', 'mae_rad', 'mae_alpha',
                 'cnt', 'a', 'b', 'c', 'd']
    test_frame.columns = columns
    test_frame.fillna(0, inplace=True)
    
    #prediction = forest_model.predict(test_frame)
    pred_proba_0 = forest_model.predict_proba(test_frame)[:, 0]
    pred_proba_1 = forest_model.predict_proba(test_frame)[:, 1]
    test_frame['pred_proba_0'] = pred_proba_0
    test_frame['pred_proba_1'] = pred_proba_1
    #good_trace = test_frame[pred_proba >= 1/2].id
    #bad_trace  = test_frame[pred_proba < 1/2].id
    #return good_trace.values, bad_trace.values
    return test_frame[['id', 'pred_proba_0', 'pred_proba_1']]