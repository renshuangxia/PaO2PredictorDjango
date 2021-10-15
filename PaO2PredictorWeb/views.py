'''
    Actual function for making predictions
    @Author: Shuangxia Ren
'''

import os
import keras
import numpy as np
from django.shortcuts import render
from joblib import load
from keras import backend

wd = os.path.join(os.path.dirname(os.path.realpath(__file__)))

def home(request):
    return render(request, 'index.html')

# Read input entries
def read_input(request):
    inputs = []
    global use_3_features, filter_sample

    use_3_features = request.GET.get('cb_feature_num') == 'on'
    filter_sample = request.GET.get('cb_exclude_sample') == 'on'

    try:
        inputs.append(float(request.GET['spo2']))
        inputs.append(float(request.GET['fio2']))
        inputs.append(float(request.GET['peep']))
    except:
        return "Unable to get input values, this may caused by missing required value."

    if not use_3_features:
        try:
            inputs.append(float(request.GET['vt']))
            inputs.append(float(request.GET['map']))
            inputs.append(float(request.GET['temp']))
            inputs.append(float(request.GET['vaso']))
        except:
            return "Error: Unable to process input values, please double check you input values"
    return np.array(inputs)


# Load pretrained scalers
def load_scalers():

    print('use 3 features? ', str(use_3_features))
    print('filter_sample? ', str(filter_sample))

    scaler_dir = 'saved_models/'
    scaler_dir += 'regressor/'
    scaler_dir += 'FilteredSpO2/' if filter_sample else 'NoSpO2Filtering/'

    in_scaler_name = 'InputScaler_'
    in_scaler_name += '3_features' if use_3_features else '7_features'

    out_scaler_name = 'OutputScaler_'
    out_scaler_name += '3_features' if use_3_features else '7_features'

    print(in_scaler_name, out_scaler_name)

    input_scaler = load(os.path.join(wd, scaler_dir + in_scaler_name + '.joblib'))
    output_scaler = load(os.path.join(wd, scaler_dir + out_scaler_name + '.joblib'))
    return input_scaler, output_scaler


# Load model
def load_model(request):
    model_dir = 'saved_models/'
    model_name = str(request.GET['model'])

    model_dir += 'regressor/'
    model_dir += 'FilteredSpO2/' if filter_sample else 'NoSpO2Filtering/'

    model_surfix = '_3_features' if use_3_features else '_7_features'
    model_name = model_name.split(' ')[0]
    model_path = model_dir + model_name + model_surfix

    if 'Neural_Network' not in model_name:
        model = load(os.path.join(wd, model_path + '.joblib'))
    else:
        model = keras.models.load_model(os.path.join(wd, model_path + '.ckpt'), custom_objects={'rmse': rmse})
    return model


# custom method for generating predictions
def getPredictions(request):
    in_features = read_input(request)
    if type(in_features) == str:
        return in_features

    model = load_model(request)
    input_scaler, output_scaler = load_scalers()

    # 3 features scaling
    if use_3_features:
        in_features = np.expand_dims(in_features, axis=0)
        in_features = input_scaler.transform(in_features)
    else:
        # 7 feature scaling
        other_features = in_features[0:-1]  # exclude vaso feature
        other_features = np.expand_dims(other_features, axis=0)
        other_features = input_scaler.transform(other_features).squeeze()
        in_features[0:-1] = other_features
        in_features = np.expand_dims(in_features, axis=0)

    pred = model.predict(in_features)

    if 'Neural_Network' in str(request.GET['model']):
        backend.clear_session()

    # Regression Model predicts PaO2 Value
    pred = np.expand_dims(pred, axis=0)
    pred = output_scaler.inverse_transform(pred).squeeze()
    pred = np.round(pred, decimals=2)
    print('Prediction PaO2 Value:', pred)
    return pred



def rmse(y_true,y_predict):
    return backend.sqrt(backend.mean(backend.square(y_predict-y_true),axis=-1))


# our result page view
def result(request):
    result = getPredictions(request)
    if type(result) is not str and result < 0:
        result = 'Negative prediciton:' + str(result) + ' (Are you using linear regression with an out-of-range input value?)'
    return render(request, 'result.html', {'result': result})


