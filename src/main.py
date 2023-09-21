import json
import itertools
import time
import os
import numpy as np
import pandas as pd
from metrics import evaluate
from preprocessing import denormalize_data, read_data
from models import create_model_ml
import sklearn 
import xgboost
import matplotlib 




def read_results_file(csv_filepath, metrics):
    try:
        results = pd.read_csv(csv_filepath, sep=";", index_col=0)
    except IOError:
        results = pd.DataFrame(
            columns=[
                "FEATURES",
                "TRAIN DATE",
                "TEST DATE",
                "MODEL",
                "MODEL_DESCRIPTION",
                "FORECAST_HORIZON",
                "PAST_HISTORY",
                "NORMALIZATION",
                "TRAINING_TIME",
                "TEST_TIME",
                *metrics,
            ]
        )
    return results


def train_ml(model_name, iter_params, x_train, y_train, x_test, norm_params, normalization_method):
    model = create_model_ml(model_name, iter_params)

    x_train1 = x_train[0] 
    #x_train2 = x_train[1] 

    x_trainf = x_train1.reshape(x_train1.shape[0], x_train1.shape[1] * x_train1.shape[2])
    #x_train2 = x_train2.reshape(x_train2.shape[0], x_train2.shape[1] * x_train2.shape[2])
    #x_trainf = np.concatenate((x_trainf, x_train2), axis=1)
    print('x_train: {} -> {}'.format(x_train1.shape, x_trainf.shape))

    training_time_0 = time.time()
    model.fit(x_trainf, y_train)
    training_time = time.time() - training_time_0

    x_test2 = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    print('x_test: {} -> {}'.format(x_test.shape, x_test2.shape))

    test_time_0 = time.time()
    test_forecast = model.predict(x_test2)
    test_time = time.time() - test_time_0

    for i in range(test_forecast.shape[0]):
        test_forecast[i] = denormalize_data(
            test_forecast[i], norm_params, method=normalization_method, feature=-1
        )

    return test_forecast, training_time, test_time


def main_ml(parameters_path, results_path):
    
    TRAIN_ML = {
        'xgb': train_ml,
        'rf': train_ml,
        'm5p': train_ml,
        
    }
    with open(parameters_path, "r") as params_file:
                parameters = json.load(params_file)

    dataset_path = parameters['dataset_path']
    models_ml = parameters['models_ml']
    metrics = parameters['metrics']
    delay = parameters['delay']
    start_train = parameters['start_train']
    end_train = parameters['end_train']
    start_test = parameters['start_test']
    end_test = parameters['end_test']
    future_variables_list = parameters['future_variables']

    print("Variable a predecir debe estar en última columna del dataset \n")
    
    for i, features in enumerate(parameters['features']):
        future_variables = future_variables_list[i]        
        for model_name in models_ml:
            for normalization_method, past_history, forecast_horizon, \
            start_train, end_train, start_test, end_test in itertools.product(
                    parameters['normalization_method'],
                    parameters['past_history'],
                    parameters['forecast_horizon'],
                    parameters['start_train'],
                    parameters['end_train'],
                    parameters['start_test'],
                    parameters['end_test']

            ): 
                csv_filepath = '{}/results_ml.csv'.format(results_path)
                results = read_results_file(csv_filepath, metrics)

                x_train, y_train, x_test, x_test_denorm, y_test, y_test_denorm, norm_params = read_data(dataset_path, 
                features,
                future_variables,
                start_train,
                end_train,
                start_test,
                end_test, 
                past_history, 
                forecast_horizon, 
                delay, 
                normalization_method)

                
                past_history = x_test.shape[1]
                forecast_horizon = y_test.shape[1]

                parameters_models = parameters['model_params'][model_name]
                list_parameters_models = []
                for parameter_field in parameters_models.keys():
                    list_parameters_models.append(parameters_models[parameter_field])

                for iter_params in itertools.product(*list_parameters_models):
                    test_forecast, training_time, test_time = TRAIN_ML[model_name](
                        model_name,
                        iter_params,
                        x_train,
                        y_train,
                        x_test,
                        norm_params,
                        normalization_method
                    )

                    if metrics:
                        test_metrics = evaluate(x_test_denorm, y_test_denorm, test_forecast, metrics)
                    else:
                        test_metrics = {}

                    num_features = len(features)
                    data_date = "Train: {} - {} | Test: {} - {}".format(
                        start_train, end_train, start_test, end_test
                    )
                    prediction_path = '{}/{}/{}/{}/{}/{}/'.format(
                        results_path,
                        data_date,
                        num_features,
                        normalization_method,
                        str(past_history),
                        model_name,
                    )

                    if not os.path.exists(prediction_path):
                        os.makedirs(prediction_path)

                    np.save(prediction_path + "real" + '.npy', y_test_denorm)
                    np.save(prediction_path + "predicted" + '.npy', test_forecast)

                    iter_params_str = "".join(str(iter_params).split(','))
                    features_str = "".join(str(features).split(','))
                    results = results._append(
                        {
                            "FEATURES": features_str,
                            "TRAIN DATE": start_train + ' - ' + end_train,
                            "TEST DATE": start_test + ' - ' + end_test,
                            "MODEL": model_name,
                            "MODEL_DESCRIPTION": iter_params_str,
                            "FORECAST_HORIZON": forecast_horizon,
                            "PAST_HISTORY": past_history,
                            "NORMALIZATION": normalization_method,
                            "TRAINING_TIME": training_time,
                            "TEST_TIME": test_time,
                            **test_metrics,
                        },
                        ignore_index=True
                    )

                    print('\nEND OF EXPERIMENT -> {}/{}/{}/{}/{}/{} \n\n'.format(
                        results_path,
                        data_date,
                        num_features,
                        normalization_method,
                        past_history,
                        model_name
                    ))
                    
                    results.to_csv(csv_filepath, sep=";")



if __name__ == '__main__':
    parameters_path = './parameters.json'
    output_path = '../results'
    main_ml(parameters_path, output_path)
    
    