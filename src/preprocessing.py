import pandas as pd
import numpy as np
import sys

def read_dataset(filename):
    """Reads the dataset from the given filename and returns it as a pandas dataframe.

    Args:
        filename (string): The name of the file to read.

    Returns:
        pandas.DataFrame: The dataset as a pandas dataframe.
    """
    return pd.read_csv(filename)


def build_differencing_ts(data, lag=1):
    """Builds the differencing time series for the given data.

    Args:
        data (numpy.array): The data to build the differencing time series for.
        lag (int, optional): The lag to use for the differencing. Defaults to 1.

    Returns:
        tuple: A tuple containing the differencing time series and the corresponding initial values.
    """
    diff_series = data[lag:] - data[:-lag]
    diff_series = np.concatenate((diff_series, np.zeros(lag)))

    return diff_series



def build_dataset(filename, features, start_train, end_train, start_test, end_test, forecast_horizon, index_name="datetime"):
    """Builds the dataset from the given filename.

    Args:
        filename (string): The name of the file to read.

    Returns:
        numpy.array: The dataset as a numpy array.
    """
    dataset = read_dataset(filename)
    dataset = dataset.set_index(index_name)

    # Select features
    dataset = dataset[features]

    fh = forecast_horizon
    for lag in range(1, fh + 1):
        precio = dataset["precio_electricidad"].values
        diff_series = build_differencing_ts(precio, lag)
        dataset["precio_electricidad_lag_{}".format(lag)] = diff_series

    dataset = dataset[:-fh]

    train, test = train_test_split(dataset, start_train, end_train, start_test, end_test)
    
    initial_values = train["precio_electricidad"].values[-fh:] # Initial values for the test dataset

    train = train.drop(columns=["precio_electricidad"])
    test = test.drop(columns=["precio_electricidad"])


    # Transform to numpy array 
    train = train.to_numpy()
    test = test.to_numpy()

    return train, test, initial_values

def train_test_split(dataset, start_train, end_train, start_test, end_test):
    """Splits the dataset into train and test datasets.

    Args:
        dataset (numpy.array): The dataset to split.
        start_train (string): The start date for the train dataset.
        end_train (string): The end date for the train dataset.
        start_test (string): The start date for the test dataset.
        end_test (string): The end date for the test dataset.

    Returns:
        numpy.array: The train dataset.
        numpy.array: The test dataset.
    """
    train = dataset[start_train:end_train]
    if end_test == " ":
        test = dataset[start_test:]
    else:
        test = dataset[start_test:end_test]
    return train, test

def normalize_data(data, params, method, feature=None):
    """Normalizes the data using the given method.

    Args:
        data (numpy.array): The data to normalize.
        params (dict): The parameters for the normalization.
        method (string): The method to use for normalization.
        feature (int): The feature to normalize. Defaults None, normalizes all features.

    Returns:
        numpy.arrays: The normalized data.
    """
    if feature is not None:
        mean = params["mean"][feature]
        std = params["std"][feature]
        min = params["min"][feature]
        max = params["max"][feature]
    
    else: 
        mean = params["mean"]
        std = params["std"]
        min = params["min"]
        max = params["max"]

    if method == "zscore":
        return (data - mean) / std
    
    elif method == "minmax":
        return (data - min) / (max - min)
    
    else:
        raise ValueError("Unknown normalization method.")
    

def denormalize_data(data, params, method, feature=None):
    """Denormalizes the data using the given method.

    Args:
        data (numpy.array): The data to denormalize.
        params (dict): The parameters for the denormalization.
        method (string): The method to use for denormalization. 
        feature (int): The feature to denormalize. Defaults None, denormalizes all features.

    Returns:
        numpy.array: The denormalized data.
    """

    if feature is not None:
        mean = params["mean"][feature]
        std = params["std"][feature]
        min = params["min"][feature]
        max = params["max"][feature]
    
    else: 
        mean = params["mean"]
        std = params["std"]
        min = params["min"]
        max = params["max"]

    if method == "zscore":
        return data * std + mean
    
    elif method == "minmax":
        return data * (max - min) + min
    
    else:
        raise ValueError("Unknown denormalization method.")
    

def get_normalization_params(data):
    """Calculates the normalization parameters for the given data.
        (Parameters for each feature)

    Args:
        data (numpy.array): The data to calculate the normalization parameters for.

    Returns:
        dict: The normalization parameters.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    return {"mean": mean, "std": std, "min": min, "max": max}


def normalize_dataset(train, test, norm_method):
    """Normalizes the train and test datasets using the given normalization method.

    Args:
        train (numpy.array): The train dataset.
        test (numpy.array): The test dataset.
        norm_method (string): The normalization method to use.

    Returns:
        numpy.array: The normalized train dataset.
        numpy.array: The normalized test dataset.
    """
    norm_params = get_normalization_params(train)
    train_norm = normalize_data(train, norm_params, norm_method)
    test_norm = normalize_data(test, norm_params, norm_method)
    return train_norm, test_norm, norm_params


def build_preprocessing_window(train, test, future_variables, past_history, forecast_horizon, delay):
    """Builds the preprocessing window for the given train and test datasets.

    Args:
        train (numpy.array): The train dataset.
        test (numpy.array): The test dataset.
        future_variables (list): The future variables to use for prediction.
        past_history (int): The number of past timesteps to use for prediction.
        forecast_horizon (int): The number of future timesteps to predict.
        delay (int): The delay to use for the prediction.

    Returns:
        numpy.array: The train dataset with the preprocessing window.
        numpy.array: The test dataset with the preprocessing window.
    """
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_train_future = []
    x_test_future = []
    
    for i in range(past_history, len(train) - delay):
        x_train.append(train[i - past_history:i])
        y_train.append(train[i + delay, -forecast_horizon:])
        
        if len(future_variables) > 0:
            x_train_future.append(train[i + delay:i + delay + forecast_horizon][:, future_variables]) 

    for i in range(past_history, len(test) - delay):
        x_test.append(test[i - past_history:i])
        y_test.append(test[i + delay, -forecast_horizon:])
        
        if len(future_variables) > 0:
            x_test_future.append(test[i + delay:i + delay + forecast_horizon][:, future_variables])
            
    x_train, y_train = np.array(x_train[:-24]), np.array(y_train[:-24])
    """
    if len(future_variables) > 0:
        x_train = np.delete(x_train, 0, axis=2)

    
    if len(future_variables) > 0:
        x_test = np.delete(x_test, 0, axis=2)
    
    """
    x_test, y_test = np.array(x_test[:-24]), np.array(y_test[:-24])
    x_train_future = np.array(x_train_future[:-24])
    x_test_future = np.array(x_test_future[:-24])
    
    if len(future_variables) > 0:
        
        return (x_train, x_train_future) , y_train, (x_test, x_test_future), y_test
    else:
        return (x_train, ) , y_train, (x_test, ), y_test


def read_data(filename, features, future_variables, start_train, end_train, 
              start_test, end_test, past_history, forecast_horizon, delay, norm_method):
    """Reads the data from the given filename and builds the preprocessing window.

    Args:
        filename (string): The name of the file to read.
        features (list): The features to use for prediction.
        future_variables (list): The future variables to use for prediction.
        start_train (string): The start date for the train dataset.
        end_train (string): The end date for the train dataset.
        start_test (string): The start date for the test dataset.
        end_test (string): The end date for the test dataset.
        past_history (int): The number of past timesteps to use for prediction.
        forecast_horizon (int): The number of future timesteps to predict.
        predictor (string): The variable to predict.
        delay (int): The delay to use for the prediction.
        norm_method (string, optional): The normalization method to use. 

    Returns:
        numpy.array: The train dataset with the preprocessing window.
        numpy.array: The test dataset with the preprocessing window.
        dict: The normalization parameters.
    """
    train, test, initial_values = build_dataset(filename, features, start_train, end_train, start_test, end_test, forecast_horizon)
    train_norm, test_norm, norm_params = normalize_dataset(train, test, norm_method)
    x_train, y_train, x_test, y_test = build_preprocessing_window(train_norm, test_norm, future_variables,
                                                                  past_history, forecast_horizon,delay)
    
    
    y_test_denorm = np.zeros(y_test.shape)
    for i in range(y_test.shape[0]):
        y_test_denorm[i] = denormalize_data(
            y_test[i], norm_params, norm_method, -1
        )
    x_test_denorm = denormalize_data(x_test[0], norm_params, norm_method)
    
    print("TRAINING DATA")
    print("Input shape:", x_train[0].shape)
    print("Output shape:", y_train.shape)

    print("\nTEST DATA")
    print("Input shape:", x_test[0].shape)
    print("Output shape:", y_test.shape)

    return x_train, y_train, x_test, x_test_denorm, y_test, y_test_denorm, norm_params, initial_values