import numpy as np
from matplotlib import pyplot as plt
from preprocessing import denormalize_data

def error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """ Percentage error """
    return error(actual, predicted) / actual

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(error(actual, predicted)))

def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(error(actual, predicted)))

def mape(actual: np.ndarray, predicted: np.ndarray):
    """Mean Absolute Percentage Error"""
    return 100*np.mean(np.abs(percentage_error(actual, predicted)))

METRICS = {
    "mse": mse,
    "mae": mae,
    "mape": mape,
}


def rebuild_original_ts_column(actual_original: np.ndarray, predicted_original: np.ndarray, 
                               actual: np.ndarray, predicted: np.ndarray, 
                               initial_values: np.ndarray, i: int):
    
    
    fh = len(initial_values)

    actual_original[:fh, i] = initial_values
    predicted_original[:fh, i] = initial_values
    for j in range(fh, actual.shape[0]):
        actual_original[j, i] = actual_original[j-fh, i] + actual[j]
        predicted_original[j, i] =  actual_original[j-fh, i] + predicted[j]

    return actual_original, predicted_original

def rebuild_original_ts(actual: np.ndarray, predicted: np.ndarray, initial_values: np.ndarray):

    actual_original = np.zeros(actual.shape)
    predicted_original = np.zeros(predicted.shape)

    for i in range(actual.shape[1]):
        actual_original, predicted_original = rebuild_original_ts_column(actual_original, 
                                                                        predicted_original, 
                                                                        actual[:, i], 
                                                                        predicted[:, i], 
                                                                        initial_values[:i+1], 
                                                                        i)

    fh = len(initial_values)
    actual_original = actual_original[fh:, :]
    predicted_original = predicted_original[fh:, :]
    return actual_original, predicted_original


def evaluate(
    x_test_denorm: np.ndarray,    
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics, 
    initial_values: np.ndarray,
):
    results = {}
    cont = 0

    actual, predicted = rebuild_original_ts(actual, predicted, initial_values)
    for name in metrics:
        try:
            res = []

            for i, (y, o) in enumerate(zip(actual, predicted)):
                res.append(METRICS[name](y, o))

                if (cont <0):  
                    precio = x_test_denorm[i, :, -1]
                    oo = np.concatenate((precio, o))
                    yy = np.concatenate((precio, y))
                    plt.plot(yy, label="real")
                    plt.plot(oo, label="predicted")
                    plt.title("{}: {}".format(name, METRICS[name](y, o)))
                    plt.legend()
                    plt.show()
                    cont += 1

            results[name] = np.mean(res)

        except Exception as err:
            results[name] = np.nan
            print("Unable to compute metric {0}: {1}".format(name, err))

    return results